"""Strict M21 provenance, optional availability and isolated review/version contracts."""
import contextlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from scripts.analyze_cv_m21 import write_variant
from src.serving.ball import BallReview
from src.serving.cv import digest
from src.serving.advanced21 import AdvancedRegistry
from src.vision.events.passes import signature
from src.vision.ball.schema import BallFrame
from src.assistant.schema import AssistantRequest


class AdvancedArtifactTests(unittest.TestCase):
    def make(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name)/'ball'
        metadata={'recording_id':'r','frames':2,'fps':25.,'width':100,'height':100,'video_sha256':'v'*64,'frames_sha256':'w'*64}
        balls=[];debug=[]
        for i in range(2):
            c={'candidate_id':'c','bbox':[10.+i,20.,8.,8.],'confidence':.9,'inference_view':'current','source':'VIDEO_CV_DETECTOR'}
            ball=BallFrame(version='m21',frame=i,timestamp=i/25,camera_epoch=0,directly_observed=True,
                detection_status='DETECTED',tracking_status='TRACKED',image_position=[14.+i,24.],image_bbox=c['bbox'],
                detection_confidence=.9,tracking_confidence=.9,candidates=[c],candidate_count=1,track_id=1)
            balls.append(ball)
            debug.append({'schema_version':'cv-m21-debug-v1','variant':'unit','frame':i,'timestamp':i/25,'camera_epoch':0,
                'association_mode':'offline','uses_future_for_ball_association':True,'event_analysis_mode':'offline_with_original_M9_participants',
                'selected_candidate_id':'c','directly_observed':True,'candidates':[{'candidate_id':'c','bbox':c['bbox'],
                    'visual_confidence':.9,'inference_view':'current','selected':True,'deterministic':{},'temporal_score':None,
                    'temporal_features':{},'score_calibrated':False}], 'reconstruction':None,'camera_refinement':{},
                'player_tracks':[],'manual_identity_propagated':False})
        write_variant(directory,'r',metadata,balls,[],{},[],debug,{'variant':'unit'},'d'*64)
        return directory,metadata

    def rewrite_debug(self,directory,change):
        path=directory/'advanced_frames.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        offsets=[]
        with path.open('wb') as f:
            for row in rows:offsets.append(f.tell());f.write((json.dumps(row)+'\n').encode())
        p=directory/'manifest.json';manifest=json.loads(p.read_text());manifest.update(advanced_sha256=digest(path),advanced_frame_offsets=offsets);p.write_text(json.dumps(manifest))

    def test_observed_diagnostics_match_current_source_and_canonical(self):
        directory,metadata=self.make();review=BallReview(directory,metadata)
        self.assertTrue(review.advanced.frame(1).directly_observed)
        self.assertEqual(review.frame(1).image_position,(15.,24.))

    def test_corrupt_debug_hash_is_rejected(self):
        directory,metadata=self.make();(directory/'advanced_frames.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):BallReview(directory,metadata)

    def test_rehashed_visual_confidence_forgery_is_rejected(self):
        directory,metadata=self.make()
        self.rewrite_debug(directory,lambda rows:rows[0]['candidates'][0].update(visual_confidence=.4))
        with self.assertRaisesRegex(ValueError,'confidence'):BallReview(directory,metadata)

    def test_future_data_cannot_be_labelled_causal(self):
        directory,metadata=self.make()
        self.rewrite_debug(directory,lambda rows:rows[0].update(association_mode='causal_ball_only'))
        with self.assertRaises(ValueError):BallReview(directory,metadata)

    def test_fusion_proposal_without_current_verification_is_rejected(self):
        directory,metadata=self.make()
        self.rewrite_debug(directory,lambda rows:rows[0]['candidates'][0].update(inference_view='PAST_FRAME_FUSION_PROPOSAL_NOT_OBSERVED'))
        with self.assertRaisesRegex(ValueError,'Fusion'):BallReview(directory,metadata)

    def test_runtime_diagnostic_change_blocks_ball_queries(self):
        directory,metadata=self.make();review=BallReview(directory,metadata)
        (directory/'advanced_frames.jsonl').write_text('{}\n')
        with self.assertRaises(HTTPException):review.frame(0)

    def test_optional_registry_missing_or_malformed_has_no_fallback(self):
        registry=AdvancedRegistry({'r':object()},path='/nonexistent-m21-registry')
        with self.assertRaises(HTTPException) as error:registry.load('r')
        self.assertEqual(error.exception.status_code,503)
        registry.entries={'r':{'root':None}}
        with self.assertRaises(HTTPException) as error:registry.load('r')
        self.assertEqual(error.exception.status_code,503)


class IndependentReviewM21Tests(unittest.TestCase):
    def test_source_only_review_has_its_own_annotation_store_and_video_ranges(self):
        from src.serving.advanced21_review import create_review_router
        with TemporaryDirectory() as directory:
            root=Path(directory);video=root/'source.mp4';video.write_bytes(b'synthetic range transport fixture')
            source={'source_video':str(video),'video_sha256':digest(video),'frames':10,'fps':25,'width':100,'height':100}
            data={'sources':{'holdout':source},'protocol':{},'tasks':[{'task_id':'ball','recording_id':'holdout','kind':'ball','frame':2}],
                  'selection_uses_model_candidates':False}
            data['fingerprint']=signature(data);task_path=root/'tasks.json';task_path.write_text(json.dumps(data))
            router,store=create_review_router(lambda:None,task_path,root/'new-labels')
            app=FastAPI();app.include_router(router);client=TestClient(app)
            response=client.get('/cv/m21/matches/holdout/cv/review/tasks')
            self.assertEqual(response.status_code,200);self.assertFalse(response.json()['selection_uses_model_candidates'])
            self.assertEqual(response.json()['labels'],[])
            response=client.get('/cv/m21/matches/holdout/video.mp4',headers={'Range':'bytes=0-3'})
            self.assertEqual(response.status_code,206);self.assertEqual(response.content,b'synt')
            self.assertFalse((root/'new-labels/labels.json').exists())
            self.assertEqual(client.get('/cv/m21/matches/../video.mp4').status_code,404)


class AssistantVersionM21Tests(unittest.TestCase):
    def test_experiment_cannot_leak_to_legacy_pipelines_or_escape_registered_names(self):
        from pydantic import ValidationError
        import uuid
        request={'question':'How many passes?','session_id':str(uuid.uuid4())}
        for extra in [{'cv_pipeline':'m20','cv_experiment':'offline_dag'},{'cv_pipeline':'m21','cv_experiment':'../../secret'}]:
            with self.assertRaises(ValidationError):AssistantRequest(**request,**extra)
        self.assertEqual(AssistantRequest(**request,cv_pipeline='m21',cv_experiment='m20_control').cv_experiment,'m20_control')

    def test_event_indexes_are_isolated_by_explicit_experiment(self):
        from src.assistant.service import UnifiedMatchAssistant
        service=object.__new__(UnifiedMatchAssistant)
        service.indexes={};service.loaded={};service.cv={};service.cv_variants={}
        layers={'one':object(),'two':object()}
        service.cv_resolver=lambda recording,experiment:SimpleNamespace(event_review=layers[experiment])
        service.catalog=lambda recording:object();service.recording_lock=lambda recording:contextlib.nullcontext()
        class Index:
            def __init__(self,*args):pass
            def attach_cv(self,layer):self.cv_layer=layer
        with patch('src.assistant.service.AssistantEventIndex',Index):
            first=service.index('r','m21','one');second=service.index('r','m21','two')
            self.assertIsNot(first,second);self.assertIs(first.cv_layer,layers['one']);self.assertIs(second.cv_layer,layers['two'])
            self.assertIs(service.index('r','m21','one'),first)


if __name__=='__main__':unittest.main()
