import contextlib
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
from src.serving.intelligent23 import IntelligentRegistry
from src.assistant.schema import AssistantRequest


class M23ServingTests(unittest.TestCase):
    def test_unconfigured_or_corrupt_optional_m23_is_unavailable(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'registry.json'
            registry=IntelligentRegistry({'r':object()},path)
            self.assertFalse(registry.available('r'))
            path.write_text(json.dumps({'r':{'root':root,'report_sha256':'a'*64}}))
            self.assertTrue(registry.configured('r'))
            self.assertEqual(registry.validation_state('r'),'pending_validation')
            self.assertFalse(registry.available('r'))
            with self.assertRaises(HTTPException) as error:registry.load('r')
            self.assertEqual(error.exception.status_code,503)
            self.assertIn('existing modes remain available',error.exception.detail)

    def test_m23_request_accepts_version_and_rejects_m21_experiment(self):
        import uuid
        request=AssistantRequest(question='How many passes?',session_id=uuid.uuid4(),cv_pipeline='m23')
        self.assertEqual(request.cv_pipeline,'m23')
        with self.assertRaises(ValidationError):AssistantRequest(**(request.model_dump()|{'cv_experiment':'temporal_offline'}))

    def test_assistant_isolates_pipeline_source_and_all_artifact_revisions(self):
        from src.assistant.service import UnifiedMatchAssistant
        service=object.__new__(UnifiedMatchAssistant)
        service.indexes={};service.loaded={};service.cv={};service.cv_variants={}
        revision=['first'];layers={}
        def resolver(recording):
            layer=layers.setdefault((recording,revision[0]),SimpleNamespace(manifest={'events_sha256':'same-events'}))
            return SimpleNamespace(ball_review=SimpleNamespace(manifest={'detector_sha256':revision[0],'ball_frames_sha256':'same-observations'}),event_review=layer)
        service.cv_m23_resolver=resolver;service.cv_m22_resolver=resolver
        service.catalog=lambda recording:object();service.recording_lock=lambda recording:contextlib.nullcontext()
        class Index:
            def __init__(self,*args):pass
            def attach_cv(self,layer):self.layer=layer
        with patch('src.assistant.service.AssistantEventIndex',Index):
            first=service.index('r','m23');self.assertIs(first,service.index('r','m23'))
            m22=service.index('r','m22');self.assertIsNot(first,m22)
            revision[0]='second';second=service.index('r','m23');other=service.index('s','m23')
            self.assertIsNot(first,second);self.assertIsNot(second,other)
            self.assertNotEqual(first.cv_model_revision,second.cv_model_revision)
            self.assertIs(second.layer,layers[('r','second')])

    def test_current_pixel_frame_schema_rejects_prediction_masquerade(self):
        from src.vision.intelligent23.schema import IntelligentFrame
        row={'schema_version':'cv-m23-frame-v1','recording_id':'synthetic','variant':'full_m23',
            'frame':2,'timestamp':.08,'camera_epoch':0,'state':'UNKNOWN','directly_observed':False,
            'selected_candidate_id':None,'candidates':[],'memory':None,'reconstruction':None,
            'trajectory_history':[],'interaction':{},'source_pixels_sha256':'a'*64,
            'model_set_sha256':'b'*64,'config_sha256':'c'*64,
            'uses_future_for_association':True,'future_visual_frames_used':False,'m22_context_frames':[0,1,2],'reasons':['NO_VISUAL_EVIDENCE']}
        IntelligentFrame.model_validate(row)
        for change in [{'directly_observed':True,'state':'OBSERVED'}, {'timestamp':float('nan')},
            {'m22_context_frames':[0,2,3]}, {'reconstruction':{'image':[2.,3.],'directly_observed':True,'support_frames':[1,3],'uses_future':True,'reason':'inferred'}}]:
            with self.subTest(change=change),self.assertRaises(ValidationError):IntelligentFrame.model_validate(row|change)

    def test_custom_m22_crop_is_bounded_deduplicated_and_current_verified(self):
        import hashlib
        import numpy as np
        import torch
        from src.vision.football22.inference import FootballDetector
        class EmptyModel:
            def __call__(self,x):
                n,_,h,w=x.shape
                return {'heatmap':torch.full((n,1,h//4,w//4),-20.),'offset':torch.zeros(n,2,h//4,w//4),'size':torch.zeros(n,2,h//4,w//4)}
        detector=object.__new__(FootballDetector)
        detector.device='cpu';detector.model=EmptyModel();detector.batch_size=4;detector.checkpoint_sha256='d'*64
        detector.config={'tile_source_size':64,'tile_overlap':.2,'input_hw':[64,96],
            'candidate_threshold':.02,'current_verification_match_iou':.1,
            'observation_threshold':.35,'current_only_verification_threshold':.35,'maximum_candidates_per_frame':20}
        images=[np.zeros((64,96,3),np.uint8)]*3
        out=detector.infer(images,2,regions=[(0,0,32,32)]*2)
        self.assertEqual(out['inference_views'],2);self.assertEqual(out['candidates'],[])
        self.assertEqual(out['current_frame_pixels_sha256'],hashlib.sha256(images[-1].tobytes()).hexdigest())
        self.assertGreater(detector.infer(images,2)['inference_views'],2)
        with self.assertRaises(ValueError):detector.infer(images,2,regions=[(90,0,30,30)])


if __name__=='__main__':unittest.main()
