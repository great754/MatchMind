"""Temporal visual truth, source mapping, review isolation and optional-mode safety."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from fastapi import HTTPException
from pydantic import ValidationError
from src.vision.football22.model import TemporalFootballNet,preprocess,targets,detection_loss,decode,context_indices,spacing
from src.vision.football22.review import DevelopmentLabel,DevelopmentReviewStore
from src.vision.football22.schema import FootballFrame
from src.vision.football22.training import split_sources
from src.serving.football22 import FootballRegistry
from src.vision.events.passes import signature


class TemporalModelM22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_start_and_cut_contexts_are_chronological_and_never_future(self):
        epochs=[0,0,1,1,1]
        self.assertEqual(context_indices(0,25,epochs),[0,0,0])
        self.assertEqual(context_indices(1,25,epochs),[0,0,1])
        self.assertEqual(context_indices(2,25,epochs),[2,2,2])
        self.assertEqual(context_indices(3,25,epochs),[2,2,3])
        self.assertEqual(context_indices(4,25,epochs),[2,3,4])
        self.assertEqual(spacing(60),2)
        with self.assertRaises(ValueError):spacing(float('nan'))

    def test_whole_portrait_source_is_preserved_by_padding(self):
        image=np.zeros((1080,700,3),np.uint8);image[-20:]=255
        x,m=preprocess([image]*3,(384,640))
        self.assertEqual(x.shape,(9,384,640));self.assertEqual(m['resized_height'],384)
        self.assertLess(m['resized_width'],640)
        self.assertGreater(float(x[-3:,380:384,:m['resized_width']].mean()),1.)

    def test_center_and_box_roundtrip_in_source_pixels(self):
        im=np.zeros((1080,1920,3),np.uint8);_,mapping=preprocess([im]*3)
        # A known source ball at 960,540 maps exactly to heatmap cell 80,45.
        out={'heatmap':torch.full((1,1,96,160),-20.),'offset':torch.zeros(1,2,96,160),'size':torch.zeros(1,2,96,160)}
        out['heatmap'][0,0,45,80]=10
        out['size'][0,:,45,80]=torch.log(torch.tensor([20/12,12/12]))
        found=decode(out,mapping)
        self.assertEqual(len(found),1);np.testing.assert_allclose(found[0]['center'],[960,540],atol=.001)
        np.testing.assert_allclose(found[0]['bbox'],[950,534,20,12],atol=.001)

    def test_padded_cells_and_source_out_of_bounds_regions_do_not_detect(self):
        im=np.zeros((360,640,3),np.uint8);_,mapping=preprocess([im]*3)
        out={'heatmap':torch.full((1,1,96,160),-20.),'offset':torch.zeros(1,2,96,160),'size':torch.zeros(1,2,96,160)}
        out['heatmap'][0,0,94,30]=10
        self.assertEqual(decode(out,mapping),[])
        with self.assertRaises(ValueError):preprocess([im]*3,region=(630,0,50,50))

    def test_unknown_visibility_has_zero_positive_and_negative_loss(self):
        im=np.zeros((64,96,3),np.uint8);_,mapping=preprocess([im]*3,(64,96))
        target={k:v[None] for k,v in targets('uncertain',None,mapping).items()}
        output={k:torch.randn(1,c,16,24,requires_grad=True) for k,c in [('heatmap',1),('offset',2),('size',2)]}
        loss,_=detection_loss(output,target);self.assertEqual(float(loss.detach()),0.);loss.backward()
        self.assertEqual(float(output['heatmap'].grad.abs().sum()),0.)
        absent={k:v[None] for k,v in targets('not_visible',None,mapping).items()}
        loss,_=detection_loss(output,absent);self.assertGreater(float(loss.detach()),0.)

    def test_nine_channel_network_trains_temporal_filters_and_has_stride_four_heads(self):
        model=TemporalFootballNet();model.eval();model.freeze_early_backbone()
        self.assertTrue(model.backbone.conv1.weight.requires_grad)
        self.assertFalse(model.backbone.layer1[0].conv1.weight.requires_grad)
        with torch.no_grad():out=model(torch.zeros(1,9,64,96))
        self.assertEqual(out['heatmap'].shape,(1,1,16,24));self.assertEqual(out['offset'].shape,(1,2,16,24))
        with self.assertRaises(ValueError):model(torch.zeros(1,3,64,96))

    def test_target_recording_cannot_enter_training_or_validation(self):
        rows=[{'recording_id':n} for n in ['test_match','team_goal','southampton']]
        train,val,split=split_sources(rows,'team_goal')
        self.assertEqual(train,[{'recording_id':'test_match'}]);self.assertEqual(val,[{'recording_id':'southampton'}])
        self.assertFalse(split['target_labels_used_for_fitting_or_early_stopping'])

    def test_two_source_temporal_validation_excludes_target_buffers_and_shared_context(self):
        from src.vision.football22.training import split_temporal_groups
        sources={n:{'fps':25.,'frames':100} for n in ['test_match','team_goal','southampton']}
        rows=[{'recording_id':n,'frame':f,'context':[{'frame':f-d} for d in [2,1,0]]} for n in sources for f in [5,20,30,50,70,90]]
        train,val,split=split_temporal_groups(rows,'team_goal',sources)
        self.assertEqual({r['recording_id'] for r in train},{'test_match','southampton'})
        self.assertEqual({r['frame'] for r in val},{20,70});self.assertFalse(any(r['frame']==30 for r in train))
        for a in train:
            for b in val:
                if a['recording_id']==b['recording_id']:
                    self.assertGreater(abs(a['frame']-b['frame'])/25,.5)
                    self.assertFalse({c['frame'] for c in a['context']} & {c['frame'] for c in b['context']})
        self.assertFalse(split['target_labels_used_for_fitting_or_early_stopping'])

    def test_hard_negative_regions_never_label_surrounding_unknown_pixels(self):
        im=np.zeros((64,96,3),np.uint8);_,mapping=preprocess([im]*3,(64,96))
        target=targets('uncertain',None,mapping,negative_regions=[[20,20,12,12]])
        self.assertGreater(float(target['valid'].sum()),0)
        self.assertEqual(float(target['valid'][0,0,0]),0.)
        self.assertEqual(float(target['heatmap'].sum()),0.)

    def test_negative_regions_outside_a_crop_do_not_wrap_around_the_mask(self):
        im=np.zeros((100,100,3),np.uint8);_,mapping=preprocess([im]*3,(384,384),(30,30,32,32))
        for box in [[24,40,4,4],[40,24,4,4],[65,40,4,4],[40,65,4,4]]:
            target=targets('uncertain',None,mapping,negative_regions=[box])
            self.assertEqual(float(target['valid'].sum()),0.)

    def test_weak_organizer_positive_has_no_unknown_background_negative_loss(self):
        import cv2
        from src.serving.cv import digest
        from src.vision.football22.training import TemporalSamples
        with TemporaryDirectory() as directory:
            path=Path(directory)/'frame.png';cv2.imwrite(str(path),np.zeros((64,96,3),np.uint8))
            row={'recording_id':'weak','frame':2,'state':'weak_positive','bbox':[20,20,8,8],
                'weak_supervision':True,'weight':.15,'context':[{'path':str(path),'sha256':digest(path),'frame':i} for i in range(3)]}
            data=TemporalSamples([row],False);_,target=data[0]
            self.assertEqual(int((target['valid']>0).sum()),1)
            self.assertAlmostEqual(float(target['valid'].sum()),.15,places=6)

    def test_human_hard_negative_crops_mask_all_unreviewed_pixels(self):
        import cv2
        from src.serving.cv import digest
        from src.vision.football22.training import TemporalSamples
        with TemporaryDirectory() as directory:
            path=Path(directory)/'frame.png';cv2.imwrite(str(path),np.zeros((360,640,3),np.uint8))
            row={'recording_id':'test_match','frame':2,'state':'visible','bbox':[500,260,12,12],
                'negative_regions':[[50,50,20,20]],'weight':1.,
                'context':[{'path':str(path),'sha256':digest(path),'frame':i} for i in range(3)]}
            dataset=TemporalSamples([row],True,repeats=24);positive=negative=0
            for i in range(len(dataset)):
                _,target=dataset[i]
                if target['mask'].sum():positive+=1
                else:
                    negative+=1;self.assertEqual(float(target['heatmap'].sum()),0.)
                    self.assertGreater(float(target['valid'].sum()),0.)
                    self.assertLess(float(target['valid'].sum()),target['valid'].numel()/10)
            self.assertGreater(negative,0);self.assertGreater(positive,0)


class DevelopmentReviewM22Tests(unittest.TestCase):
    def make(self):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        source={'width':100,'height':80,'fps':25,'frames':100,'video_sha256':'a'*64}
        task={'task_id':'t','recording_id':'team_goal','frame':10,'proposals':[{'proposal_id':'p','bbox':[10,10,5,5]}]}
        d={'reference_role':'development_only','model_proposals_are_truth':False,'sources':{n:source for n in ['test_match','team_goal','southampton']},'tasks':[task],
            'source_suitability':{},'selection':{}}
        d['fingerprint']=signature(d);p=root/'tasks.json';p.write_text(json.dumps(d))
        return DevelopmentReviewStore(p,root/'new-labels'),root

    def label(self,**extra):
        return DevelopmentLabel(task_id='t',reviewer_id='AB',state='visible',bbox=[10,10,5,5],reviewed_from_source=True,**extra)

    def test_proposals_are_not_labels_and_reads_do_not_autosave(self):
        store,root=self.make();self.assertEqual(store.labels(),[])
        self.assertFalse(store.public('team_goal')['model_proposals_are_truth']);self.assertFalse((root/'new-labels').exists())

    def test_explicit_acceptance_saves_assisted_development_provenance_and_replaces_one_reviewer_task(self):
        store,root=self.make();label=self.label(accepted_proposal='p');self.assertTrue(store.save('team_goal',label)['saved'])
        store.save('team_goal',label);rows=store.labels();self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]['model_assisted']);self.assertEqual(rows[0]['reference_role'],'development_only')
        self.assertEqual(rows[0]['source_video_sha256'],'a'*64)

    def test_rejected_region_is_not_whole_frame_absence(self):
        store,root=self.make();label=DevelopmentLabel(task_id='t',reviewer_id='AB',state='uncertain',reviewed_from_source=True,rejected_proposals=['p'])
        row=store.validate('team_goal',label);self.assertEqual(row['state'],'uncertain');self.assertEqual(row['hard_negative_regions'][0]['bbox'],[10,10,5,5])

    def test_completed_proposal_review_preserves_explicit_coverage_and_legacy_labels(self):
        store,_=self.make()
        self.assertTrue(store.public('team_goal')['supports_reviewed_proposals'])
        self.assertEqual(store.validate('team_goal',self.label())['reviewed_proposals'],[])
        store.save('team_goal',self.label(accepted_proposal='p',reviewed_proposals=['p']))
        self.assertEqual(store.labels()[0]['reviewed_proposals'],['p'])
        for ids in [['p','p'],['unknown']]:
            with self.assertRaises(ValueError):store.validate('team_goal',self.label(reviewed_proposals=ids))

    def test_wrong_source_outside_box_unknown_proposal_or_overlap_are_rejected(self):
        store,_=self.make()
        for recording,label in [('southampton',self.label()),('team_goal',self.label(accepted_proposal='unknown')),('team_goal',self.label(rejected_proposals=['p']))]:
            with self.assertRaises(ValueError):store.validate(recording,label)
        with self.assertRaises(ValueError):store.validate('team_goal',DevelopmentLabel(task_id='t',reviewer_id='AB',state='visible',bbox=[99,79,5,5],reviewed_from_source=True))
        with self.assertRaises(ValidationError):DevelopmentLabel(task_id='t',reviewer_id='AB',state='visible',bbox=[float('nan'),0,5,5],reviewed_from_source=True)

    def test_unknown_and_absent_cannot_have_boxes_or_false_source_attestations(self):
        for state in ['uncertain','not_visible']:
            with self.assertRaises(ValidationError):DevelopmentLabel(task_id='t',reviewer_id='AB',state=state,bbox=[10,10,5,5],reviewed_from_source=True)
        with self.assertRaises(ValidationError):DevelopmentLabel(task_id='t',reviewer_id='AB',state='uncertain',reviewed_from_source=False)

    def test_missing_optional_model_does_not_fall_back_to_control(self):
        registry=FootballRegistry({'r':object()},path='/nonexistent-m22-registry')
        self.assertFalse(registry.configured('r'))
        with self.assertRaises(HTTPException) as e:registry.load('r')
        self.assertEqual(e.exception.status_code,503)

    def test_assistant_accepts_version_but_not_an_m21_experiment_for_m22(self):
        import uuid
        from src.assistant.schema import AssistantRequest
        value={'question':'How many passes?','session_id':uuid.uuid4(),'cv_pipeline':'m22'}
        self.assertEqual(AssistantRequest(**value).cv_pipeline,'m22')
        with self.assertRaises(ValidationError):AssistantRequest(**value,cv_experiment='offline_dag')


class EvidenceIsolationM22Tests(unittest.TestCase):
    def fixture(self):
        from src.serving.cv import digest
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        checkpoint=root/'model.pt';checkpoint.write_bytes(b'local fixture, not a neural model')
        row={'schema_version':'cv-m22-frame-v1','recording_id':'r','frame':0,'timestamp':0.,'camera_epoch':0,
            'context_frames':[0,0,0],'checkpoint_sha256':digest(checkpoint),'directly_observed':True,
            'candidates':[{'candidate_id':'c','bbox':[10.,10.,5.,5.],'confidence':.8,'current_only_confidence':.7,
                'observation_eligible':True,'decision_reason':'CURRENT_PIXELS','selected':True}],
            'heatmap':[[0.]*40 for _ in range(24)],'heatmap_is_calibrated':False,
            'association':'existing_m21_offline_dag','uses_future_for_association':True,'reconstruction':None,'interaction':{}}
        path=root/'football_frames.jsonl';path.write_text(json.dumps(row)+'\n')
        config={'observation_threshold':.35,'current_only_verification_threshold':.35}
        ball=SimpleNamespace(directory=root,manifest={'recording_id':'r','frames':1,'width':100,'height':80,
            'football_frame_offsets':[0],'football_sha256':digest(path),'config':config,'config_sha256':signature(config),
            'trained_checkpoint':str(checkpoint),'detector_sha256':digest(checkpoint)},
            frame=lambda i:SimpleNamespace(timestamp=0.,camera_epoch=0,directly_observed=True,image_bbox=(10.,10.,5.,5.),detection_confidence=.8))
        return ball,row,path,checkpoint

    def test_diagnostic_hash_and_rehashed_observation_forgery_are_rejected(self):
        from src.serving.cv import digest
        from src.serving.football22 import FootballReview
        ball,row,path,_=self.fixture();self.assertTrue(FootballReview(ball).frame(0).directly_observed)
        row['candidates'][0]['confidence']=.9;path.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError,'mismatch'):FootballReview(ball)
        ball.manifest['football_sha256']=digest(path)
        with self.assertRaisesRegex(ValueError,'differs from observation'):FootballReview(ball)

    def test_checkpoint_modification_is_rejected_at_load_and_after_loading(self):
        from src.serving.football22 import FootballReview
        ball,_,_,checkpoint=self.fixture();review=FootballReview(ball);checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):review.frame(0)
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):FootballReview(ball)

    def test_inferred_or_unverified_candidates_cannot_be_direct_observations(self):
        _,row,_,_=self.fixture();row['candidates'][0]['current_only_confidence']=.1
        with self.assertRaises(ValidationError):FootballFrame.model_validate(row)
        row['candidates'][0]['current_only_confidence']=.7;row['reconstruction']={'image':[12.,12.]}
        with self.assertRaises(ValidationError):FootballFrame.model_validate(row)

    def test_assistant_indexes_are_isolated_by_artifact_revision_and_recording(self):
        import contextlib
        from unittest.mock import patch
        from src.assistant.service import UnifiedMatchAssistant
        service=object.__new__(UnifiedMatchAssistant);service.indexes={};service.loaded={};service.cv={};service.cv_variants={}
        revision=['first'];layers={}
        def resolver(recording):
            layer=layers.setdefault((recording,revision[0]),SimpleNamespace(manifest={'events_sha256':'same-events'}))
            return SimpleNamespace(ball_review=SimpleNamespace(manifest={'detector_sha256':revision[0],'ball_frames_sha256':'same-observations'}),event_review=layer)
        service.cv_m22_resolver=resolver;service.catalog=lambda recording:object();service.recording_lock=lambda recording:contextlib.nullcontext()
        class Index:
            def __init__(self,*args):pass
            def attach_cv(self,layer):self.layer=layer
        with patch('src.assistant.service.AssistantEventIndex',Index):
            first=service.index('r','m22');self.assertIs(first,service.index('r','m22'))
            revision[0]='second';second=service.index('r','m22');other=service.index('s','m22')
            self.assertIsNot(first,second);self.assertIsNot(other,second)
            self.assertNotEqual(first.cv_model_revision,second.cv_model_revision)
            self.assertIs(second.layer,layers[('r','second')])


class AssistedDevelopmentMetricsM22Tests(unittest.TestCase):
    def make(self):
        tasks=[{'task_id':'t','recording_id':'r','frame':0,'kind':'ball'}]
        source={'r':{'video_sha256':'a'*64,'fps':25.,'frames':1,'width':100,'height':80}}
        row={'task_id':'t','frame':0,'timestamp':0.,'source_video_sha256':'a'*64,
            'origin':'human','reviewed_from_source':True,'model_assisted':True,'state':'visible','bbox':[10.,10.,5.,5.]}
        balls={'r':[{'directly_observed':True,'image_bbox':[10.,10.,5.,5.]}]}
        return tasks,source,row,balls

    def test_assisted_labels_are_explicitly_development_and_still_rejected_by_independent_evaluator(self):
        from src.evaluation.cv_m22_development import ball_metrics
        from src.evaluation.cv_m19 import evaluate
        tasks,sources,row,balls=self.make();result=ball_metrics(tasks,[row],balls,sources)
        self.assertEqual(result['ball']['tp'],1);self.assertEqual(result['reference_role'],'development_only');self.assertFalse(result['independent_final_test'])
        with self.assertRaises(ValueError):evaluate(tasks,[row],balls,{}, {'sources':sources,'ball_box_match_iou':.25})

    def test_predicted_positions_and_unknown_visibility_cannot_inflate_observation_accuracy(self):
        from src.evaluation.cv_m22_development import ball_metrics
        tasks,sources,row,balls=self.make();balls['r'][0]={'directly_observed':False,'image_bbox':None,'predicted_image_position':[12.,12.]}
        result=ball_metrics(tasks,[row],balls,sources);self.assertEqual(result['ball']['fn'],1);self.assertEqual(result['ball']['tp'],0)
        row.update(state='uncertain',bbox=None);result=ball_metrics(tasks,[row],balls,sources)
        self.assertEqual(result['ball']['unknown_visibility_frames'],1);self.assertEqual(result['ball']['reviewed_frames'],0)

    def test_incorrect_boxes_count_as_false_and_missed_and_absence_is_explicit(self):
        from src.evaluation.cv_m22_development import ball_metrics
        tasks,sources,row,balls=self.make();balls['r'][0]['image_bbox']=[50.,50.,5.,5.]
        result=ball_metrics(tasks,[row],balls,sources);self.assertEqual((result['ball']['fp'],result['ball']['fn']),(1,1))
        row.update(state='not_visible',bbox=None);result=ball_metrics(tasks,[row],balls,sources);self.assertEqual(result['ball']['fp'],1)

    def test_wrong_source_duplicate_and_nonhuman_provenance_are_rejected(self):
        from src.evaluation.cv_m22_development import ball_metrics
        tasks,sources,row,balls=self.make()
        with self.assertRaises(ValueError):ball_metrics(tasks,[row,row],balls,sources)
        for override in [{'source_video_sha256':'b'*64},{'model_assisted':False},{'origin':'model'},{'frame':1}]:
            with self.assertRaises(ValueError):ball_metrics(tasks,[{**row,**override}],balls,sources)


class VerifiedPixelCacheM22Tests(unittest.TestCase):
    def make(self,root):
        import cv2
        from src.serving.cv import digest
        pixels=np.full((32,48,3),123,np.uint8);image=root/'source.png';cv2.imwrite(str(image),pixels)
        dataset=root/'dataset.json';dataset.write_text(json.dumps({'rows':[{'context':[{'path':str(image),'sha256':digest(image)}]}]}))
        return pixels,image,dataset

    def test_cache_uses_identical_pixels_after_original_is_unavailable_without_global_opencv_patch(self):
        import cv2
        from src.vision.football22 import training
        from src.vision.football22.pixel_cache import PixelCache
        with TemporaryDirectory() as directory:
            root=Path(directory);pixels,image,dataset=self.make(root);cache=PixelCache(dataset,root/'cache');original=cv2.imread;sha=cache.digest(image);image.unlink()
            with cache.training_io():
                np.testing.assert_array_equal(training.cv2.imread(str(image)),pixels)
                self.assertEqual(training.digest(image),sha);self.assertIs(cv2.imread,original)
            self.assertIs(training.cv2,cv2)

    def test_corrupted_cache_and_unknown_image_are_rejected(self):
        from src.vision.football22.pixel_cache import PixelCache
        with TemporaryDirectory() as directory:
            root=Path(directory);_,image,dataset=self.make(root);cache=PixelCache(dataset,root/'cache')
            with self.assertRaises(ValueError):cache.imread(root/'unrelated.png')
            next(iter(cache.paths.values())).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):PixelCache(dataset,root/'cache')


if __name__=='__main__':unittest.main()
