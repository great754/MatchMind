"""M15 causal motion, verified observations, expiry and metric-provenance regressions."""
import unittest,math
import numpy as np
from pydantic import ValidationError
from src.vision.ball.schema import BallCandidate,BallFrame,canonical
from src.vision.ball15.motion import MotionHistory,features,extrapolate
from src.vision.ball15.association import evidence,score,rank,logistic
from src.vision.ball15.tracker import GuidedTracker
from ml.ball15.detector import search_region

CFG={'minimum_visual':.4,'prediction_seconds':.12,'margin':.08,'score_threshold':.4,'association_kind':'interpretable','motion_mode':'velocity','motion_kind':'velocity','radius90_px':12,'ground_validation_passed':False}
def candidate(x=100,y=100,conf=.9,id='a'):return {'candidate_id':id,'bbox':[x-4,y-4,8,8],'confidence':conf,'inference_view':'full','source':'VIDEO_CV_DETECTOR'}
def camera(epoch=0,cut=False):return {'epoch':epoch,'calibration_valid':False,'continuity':'cut' if cut else 'continuous','motion_evidence':{}}
class MotionTests(unittest.TestCase):
 def test_constant_velocity_next_position(self):self.assertTrue(np.allclose(extrapolate([[0,0],[1,2],[2,4]],[0,.04,.08],.12),[3,6]))
 def test_acceleration_baseline(self):self.assertTrue(np.allclose(extrapolate([[0,0],[1,1],[4,4],[9,9]],[0,1,2,3],4,'acceleration'),[16,16]))
 def test_history_bounded(self):
  h=MotionHistory()
  for i in range(12):h.append([i,i],i*.04,.8)
  self.assertEqual(len(h.points),8)
 def test_history_rejects_backwards_clock(self):
  h=MotionHistory();h.append([0,0],1,.9)
  with self.assertRaises(ValueError):h.append([1,1],.5,.9)
 def test_feature_shape_and_relative_origin(self):
  f=features([[0,0],[2,4]],[0,.04],[1,.9],100);self.assertEqual(f.shape,(2,8));self.assertTrue(np.all(f[-1,:2]==0));self.assertTrue(np.isfinite(f).all())
 def test_empty_history_no_prediction(self):self.assertIsNone(MotionHistory().predict(1))
 def test_local_search_expands_with_uncertainty(self):self.assertGreater(search_region([320,180],70,.08,640,360)[2],search_region([320,180],10,.04,640,360)[2])
 def test_search_clamps_source_bounds(self):
  x,y,w,h=search_region([635,355],30,.08,640,360);self.assertGreaterEqual(x,0);self.assertGreaterEqual(y,0);self.assertLessEqual(x+w,640);self.assertLessEqual(y+h,360)
 def test_no_prediction_uses_global(self):self.assertIsNone(search_region(None,None,0,640,360))
class AssociationTests(unittest.TestCase):
 def test_near_moderate_visual_beats_far_high_visual(self):
  cs=[candidate(104,103,.75,'near'),candidate(500,50,.95,'far')];r=rank(cs,[100,100],8,[95,95],[],640,360,.04);self.assertEqual(r[0]['candidate_id'],'near')
 def test_nearest_weak_visual_cannot_be_confirmed(self):
  t=GuidedTracker(640,360,CFG)
  for i in range(5):s=t.step(i,i*.04,[candidate(100+i,100,.1)],camera())
  self.assertFalse(s.directly_observed)
 def test_player_proximity_not_proof(self):
  f=evidence(candidate(conf=.05),[100,100],4,[99,100],[{'bbox':[95,20,10,80]}],640,360);self.assertLess(score(f),.4)
 def test_serialized_logistic_finite(self):self.assertAlmostEqual(logistic([0,0],{'mean':[0,0],'scale':[1,1],'coef':[1,1],'intercept':0}),.5)
 def test_ties_ambiguous(self):
  t=GuidedTracker(640,360,CFG);s=t.step(0,0,[candidate(id='a'),candidate(105,100,id='b')],camera());self.assertEqual(s.tracking_status,'AMBIGUOUS');self.assertIsNone(s.image_position)
class TrackerTests(unittest.TestCase):
 def track(self):
  t=GuidedTracker(640,360,CFG)
  for i in range(6):s=t.step(i,i*.04,[candidate(100+i,100)],camera())
  self.assertTrue(s.directly_observed);return t,s
 def test_version_and_actual_observation(self):
  t,s=self.track();self.assertEqual(s.version,'m15');self.assertEqual(tuple(s.image_position),tuple(s.candidates[0].center));self.assertEqual(s.candidate_verification[0].candidate_id,'a')
 def test_missing_is_not_observed(self):
  t,s=self.track();s=t.step(6,.24,[],camera());self.assertFalse(s.directly_observed);self.assertIsNone(s.image_position);self.assertIsNotNone(s.predicted_image_position)
 def test_prediction_expires(self):
  t,s=self.track()
  for i in range(6,12):s=t.step(i,i*.04,[],camera())
  self.assertEqual(s.tracking_status,'LOST');self.assertIsNone(s.predicted_image_position)
 def test_uncertainty_grows(self):
  t,s=self.track();a=t.step(6,.24,[],camera());b=t.step(7,.28,[],camera());self.assertGreaterEqual(b.prediction_uncertainty_px,a.prediction_uncertainty_px)
 def test_camera_epoch_reset(self):
  t,s=self.track();s=t.step(6,.24,[],camera(1));self.assertEqual(s.tracking_status,'LOST');self.assertIsNone(s.prediction_region)
 def test_camera_cut_reset(self):
  t,s=self.track();s=t.step(6,.24,[],camera(cut=True));self.assertIsNone(s.predicted_image_position)
 def test_reacquisition_requires_real_candidate(self):
  t,s=self.track();t.step(6,.24,[],camera());s=t.step(7,.28,[candidate(107,100)],camera());self.assertTrue(s.directly_observed);self.assertEqual(tuple(s.image_position),(107,100))
 def test_impossible_distant_distractor_not_observed(self):
  t,s=self.track();s=t.step(6,.24,[candidate(600,300,.99)],camera());self.assertFalse(s.directly_observed)
 def test_confidence_decays_in_missing_interval(self):
  t,s=self.track();a=t.step(6,.24,[],camera());b=t.step(7,.28,[],camera());self.assertLess(b.tracking_confidence,a.tracking_confidence)
 def test_unknown_height_no_metric(self):
  t,s=self.track();self.assertIsNone(s.pitch_position)
 def test_canonical_m15_provenance(self):
  t,s=self.track();o=canonical('clip',25,s,'hash');self.assertEqual(o.schema_version,'ball-m15-v1');self.assertTrue(o.directly_observed)
 def test_prediction_canonical_no_observed_image(self):
  t,s=self.track();s=t.step(6,.24,[],camera());o=canonical('clip',25,s,'hash');self.assertIsNone(o.image);self.assertIsNotNone(o.predicted_image)
class SchemaTests(unittest.TestCase):
 def test_nonfinite_region_rejected(self):
  with self.assertRaises(ValidationError):BallFrame(version='m15',frame=0,timestamp=0,camera_epoch=0,prediction_region={'center':[math.nan,0],'radius90_px':10,'calibration':'test'})
 def test_negative_region_radius_rejected(self):
  with self.assertRaises(ValidationError):BallFrame(version='m15',frame=0,timestamp=0,camera_epoch=0,prediction_region={'center':[0,0],'radius90_px':-1,'calibration':'test'})
 def test_unreferenced_verification_rejected(self):
  with self.assertRaises(ValidationError):BallFrame(version='m15',frame=0,timestamp=0,camera_epoch=0,candidate_verification=[{'candidate_id':'fake','score':.9,'features':[]}])
 def test_unsupported_metric_rejected(self):
  with self.assertRaises(ValidationError):BallFrame(version='m15',frame=0,timestamp=0,camera_epoch=0,pitch_position=[20,30])


class SearchIntegrationTests(unittest.TestCase):
 def test_causal_local_search_and_periodic_global_recovery(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  from ml.ball15.search import run_search
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'cache').mkdir();rows=[{'sample_id':f'clip:{i}','recording_id':'clip','frame':i,'timestamp':i*.04,'width':640,'height':360,'people':[]} for i in range(12)];calls=[]
   def detector(model,image,tile,region=None,generic=False):
    i=len(calls);calls.append(region);return [candidate(100+min(i,11),100)],{'seconds':.001,'views':1}
   with patch('ml.ball15.search.ROOT',root),patch('ml.ball15.search.infer',detector):
    ps,states=run_search(rows,[camera() for r in rows],None,{'tile':320,'kind':'generic','checkpoint_sha256':'test'},CFG,'unit',image_loader=lambda r:np.zeros((360,640,3),np.uint8))
   self.assertTrue(any(p['search_stage']=='LOCAL_GUIDED' for p in ps));self.assertIsNone(calls[8]);self.assertEqual(states[-1]['version'],'m15');self.assertTrue(states[-1]['directly_observed'])
 def test_failed_local_search_returns_to_global_same_frame(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  from ml.ball15.search import run_search
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'cache').mkdir();rows=[{'sample_id':f'clip:{i}','recording_id':'clip','frame':i,'timestamp':i*.04,'width':640,'height':360,'people':[]} for i in range(8)];local=[]
   def detector(model,image,tile,region=None,generic=False):
    if region is not None:local.append(region);return [],{'seconds':.001,'views':1}
    return [candidate()],{'seconds':.001,'views':1}
   with patch('ml.ball15.search.ROOT',root),patch('ml.ball15.search.infer',detector):ps,states=run_search(rows,[camera() for r in rows],None,{'tile':320,'kind':'generic','checkpoint_sha256':'test'},CFG,'unit',image_loader=lambda r:np.zeros((360,640,3),np.uint8))
   self.assertTrue(local);self.assertTrue(any(p['search_stage']=='GLOBAL_RECOVERY' for p in ps));self.assertTrue(states[-1]['directly_observed'])
 def test_ground_cues_never_certify_height_from_size_alone(self):
  from ml.ball15.ground import ground_cues
  c=BallCandidate.model_validate(candidate());self.assertFalse(ground_cues(c,camera(),[])['independent_ground_model_validated'])


class VersionedServingTests(unittest.TestCase):
 def setUp(self):
  from tests.test_cv_pitch_review import ServingTests
  ServingTests.setUp(self)
 def sidecar(self,version):
  import json
  from src.serving.cv import digest
  directory=self.root/version;directory.mkdir();manifest=json.loads((self.root/'manifest.json').read_text());states=[BallFrame(version=version,frame=i,timestamp=i/25,camera_epoch=0) for i in range(3)];offsets=[]
  with (directory/'ball_frames.jsonl').open('wb') as f:
   for s in states:offsets.append(f.tell());f.write((s.model_dump_json()+'\n').encode())
  (directory/'video_ball_observations.jsonl').write_text(''.join(canonical('cv_test',25,s,'model').model_dump_json()+'\n' for s in states));m={**manifest,'version':version,'detector_sha256':'model','source_m9_frames_sha256':manifest['frames_sha256'],'ball_frames_sha256':digest(directory/'ball_frames.jsonl'),'canonical_sha256':digest(directory/'video_ball_observations.jsonl'),'frame_offsets':offsets};(directory/'manifest.json').write_text(json.dumps(m));return directory
 def variant_client(self,m14,m15):
  from unittest.mock import patch
  from fastapi.testclient import TestClient
  from src.serving.app import create_app
  with patch('src.serving.ball.read_ball_registry',side_effect=lambda p:{'cv_test':str(m15 if p.name=='cv_ball_m15.json' else m14)}):return TestClient(create_app({},cv_registry={'cv_test':self.root}))
 def test_default_m14_and_explicit_m15_are_separate(self):
  c=self.variant_client(self.sidecar('m14'),self.sidecar('m15'));self.assertEqual(c.get('/matches/cv_test/cv').json()['ball_model_version'],'m14');self.assertEqual(c.get('/matches/cv_test/cv',params={'ball_pipeline':'m15'}).json()['ball_model_version'],'m15');self.assertEqual(c.get('/matches/cv_test/cv/frame').json()['ball_world']['version'],'m14');self.assertEqual(c.get('/matches/cv_test/cv/frame',params={'ball_pipeline':'m15'}).json()['ball_world']['version'],'m15')
 def test_corrupt_m15_does_not_remove_m14_or_ready(self):
  a=self.sidecar('m14');b=self.sidecar('m15');(b/'ball_frames.jsonl').write_text('corrupt');c=self.variant_client(a,b);self.assertEqual(c.get('/ready').status_code,200);self.assertEqual(c.get('/matches/cv_test/cv/frame').status_code,200);self.assertEqual(c.get('/matches/cv_test/cv/frame',params={'ball_pipeline':'m15'}).status_code,503)
 def test_invalid_pipeline_not_path_input(self):
  c=self.variant_client(self.sidecar('m14'),self.sidecar('m15'));self.assertEqual(c.get('/matches/cv_test/cv',params={'ball_pipeline':'../../other'}).status_code,422)
 def test_version_mismatch_rejected(self):
  import json
  a=self.sidecar('m14');b=self.sidecar('m15');p=b/'manifest.json';m=json.loads(p.read_text());m['version']='m14';p.write_text(json.dumps(m));c=self.variant_client(a,b);self.assertEqual(c.get('/matches/cv_test/cv',params={'ball_pipeline':'m15'}).status_code,503)
if __name__=='__main__':unittest.main()
