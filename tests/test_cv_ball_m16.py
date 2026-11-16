import unittest,math
import numpy as np
from src.vision.ball16.data_contract import validate_annotation,check_splits
from src.vision.ball16.tracker import MultiHypothesisTracker
from src.vision.ball.schema import canonical
from ml.ball16.detector import merge_additive,recovery_region

def c(x=100,y=100,score=.9,id='a'):return {'candidate_id':id,'bbox':[x-4,y-4,8,8],'confidence':score,'source':'VIDEO_CV_DETECTOR','inference_view':'global'}
def camera(epoch=0,cut=False):return {'epoch':epoch,'continuity':'cut' if cut else 'continuous','calibration_valid':False,'motion_evidence':{}}
def row():return {'sample_id':'a:1','match_id':'match1','split':'train','source_sha256':'hash','width':640,'height':360,'timestamp':0,'balls':[{'bbox':[10,20,5,5]}],'visibility':'UNKNOWN','verified_negative':False}
class AnnotationTests(unittest.TestCase):
 def test_unknown_absence_is_not_negative(self):
  r=row();r.update(balls=[],verified_negative=True)
  with self.assertRaises(ValueError):validate_annotation(r)
 def test_negative_visibility_cannot_keep_ball_boxes(self):
  r=row();r.update(verified_negative=True,visibility='NO_VISIBLE_BALL',independent_review='review')
  with self.assertRaises(ValueError):validate_annotation(r)
 def test_reviewed_negative_is_explicit(self):
  r=row();r.update(balls=[],verified_negative=True,visibility='NO_VISIBLE_BALL',independent_review='review123');self.assertIs(validate_annotation(r),r)
 def test_box_outside_image_rejected(self):
  r=row();r['balls']=[{'bbox':[639,10,5,5]}]
  with self.assertRaises(ValueError):validate_annotation(r)
 def test_nonfinite_box_rejected(self):
  r=row();r['balls']=[{'bbox':[math.nan,10,5,5]}]
  with self.assertRaises(ValueError):validate_annotation(r)
 def test_match_leakage_rejected(self):
  a=row();b={**a,'sample_id':'a:2','source_sha256':'other','split':'test'}
  with self.assertRaises(ValueError):check_splits([a,b])
 def test_historical_test_not_train(self):
  with self.assertRaises(ValueError):check_splits([row()],['a:1'])
 def test_duplicate_across_matches_rejected(self):
  a=row();b={**a,'match_id':'other','split':'test'}
  with self.assertRaises(ValueError):check_splits([a,b])
 def test_unknown_height_has_no_metric_reference(self):
  r=row();r['metric_reference']=[20,30]
  with self.assertRaises(ValueError):validate_annotation(r)
class CandidateTests(unittest.TestCase):
 def test_additive_local_preserves_global_visual_evidence(self):
  global_cs=[c()];local=[c(101,100,.99,'local'),c(150,100,.5,'other')];merged=merge_additive(global_cs,local);self.assertEqual(merged[0]['bbox'],global_cs[0]['bbox']);self.assertEqual(merged[0]['confidence'],.9);self.assertEqual(len(merged),2)
 def test_local_context_not_smaller_than_training_context(self):self.assertGreaterEqual(recovery_region([320,180],5,640,360)[2],256)
 def test_region_within_source_bounds(self):
  x,y,w,h=recovery_region([639,359],500,640,360);self.assertGreaterEqual(x,0);self.assertGreaterEqual(y,0);self.assertLessEqual(x+w,640);self.assertLessEqual(y+h,360)
 def test_missing_forecast_means_no_local_crop(self):self.assertIsNone(recovery_region(None,None,640,360))
class MultiTests(unittest.TestCase):
 def history(self):
  t=MultiHypothesisTracker(640,360,.35)
  for i in range(6):s=t.step(i,i*.04,[c(100+i)],camera())
  self.assertTrue(s.directly_observed);return t,s
 def test_actual_retained_candidate_required(self):
  t,s=self.history();self.assertEqual(s.version,'m16');self.assertEqual(s.image_position,s.candidates[0].center)
 def test_low_visual_score_cannot_confirm(self):
  t=MultiHypothesisTracker(640,360,.35)
  for i in range(8):s=t.step(i,i*.04,[c(score=.1)],camera())
  self.assertFalse(s.directly_observed)
 def test_multiple_hypotheses_are_bounded(self):
  t=MultiHypothesisTracker(640,360,.35);t.step(0,0,[c(20+i*20,id=str(i)) for i in range(10)],camera());self.assertLessEqual(len(t.hypotheses),4)
 def test_competing_repeat_candidates_stay_ambiguous(self):
  t=MultiHypothesisTracker(640,360,.35)
  for i in range(3):s=t.step(i,i*.04,[c(100,id='a'),c(180,id='b')],camera())
  self.assertEqual(s.tracking_status,'AMBIGUOUS');self.assertIsNone(s.image_position)
 def test_cut_resets_continuity(self):
  t,s=self.history();s=t.step(6,.24,[],camera(1,True));self.assertFalse(s.directly_observed);self.assertIsNone(s.predicted_image_position)
 def test_prediction_expires(self):
  t,s=self.history()
  for i in range(6,14):s=t.step(i,i*.04,[],camera())
  self.assertIsNone(s.predicted_image_position);self.assertEqual(s.tracking_status,'LOST')
 def test_reappearance_needs_actual_visual_evidence(self):
  t,s=self.history();t.step(6,.24,[],camera());s=t.step(7,.28,[c(107)],camera());self.assertTrue(s.directly_observed)
 def test_metric_unknown_does_not_hide_image(self):
  t,s=self.history();self.assertIsNotNone(s.image_position);self.assertIsNone(s.pitch_position)
 def test_canonical_version_provenance(self):
  t,s=self.history();o=canonical('clip',25,s,'hash');self.assertEqual(o.schema_version,'ball-m16-v1');self.assertTrue(o.directly_observed)
 def test_clock_monotonic(self):
  t,s=self.history()
  with self.assertRaises(ValueError):t.step(0,0,[],camera())
class VisualTests(unittest.TestCase):
 def test_temporal_network_probabilities_finite(self):
  import torch
  from ml.ball16.temporal import VisualPair
  m=VisualPair().eval()
  with torch.no_grad():p=m(torch.rand(2,6,48,48)).sigmoid()
  self.assertEqual(tuple(p.shape),(2,));self.assertTrue(torch.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())


class VersionTests(unittest.TestCase):
 def setUp(self):
  from tests.test_cv_ball_m15 import VersionedServingTests
  VersionedServingTests.setUp(self)
 def make(self,version):
  from tests.test_cv_ball_m15 import VersionedServingTests
  return VersionedServingTests.sidecar(self,version)
 def variants(self,paths):
  from unittest.mock import patch
  from fastapi.testclient import TestClient
  from src.serving.app import create_app
  def registry(p):return {'cv_test':str(paths['m16' if p.name=='cv_ball_m16.json' else 'm15' if p.name=='cv_ball_m15.json' else 'm14'])}
  with patch('src.serving.ball.read_ball_registry',side_effect=registry):return TestClient(create_app({},cv_registry={'cv_test':self.root}))
 def test_three_versions_separately_queryable_default_m14(self):
  c=self.variants({v:self.make(v) for v in ['m14','m15','m16']});self.assertEqual(c.get('/matches/cv_test/cv').json()['ball_model_version'],'m14')
  for version in ['m14','m15','m16']:
   self.assertEqual(c.get('/matches/cv_test/cv/frame',params={'ball_pipeline':version}).json()['ball_world']['version'],version)
 def test_direct_serialization_preserves_full_validated_frame(self):
  from src.serving.cv import CVReview
  paths={v:self.make(v) for v in ["m14","m15","m16"]};client=self.variants(paths)
  for version in paths:
   review=CVReview(self.root);review.attach_ball(paths[version])
   response=client.get("/matches/cv_test/cv/frame",params={"ball_pipeline":version})
   self.assertEqual(response.headers["content-type"],"application/json")
   self.assertEqual(response.json(),review.frame(0).model_dump(mode="json"))
 def test_corrupt_m16_does_not_fall_back_silently(self):
  paths={v:self.make(v) for v in ['m14','m15','m16']};(paths['m16']/'ball_frames.jsonl').write_text('corrupt');c=self.variants(paths);self.assertEqual(c.get('/matches/cv_test/cv',params={'ball_pipeline':'m16'}).status_code,503);self.assertEqual(c.get('/matches/cv_test/cv/frame').status_code,200);self.assertEqual(c.get('/ready').status_code,200)


class HistoricalReplayTests(unittest.TestCase):
 def test_frozen_m15_history_does_not_reset_each_frame(self):
  import json
  from unittest.mock import patch
  from ml.ball16.evaluate import replay
  from tests.test_cv_ball_m15 import CFG
  rows=[{'recording_id':'same','frame':i,'timestamp':i*.04,'width':640,'height':360} for i in range(6)];predictions=[{'candidates':[c(100+i)]} for i in range(6)]
  with patch('pathlib.Path.read_text',return_value=json.dumps({'selected':{'config':CFG},'verification_enabled':False})):states=replay(rows,predictions,.4,'m15')
  self.assertTrue(states[-1]['directly_observed']);self.assertEqual(states[-1]['version'],'m15')


class CacheIntegrityTests(unittest.TestCase):
 def test_cached_candidates_cannot_change_weights_silently(self):
  import tempfile,cv2
  from pathlib import Path
  from unittest.mock import patch
  from ml.ball16.evaluate import predictions
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);image=root/'source.jpg';cv2.imwrite(str(image),np.zeros((40,40,3),np.uint8));rows=[{'sample_id':'a:1','path':str(image)}]
   with patch('ml.ball16.evaluate.ROOT',root),patch('ml.ball16.evaluate.sha',return_value='dataset'),patch('ml.ball16.evaluate.load',return_value={'checkpoint_sha256':'weights1'}),patch('ml.ball16.evaluate.infer',return_value=([c(20,20)],{'seconds':.001,'views':1})):
    self.assertEqual(len(predictions(rows,None,640,'cache')),1)
   with patch('ml.ball16.evaluate.ROOT',root),patch('ml.ball16.evaluate.sha',return_value='dataset'),patch('ml.ball16.evaluate.load',return_value={'checkpoint_sha256':'weights2'}):
    with self.assertRaises(ValueError):predictions(rows,None,640,'cache')


class IndependentGeometryTests(unittest.TestCase):
 def cal(self,x=0):return {'rotation_matrix':np.eye(3).tolist(),'position_meters':[x,0,-10],'x_focal_length':100,'y_focal_length':100,'principal_point':[100,100]}
 def test_other_view_triangulation_does_not_use_heldout_point(self):
  from ml.ball16.metric_geometry import projection,image,triangulate
  X=np.array([3,4,0.]);a=projection(self.cal());b=projection(self.cal(2));result=triangulate([{'P':a,'point':image(a,X)},{'P':b,'point':image(b,X)}]);self.assertTrue(np.allclose(result,X,atol=1e-7))
 def test_field_region_is_clipped_to_the_image_before_mapping(self):
  from ml.ball16.metric_geometry import projection,visible_field_polygon
  P=projection(self.cal());region=np.asarray(visible_field_polygon(P,200,200));self.assertGreaterEqual(len(region),3);self.assertTrue(np.isfinite(region).all());self.assertTrue((region>=-1e-5).all());self.assertTrue((region<=199+1e-5).all())
if __name__=='__main__':unittest.main()
