import unittest,json,tempfile
from pathlib import Path
import numpy as np
from ml.reid.protocol import validate_splits,retrieval_sets,integration_gate
from ml.reid.tracklet import pool
from ml.reid.sampling import pk_batches
from ml.reid.dataset import quality,crop_regions,perceptual_hash
try:
 import torch
 from ml.reid.losses import hard_triplet
except ImportError:torch=None

def row(split='train',match='a',identity='a:1',team='t1',pid=1,frame=0):
 return {'split':split,'match_id':match,'recording_id':match+'_clip','player_identity':identity,'team_identity':team,'registered_player_id':pid,'source_sha256':match,'sample_id':f'{match}:{identity}:{frame}','timestamp':frame*2,'quality':{'score':.7,'height':50,'size_bucket':'medium'},'frame':frame}
class DatasetTests(unittest.TestCase):
 def test_match_leak_rejected(self):
  with self.assertRaises(ValueError):validate_splits([row(),row('test','a','x','t2',2)])
 def test_identity_leak_rejected(self):
  with self.assertRaises(ValueError):validate_splits([row(),row('test','b','a:1','t2',2)])
 def test_registered_person_leak_rejected(self):
  with self.assertRaises(ValueError):validate_splits([row(),row('test','b','b:1','t2',1)])
 def test_team_leak_rejected(self):
  with self.assertRaises(ValueError):validate_splits([row(),row('test','b','b:1','t1',2)])
 def test_source_hash_leak_rejected(self):
  a,b=row(),row('test','b','b:1','t2',2);b['source_sha256']='a'
  with self.assertRaises(ValueError):validate_splits([a,b])
 def test_m9_eval_video_excluded(self):
  a=row();a['recording_id']='test_match'
  with self.assertRaises(ValueError):validate_splits([a])
 def test_disjoint_split_valid(self):self.assertIn('test',validate_splits([row(),row('test','b','b:1','t2',2)]))
 def test_positive_has_long_time_gap(self):
  rows=[row(frame=i) for i in range(120)];q,g=retrieval_sets(rows);self.assertTrue(q and g);self.assertGreater(min(x[0]['timestamp'] for x in g)-max(x[0]['timestamp'] for x in q),90)
 def test_tracklet_lengths_bounded(self):
  q,g=retrieval_sets([row(frame=i) for i in range(120)],True);self.assertTrue(all(10<=len(t)<=30 for t in q+g))
 def test_pk_positive_temporal_quartiles(self):
  rows=[row(identity=f'a:{p}',pid=p,frame=f) for p in range(10) for f in range(40)];indices=next(pk_batches(rows,1,5));self.assertEqual(len(indices),32)
  for key in {rows[i]['player_identity'] for i in indices}:
   frames=[rows[i]['frame'] for i in indices if rows[i]['player_identity']==key];self.assertEqual(len(frames),4);self.assertGreater(max(frames)-min(frames),20)
 def test_pk_same_team_negatives_dominate(self):
  rows=[row(identity=f'a:{p}',team='a' if p<10 else 'b',pid=p,frame=f) for p in range(20) for f in range(20)];ix=next(pk_batches(rows,1,6));teams=[rows[i]['team_identity'] for i in ix];self.assertEqual(max(teams.count('a'),teams.count('b')),28)
 def test_sampling_is_deterministic(self):
  rows=[row(identity=f'a:{p}',pid=p,frame=f) for p in range(10) for f in range(20)];self.assertEqual(list(pk_batches(rows,3,4)),list(pk_batches(rows,3,4)))
 def test_crop_quality_boundary_and_small_size(self):
  image=np.random.default_rng(3).integers(0,255,(60,80,3),dtype=np.uint8);q,c=quality(image,[-4,5,20,30],[]);self.assertTrue(q['boundary_truncated']);self.assertGreater(q['edge_clipping_fraction'],0)
 def test_full_crop_masks_face_region(self):
  a=np.ones((100,40,3),np.uint8)*10;b=crop_regions(a,'full_masked')[0];self.assertTrue(np.all(b[:18]==127));self.assertTrue(np.all(a==10))
 def test_crop_regions_count(self):self.assertEqual(len(crop_regions(np.ones((100,40,3),np.uint8),'multi_region')),3)
 def test_duplicate_hash_deterministic(self):
  a=np.random.default_rng(4).integers(0,255,(80,30,3),dtype=np.uint8);self.assertEqual(perceptual_hash(a),perceptual_hash(a.copy()))
class PoolAndGateTests(unittest.TestCase):
 def test_normalized_pool(self):self.assertAlmostEqual(np.linalg.norm(pool([[1,0],[0,1]])),1)
 def test_quality_weighted_suppresses_bad_crop(self):self.assertGreater(pool([[1,0],[0,1]],[1,.01])[0],.99)
 def test_empty_pool_rejected(self):
  with self.assertRaises(ValueError):pool([])
 def test_bad_pool_shape_rejected(self):
  with self.assertRaises(ValueError):pool([[1,0],[0,1]],[1])
 def test_gate_rejects_loss_only_improvement(self):
  b={k:.5 for k in ['same_team_rank1','same_team_map','tracklet_map','rank1','map']};self.assertFalse(integration_gate(b,b)['passed'])
 def test_gate_requires_tracklet_gain(self):
  b={k:.5 for k in ['same_team_rank1','same_team_map','tracklet_map','rank1','map']};a={k:.6 for k in b};a['tracklet_map']=.51;self.assertFalse(integration_gate(b,a)['passed'])
 def test_gate_accepts_all_required_improvements(self):
  b={k:.5 for k in ['same_team_rank1','same_team_map','tracklet_map','rank1','map']};self.assertTrue(integration_gate(b,{k:.6 for k in b})['passed'])
 def test_test_is_sealed_before_selection(self):
  if torch is None:self.skipTest('optional torch unavailable')
  import ml.reid.evaluate as e
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d,patch.object(e,'ROOT',Path(d)):
   with self.assertRaisesRegex(ValueError,'sealed'):e.embedding_cache([],'hash',split='test')
@unittest.skipIf(torch is None,'optional torch unavailable')
class MetricTests(unittest.TestCase):
 def test_triplet_uses_same_team_hard_negative(self):
  z=torch.nn.functional.normalize(torch.tensor([[1.,0],[.9,.1],[.95,.05],[0.,1]]),dim=1);labels=torch.tensor([0,0,1,2]);teams=torch.tensor([0,0,0,1]);self.assertGreater(float(hard_triplet(z,labels,teams)),0)
 def test_perfect_retrieval(self):
  from ml.reid.evaluate import metrics_for
  q=[{**row(),'sample_id':'q'},{**row(identity='b',pid=2),'sample_id':'q2'}];g=[{**x,'sample_id':x['sample_id']+'g'} for x in q];m=metrics_for(q,g,np.eye(2),np.eye(2),True,subgroups=False);self.assertEqual(m['rank1'],1);self.assertEqual(m['map'],1)
 def test_similarity_not_identity_certainty(self):
  from src.vision.world.player_identity import candidate
  from tests.test_cv_players_m9 import segment
  c=candidate(segment('a',0,10),segment('b',12,20,x=100),25);self.assertFalse(c.eligible)
 def test_global_assignment_stays_deterministic(self):
  from src.vision.world.player_identity import globally_associate
  from tests.test_cv_players_m9 import segment
  a=[segment('a',0,10),segment('b',30,40)];x=globally_associate(a,25);y=globally_associate(a,25);self.assertEqual(x[0],y[0]);self.assertEqual(x[3],y[3])

class RegistryAndProtocolTests(unittest.TestCase):
 def test_model_version_changes_fingerprint(self):
  from ml.reid.registry import fingerprint
  a={'model_name':'v1','architecture':'osnet','checkpoint_sha256':'a','training_manifest_sha256':'b','training_configuration_sha256':'c','created_date':'2026-11-02'};b={**a,'model_name':'v2'};self.assertNotEqual(fingerprint(a),fingerprint(b))
 def test_model_config_changes_fingerprint(self):
  from ml.reid.registry import fingerprint
  a={'model_name':'v1','architecture':'osnet','checkpoint_sha256':'a','training_manifest_sha256':'b','training_configuration_sha256':'c','created_date':'2026-11-02'};self.assertNotEqual(fingerprint(a),fingerprint({**a,'training_configuration_sha256':'d'}))
 def test_frozen_model_checkpoint_change_rejected(self):
  from ml.reid.registry import verify_hashes
  from src.vision.world.player_appearance import digest
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'checkpoint';p.write_bytes(b'original');sha=digest(p);p.write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_hashes({str(p):sha})
 def test_same_vectors_give_same_generic_and_soccer_metrics(self):
  if torch is None:self.skipTest('optional torch unavailable')
  from ml.reid.evaluate import metrics_for
  q=[{**row(),'sample_id':'q'}];g=[{**row(),'sample_id':'g'}];a=metrics_for(q,g,np.array([[1.,0]]),np.array([[1.,0]]),subgroups=False);self.assertEqual(a,metrics_for(q,g,np.array([[1.,0]]),np.array([[1.,0]]),subgroups=False))
@unittest.skipIf(torch is None,'optional torch unavailable')
class InferenceTests(unittest.TestCase):
 @unittest.skipUnless(Path('outputs/models/m9/osnet_x0_25_msmt17.pth').exists(),'optional generic checkpoint unavailable')
 def test_actual_encoder_normalized_and_deterministic(self):
  from ml.reid.model import SoccerReID,seed_all
  seed_all(5);model=SoccerReID(2).eval();x=torch.ones(2,1,3,256,128)
  with torch.inference_mode():a=model(x)[1];b=model(x)[1]
  torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(a.norm(dim=1),torch.ones(2),atol=1e-6,rtol=0)

class QualityMetadataTests(unittest.TestCase):
 def test_fractional_interior_bbox_cannot_report_negative_clipping(self):
  image=np.random.default_rng(10).integers(0,255,(100,100,3),dtype=np.uint8);q,_=quality(image,[10.9,10.9,19.76,53.48],[]);self.assertGreaterEqual(q['edge_clipping_fraction'],0);self.assertLessEqual(q['edge_clipping_fraction'],1)

if __name__=='__main__':unittest.main()
