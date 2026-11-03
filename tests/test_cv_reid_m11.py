"""M11 research guards: tracklet evidence, uncertainty, calibration and assignment."""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ml.reid11.data import diverse,tracks
try:
 import torch
except ImportError:torch=None

def row(i,identity='p',team='a'):
 return {'timestamp':i*2.,'sample_id':f'{identity}:{i}','player_identity':identity,'team_identity':team,'match_id':'match','perceptual_hash':format(i%255,'064b'),'quality':{'score':.7,'height':50+i,'brightness':80+i,'contrast':30}}

def track(name='a',start=0,end=1,**kw):
 return dict(tracklet_id=name,player_identity=name,match_id='m',start=start,end=end,team_identity='t',quality=.8,samples=list(range(8)),**kw)

class TrackletTests(unittest.TestCase):
 def test_time_diverse_sampling(self):
  a=diverse([row(i) for i in range(100)],12);self.assertEqual(len(a),12);self.assertGreater(a[-1]['timestamp']-a[0]['timestamp'],170)
 def test_deterministic_sampling(self):
  a=[row(i) for i in range(100)];self.assertEqual(diverse(a),diverse(a))
 def test_training_tracklets_have_unique_identity(self):
  rows=[row(i,p) for p in ['p','q'] for i in range(120)];ts=tracks(rows,True);by={r['sample_id']:r for r in rows}
  for t in ts:self.assertEqual({by[k]['player_identity'] for k in t['samples']},{t['player_identity']})
 def test_tracklet_length_8_12(self):
  ts=tracks([row(i) for i in range(120)],True);self.assertTrue(ts);self.assertTrue(all(8<=len(t['samples'])<=12 for t in ts))
 def test_short_track_not_invented_by_padding(self):self.assertEqual(tracks([row(i) for i in range(7)],True),[])
 def test_eval_windows_disjoint_and_separated(self):
  ts=tracks([row(i) for i in range(120)]);self.assertEqual(len(ts),2);self.assertGreater(ts[1]['start']-ts[0]['end'],90);self.assertFalse(set(ts[0]['samples'])&set(ts[1]['samples']))

@unittest.skipIf(torch is None,'optional CV training runtime')
class TemporalAndCueTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(1);torch.set_num_threads(2)
 def test_temporal_normalized_and_deterministic(self):
  from ml.reid11.model import Temporal
  m=Temporal(dim=16).eval();x=torch.randn(2,8,16);q=torch.ones(2,8);t=torch.arange(8.).repeat(2,1)
  with torch.no_grad():a=m(x,q,t);b=m(x,q,t)
  self.assertTrue(torch.allclose(a[1].norm(dim=-1),torch.ones(2)));self.assertTrue(torch.equal(a[1],b[1]))
 def test_padding_receives_no_attention(self):
  from ml.reid11.model import Temporal
  m=Temporal(dim=16).eval();x=torch.randn(1,10,16);q=torch.ones(1,10);t=torch.arange(10.)[None];mask=torch.tensor([[True]*8+[False]*2])
  with torch.no_grad():a=m(x,q,t,mask);x[:,8:]=1000;b=m(x,q,t,mask)
  self.assertEqual(float(a[2][:,8:].sum()),0);self.assertTrue(torch.allclose(a[1],b[1],atol=1e-6))
 def test_low_quality_gets_less_attention(self):
  from ml.reid11.model import Temporal
  m=Temporal(dim=16).eval();q=torch.tensor([[1.,.1]]);_,_,w=m(torch.randn(1,2,16),q,torch.zeros(1,2));self.assertGreater(float(w[0,0]),float(w[0,1]))
 def test_empty_tracklet_rejected(self):
  from ml.reid11.model import Temporal
  with self.assertRaisesRegex(ValueError,'Empty'):Temporal(dim=16)(torch.randn(1,2,16),torch.ones(1,2),torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool))
 def test_head_pixels_do_not_enter_encoder(self):
  from ml.reid11.model import image_array
  a=np.ones((100,40,3),np.uint8)*50;b=a.copy();b[:18]=255;self.assertTrue(np.array_equal(image_array(a),image_array(b)))
 def test_missing_geometry_preserves_nan(self):
  from ml.reid11.scorer import cues
  x=cues(track(),track('b',2,3),.9);self.assertTrue(np.isnan(x[3:7]).all());self.assertEqual(x[10],0);self.assertEqual(x[11],0)
 def test_speed_and_trajectory_units(self):
  from ml.reid11.scorer import cues
  a=track(end_pitch=[1,1],end_velocity=[2,0],uncertainty=1);b=track('b',3,4,start_pitch=[5,1],uncertainty=2);x=cues(a,b,.9);self.assertEqual(x[2],2);self.assertEqual(x[3],4);self.assertEqual(x[4],2);self.assertEqual(x[6],0);self.assertAlmostEqual(x[5],np.sqrt(5))
 def test_impossible_move_rejected_despite_similarity(self):
  from ml.reid11.scorer import physical_gate
  a=track(end_pitch=[0,0],uncertainty=1);b=track('b',1.04,2,start_pitch=[100,0],uncertainty=1);self.assertEqual(physical_gate(a,b),(False,'PHYSICALLY_IMPLAUSIBLE_REASSOCIATION'))
 def test_unknown_uncertainty_not_zero_certificate(self):
  from ml.reid11.scorer import physical_gate
  a=track(end_pitch=[0,0],uncertainty=None);b=track('b',2,3,start_pitch=[100,0],uncertainty=None);self.assertEqual(physical_gate(a,b)[1],'NO_KNOWN_PHYSICAL_CONTRADICTION')
 def test_overlapping_players_never_join(self):
  from ml.reid11.scorer import physical_gate
  self.assertFalse(physical_gate(track(),track('b',.5,2))[0])
 def test_camera_epoch_gate(self):
  from ml.reid11.scorer import physical_gate
  self.assertFalse(physical_gate(track(epoch=1),track('b',2,3,epoch=2))[0])
 def test_strong_team_conflict_gate(self):
  from ml.reid11.scorer import physical_gate
  a=track(team_strength=.9);b=track('b',2,3,team_strength=.9);b['team_identity']='other';self.assertFalse(physical_gate(a,b)[0])
 def test_stale_gate(self):
  from ml.reid11.scorer import physical_gate
  self.assertFalse(physical_gate(track(),track('b',20,21))[0])
 def test_threshold_obeys_validation_fmr(self):
  from ml.reid11.scorer import choose_threshold
  y=np.array([True,True,False,False]);p=np.array([.9,.6,.7,.2]);t=choose_threshold(y,p,0);self.assertEqual(np.mean(p[~y]>=t),0);self.assertEqual(np.mean(p[y]>=t),.5)
 def test_test_cache_sealed_before_selection(self):
  import ml.reid11.features as f
  with tempfile.TemporaryDirectory() as d,patch.object(f,'ROOT',Path(d)):
   with self.assertRaisesRegex(ValueError,'sealed'):f.cache('test')
 def test_scorer_no_identity_features(self):
  from ml.reid11.scorer import NAMES
  self.assertFalse(any('player_id' in n or 'team_id' in n or n in ['start_pitch','end_pitch'] for n in NAMES))
 def test_assignment_replays_production(self):
  from tests.test_cv_players_m9 import segment
  from src.vision.world.player_identity import globally_associate
  from src.vision.world.player_state import PlayerWorldConfig
  from ml.reid11.shadow import assign
  segments=[segment('a',0,10),segment('b',30,40),segment('c',30,40)];cfg=PlayerWorldConfig();original=globally_associate(segments,25,cfg);shadow=assign(segments,original[1],cfg.minimum_score,cfg.ambiguity_margin)
  self.assertEqual(original[2],shadow[2]);self.assertEqual([(e['from_segment'],e['to_segment']) for e in original[3]],[(e['from_segment'],e['to_segment']) for e in shadow[3]])
 def test_assignment_cannot_accept_ineligible_high_probability(self):
  from tests.test_cv_players_m9 import segment
  from src.vision.world.player_identity import globally_associate
  from src.vision.world.player_state import PlayerWorldConfig
  from ml.reid11.shadow import assign
  segments=[segment('a',0,10),segment('b',12,20,x=100)];cfg=PlayerWorldConfig();original=globally_associate(segments,25,cfg);ev={k:[e.model_copy(update={'score':1.}) for e in v] for k,v in original[1].items()};self.assertEqual(assign(segments,ev,.8,.06)[3],[])

if __name__=='__main__':unittest.main()
