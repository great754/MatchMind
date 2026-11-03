import unittest,io,json,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from ml.reid13.remote import RangeFile,member_bytes
from ml.reid13.data import allowed,validate
try:
 import torch
except ImportError:torch=None

def row(split='train',match='a',person='p',team='t',timestamp=0):
 return {'split':split,'match_id':match,'player_identity':person,'team_identity':team,'crop_sha256':person+match,'source_sha256':match,'sample_id':f'{person}:{timestamp}','timestamp':timestamp,'quality':{'score':.8,'height':50,'size_bucket':'medium','sharpness':100,'occlusion_proxy':0}}
class SourceTests(unittest.TestCase):
 def test_original_metadata_only_numeric_outfield(self):
  m={'gameid':'6','trackletid_1':'player team left;10','trackletid_2':'player team left;A','trackletid_3':'ball;1','trackletid_4':'referee;main','trackletid_5':'goalkeeper team left;1','trackletid_6':'player team right;9'};self.assertEqual(set(allowed(m,['left'])),{1})
 def test_half_and_side_scope_not_global(self):
  p=allowed({'gameid':'6','trackletid_1':'player team left;10'},['left'])[1];self.assertEqual(p['identity'],'SN:game6:left:10')
 def test_match_leak_rejected(self):
  with self.assertRaises(ValueError):validate([row(),row('test','a','x','y')])
 def test_person_leak_rejected(self):
  with self.assertRaises(ValueError):validate([row(),row('test','b','p','y')])
 def test_team_leak_rejected(self):
  with self.assertRaises(ValueError):validate([row(),row('test','b','x','t')])
 def test_crop_duplicate_rejected(self):
  a=row();b=row('test','b','x','y');b['crop_sha256']=a['crop_sha256']
  with self.assertRaises(ValueError):validate([a,b])
 def test_disjoint_data_accepted(self):self.assertEqual(validate([row(),row('test','b','x','y')])['test']['people'],1)
 def test_remote_budget_before_network(self):
  f=RangeFile('https://example.invalid/file',1000,10)
  with self.assertRaisesRegex(ValueError,'budget'):f.read(11)
 def test_remote_negative_seek(self):
  with self.assertRaises(ValueError):RangeFile('https://example.invalid/file',100).seek(-1)
 def test_remote_refuses_full_archive_response(self):
  f=RangeFile('https://example.invalid/file',100);r=Mock(status_code=200,headers={});f.session.get=Mock(return_value=r)
  with self.assertRaisesRegex(ValueError,'Range contract'):f.read(5)
  r.close.assert_called_once()
 def test_remote_refuses_wrong_offset(self):
  f=RangeFile('https://example.invalid/file',100);r=Mock(status_code=206,headers={'Content-Range':'bytes 1-5/100'});f.session.get=Mock(return_value=r)
  with self.assertRaises(ValueError):f.read(5)
 def test_remote_correct_exact_range(self):
  f=RangeFile('https://example.invalid/file',100);r=Mock(status_code=206,headers={'Content-Range':'bytes 0-4/100'},raw=io.BytesIO(b'abcde'));f.session.get=Mock(return_value=r);self.assertEqual(f.read(5),b'abcde');self.assertEqual(f.bytes,5)
 def test_crc_verified_member(self):
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('gt/gt.txt',b'original ground truth')
  raw=b.getvalue()
  with zipfile.ZipFile(io.BytesIO(raw)) as z:i=z.infolist()[0];entry={'path':i.filename,'offset':i.header_offset,'compressed':i.compress_size,'size':i.file_size,'crc':i.CRC}
  f=Mock();f.read.return_value=raw;f.bytes=len(raw)
  with patch('ml.reid13.remote.RangeFile',return_value=f):self.assertEqual(member_bytes('url',len(raw),entry)[0],b'original ground truth')
  with patch('ml.reid13.remote.RangeFile',return_value=f),self.assertRaisesRegex(ValueError,'CRC'):member_bytes('url',len(raw),{**entry,'crc':0})

@unittest.skipIf(torch is None,'optional isolated ML runtime')
class ModelTests(unittest.TestCase):
 def test_new_protocol_has_no_adjacent_positives(self):
  from ml.reid13.protocol import retrieval
  rows=[row(timestamp=i*.48) for i in range(63)];q,g,tq,tg=retrieval(rows);self.assertEqual(len(q),3);self.assertGreater(min(r['timestamp'] for r in g)-max(r['timestamp'] for r in q),10);self.assertTrue(all(4<=len(t)<=8 for t in tq+tg))
 def test_no_old_long_window_assumption(self):
  from ml.reid13.protocol import retrieval
  self.assertTrue(retrieval([row(timestamp=i*.48) for i in range(63)])[0])
 def test_supcon_finite_and_gradient(self):
  from ml.reid13.model import supcon
  z=torch.randn(8,16,requires_grad=True);y=torch.arange(4).repeat_interleave(2);teams=torch.zeros(8,dtype=torch.int64);loss=supcon(z,y,teams);self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
 def test_supcon_no_positives_zero(self):
  from ml.reid13.model import supcon
  z=torch.randn(4,16,requires_grad=True);self.assertEqual(float(supcon(z,torch.arange(4),torch.arange(4))),0)
 def test_pool_padding_and_normalization(self):
  from ml.reid13.model import Pooler
  torch.manual_seed(3);m=Pooler().eval();x=torch.randn(1,8,512);q=torch.ones(1,8);mask=torch.tensor([[True]*6+[False]*2]);a,w=m(x,q,mask);x[:,6:]=500;b,_=m(x,q,mask);self.assertTrue(torch.allclose(a,b,atol=1e-6));self.assertAlmostEqual(float(a.detach().norm()),1,places=5);self.assertEqual(float(w.detach()[:,6:].sum()),0)
 def test_pool_rejects_empty_evidence(self):
  from ml.reid13.model import Pooler
  with self.assertRaises(ValueError):Pooler()(torch.zeros(1,2,512),torch.ones(1,2),torch.zeros(1,2,dtype=torch.bool))
 def test_low_quality_pool_evidence_downweighted(self):
  from ml.reid13.model import Pooler
  _,w=Pooler()(torch.randn(1,2,512),torch.tensor([[1.,.01]]));self.assertGreater(float(w.detach()[0,0]),float(w.detach()[0,1]))
 def test_final_test_sealed(self):
  import ml.reid13.evaluate as e
  with tempfile.TemporaryDirectory() as d,patch.object(e,'ROOT',Path(d)):
   with self.assertRaisesRegex(ValueError,'sealed'):e.vectors([row('test')],'m9')
 def test_model_selection_refuses_existing_history(self):
  import ml.reid13.train as t
  with tempfile.TemporaryDirectory() as d,patch.object(t,'ROOT',Path(d)):
   (Path(d)/'training_history.json').write_text('[]')
   with self.assertRaisesRegex(ValueError,'already started'):t.train()
if __name__=='__main__':unittest.main()
