"""M25 image coordinates, isolated revisions and optional-route safety."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.serving.dataset25 import create_router
from src.serving.cv import digest
from scripts.annotate_cv_m25 import convert

class DatasetTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.image=self.root/'image.png';self.image.write_bytes(b'unit test source, not a real training image')
  self.row={'frame_id':'fixture','recording_id':'test','frame':4,'width':1280,'height':720,'image':str(self.image),'image_sha256':digest(self.image),'proposals':[],'existing_reference':None}
  (self.root/'pilot_20.json').write_text(json.dumps({'schema_version':'cv-m25-pilot-v1','frames':[self.row]}));app=FastAPI();app.include_router(create_router(lambda:None,self.root));self.client=TestClient(app)
 def body(self):return {'task_id':'fixture','reviewer_id':'test-human','state':'visible','bbox':[10,20,3,4],'reviewed_from_source':True,'accepted_proposal':None,'rejected_proposals':[],'reviewed_proposals':[]}
 def test_read_does_not_approve(self):
  self.assertEqual(self.client.get('/cv/m25/batch').status_code,200);self.assertFalse((self.root/'human_revisions.json').exists())
 def test_revisions_are_append_only(self):
  body=self.body();self.assertEqual(self.client.post('/cv/m25/frames/fixture/labels',json=body).status_code,200);body['bbox']=[20,30,4,5];self.client.post('/cv/m25/frames/fixture/labels',json=body)
  rows=json.loads((self.root/'human_revisions.json').read_text());self.assertEqual(len(rows),2);self.assertEqual(rows[0]['bbox'],[10,20,3,4]);self.assertEqual(rows[1]['revision'],2)
 def test_frame_identity_and_bounds_rejected(self):
  body=self.body();body['bbox']=[1279,0,5,5];self.assertEqual(self.client.post('/cv/m25/frames/fixture/labels',json=body).status_code,422)
  body=self.body();body['task_id']='different';self.assertEqual(self.client.post('/cv/m25/frames/fixture/labels',json=body).status_code,422)
 def test_changed_image_not_accepted(self):
  self.image.write_bytes(b'changed');self.assertEqual(self.client.post('/cv/m25/frames/fixture/labels',json=self.body()).status_code,503)
 def test_unknown_proposal_not_accepted(self):
  body=self.body();body['accepted_proposal']='fabricated';self.assertEqual(self.client.post('/cv/m25/frames/fixture/labels',json=body).status_code,422)
 def test_gemini_yx_mapping(self):self.assertEqual(convert({'ball_visible':True,'box_2d':[100,200,120,220],'uncertain':False},1280,720),[256,72,25.6,14.4])
 def test_gemini_invalid_and_uncertain(self):
  self.assertIsNone(convert({'ball_visible':False,'box_2d':None,'uncertain':True},1280,720))
  for box in [[200,100,100,200],[0,0,1,1001],[0,0,float('nan'),20]]:
   with self.assertRaises(ValueError):convert({'ball_visible':True,'box_2d':box,'uncertain':False},1280,720)
 def test_gemini_resume_never_sends_verified_frames(self):
  from scripts import resume_cv_m25_gemini as resume
  frames=[{**self.row,'frame_id':'trusted'+str(i),'existing_reference':{'state':'visible'},'proposals':[]} for i in range(20)]
  (self.root/'pilot_20.json').write_text(json.dumps({'frames':frames,'summary':{}}))
  with patch.object(resume,'ROOT',self.root),patch.object(resume.requests,'post',side_effect=AssertionError('Verified image must not be sent')) as post:
   resume.run();post.assert_not_called()
 def test_shot_exclusion_persists_and_blocks_ball_labels(self):
  result=self.client.post('/cv/m25/frames/fixture/shot',json={'shot_type':'replay','reviewer_id':'test-human','reviewed_from_source':True});self.assertEqual(result.status_code,200)
  self.assertEqual(self.client.get('/cv/m25/batch').json()['frames'][0]['shot_type'],'replay')
  self.assertEqual(self.client.post('/cv/m25/frames/fixture/labels',json=self.body()).status_code,422)
  self.assertFalse((self.root/'human_revisions.json').exists())
 def test_missing_pilot_optional(self):
  (self.root/'pilot_20.json').unlink();self.assertEqual(self.client.get('/cv/m25/batch').status_code,503)

if __name__=='__main__':unittest.main()
