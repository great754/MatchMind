"""Source-specific development labels stay isolated from the completed M22 set."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.vision.events.passes import signature


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.path=self.root/'tasks.json'
        self.data={'schema_version':'cv-m23-source-review-v1','reference_role':'development_only','model_proposals_are_truth':False,
            'publication_status':'source_sequences_visually_inspected',
            'sources':{'giroud':{'recording_id':'giroud','width':100,'height':80,'fps':25,'frames':100,'video_sha256':'a'*64}},
            'tasks':[{'task_id':'giroud:0','recording_id':'giroud','frame':0,'timestamp':0.,'proposals':[
                {'proposal_id':'p','bbox':[10,10,5,5],'confidence':.6},{'proposal_id':'q','bbox':[40,40,5,5],'confidence':.5}]}],
            'source_suitability':{},'selection':{}}
        self.write(self.data)
        self.store=DevelopmentReviewStore(self.path,self.root/'new-labels',approved_sources={'giroud'})

    def write(self,data):
        data={k:v for k,v in data.items() if k!='fingerprint'};data['fingerprint']=signature(data)
        self.path.write_text(json.dumps(data))

    def label(self,**updates):
        return DevelopmentLabel(**({'task_id':'giroud:0','reviewer_id':'AB','state':'visible','bbox':[10,10,5,5],
            'accepted_proposal':'p','rejected_proposals':['q'],'reviewed_proposals':['p','q'],'reviewed_from_source':True}|updates))

    def test_opening_review_does_not_create_any_labels(self):
        self.assertEqual(self.store.public('giroud')['labels'],[])
        self.assertFalse((self.root/'new-labels').exists())

    def test_new_source_requires_explicit_scope_and_own_schema(self):
        with self.assertRaises(ValueError):DevelopmentReviewStore(self.path,self.root/'wrong')
        with self.assertRaises(ValueError):DevelopmentReviewStore(self.path,self.root/'wrong',approved_sources={'team_goal'})
        self.write(self.data|{'schema_version':'cv-m22-development-review-v1'})
        with self.assertRaises(ValueError):DevelopmentReviewStore(self.path,self.root/'wrong',approved_sources={'giroud'})

    def test_incomplete_proposal_review_cannot_be_saved(self):
        for reviewed in [[],['p'],['p','p','q']]:
            with self.subTest(reviewed=reviewed),self.assertRaises(ValueError):self.store.save('giroud',self.label(reviewed_proposals=reviewed))

    def test_accepted_box_and_rejection_regions_must_match_explicit_decisions(self):
        for updates in [{'bbox':[12,12,5,5]},{'accepted_proposal':'missing'},
            {'accepted_proposal':None,'bbox':[40,40,5,5]}]:
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.store.save('giroud',self.label(**updates))

    def test_uncertain_rejected_region_is_not_whole_frame_absence(self):
        self.store.save('giroud',self.label(state='uncertain',bbox=None,accepted_proposal=None))
        row=self.store.labels()[0];self.assertEqual(row['state'],'uncertain')
        self.assertEqual(row['hard_negative_regions'][0]['bbox'],[40,40,5,5])
        self.assertEqual(row['reference_role'],'development_only');self.assertTrue(row['model_assisted'])

    def test_correction_replaces_only_same_reviewer_and_task(self):
        self.store.save('giroud',self.label());self.store.save('giroud',self.label(reviewer_id='CD'))
        self.store.save('giroud',self.label(state='not_visible',bbox=None,accepted_proposal=None,rejected_proposals=['p','q']))
        rows=self.store.labels();self.assertEqual(len(rows),2)
        self.assertEqual(next(r for r in rows if r['reviewer_id']=='AB')['state'],'not_visible')
        self.assertEqual(next(r for r in rows if r['reviewer_id']=='CD')['state'],'visible')

    def test_source_mismatch_bounds_and_modified_task_version_are_rejected(self):
        with self.assertRaises(ValueError):self.store.save('team_goal',self.label())
        for change in [{'frame':100},{'timestamp':.4}]:
            data=copy.deepcopy(self.data);data['tasks'][0].update(change);self.write(data)
            with self.assertRaises(ValueError):DevelopmentReviewStore(self.path,self.root/'wrong',approved_sources={'giroud'})
        data=copy.deepcopy(self.data);data['tasks'][0]['proposals'][0]['bbox']=[99,10,5,5];self.write(data)
        with self.assertRaises(ValueError):DevelopmentReviewStore(self.path,self.root/'wrong',approved_sources={'giroud'})
        with self.assertRaises(ValueError):self.store.save('giroud',self.label())

    def test_invalid_source_clock_is_a_validation_error(self):
        for change in [{'fps':0},{'fps':-1},{'width':0},{'frames':-1}]:
            data=copy.deepcopy(self.data);data['sources']['giroud'].update(change);self.write(data)
            with self.subTest(change=change),self.assertRaises(ValueError):DevelopmentReviewStore(self.path,self.root/'wrong',approved_sources={'giroud'})

    def test_route_saves_and_exports_only_to_its_isolated_label_store(self):
        from src.serving.football22_review import create_review_router
        router,store=create_review_router(lambda:None,{'giroud':SimpleNamespace(manifest=self.data['sources']['giroud'])},
            manifest=self.path,root=self.root/'http-labels',prefix='/cv/m23',approved_sources={'giroud'})
        app=FastAPI();app.include_router(router);client=TestClient(app)
        base='/cv/m23/matches/giroud/review'
        self.assertEqual(client.get(base+'/tasks').json()['labels'],[])
        self.assertFalse((self.root/'http-labels').exists())
        response=client.post(base+'/labels',json=self.label().model_dump(mode='json'))
        self.assertEqual(response.status_code,200,response.text);self.assertTrue(response.json()['saved'])
        self.assertEqual(len(store.labels()),1);self.assertEqual(store.labels()[0]['source_video_sha256'],'a'*64)
        export=client.get('/cv/m23/review/export').json()
        self.assertEqual(export['schema_version'],'cv-m23-temporal-supervision-v1');self.assertEqual(len(export['annotations']),1)
        self.assertEqual(client.post('/cv/m23/matches/team_goal/review/labels',json=self.label().model_dump(mode='json')).status_code,404)
        self.assertFalse((self.root/'new-labels').exists())


if __name__=='__main__':unittest.main()
