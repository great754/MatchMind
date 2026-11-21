"""Frozen annotations and exact served revisions must reproduce published scores."""
import copy
import json
from pathlib import Path
import shutil
import unittest
from tests import test_cv_source_m23 as source_fixture
from src.serving.cv import digest
from src.serving.source_review_evaluation import completed_review
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write
from src.evaluation.cv_m22_development import ball_metrics
from scripts.evaluate_cv_source_review import failure_traces


class CompletedEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture=source_fixture.NewSourceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.root=fixture.root;self.analysis=self.root/'analysis';self.review=fixture.registry.load('new')
        evaluation=self.root/'evaluation';self.snapshot=evaluation/'snapshot';self.snapshot.mkdir(parents=True)
        source={k:self.review.manifest[k] for k in ['recording_id','video_sha256','frames','fps','width','height']}
        tasks=[{'task_id':f'frame:{i}','recording_id':'new','frame':i,'timestamp':i/25,'proposals':[]} for i in range(3)]
        data={'schema_version':'cv-m23-source-review-v1','reference_role':'development_only','model_proposals_are_truth':False,
            'sources':{'new':source},'tasks':tasks,'source_suitability':{},'selection':{}}
        data['fingerprint']=signature(data);write(self.snapshot/'tasks.json',data)
        store=DevelopmentReviewStore(self.snapshot/'tasks.json',self.snapshot,approved_sources={'new'})
        for t in tasks:store.save('new',DevelopmentLabel(task_id=t['task_id'],reviewer_id='Synthetic test fixture',state='not_visible',reviewed_from_source=True))
        live=self.root/'live_labels.json';shutil.copyfile(self.snapshot/'labels.json',live)
        self.report={'schema_version':'cv-source-review-evaluation-v1','recording_id':'new','source':source|{'frames_sha256':self.review.manifest['frames_sha256']},
            'reference_role':'development_only','model_assisted':True,'independent_final_test':False,
            'analysis_report_sha256':digest(self.analysis/'report.json'),'perception_records_sha256':self.review.ball_review.manifest['perception_records_sha256'],
            'snapshot_root':str(self.snapshot),'annotation_snapshot_sha256':digest(self.snapshot/'labels.json'),
            'task_manifest_sha256':digest(self.snapshot/'tasks.json'),'dataset_fingerprint':data['fingerprint'],
            'completed_tasks':3,'live_labels_path':str(live),'evaluated_at_utc':'2026-11-21T00:00:00+00:00','failure_counts':{},'systems':{}}
        for method in ['full_m23','m21_temporal_offline']:
            directory=self.analysis/'new'/method;manifest=json.loads((directory/'manifest.json').read_text())
            with (directory/'ball_frames.jsonl').open() as stream:balls=[json.loads(line) for line in stream]
            self.report['systems'][method]={'metrics':ball_metrics(tasks,store.labels(),{'new':balls},data['sources']),
                'ball_frames_sha256':manifest['ball_frames_sha256'],'model_set_sha256':manifest['detector_sha256']}
        self.path=evaluation/'report.json';self.publish(self.report)

    def publish(self,report):
        write(self.path,report);self.entry={'path':str(self.path),'sha256':digest(self.path)}

    def load(self):return completed_review(self.entry,self.review,self.analysis)

    def test_scores_reproduce_without_exposing_raw_annotations(self):
        result=self.load();self.assertEqual(result['completed_tasks'],3)
        self.assertEqual(result['absent_frames'],3);self.assertIsNone(result['systems'][0]['ball']['precision'])
        self.assertFalse(result['labels_updated_since_snapshot']);self.assertNotIn('annotations',result)

    def test_changed_metrics_or_prediction_revision_are_rejected(self):
        for key in ['metrics','ball_frames_sha256']:
            report=copy.deepcopy(self.report)
            if key=='metrics':report['systems']['full_m23']['metrics']['ball']['tp']=1
            else:report['systems']['full_m23'][key]='a'*64
            self.publish(report)
            with self.assertRaises(ValueError):self.load()

    def test_changed_snapshot_or_source_are_rejected(self):
        report=copy.deepcopy(self.report);report['source']['video_sha256']='a'*64;self.publish(report)
        with self.assertRaises(ValueError):self.load()
        self.publish(self.report)
        with (self.snapshot/'labels.json').open('a') as stream:stream.write(' ')
        with self.assertRaises(ValueError):self.load()

    def test_live_corrections_mark_frozen_scores_stale(self):
        path=Path(self.report['live_labels_path']);path.write_text(path.read_text()+' ')
        self.assertTrue(self.load()['labels_updated_since_snapshot'])

    def test_bad_optional_evaluation_keeps_model_available(self):
        registry=self.fixture.registry;entry=registry.entries()['new'];entry['completed_review']={'path':str(self.path),'sha256':'a'*64}
        write(self.fixture.path,{'new':entry})
        self.assertTrue(registry.available('new'));meta=registry.metadata('new')
        self.assertIsNone(meta['completed_review']);self.assertIn('unavailable',meta['completed_review_diagnostic'])


class FailureTraceTests(unittest.TestCase):
    def test_missing_visual_weak_visual_and_anchor_policy_remain_distinct(self):
        def candidate(origin,score,eligible):return {'bbox':[1,1,4,4],'confidence':score,'observation_eligible':eligible,'origin':origin}
        pools=[[],[candidate('primary',.2,False)],[candidate('complement',.7,True)],[candidate('primary',.7,True)]]
        records=[{'methods':{'full_m23':{'candidates':p}}} for p in pools]
        tasks=[{'task_id':str(i),'frame':i,'selection_kind':'uniform_source_sequence'} for i in range(4)]
        labels=[{'task_id':str(i),'state':'visible','bbox':[1,1,4,4]} for i in range(4)]
        balls=[{'directly_observed':False,'image_bbox':None} for _ in range(4)]
        rows=failure_traces(tasks,labels,records,balls)
        self.assertEqual([r['reason'] for r in rows],['no_matching_raw_candidate','matching_candidate_below_fixed_visual_gate',
            'complement_only_gap_policy_did_not_admit_match','eligible_primary_not_selected_by_offline_association'])


if __name__=='__main__':unittest.main()
