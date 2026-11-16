"""Synthetic safety regressions only; these tests do not measure real-world accuracy."""
import copy,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from fastapi import HTTPException
from pydantic import ValidationError
from src.vision.ball19.tracker import VisualBallTracker,AssociationConfig
from src.vision.events.passes import PassConfig,signature
from src.vision.events.passes_m19 import detect_m19,existing_m9_receiver
from src.vision.events.m19_players import short_links
from src.serving.cv_review_labels import HumanReviewStore,HumanAnnotation
from src.evaluation.cv_m19 import evaluate,counts
from tests.test_cv_passes import footage
from tests import test_cv_passes as pass_fixtures

CAMERA={'epoch':0,'continuity':'continuous','calibration_valid':False,'calibration':None,'motion_evidence':{}}
def candidate(x=50,confidence=.9,key='ball'):
    return {'candidate_id':key,'bbox':[x-2,48,4,4],'confidence':confidence,'inference_view':'global'}

class BallM19Tests(unittest.TestCase):
    def tracker(self,**kw):return VisualBallTracker(300,200,.35,AssociationConfig(**kw))
    def test_one_gate_requires_two_actual_observations(self):
        t=self.tracker();a=t.step(0,0,[candidate()],CAMERA);b=t.step(1,.04,[candidate(51)],CAMERA)
        self.assertFalse(a.directly_observed);self.assertTrue(b.directly_observed);self.assertEqual(b.image_position,(51,50));self.assertIsNone(b.pitch_position)
    def test_predictions_never_become_observed_positions(self):
        t=self.tracker()
        for i in range(2):t.step(i,i*.04,[candidate()],CAMERA)
        b=t.step(2,.08,[],CAMERA);self.assertFalse(b.directly_observed);self.assertIsNone(b.image_position);self.assertIsNone(b.image_bbox);self.assertIsNotNone(b.predicted_image_position);self.assertIsNone(b.pitch_position)
        c=t.step(3,.5,[],CAMERA);self.assertEqual(c.tracking_status,'LOST');self.assertIsNone(c.predicted_image_position)
    def test_single_candidate_cannot_update_multiple_hypotheses(self):
        t=self.tracker();t.step(0,0,[candidate(50,key='a'),candidate(55,key='b')],CAMERA);t.step(1,.04,[candidate(51)],CAMERA)
        self.assertEqual(sum(h['candidate'] is not None for h in t.hypotheses),1);self.assertEqual(sum(h['confirmed'] for h in t.hypotheses),1)
    def test_competing_supported_trajectories_remain_ambiguous(self):
        t=self.tracker()
        for i in range(2):b=t.step(i,.04*i,[candidate(40,key='a'),candidate(180,key='b')],CAMERA)
        self.assertEqual(b.tracking_status,'AMBIGUOUS');self.assertFalse(b.directly_observed);self.assertIsNone(b.image_position)
    def test_cut_resets_birth_and_prediction(self):
        t=self.tracker()
        for i in range(2):t.step(i,.04*i,[candidate()],CAMERA)
        c=t.step(2,.08,[candidate()],{**CAMERA,'epoch':1,'continuity':'cut'})
        self.assertFalse(c.directly_observed);self.assertIsNone(c.predicted_image_position)
    def test_frozen_confidence_threshold_is_respected(self):
        t=self.tracker()
        for i in range(4):b=t.step(i,.04*i,[candidate(confidence=.34)],CAMERA)
        self.assertFalse(b.directly_observed);self.assertEqual(len(t.hypotheses),0)
    def test_occluded_prediction_and_reacquisition_share_track(self):
        t=self.tracker()
        for i in range(2):a=t.step(i,.04*i,[candidate()],CAMERA)
        b=t.step(2,.08,[],CAMERA,[{'bbox':[40,0,20,100]}]);c=t.step(3,.12,[candidate()],CAMERA)
        self.assertEqual(b.tracking_status,'OCCLUDED');self.assertEqual(a.track_id,c.track_id);self.assertTrue(c.directly_observed)
    def test_birth_hits_must_be_consecutive(self):
        t=self.tracker();t.step(0,0,[candidate()],CAMERA);t.step(1,.04,[],CAMERA);b=t.step(2,.08,[candidate()],CAMERA);self.assertFalse(b.directly_observed)
    def test_camera_warp_and_variable_elapsed_time(self):
        t=self.tracker()
        with patch('src.vision.ball19.tracker.camera_motion',return_value=(None,'NONE')):t.step(0,0,[candidate()],CAMERA)
        import numpy as np
        with patch('src.vision.ball19.tracker.camera_motion',return_value=(np.array([[1,0,20],[0,1,0],[0,0,1]]),'TEST')):
            b=t.step(1,.1,[candidate(70)],CAMERA);c=t.step(2,.3,[candidate(90)],CAMERA)
        self.assertTrue(b.directly_observed);self.assertTrue(c.directly_observed);self.assertLess(abs(c.estimated_velocity[0]),1e-6)
    def test_invalid_clock_and_bounds_rejected(self):
        t=self.tracker();t.step(0,0,[],CAMERA)
        with self.assertRaises(ValueError):t.step(1,0,[],CAMERA)
        with self.assertRaises(ValueError):self.tracker().step(0,0,[candidate(500)],CAMERA)
    def test_birth_gate_cannot_be_disabled(self):
        with self.assertRaises(ValueError):self.tracker(minimum_birth_hits=1)

class PassM19Tests(unittest.TestCase):
    def test_config_roundtrip_preserves_event_hash(self):
        a=PassConfig().model_dump();self.assertEqual(signature(a),signature(PassConfig.model_validate(a).model_dump()))
    def test_same_numeric_gates(self):
        self.assertEqual(PassConfig().minimum_ball_observed_fraction,.5)
        w,b=footage();events,stats,_=detect_m19('r',w,b,10);self.assertEqual(stats['completed_pass_estimates'],1);self.assertTrue(events[0]['ball_frames'])
    def test_lookup_only_unique_existing_identity(self):
        receiver={'entity':'r'};p={'entity':'r','ref':'old'}
        self.assertEqual(existing_m9_receiver({'people':{'x':p}},receiver),p)
        self.assertIsNone(existing_m9_receiver({'people':{'x':p,'y':p}},receiver));self.assertIsNone(existing_m9_receiver({'people':{'x':{'entity':'different'}}},receiver))
    def test_receiver_segment_change_preserves_explicit_existing_entity(self):
        w,b=footage()
        for r in w[:8]:r['player_world']['players'][1]['current_segment_id']='b-old'
        events,stats,_=detect_m19('r',w,b,10);self.assertEqual(stats['completed_pass_estimates'],1);self.assertEqual(events[0]['receiver_release_ref'],'segment:b-old');self.assertEqual(events[0]['receiver_release_entity'],'b')
    def test_gate_replay_rejects_coherently_rehashed_forgery(self):
        fixture=pass_fixtures.EventReviewTests();fixture.setUp();p,r,m=fixture.make()
        try:
            from src.serving.cv_events import CVEventReview
            from src.serving.cv import digest
            w,b=footage();events,_,evidence=detect_m19('r',w,b,10);(p/'evidence_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in evidence));events[0]['start_image']=[100,100];(p/'events.json').write_text(json.dumps(events));m.update(algorithm_version='cv-pass-m19-v1',events_sha256=digest(p/'events.json'),evidence_sha256=digest(p/'evidence_frames.jsonl'));(p/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'gate replay'):CVEventReview(p,r)
        finally:fixture.doCleanups()

class PlayerM19Tests(unittest.TestCase):
    def run_links(self,entities=('p','p'),teams=('TEAM_A','TEAM_A'),embeddings=(1,1),collision=False):
        import numpy as np
        cfg=json.loads(Path('configs/vision/cv_m19.json').read_text())['player_links']
        def seg(key,start,end,embedding):return SimpleNamespace(key=key,start=start,end=end,epoch=0,embedding=np.array([embedding,1.]),histogram=np.array([1.,0]),quality=.9,evidence_frames=[start,start+1,end],observations=[{'foot':[10,100],'bbox':[0,0,20,100]}])
        segments=[seg('a',0,2,embeddings[0]),seg('b',4,6,embeddings[1])]
        people=[{'current_segment_id':key,'entity_id':entity,'team':{'label':team,'confidence':.9}} for key,entity,team in zip(['a','b'],entities,teams)]
        if collision:segments.append(seg('c',4,6,1));people.append({**people[1],'current_segment_id':'c'})
        worlds=[{'player_world':{'players':people}}];original=copy.deepcopy(worlds)
        with patch('src.vision.events.m19_players.build_segments',return_value=(segments,None)):result=short_links([],{},worlds,25,cfg)
        self.assertEqual(worlds,original);return result
    def test_existing_identity_support_never_rewrites_manual_identity(self):
        r=self.run_links();self.assertEqual(r['supported_existing_identity_links'],1);self.assertFalse(r['links'][0]['manual_identity_propagated']);self.assertEqual(r['actual_m9_identity_rewrites'],0)
    def test_cross_entity_is_a_review_proposal_not_a_merge(self):
        r=self.run_links(entities=('p','q'));self.assertEqual(r['links'][0]['status'],'REVIEW_REQUIRED_NO_IDENTITY_MERGE');self.assertEqual(r['actual_m9_identity_rewrites'],0)
    def test_conflicting_teams_or_weak_appearance_cannot_link(self):
        self.assertEqual(self.run_links(teams=('TEAM_A','TEAM_B'))['links'],[]);self.assertEqual(self.run_links(embeddings=(1,-1))['links'],[])
    def test_competing_successors_do_not_duplicate_identity(self):
        r=self.run_links(collision=True);self.assertEqual(r['links'],[]);self.assertEqual(len(r['ambiguous']),2)

class HumanReviewM19Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.store=HumanReviewStore(root=self.tmp.name);self.task=next(t for t in self.store.dataset['tasks'] if t['kind']=='ball');self.name=self.task['recording_id']
    def label(self,**kw):return HumanAnnotation.model_validate({'task_id':self.task['task_id'],'reviewer_id':'TEST','reviewed_from_source':True,'model_assisted':False,'state':'visible','bbox':[1,2,3,4],**kw})
    def test_sealed_source_independent_tasks(self):
        self.assertEqual(len(self.store.dataset['tasks']),60);self.assertFalse(self.store.dataset['selection_uses_model_candidates']);self.assertEqual(self.store.labels(),[])
    def test_save_isolated_labels_and_hash_does_not_change_model(self):
        self.store.save(self.name,self.label());rows=self.store.labels();self.assertEqual(len(rows),1);self.assertEqual(rows[0]['source_video_sha256'],self.store.dataset['sources'][self.name]['video_sha256']);self.store.save(self.name,self.label(notes='revised'));self.assertEqual(len(self.store.labels()),1)
    def test_no_prediction_generated_or_unreviewed_labels(self):
        for kw in [{'model_assisted':True},{'origin':'model'},{'reviewed_from_source':False}]:
            with self.assertRaises(ValidationError):self.label(**kw)
    def test_invalid_or_nonfinite_box(self):
        for box in [[-1,2,3,4],[1,2,0,4],[1,2,10000,4],[math.nan,2,3,4]]:
            with self.assertRaises((HTTPException,ValidationError)):self.store.save(self.name,self.label(bbox=box))
    def test_wrong_recording_rejected(self):
        with self.assertRaises(HTTPException):self.store.save('other',self.label())
    def test_visible_box_required_hidden_ball_must_be_unboxed(self):
        for kw in [{'bbox':None},{'state':'not_visible'}]:
            with self.assertRaises(HTTPException):self.store.save(self.name,self.label(**kw))
    def test_unknown_visibility_is_valid_but_not_a_negative(self):
        self.store.save(self.name,self.label(state='uncertain',bbox=None));self.assertEqual(self.store.labels()[0]['state'],'uncertain')
    def test_pass_reference_must_fall_inside_window(self):
        t=next(t for t in self.store.dataset['tasks'] if t['kind']=='passes');label=self.label(task_id=t['task_id'],state='reviewed',bbox=None,exhaustive=True,passes=[{'release_frame':0,'receipt_frame':9999}])
        with self.assertRaises(HTTPException):self.store.save(t['recording_id'],label)

class IndependentEvaluationM19Tests(unittest.TestCase):
    def setUp(self):
        self.tasks=[{'task_id':'b','recording_id':'r','kind':'ball','frame':0},{'task_id':'p','recording_id':'r','kind':'passes','start_frame':0,'end_frame':30,'start':0,'end':3}];self.protocol={'sources':{'r':{'fps':10,'video_sha256':'source'}},'ball_box_match_iou':.25,'pass_endpoint_tolerance_seconds':.25};self.balls={'r':[{'directly_observed':True,'image_bbox':[1,1,4,4]}]}
    def label(self,**kw):return {'task_id':'b','origin':'human','reviewed_from_source':True,'model_assisted':False,'source_video_sha256':'source','state':'visible','bbox':[1,1,4,4],**kw}
    def evaluate(self,rows,events=None):return evaluate(self.tasks,rows,self.balls,events or {},self.protocol)
    def test_zero_labels_is_not_accuracy(self):
        r=self.evaluate([]);self.assertEqual(r['status'],'awaiting_independent_human_labels');self.assertIsNone(r['ball']['precision']);self.assertIsNone(r['passes']['recall']);self.assertIsNone(r['identity']['pair_accuracy'])
    def test_unknown_ball_excluded(self):
        r=self.evaluate([self.label(state='uncertain')]);self.assertEqual(r['ball']['reviewed_frames'],0);self.assertEqual(r['ball']['unknown_visibility_frames'],1)
    def test_source_hash_and_model_label_rejected(self):
        for kw in [{'source_video_sha256':'wrong'},{'origin':'model'},{'model_assisted':True}]:
            with self.assertRaises(ValueError):self.evaluate([self.label(**kw)])
    def test_duplicate_references_require_adjudication(self):
        with self.assertRaises(ValueError):self.evaluate([self.label(),self.label()])
    def test_synthetic_metric_arithmetic_not_accuracy_claim(self):
        r=self.evaluate([self.label()]);self.assertEqual(r['ball']['tp'],1);self.assertIsNotNone(r['ball']['precision_wilson95']);self.assertLess(r['ball']['precision_wilson95'][0],1)
    def test_predicted_position_cannot_match_human_reference(self):
        self.balls['r'][0].update(directly_observed=False,image_bbox=None,predicted_image_position=[3,3]);r=self.evaluate([self.label()]);self.assertEqual(r['ball']['fn'],1);self.assertEqual(r['ball']['tp'],0)
    def test_candidate_events_excluded_and_exhaustive_negative_counts_false_positive(self):
        event={'status':'estimated','start_frame':5,'end_frame':10,'start':.5,'end':1};candidate={**event,'status':'candidate'};label=self.label(task_id='p',state='reviewed',exhaustive=True,passes=[])
        r=self.evaluate([label],{'r':[event,candidate]});self.assertEqual(r['passes']['fp'],1);self.assertEqual(r['passes']['exhaustively_reviewed_windows'],1)
    def test_incomplete_and_uncertain_pass_windows_excluded(self):
        for extra in [{'exhaustive':False,'passes':[]},{'exhaustive':True,'passes':[{'outcome':'uncertain'}]}]:
            r=self.evaluate([self.label(task_id='p',state='reviewed',**extra)]);self.assertEqual(r['passes']['exhaustively_reviewed_windows'],0);self.assertIsNone(r['passes']['recall'])
    def test_one_to_one_pass_endpoint_matching(self):
        event={'status':'estimated','start_frame':5,'end_frame':10,'start':.5,'end':1};label=self.label(task_id='p',state='reviewed',exhaustive=True,passes=[{'outcome':'completed','release_frame':5,'receipt_frame':10}]);r=self.evaluate([label],{'r':[event,event]});self.assertEqual(r['passes']['tp'],1);self.assertEqual(r['passes']['fp'],1)
    def test_metrics_with_matches_are_json_serializable(self):
        event={'status':'estimated','start_frame':5,'end_frame':10,'start':.5,'end':1};label=self.label(task_id='p',state='reviewed',exhaustive=True,passes=[{'outcome':'completed','release_frame':5,'receipt_frame':10}]);result=self.evaluate([label],{'r':[event]});json.dumps(result,allow_nan=False)
    def test_identity_pairs_use_unique_box_matches_and_exclude_unmatched(self):
        task={'task_id':'i','recording_id':'r','kind':'identity','frame':0,'other_frame':1};label=self.label(task_id='i',state='same',bbox=[0,0,10,20],other_bbox=[1,0,10,20]);players={'r':{0:[{'entity':'p','bbox':[0,0,10,20]}],1:[{'entity':'p','bbox':[1,0,10,20]}]}}
        result=evaluate([task],[label],self.balls,{},self.protocol,players);self.assertEqual(result['identity']['pair_accuracy'],1)
        players['r'][1].append({'entity':'q','bbox':[1,0,10,20]});result=evaluate([task],[label],self.balls,{},self.protocol,players);self.assertIsNone(result['identity']['pair_accuracy']);self.assertEqual(result['identity']['unknown_or_unmatched_pairs'],1)
    def test_zero_denominators_remain_undefined(self):
        self.assertIsNone(counts(0,0,0)['f1']);self.assertIsNone(counts(0,0,1)['precision'])

if __name__=='__main__':unittest.main()
