import copy,json,tempfile,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from pydantic import ValidationError
from src.vision.events.passes import PassConfig,VideoPass,detect_passes,signature
from src.serving.cv_events import CVEventReview
from src.serving.cv import digest


def footage(frames=20,fps=10):
    worlds=[];balls=[]
    for i in range(frames):
        people=[];entities=[]
        for ref,x in [('a',0),('b',200)]:
            people.append({'entity_id':ref,'current_segment_id':ref,'current_track_id':ref,'visibility_state':'VISIBLE','identity_status':'PROVISIONAL','team':{'label':'TEAM_A','confidence':.9},'current_position':None,'position_provenance':'UNKNOWN','position_uncertainty_m':None})
            entities.append({'track_id':ref,'image_bbox':[x,0,20,100],'visibility':'visible'})
        worlds.append({'frame':i,'timestamp':i/fps,'camera':{'epoch':0,'continuity':'continuous'},'entities':entities,'unassigned':[],'player_world':{'players':people}})
        x=10 if i<3 else 210 if i>12 else 10+(i-2)*200/11
        balls.append({'frame':i,'timestamp':i/fps,'directly_observed':True,'tracking_status':'TRACKED','image_position':[x,100],'track_id':1})
    return worlds,balls

class PassDetectionTests(unittest.TestCase):
    def run_case(self,worlds=None,balls=None):
        a,b=footage();return detect_passes('r',worlds or a,balls or b,10)
    def test_supported_release_flight_and_receiver_produce_one_pass(self):
        events,stats,_=self.run_case();self.assertEqual(stats['completed_pass_estimates'],1);self.assertEqual(len(events),1);e=events[0];self.assertEqual((e['sender_ref'],e['receiver_ref']),('segment:a','segment:b'));self.assertEqual(e['status'],'estimated');self.assertEqual(e['source'],'VIDEO_CV');self.assertEqual(e['ball_observed_fraction'],1)
    def test_image_pass_does_not_invent_metric_coordinates_or_goal_counts(self):
        events,stats,_=self.run_case();self.assertIsNone(events[0]['start_pitch']);self.assertIsNone(events[0]['end_pitch'])
        for key in ['goal_count','shot_count','pass_attempts','pass_accuracy']:self.assertIsNone(stats[key])
    def test_predicted_ball_is_never_accepted_as_visual_flight(self):
        w,b=footage()
        for r in b[5:11]:r.update(directly_observed=False,image_position=None,tracking_status='PREDICTED',predicted_image_position=[100,100])
        events,stats,_=self.run_case(w,b);self.assertEqual(stats['completed_pass_estimates'],0);self.assertTrue(events);self.assertIn('NO_OBSERVED_RELEASE_AND_FLIGHT',events[0]['reasons'])
    def test_long_ball_gap_does_not_become_a_complete_pass(self):
        w,b=footage()
        for r in b[5:10]:r.update(directly_observed=False,image_position=None,tracking_status='LOST')
        events,stats,_=self.run_case(w,b);self.assertEqual(stats['completed_pass_estimates'],0);self.assertIn('BALL_OBSERVATION_GAP',events[0]['reasons'])
    def test_unknown_team_and_opponent_transfer_are_candidates(self):
        for team in ['UNKNOWN','TEAM_B']:
            w,b=footage()
            for r in w:r['player_world']['players'][1]['team']['label']=team
            events,stats,_=self.run_case(w,b);self.assertEqual(stats['completed_pass_estimates'],0);self.assertEqual(events[0]['status'],'candidate')
    def test_cut_and_ball_track_change_do_not_count(self):
        w,b=footage();w[8]['camera'].update(epoch=1,continuity='cut')
        for r in w[9:]:r['camera']['epoch']=1
        for r in b[8:]:r['track_id']=2
        events,stats,_=self.run_case(w,b);self.assertEqual(stats['completed_pass_estimates'],0);self.assertIn('CAMERA_CUT_OR_EPOCH_CHANGE',events[0]['reasons']);self.assertIn('BALL_TRACK_DISCONTINUITY',events[0]['reasons'])
    def test_ambiguous_identity_or_self_reassociation_cannot_pass(self):
        for ambiguous in [True,False]:
            w,b=footage()
            for row in w:
                if ambiguous:row['player_world']['players'][0]['identity_status']='AMBIGUOUS'
                else:row['player_world']['players'][1]['entity_id']='a'
            _,stats,_=self.run_case(w,b);self.assertEqual(stats['completed_pass_estimates'],0)
    def test_ambiguous_nearby_players_are_not_nearest_actor_attribution(self):
        w,b=footage()
        for r in w:
            p=copy.deepcopy(r['player_world']['players'][0]);p.update(entity_id='c',current_segment_id='c',current_track_id='c');r['player_world']['players'].append(p);r['entities'].append({'track_id':'c','visibility':'visible','image_bbox':[5,0,20,100]})
        _,stats,_=self.run_case(w,b);self.assertEqual(stats['completed_pass_estimates'],0);self.assertGreater(stats['frame_diagnostics'].get('AMBIGUOUS_CONTACT',0),0)
    def test_clock_mismatch_rejected(self):
        w,b=footage();b[5]['timestamp']=100
        with self.assertRaises(ValueError):self.run_case(w,b)
    def test_candidate_cannot_be_relabelled_as_a_supported_pass(self):
        e=self.run_case()[0][0];e.update(status='estimated',reasons=['TEAM_UNCERTAIN'])
        with self.assertRaises(ValidationError):VideoPass.model_validate(e)

class EventReviewTests(unittest.TestCase):
    def make(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);p=Path(self.tmp.name);w,b=footage();events,summary,_=detect_passes('r',w,b,10);self.events=events
        (p/'events.json').write_text(json.dumps(events));(p/'evidence_frames.jsonl').write_text('{}\n');(p/'world').write_text('world');(p/'video').write_text('video');(p/'balls').write_text('balls');(p/'canonical').write_text('canonical')
        base={'recording_id':'r','fps':10,'frames':20,'width':300,'height':120,'video_sha256':'v'*64,'frames_sha256':'w'*64};bm={'version':'m14','ball_frames_sha256':'b'*64,'detector_sha256':'d'*64}
        cfg=PassConfig().model_dump();m={**base,'schema_version':'cv-passes-v1','world_frames_sha256':base['frames_sha256'],'ball_version':'m14',**{k:bm[k] for k in ['ball_frames_sha256']},'ball_detector_sha256':bm['detector_sha256'],'config':cfg,'config_sha256':signature(cfg),'events_sha256':digest(p/'events.json'),'evidence_sha256':digest(p/'evidence_frames.jsonl'),'statistics':summary,'unavailable':['goals','shots']};m.pop('frames_sha256');(p/'manifest.json').write_text(json.dumps(m))
        people=[SimpleNamespace(entity_id=who,current_segment_id=who,visibility_state='VISIBLE',identity_status='PROVISIONAL') for who in ['a','b']]
        ball=SimpleNamespace(manifest=bm,frame=lambda _:SimpleNamespace(directly_observed=True,track_id=1),path=p/'balls',canonical=p/'canonical',stamps=((p/'balls').stat().st_mtime_ns,(p/'canonical').stat().st_mtime_ns))
        review=SimpleNamespace(manifest=base,ball_review=ball,frame=lambda _:SimpleNamespace(player_world=SimpleNamespace(players=people)),data=p/'world',data_stamp=(p/'world').stat().st_mtime_ns,video=p/'video',video_stamp=(p/'video').stat().st_mtime_ns)
        return p,review,m
    def test_source_validated_events_load_and_keep_unknown_stats(self):
        p,r,m=self.make();layer=CVEventReview(p,r);d=layer.public();self.assertEqual(d['statistics']['completed_pass_estimates'],1);self.assertIsNone(d['statistics']['goal_count'])
    def test_corruption_and_source_mismatch_are_rejected(self):
        for corrupt in [True,False]:
            p,r,m=self.make()
            if corrupt:(p/'events.json').write_text('[]')
            else:r.ball_review.manifest['ball_frames_sha256']='changed'
            with self.assertRaises(ValueError):CVEventReview(p,r)
    def test_changed_event_file_is_unavailable_after_validation(self):
        from fastapi import HTTPException
        p,r,m=self.make();layer=CVEventReview(p,r);(p/'events.json').write_text('[]')
        with self.assertRaises(HTTPException):layer.public()
    def test_unsupported_goal_count_cannot_become_valid_by_manifest(self):
        p,r,m=self.make();m['statistics']['goal_count']=0;(p/'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError):CVEventReview(p,r)

class CVAssistantTests(unittest.TestCase):
    def setUp(self):
        from src.assistant.service import UnifiedMatchAssistant
        from src.highlights.service import HighlightService
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        w,b=footage();events,summary,_=detect_passes('r',w,b,10)
        candidate={**events[0],'event_id':'cv_candidate','status':'candidate','event_type':'transfer_candidate','reasons':['BALL_OBSERVATION_GAP']}
        self.layer=SimpleNamespace(events=events+[candidate],manifest={'fps':10,'frames':20,'ball_version':'m14','events_sha256':'a'*64,'world_frames_sha256':'b'*64,'ball_frames_sha256':'c'*64},check=Mock())
        players=[{'player_ref':'segment:'+who,'label':'Player '+label,'actor_id':None,'mot_id':None,'entity_id':who,'segment_id':who,'team':'left','start':0.,'end':2.,'source_signature':'s'} for who,label in [('a','A'),('b','B')]]
        cat=SimpleNamespace(players=players,by_ref={p['player_ref']:p for p in players},recording='r',tools=None,boxes={},source_signature='s',fps=10,frames=20)
        cv=SimpleNamespace(event_review=self.layer)
        h=HighlightService({},{} ,root=Path(self.tmp.name)/'highlights');self.addCleanup(h.executor.shutdown,wait=True)
        self.a=UnifiedMatchAssistant({}, {'r':cv},h,root=Path(self.tmp.name)/'assistant');self.a.catalogs['r']=cat;self.session=uuid.uuid4()
    def ask(self,q,**kw):
        from src.assistant.schema import AssistantRequest
        return self.a.ask('r',AssistantRequest(question=q,session_id=self.session,provider='local',**kw))
    def test_segment_refs_do_not_all_match_null_actor_ids(self):
        idx=self.a.index('r');self.assertEqual(len(idx.select(['pass'],['segment:a'])),1);self.assertEqual(idx.select(['pass'],['segment:b']),[]);self.assertEqual(len(idx.select(['reception'],['segment:b'])),1)
    def test_profile_has_pass_count_but_unknown_goals_and_accuracy(self):
        r=self.ask('Show me their stats',selected_player='segment:a');stats={s['key']:s['value'] for s in r['player_profile']['stats']};self.assertEqual(stats['passes'],1);self.assertIsNone(stats['goals']);self.assertIsNone(stats['pass_accuracy']);self.assertNotIn('scored 0',r['text'])
    def test_global_pass_count_and_unsupported_goals_are_distinct(self):
        r=self.ask('How many passes were detected?');self.assertEqual(r['metrics'][0]['value'],1)
        r=self.ask('How many goals?');self.assertIsNone(r['highlight']);self.assertEqual(r['metrics'],[]);self.assertIn('not available',r['text'])
    def test_candidate_review_is_explicit_and_uses_clip_clock_without_a_fake_half(self):
        with patch.object(self.a.highlights,'submit',return_value={'job_id':'review'}) as submit:
            r=self.ask('Make a review video of possible passes');plan,rows,idx,overlay=submit.call_args.kwargs['selection'];self.assertTrue(plan.review);self.assertTrue(any(not e['safe_for_automatic'] for e in rows));self.assertIsNone(idx.timebase.period);self.assertTrue(overlay['candidate_review']);self.assertIn('not confirmed',r['text'])
    def test_candidate_changes_invalidate_the_event_index_hash(self):
        idx=self.a.index('r');old=idx.hash;idx.candidates[0]['validation_state']='changed';idx.rehash();self.assertNotEqual(old,idx.hash)

    def test_cv_clock_does_not_claim_an_unverified_match_half(self):
        r=self.ask('Show passes in the first half');self.assertIn('can’t verify which half',r['text']);self.assertIsNone(r['highlight'])

    def test_hosted_plan_cannot_silently_drop_explicit_candidate_review(self):
        from src.assistant.planner import IntentAndQueryPlanner
        client=SimpleNamespace(client=SimpleNamespace(generate_structured=lambda *args:json.dumps({'operation':'actions','actions':['pass'],'video':True,'review':False})))
        with patch('src.providers.factory.build_provider',return_value=client):
            plan,_,provider,_=IntentAndQueryPlanner().plan('Make a review video of possible passes',provider='gemini',roster=[])
        self.assertTrue(plan.review);self.assertEqual(provider,'local_fallback')
