import json,math,tempfile,threading,time,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.highlights.schema import HighlightQueryPlan as Plan,HighlightRequest,VideoTimebase,HighlightManifest
from src.highlights.agent import HighlightAgent
from src.highlights.index import MatchEventIndex
from src.highlights.boundaries import ClipBoundaryEngine
from src.highlights.render import HighlightRenderer,HighlightError,probe,run
from src.highlights.service import HighlightService
from src.intelligence.schema import CanonicalEvent

def event(kind='GOAL',frame=100,**kwargs):
    return CanonicalEvent(event_id='e'+str(frame)+kind,group_id='g',timestamp=frame/25,frame=frame,type=kind,source='bas',validated_by_bas=True,validation_state='bas_reference',bas_annotation_id=frame,significance=100,metrics={'action_label':'GOAL' if kind=='GOAL' else 'SHOT'},**kwargs)
def tools(events=None):return SimpleNamespace(summary={'fps':25,'frames':500,'half':1,'source_offset_frames':5999,'scope':'clip'},events=events if events is not None else [event(),event(frame=300)])

class PlanTests(unittest.TestCase):
    def test_synonyms(self):
        a=HighlightAgent()
        for q,kind,selection in [('Show me all the goals','goal','all'),('Every time they scored','goal','all'),('Show the pass that went the farthest','pass','maximum'),('Show the buildup to the second goal','goal','ordinal'),('Show every attempt','shot','all')]:
            with self.subTest(q=q):p=a.parse(q);self.assertEqual(p.event_types,[kind]);self.assertEqual(p.selection,selection)
    def test_filters_range_duration_and_combining(self):
        p=HighlightAgent().parse('Show goals and shots by the left team between 1:00 and 2:30');self.assertEqual(p.event_types,['goal','shot']);self.assertEqual((p.team,p.start,p.end),('left',60,150))
        p=HighlightAgent().parse('Make a 60-second highlight video');self.assertEqual((p.selection,p.duration),('recap',60))
    def test_followup(self):
        a=HighlightAgent();p=a.parse('Show all goals');self.assertEqual(a.parse('Just show the second one',p).ordinal,2)
        self.assertEqual(a.parse('Who made that pass?',a.parse('What was the longest pass?')).intent,'identity')
    def test_no_unrecognized_filter_broadening(self):
        self.assertEqual(HighlightAgent().parse('Show goals by Liverpool').intent,'unsupported')
    def test_closed_schema_and_limits(self):
        for value in [{'timestamp':12},{'duration':999},{'selection':'ordinal'},{'start':20,'end':10},{'duration':float('nan')},{'event_ids':['invented']}]:
            with self.subTest(value=value),self.assertRaises(ValidationError):Plan.model_validate(value)
    def test_provider_invalid_plan_falls_back(self):
        with patch('src.providers.factory.build_provider',side_effect=RuntimeError('secret')):
            p,name,diagnostic=HighlightAgent().interpret('Show goals',provider='gemini');self.assertEqual(name,'local_fallback');self.assertNotIn('secret',diagnostic)

class ProviderContractTests(unittest.TestCase):
    def test_microsoft_uses_the_existing_structured_tool_contract(self):
        from src.providers.base import ToolPlan
        expected=HighlightAgent().parse('Show all goals')
        fake=SimpleNamespace(plan=lambda q,s:ToolPlan(calls=[{'name':'highlight_query','arguments':expected.model_dump()}]))
        with patch('src.providers.factory.build_provider',return_value=fake):
            p,name,diagnostic=HighlightAgent().interpret('Show all goals',provider='microsoft');self.assertEqual(name,'microsoft');self.assertEqual(p,expected);self.assertIsNone(diagnostic)
    def test_hosted_provider_cannot_invent_a_filter(self):
        from src.providers.base import ToolPlan
        bad=HighlightAgent().parse('Show all goals').model_copy(update={'player':999})
        fake=SimpleNamespace(plan=lambda q,s:ToolPlan(calls=[{'name':'highlight_query','arguments':bad.model_dump()}]))
        with patch('src.providers.factory.build_provider',return_value=fake):
            p,name,_=HighlightAgent().interpret('Show all goals',provider='microsoft');self.assertEqual(name,'local_fallback');self.assertIsNone(p.player)

class IndexTests(unittest.TestCase):
    def test_provenance_and_timestamp_offset(self):
        idx=MatchEventIndex('r',tools());row=idx.events[0];self.assertEqual(row['start'],4);self.assertEqual(row['period_relative_seconds'],(5999+100)/25);self.assertTrue(row['safe_for_automatic']);self.assertEqual(row['verification'],'annotated')
    def test_invalid_event_clock_excluded(self):
        e=event().model_copy(update={'timestamp':10});self.assertEqual(MatchEventIndex('r',tools([e])).events,[])
    def test_no_cv_football(self):
        idx=MatchEventIndex('cv');rows,message=idx.retrieve(Plan(event_types=['goal']));self.assertEqual(rows,[]);self.assertIn('couldn’t confirm',message)
    def test_candidate_only_in_explicit_review(self):
        e=event('SHOT_CANDIDATE').model_copy(update={'source':'trajectory','validated_by_bas':False,'validation_state':'estimated'});idx=MatchEventIndex('r',tools([e]));self.assertEqual(idx.retrieve(Plan(event_types=['shot']))[0],[]);self.assertEqual(len(idx.retrieve(Plan(event_types=['shot'],review=True))[0]),1)
    def test_no_jersey_or_home_guess(self):
        idx=MatchEventIndex('r',tools())
        for p in [Plan(event_types=['goal'],team='home'),Plan(event_types=['goal'],player=10),Plan(event_types=['shot'],selection='maximum',metric='shot_speed')]:self.assertFalse(idx.retrieve(p)[0])
    def test_ranking_pass_distance(self):
        es=[]
        for i,d in [(100,20),(200,42)]:
            e=event(frame=i).model_copy(update={'type':__import__('src.intelligence.schema',fromlist=['EventType']).EventType.PASS_COMPLETED,'source':'transfer','metrics':__import__('src.intelligence.schema',fromlist=['EventMetrics']).EventMetrics(distance_m=d,outcome='completed',observed_step_fraction=1,start_frame=i-20,end_frame=i) });es.append(e)
        rows,_=MatchEventIndex('r',tools(es)).retrieve(Plan(event_types=['pass'],selection='maximum',metric='distance_m'));self.assertEqual(rows[0]['metrics']['distance_m'],42)
    def test_tracker_filter_and_receiver(self):
        idx=MatchEventIndex('r',tools([event(player_id=10)]));self.assertEqual(len(idx.retrieve(Plan(event_types=['goal'],player=10,player_namespace='tracker'))[0]),1)
    def test_ordinal(self):self.assertEqual(MatchEventIndex('r',tools()).retrieve(Plan(event_types=['goal'],selection='ordinal',ordinal=2))[0][0]['start'],12)
    def test_unknown_period_offset_not_assumed(self):
        tb=VideoTimebase(fps=25,frames=100,period=2);self.assertIsNone(tb.match_seconds(3))
        with self.assertRaises(ValueError):tb.match_to_clip(2703)
    def test_timebase_roundtrip(self):
        tb=VideoTimebase(fps=25,frames=6000,source_offset_frames=5999,period_start_match_seconds=2700,period=2)
        for t in [0,3.4,173.92]:self.assertAlmostEqual(tb.match_to_clip(tb.match_seconds(t)),t);self.assertAlmostEqual(tb.period_to_clip(tb.period_seconds(t)),t)

class BoundaryTests(unittest.TestCase):
    def test_bounds_merge_and_evidence(self):
        idx=MatchEventIndex('r',tools());rows=idx.events;s=ClipBoundaryEngine().build(rows,Plan(event_types=['goal']),20,25);self.assertEqual(len(s),1);self.assertEqual(s[0]['event_ids'],[r['event_id'] for r in rows]);self.assertTrue(0<=s[0]['start']<s[0]['end']<=20)
    def test_duration_does_not_cut_event(self):
        e=MatchEventIndex('r',tools()).events;s=ClipBoundaryEngine().build(e,Plan(duration=5),20,25);self.assertLessEqual(sum(x['end']-x['start'] for x in s),5);self.assertTrue(all(x['start']<=x['event_start']<=x['event_end']<=x['end'] for x in s))
    def test_no_cross_period_merge(self):
        e=MatchEventIndex('r',tools()).events;e[1]['timebase']['period']=2;s=ClipBoundaryEngine().build(e,Plan(),20,25);self.assertEqual(len(s),2)

class ServiceTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.s=HighlightService({'r':tools()}, {},root=self.temp.name);self.addCleanup(self.s.executor.shutdown,wait=True)
    def req(self,q='Show goals',**kw):return HighlightRequest(question=q,session_id=kw.pop('session_id',uuid.uuid4()),provider='local',**kw)
    def test_wrong_match_and_session_context_rejected(self):
        j=self.s.submit('r',self.req('Show saves'));sid=uuid.uuid4()
        with self.assertRaises(Exception):self.s.submit('r',self.req('Second one',session_id=sid,previous_job=j['job_id']))
    def test_optional_highlights_do_not_break_legacy_recording_startup(self):
        t=tools();t.summary['half']=None
        with tempfile.TemporaryDirectory() as d:
            service=HighlightService({'legacy':t},{},root=d)
            try:
                job=service.submit('legacy',self.req());self.assertEqual(job['state'],'COMPLETED');self.assertEqual(job['events'],[]);self.assertIn('clock',job['message'])
            finally:service.executor.shutdown(wait=True)
    def test_no_job_for_unsupported_evidence(self):
        j=self.s.submit('r',self.req('Show saves'));self.assertEqual(j['state'],'COMPLETED');self.assertIsNone(j['video_url'])
    def test_identity_uses_same_event_only(self):
        req=self.req('Show saves');j=self.s.submit('r',req);self.s.jobs[j['job_id']]['events']=[MatchEventIndex('r',tools([event(player_id=10)])).events[0]]
        r=self.s.submit('r',self.req('Who made that pass?',session_id=req.session_id,previous_job=j['job_id']));self.assertIn('tracked player 10',r['message'])
    def test_invalid_duration_returns_client_error(self):
        with self.assertRaises(Exception) as c:self.s.submit('r',self.req('Make a 900-second highlight video'))
        self.assertEqual(c.exception.status_code,422)
    def test_source_arbitrary_path_not_an_input(self):
        with self.assertRaises(ValidationError):HighlightRequest(question='show goals',session_id=uuid.uuid4(),source='/etc/passwd')
        with self.assertRaises(HighlightError):self.s.source.resolve('/etc/passwd',threading.Event())
    def test_ffmpeg_failure_safe(self):
        self.s.source.resolve=lambda *a: (_ for _ in ()).throw(HighlightError('source_unavailable'));j=self.s.submit('r',self.req());self.s.executor.shutdown(wait=True);self.assertEqual(self.s.jobs[j['job_id']]['state'],'FAILED');self.assertNotIn(self.temp.name,self.s.jobs[j['job_id']]['message'])

class JobIntegrationTests(unittest.TestCase):
    def test_real_job_cache_playback_rights_and_corruption(self):
        from fastapi import FastAPI
        from src.highlights.render import file_hash
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'registered.mp4';cancel=threading.Event()
            run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','testsrc2=size=160x90:rate=25','-t','3','-c:v','libx264','-threads','1',str(source)],cancel)
            t=tools([event(frame=25)]);t.summary['frames']=75
            service=HighlightService({'r':t},{},root=d/'cache');service.downloads=True;service.source.resolve=lambda *args:(source,file_hash(source),{'origin':'registered_local'})
            app=FastAPI();app.include_router(service.router(lambda:None));client=TestClient(app);session=str(uuid.uuid4())
            def submit():
                job=client.post('/matches/r/highlights',json={'question':'Show goals','session_id':session,'provider':'local'}).json()
                deadline=time.monotonic()+10
                while service.jobs[job['job_id']]['state'] in ['QUEUED','PROCESSING'] and time.monotonic()<deadline:time.sleep(.01)
                return service.jobs[job['job_id']]
            try:
                first=submit();self.assertEqual(first['state'],'COMPLETED');self.assertFalse(first['cache_hit']);second=submit();self.assertTrue(second['cache_hit'])
                url=second['video_url'];self.assertEqual(client.get(url,headers={'Range':'bytes=0-31'}).status_code,206);download=client.get(url+'&download=true');self.assertEqual(download.status_code,200);self.assertIn('attachment',download.headers['content-disposition']);service.downloads=False;self.assertEqual(client.get(url+'&download=true').status_code,403);service.downloads=True
                self.assertEqual(client.get(url.replace(session,str(uuid.uuid4()))).status_code,404)
                movie=Path(second['directory'])/'highlight.mp4';movie.write_bytes(b'corrupt');self.assertEqual(client.get(url).status_code,503)
                third=submit();self.assertEqual(third['state'],'COMPLETED');self.assertFalse(third['cache_hit'])
            finally:service.executor.shutdown(wait=True)
    def test_cancellation_and_reference_subset(self):
        with tempfile.TemporaryDirectory() as d:
            service=HighlightService({'r':tools()},{},root=d);sid=uuid.uuid4()
            def resolve(recording,cancel):
                cancel.wait(2);raise HighlightError('cancelled')
            service.source.resolve=resolve
            req=HighlightRequest(question='Show goals',session_id=sid,provider='local');job=service.submit('r',req);service.jobs[job['job_id']]['cancel'].set();service.executor.shutdown(wait=True);self.assertEqual(service.jobs[job['job_id']]['state'],'CANCELLED')
            previous=service.submit('r',HighlightRequest(question='Show saves',session_id=sid,provider='local'));service.jobs[previous['job_id']]['events']=[service.indexes['r'].events[0]]
            nextjob=service.submit('r',HighlightRequest(question='Show the second one',session_id=sid,previous_job=previous['job_id'],provider='local'));self.assertEqual(nextjob['events'],[]);self.assertIn('moment number 2',nextjob['message'])

class RenderTests(unittest.TestCase):
    def test_corrupt_media_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.mp4';p.write_bytes(b'not a movie')
            with self.assertRaises(HighlightError):probe(p)
    def test_failed_process(self):
        with self.assertRaises(HighlightError):run(['ffmpeg','-nostdin','-v','error','-invalid-option'],threading.Event(),5)
    def test_cancel_process(self):
        c=threading.Event();c.set()
        with self.assertRaises(HighlightError):run(['ffmpeg','-nostdin','-v','error','-re','-f','lavfi','-i','testsrc=size=32x32:rate=1','-t','20','-f','null','-'],c,5)
    def test_real_concat_audio_and_silent_input(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);c=threading.Event();p=d/'source.mp4'
            run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','testsrc2=size=160x90:rate=25','-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','3','-c:v','libx264','-threads','1','-c:a','aac',str(p)],c)
            segments=[{'start':.4,'end':1.2},{'start':2,'end':2.8}];r=HighlightRenderer().render(p,segments,d/'out',c);self.assertAlmostEqual(r['duration'],1.6,delta=.15);self.assertTrue(r['original_audio']);self.assertTrue((d/'out/thumbnail.jpg').is_file())
            silent=d/'silent.mp4';run(['ffmpeg','-nostdin','-v','error','-y','-i',str(p),'-an','-c:v','copy',str(silent)],c);r=HighlightRenderer().render(silent,segments,d/'silentout',c);self.assertFalse(r['original_audio'])

if __name__=='__main__':unittest.main()
