"""Known-answer intelligence tests; hosted calls are always mocked."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
from pydantic import ValidationError
from src.analytics.config import AnalyticsConfig
from src.analytics.pipeline import analyze_recording,write_outputs
from src.intelligence.config import IntelligenceConfig
from src.intelligence.schema import EventType as T,EventMetrics,CanonicalEvent,make_event
from src.intelligence.events import build_events
from src.intelligence.context import ContextBuilder
from src.intelligence.labels import PlayerLabel
from src.intelligence.commentary.base import TemplateProvider,CommentaryError
from src.intelligence.commentary.gemini_provider import GeminiCommentaryProvider
from src.intelligence.commentary.service import cache_key,request_for,generate_commentary,select_moments,validate_plan


def event(frame=500,kind=T.PASS_COMPLETED,group='transfer:1',score=60,sender=1,receiver=2):
    return make_event('fixture',frame=frame,fps=25,type=kind,group_id=group,source='transfer',significance=score,
        player_id=sender,receiver_id=receiver,team='left',metrics=EventMetrics(distance_m=30,forward_progression_m=20,outcome='completed'))

def fixture():
    n=2000
    positions=np.tile([[10.,34.],[30.,34.]],(n,1,1))
    ball=np.tile([10.,34.],(n,1))
    r=SimpleNamespace(positions=positions,ball=ball,ball_status=np.ones(n),player_ids=np.array([100,200]),player_teams=['left','right'],mot_ids={100:1,200:2},events=[],fps=25.,source_offset=5999,half=1,scope='clip',player_source='gsr',source_files=[],sync={'match_id':'fixture','method':'fixture','side_to_color':{'0':1,'1':0}})
    result=analyze_recording(r,AnalyticsConfig())
    owners=np.where(np.arange(n)<1000,100,200).astype(object);owners[750:760]=None
    result['possession']=pd.DataFrame({'frame':np.arange(n),'possessing_player_id':pd.array(owners,dtype='Int64'),
        'possessing_team':['left' if o==100 else 'right' if o==200 else None for o in owners],
        'state':['unknown' if o is None else 'controlled' for o in owners],
        'control_confidence':np.ones(n),'control_start_frame':np.zeros(n)})
    return r,result

class IntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.recording,cls.result=fixture()
    def builder(self,events=()): return ContextBuilder(self.recording,self.result,list(events))
    def test_serialization_and_stable_id(self):
        a=event();self.assertEqual(a.event_id,event().event_id)
        self.assertEqual(CanonicalEvent.model_validate_json(a.model_dump_json()),a)
        self.assertEqual(a.timestamp,a.frame/25)
        self.assertNotIn('speed_kmh',a.model_dump(exclude_none=True)['metrics'])
    def test_closed_event_rejects_unsupported_identity(self):
        with self.assertRaises(ValidationError): CanonicalEvent.model_validate({**event().model_dump(),'player_name':'Invented'})
        with self.assertRaises(ValidationError): EventMetrics(speed_kmh=float('nan'))
    def test_events_sorted_and_namespaced(self):
        events=build_events(self.recording,self.result)
        self.assertEqual([e.frame for e in events],sorted(e.frame for e in events))
        changed=next(e for e in events if e.type==T.POSSESSION_CHANGED)
        self.assertEqual((changed.player_id,changed.actor_id),(2,200))
    def test_transfer_waits_for_confirmed_owner(self):
        result=copy.deepcopy(self.result)
        row={column:None for column in result['passes'].columns}
        row.update(pass_id=1,start_frame=80,end_frame=100,sender_id=100,receiver_id=200,sender_team='left',receiver_team='right',
                   outcome='intercepted',start_x_m=10,start_y_m=34,end_x_m=30,end_y_m=34,distance_m=20,duration_seconds=.8,
                   ball_speed_peak_kmh=30,ball_speed_median_mps=5,observed_step_fraction=1,termination_reason='received')
        result['passes']=pd.DataFrame([row])
        events=build_events(self.recording,result)
        transfer=next(e for e in events if e.type==T.TRANSFER_INTERCEPTED)
        self.assertEqual(transfer.frame,1000)
        self.assertEqual(transfer.metrics.end_frame,100)
        self.assertEqual(ContextBuilder(self.recording,result,events).at(20)['transfer_count'],0)
    def test_15_second_window(self):
        c=self.builder().at(60,15);self.assertEqual(c['frames'],375);self.assertEqual(c['possession_percent']['right'],100)
    def test_30_second_window_and_unknown_denominator(self):
        c=self.builder().at(60,30);self.assertEqual(c['frames'],750)
        self.assertAlmostEqual(c['possession_percent']['left'],240/750*100)
        self.assertAlmostEqual(c['unknown_percent'],9/750*100)
        self.assertAlmostEqual(sum(c['possession_percent'].values())+c['unknown_percent'],100)
    def test_60_second_window(self):
        c=self.builder().at(60,60);self.assertEqual(c['start_frame'],1);self.assertEqual(c['frames'],1500)
        self.assertAlmostEqual(c['possession_percent']['left'],989/1500*100)
    def test_frame_grid_window_avoids_float_rounding_drift(self):
        for frame in range(800,1800,7):
            self.assertEqual(self.builder().at(frame/25,30)['frames'],750)
    def test_invalid_intelligence_thresholds(self):
        for values in ({'player_speed_kmh':0},{'ball_speed_kmh':float('nan')},{'end_seconds':-1},{'max_provider_calls':-1},{'context_window_seconds':20}):
            with self.assertRaises(ValueError):IntelligenceConfig(**values)
    def test_startup_window_and_fractional_timestamp(self):
        c=self.builder().at(0,60);self.assertEqual(c['frames'],1)
        self.assertEqual(self.builder().at(10.039,15)['frame'],250)
    def test_invalid_context_requests(self):
        for timestamp,window in [(-1,15),(80,15),(float('nan'),30),(1,20)]:
            with self.assertRaises(ValueError):self.builder().at(timestamp,window)
    def test_sequences_and_no_future_outcomes(self):
        events=[event(500,sender=1,receiver=2),event(600,group='second',sender=2,receiver=3),event(800,group='future',sender=3,receiver=4)]
        c=self.builder(events).at(25,30)
        self.assertEqual(c['recent_pass_sequence'],[1,2,3]);self.assertEqual(c['completed_passes']['left'],2)
        self.assertEqual(c['transfer_count'],2)
    def test_failed_transfer_breaks_sequence(self):
        e=[event(500),event(600,kind=T.TRANSFER_INTERCEPTED,group='second')]
        c=self.builder(e).at(25,30)
        self.assertEqual(c['recent_pass_sequence'],[]);self.assertEqual(c['interceptions']['left'],1)
    def test_streak_and_fastest_player(self):
        c=self.builder().at(60,30)
        self.assertEqual(c['current_possession']['player_id'],2)
        self.assertAlmostEqual(c['current_possession']['streak_seconds'],501/25)
        self.assertTrue(c['fastest_players'])
    def test_missing_ball_interval_has_no_displacement(self):
        result=copy.deepcopy(self.result);result['ball'].positions[100]=np.nan
        self.assertIsNone(ContextBuilder(self.recording,result,[]).at(10,15)['ball_displacement_m'])
    def test_player_label_precedence(self):
        self.assertEqual(PlayerLabel(7,10,'Name').display,'Player 7')
        self.assertEqual(PlayerLabel(7,10,identity_verified=True,verification_source='roster').display,'#10')
        self.assertEqual(PlayerLabel(7,10,'Name',True,'roster').display,'Name')
        with self.assertRaises(ValueError):PlayerLabel(7,identity_verified=True)
    def speed_events(self,speeds,cooldown=15):
        result=copy.deepcopy(self.result);result['players'][0].smoothed_speed_mps[:]=0
        result['players'][0].smoothed_speed_mps[:len(speeds)]=np.array(speeds)/3.6
        return [e for e in build_events(self.recording,result,IntelligenceConfig(threshold_cooldown_seconds=cooldown)) if e.type==T.PLAYER_SPEED_THRESHOLD]
    def test_speed_requires_persistence(self):
        self.assertEqual(self.speed_events([0,26,26,0]),[])
        self.assertEqual(self.speed_events([0,26,26,26,26])[0].frame,3)
    def test_sustained_speed_not_repeated_per_frame(self):
        self.assertEqual(len(self.speed_events([26]*1000)),1)
    def test_threshold_cooldown_and_rearming(self):
        self.assertEqual(len(self.speed_events([26]*3+[0]+[26]*3)),1)
        self.assertEqual(len(self.speed_events([26]*3+[0]*400+[26]*3)),2)
    def test_distance_milestone(self):
        result=copy.deepcopy(self.result);result['players'][0].distance_m[:]=np.arange(2000)*.2
        e=[e for e in build_events(self.recording,result) if e.type==T.PLAYER_DISTANCE_MILESTONE]
        self.assertEqual([x.metrics.milestone_m for x in e],[100,200,300])
    def test_significance_grouping_cooldown_and_limit(self):
        events=[event(100,score=20),event(200),event(201,kind=T.LONG_PASS,score=70),event(300,group='second'),event(500,kind=T.GOAL,score=100,group='goal')]
        selected=select_moments(events,IntelligenceConfig())
        self.assertEqual(len(selected),2);self.assertEqual(selected[0][0].type,T.LONG_PASS)
        self.assertEqual(select_moments(events,IntelligenceConfig(max_commentary_moments=0)),[])
        self.assertEqual(len(select_moments(events,IntelligenceConfig(max_commentary_moments=1))),1)
    def request(self,mode='analyst'): return request_for([event()],self.builder([event()]).at(20,30),mode)
    def test_request_construction_and_modes(self):
        for mode in ('beginner','fan','analyst'):
            request=self.request(mode);self.assertEqual(request['mode'],mode)
            self.assertIn('Unsupported identities',request['instruction'])
        self.assertIn('not completion probability',request_for([event().model_copy(update={'metrics':EventMetrics(distance_m=30,difficulty_score_0_100=70)})],self.builder().at(20),'analyst')['sentences'][event().event_id])
    def test_unsupported_context_never_enters_prompt(self):
        context=self.builder().at(20);context['invented_score']='4-0'
        with self.assertRaises(ValidationError):request_for([event()],context,'fan')
    def test_cache_key_stability_and_invalidation(self):
        request=self.request();configuration=TemplateProvider.configuration
        self.assertEqual(cache_key(request,configuration),cache_key(dict(reversed(list(request.items()))),configuration))
        self.assertNotEqual(cache_key(request,configuration),cache_key(self.request('fan'),configuration))
        self.assertNotEqual(cache_key(request,configuration),cache_key(request,{'provider':'other'}))
    def test_invalid_model_prose_rejected(self):
        for raw in ['{"commentary":"Player 7 scored twice"}','{"sentence_ids":["invented"]}','{"sentence_ids":[]}','{"sentence_ids":[1]}']:
            with self.assertRaises(ValueError):validate_plan(raw,self.request())
    def test_cache_reuses_all_three_modes_and_budget(self):
        provider=Mock();provider.configuration=TemplateProvider.configuration;provider.generate.side_effect=TemplateProvider().generate
        config=IntelligenceConfig(max_provider_calls=3)
        with tempfile.TemporaryDirectory() as directory:
            a=generate_commentary([event()],self.builder([event()]),config,provider,directory)
            b=generate_commentary([event()],self.builder([event()]),config,provider,directory)
        self.assertEqual(len(a['items']),3);self.assertEqual(b['cache_hits'],3);self.assertEqual(provider.generate.call_count,3)
    def test_disabled_provider_and_call_limit(self):
        self.assertEqual(generate_commentary([],self.builder(),IntelligenceConfig())['provider_calls'],0)
        a=generate_commentary([event()],self.builder(),IntelligenceConfig(max_provider_calls=1),TemplateProvider())
        self.assertEqual(len(a['items']),1)
    def test_provider_failures_not_persisted(self):
        provider=Mock();provider.configuration={'provider':'mock'};provider.generate.return_value='{"commentary":"Unsupported"}'
        a=generate_commentary([event()],self.builder(),IntelligenceConfig(max_provider_calls=1),provider)
        self.assertEqual(a['items'],[]);self.assertNotIn('Unsupported',json.dumps(a))
    def test_missing_gemini_key(self):
        with patch.dict(os.environ,{},clear=True), patch('src.intelligence.commentary.gemini_provider.load_dotenv'):
            with self.assertRaisesRegex(CommentaryError,'GEMINI_API_KEY is missing'):GeminiCommentaryProvider()
    def test_gemini_interface_and_secret_isolation(self):
        session=Mock();response=session.post.return_value;response.status_code=200;response.content=b'{}'
        response.json.return_value={'candidates':[{'content':{'parts':[{'text':'{"sentence_ids":["'+event().event_id+'"]}'}]}}]}
        # Ephemeral synthetic credential, not an actual API key or fixture file.
        with patch.dict(os.environ,{'GEMINI_API_KEY':'test-'+os.urandom(8).hex()}):
            provider=GeminiCommentaryProvider(session=session);raw=provider.generate(self.request())
            self.assertTrue(validate_plan(raw,self.request()))
            args,kwargs=session.post.call_args
            self.assertNotIn(provider._key,args[0]);self.assertNotIn(provider._key,json.dumps(kwargs['json']))
            self.assertNotIn(provider._key,json.dumps(provider.configuration));self.assertFalse(kwargs['allow_redirects'])
    def test_gemini_exception_sanitization(self):
        with patch.dict(os.environ,{'GEMINI_API_KEY':'test-'+os.urandom(8).hex()}):
            session=Mock();provider=GeminiCommentaryProvider(session=session)
            session.post.side_effect=RuntimeError(provider._key)
            with self.assertRaises(CommentaryError) as raised:provider.generate(self.request())
            self.assertNotIn(provider._key,str(raised.exception))

class ReportIntegrationTests(unittest.TestCase):
    def test_exports_and_report_synchronization(self):
        recording,result=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            from src.query.report import catalogue
            sentinel='test-secret-'+os.urandom(16).hex()
            unsafe='</script><img src=x onerror=alert(1)>'
            def injected_catalogue(output):
                data=catalogue(output)
                data['answers']['unsupported']['statements'][0]['text']+=unsafe
                return data
            with patch.dict(os.environ,{'GEMINI_API_KEY':sentinel}),patch('src.query.report.catalogue',side_effect=injected_catalogue):
                write_outputs(recording,result,AnalyticsConfig(),path,True,IntelligenceConfig(max_commentary_moments=2),TemplateProvider())
            for filename in ('events.json','rolling_context.json','commentary.json','summary.json','report.html','motion.png','passes.png'):
                self.assertTrue((path/filename).exists(),filename)
            page=(path/'report.html').read_text()
            self.assertNotIn(sentinel,page)
            self.assertNotIn(unsafe,page)
            self.assertNotIn('.innerHTML',page)
            self.assertIn('ask-form',page)
            self.assertIn('commentary-mode',page);self.assertIn('updateIntelligence(t)',page)
            self.assertIn('button.addEventListener(\'click\',()=>seek(item.timestamp))',page)
            script=page.split('<script>',1)[1].split('</script>',1)[0]
            # Execute the generated browser script against a small DOM/video stub.
            harness='''const vm=require('vm'),fs=require('fs');const elements={};function node(){return {children:[],value:0,checked:true,dataset:{},style:{},handlers:{},textContent:'',getContext(){return new Proxy({}, {get:()=>()=>{}})},addEventListener(k,v){this.handlers[k]=v},appendChild(v){this.children.push(v)},replaceChildren(){this.children=[]},setAttribute(k,v){this[k]=v},removeAttribute(k){delete this[k]}}}for(const id of ['time','video','pitch','labels','state','possession-team','ask-form','ask-question','ask-mode','ask-answer','ask-suggestions','ask-events','commentary-mode','commentary-list','commentary-status','callouts','live-callouts','tactical-status','tactical-pressure','tactical-formation','tactical-centroids','tactical-chart','tactical-metric','network-team','network-window','network-status','network-chart','tactical-timeline'])elements[id]=node();elements['ask-mode'].value='fan';elements['commentary-mode'].value='beginner';elements['tactical-metric'].value='width_m';elements['network-team'].value='left';elements['network-window'].value='rolling';for(const id of ['tactical-chart','network-chart']){elements[id].width=840;elements[id].height=230;elements[id].getBoundingClientRect=()=>({left:0,width:840});}const document={getElementById:id=>elements[id],querySelectorAll:()=>[],createElement:()=>node()};const context={document,console};vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);if(elements['possession-team'].textContent!=='● Red team'||elements['possession-team'].style.color!=='#ff6363')throw Error('Initial possession color');vm.runInContext('seek(30)',context);if(elements['possession-team'].textContent!=='No confirmed possession')throw Error('Unknown possession');vm.runInContext('seek(0)',context);vm.runInContext("askQuestion('Who was fastest?')",context);if(!elements['ask-answer'].children.some(p=>p.textContent.includes('peak estimated speed')))throw Error('Ask fastest missing');vm.runInContext("askQuestion('Was the player tired?')",context);if(!elements['ask-answer'].children.some(p=>p.textContent.includes('cannot determine')))throw Error('Unsupported answer missing');if(elements['ask-events'].children.length){const button=elements['ask-events'].children[0];if(!vm.runInContext('data.askMatchMind.explanations',context)[button.dataset.eventId])throw Error('Explain ID mismatch');button.handlers.click();const timestampButton=elements['ask-answer'].children.flatMap(p=>p.children).find(b=>b.dataset.time!==undefined);if(timestampButton){timestampButton.handlers.click();if(elements.video.currentTime!==Number(timestampButton.dataset.time))throw Error('Ask timestamp seek mismatch');}}const list=elements['commentary-list'];if(!list.children.length)throw Error('No commentary');list.children[0].handlers.click();const t=Number(list.children[0].dataset.time);if(elements.video.currentTime!==t||Number(elements.time.value)!==t)throw Error('Seek mismatch');if(list.children[0]['aria-current']!=='true')throw Error('Missing highlight');elements['commentary-mode'].value='analyst';elements['commentary-mode'].handlers.change();if(!list.children[0].textContent.includes('analyst'))throw Error('Mode switch');elements.video.currentTime=50;elements.video.handlers.timeupdate();if(Number(elements.time.value)!==50)throw Error('Video scrub mismatch');if(elements['possession-team'].textContent!=='● Blue team'||elements['possession-team'].style.color!=='#4695ff')throw Error('Possession color after seek');elements['tactical-chart'].handlers.click({clientX:430});if(Math.abs(elements.video.currentTime-39.98)>.01)throw Error('Tactical chart seek mismatch');elements['network-window'].value='full';elements['network-window'].handlers.change();if(!elements['network-status'].textContent.includes('whole recording'))throw Error('Full network label missing');'''
            (path/'script.js').write_text(script);(path/'verify.js').write_text(harness)
            if not shutil.which('node'):
                return  # Python exports remain tested without a Node installation.
            run=subprocess.run(['node',str(path/'verify.js'),str(path/'script.js')],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)

if __name__=='__main__':unittest.main()
