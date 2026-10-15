import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import requests
from fastapi.testclient import TestClient
from src.artifacts.store import LocalArtifactStore,AzureBlobArtifactStore,ArtifactError
from src.artifacts.__main__ import export as bundle_export
from src.fabric.__main__ import export as fabric_export,SCHEMAS
from src.providers.agent import ask
from src.providers.base import ToolPlan,ProviderError
from src.providers.conversation import ConversationState
from src.providers.microsoft import MicrosoftFoundryProvider
from src.providers.gemini import GeminiQueryProvider
from src.query import MatchMindTools,query_question
from src.serving.app import create_app
from tests.helpers import write_query_artifacts

class ProductTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.path=self.root/'bundle';self.path.mkdir()
        self.event=write_query_artifacts(self.path);self.tools=MatchMindTools(self.path)
    def test_local_store_loads_known_bundle(self):self.assertEqual(LocalArtifactStore({'demo':self.path}).resolve('demo'),self.path.resolve())
    def test_missing_bundle(self):
        with self.assertRaises(ArtifactError):LocalArtifactStore({'demo':self.root/'missing'}).resolve('demo')
    def test_missing_recording(self):
        with self.assertRaises(KeyError):LocalArtifactStore({'demo':self.path}).resolve('missing')
    def test_corrupt_artifact(self):
        for name in ['commentary.json','events.json']:
            (self.path/name).write_text('{')
            with self.assertRaises(ArtifactError):LocalArtifactStore({'demo':self.path}).resolve('demo')
            (self.path/name).unlink()
    def test_malformed_closed_event(self):
        (self.path/'events.json').write_text('{"events":[{"invented":true}]}')
        with self.assertRaises(ArtifactError):LocalArtifactStore({'demo':self.path}).resolve('demo')
    def test_traversal_rejected(self):
        for key in ['../demo','a/b','..','x%2Fy','/tmp']:
            with self.assertRaises(ArtifactError):LocalArtifactStore({'demo':self.path}).resolve(key)
    def test_symlink_escape_rejected(self):
        outside=self.root/'outside';outside.write_text('{}');(self.path/'commentary.json').symlink_to(outside)
        with self.assertRaises(ArtifactError):LocalArtifactStore({'demo':self.path}).resolve('demo')
    def test_bundle_export_is_allowlisted(self):
        (self.path/'.env').write_text('secret');(self.path/'raw_video.mp4').write_bytes(b'video')
        out=self.root/'export';bundle_export({'demo':self.path},out)
        manifest=json.loads((out/'matches/demo/bundle.json').read_text())
        self.assertNotIn('.env',manifest['files']);self.assertNotIn('raw_video.mp4',manifest['files'])
    def blob(self,mutate=None):
        blobs={p.name:p.read_bytes() for p in self.path.iterdir() if p.name in ('summary.json','events.json')}
        manifest=dict(schema_version='1.0',files={name:hashlib.sha256(data).hexdigest() for name,data in blobs.items()})
        if mutate:mutate(manifest,blobs)
        def download(name):
            data=json.dumps(manifest).encode() if name.endswith('bundle.json') else blobs[name.split('/')[-1]]
            return SimpleNamespace(readall=lambda:data)
        client=Mock();client.download_blob.side_effect=download
        return AzureBlobArtifactStore(client,['demo'],self.root/'cache'),client
    def test_azure_store_verified_download_and_cache(self):
        store,client=self.blob();p=store.resolve('demo');self.assertTrue((p/'events.json').exists())
        client.reset_mock();store.resolve('demo');self.assertEqual(client.download_blob.call_count,1)
    def test_azure_store_corrupt_hash(self):
        store,_=self.blob(lambda m,b:m['files'].update({'events.json':'0'*64}))
        with self.assertRaises(ArtifactError):store.resolve('demo')
    def test_azure_store_manifest_traversal(self):
        store,_=self.blob(lambda m,b:m['files'].update({'../secret':'0'*64}))
        with self.assertRaises(ArtifactError):store.resolve('demo')
    def test_azure_store_network_error_safe(self):
        store,client=self.blob();client.download_blob.side_effect=RuntimeError('server-secret')
        with self.assertRaises(ArtifactError) as e:store.resolve('demo')
        self.assertNotIn('server-secret',str(e.exception))
    def test_fabric_schemas_ids_and_counts(self):
        out=self.root/'fabric';m=fabric_export({'demo':self.path},out)
        self.assertEqual(m['tables']['Match']['rows'],1);self.assertEqual(m['tables']['Event']['rows'],1)
        self.assertEqual(m['tables']['Event']['primary_key'],['recording_id','event_id'])
        self.assertEqual(m['tables']['Event']['column_types']['timestamp'],'float64')
        import csv
        with (out/'Event.csv').open() as file:row=list(csv.DictReader(file))[0]
        self.assertEqual(row['actor_id'],'101');self.assertEqual(row['tracker_id'],'2');self.assertEqual(float(row['timestamp']),4.)
    def test_fabric_determinism(self):
        a=fabric_export({'demo':self.path},self.root/'a');b=fabric_export({'demo':self.path},self.root/'b');self.assertEqual(a,b)
    def test_fabric_duplicate_primary_keys_rejected(self):
        p=self.path/'events.json';x=json.loads(p.read_text());x['events']*=2;p.write_text(json.dumps(x))
        with self.assertRaises(ValueError):fabric_export({'demo':self.path},self.root/'bad')
    def provider(self):
        p=Mock();p.configuration={'provider':'microsoft'};p.plan.return_value=ToolPlan(calls=[dict(name='get_fastest_players',arguments={'limit':1})]);p.select.return_value={'statement_ids':[0]};return p
    def test_provider_facts_stable_and_evidence_propagates(self):
        r=ask(self.tools,'Who was the fastest player?',provider=self.provider(),recording_id='demo')
        local=ask(self.tools,'Who was the fastest player?',recording_id='demo')
        self.assertEqual(r.answer.facts,local.answer.facts);self.assertEqual(r.evidence,local.evidence);self.assertEqual(r.tools,['get_fastest_players'])
        self.assertTrue(all(e.fact_id and e.source for e in r.evidence))
    def test_unrelated_tool_cannot_change_answer(self):
        p=self.provider();p.plan.return_value=ToolPlan(calls=[dict(name='get_match_summary',arguments={})])
        r=ask(self.tools,'Who was the fastest player?',provider=p);self.assertTrue(r.fallback)
        self.assertEqual(r.answer.facts,query_question(self.tools,'Who was the fastest player?').facts)
    def test_unsupported_tool_arguments_fall_back(self):
        p=self.provider();p.plan.return_value=ToolPlan(calls=[dict(name='get_fastest_players',arguments={'limit':1,'invented_speed':99})])
        self.assertTrue(ask(self.tools,'Who was the fastest player?',provider=p).fallback)
    def test_provider_cannot_invent_numbers(self):
        p=self.provider();p.select.return_value={'statement_ids':[0],'speed':1000}
        r=ask(self.tools,'Who was the fastest player?',provider=p);self.assertTrue(r.fallback);self.assertNotIn('1000',r.answer.model_dump_json())
    def test_gemini_uses_same_facts(self):
        fake=Mock();fake.configuration={'model':'fake'};fake.generate.side_effect=['{"sentence_ids":["get_fastest_players"]}','{"sentence_ids":["0"]}']
        r=ask(self.tools,'Who was the fastest player?',provider=GeminiQueryProvider(client=fake))
        self.assertFalse(r.fallback);self.assertEqual(r.answer.facts,query_question(self.tools,'Who was the fastest player?').facts)
    def test_gemini_outage_local_fallback(self):
        fake=Mock();fake.configuration={'model':'fake'};fake.generate.side_effect=RuntimeError('secret')
        self.assertTrue(ask(self.tools,'Who was the fastest player?',provider=GeminiQueryProvider(client=fake)).fallback)
    def test_event_followup_why_and_seek(self):
        state=ConversationState(recording_id='demo',last_event_id=self.event.event_id)
        why=ask(self.tools,'Why was it difficult?',conversation=state,recording_id='demo')
        seek=ask(self.tools,'Take me there.',conversation=why.conversation,recording_id='demo')
        self.assertTrue(why.evidence);self.assertEqual(seek.conversation.last_timestamp,4.)
    def test_player_followup(self):
        first=ask(self.tools,'Who was the fastest player?',recording_id='demo')
        more=ask(self.tools,'Tell me more about that player',conversation=first.conversation,recording_id='demo')
        self.assertEqual(more.answer.status,'ok');self.assertTrue(any(f.value==101 for f in more.answer.facts))
    def test_stale_context_cannot_override_evidence(self):
        state=ConversationState(recording_id='other',last_event_id=self.event.event_id)
        self.assertEqual(ask(self.tools,'Why?',conversation=state,recording_id='demo').answer.status,'unknown')
    def test_unknown_event_followup(self):
        state=ConversationState(recording_id='demo',last_event_id='missing')
        self.assertEqual(ask(self.tools,'Why?',conversation=state,recording_id='demo').answer.status,'unknown')
    def test_forged_player_timestamp_does_not_create_fact(self):
        state=ConversationState(recording_id='demo',last_player_id=999,last_timestamp=4.)
        self.assertEqual(ask(self.tools,'Take me there',conversation=state,recording_id='demo').answer.status,'unknown')
    def test_causality_unknown(self):self.assertEqual(ask(self.tools,'Did pressure cause this turnover?').answer.status,'unsupported')
    def test_microsoft_missing_credentials(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ProviderError):MicrosoftFoundryProvider()
    def test_microsoft_timeout_and_malformed_response(self):
        for response in [requests.Timeout('secret'),SimpleNamespace(status_code=200,content=b'{}',json=lambda:{})]:
            s=Mock()
            if isinstance(response,Exception):s.post.side_effect=response
            else:s.post.return_value=response
            p=MicrosoftFoundryProvider('https://test.openai.azure.com','d','secret',session=s)
            with self.assertRaises(ProviderError) as e:p.plan('summary',[])
            self.assertNotIn('secret',str(e.exception))
    def test_service_startup_and_readiness(self):
        client=TestClient(create_app({'demo':self.path}));self.assertEqual(client.get('/ready').status_code,200)
        self.assertEqual(client.get('/matches/demo/summary').status_code,200)
    def test_service_empty_readiness(self):self.assertEqual(TestClient(create_app({})).get('/ready').status_code,503)
    def test_service_bad_bundle_fails_startup(self):
        (self.path/'events.json').unlink()
        with self.assertRaises(ArtifactError):create_app({'demo':self.path})
    def test_service_missing_provider_and_bound_body(self):
        client=TestClient(create_app({'demo':self.path}))
        with patch('src.serving.app.build_provider',side_effect=ProviderError('secret')):
            result=client.post('/matches/demo/ask',json={'question':'Who was the fastest player?','provider':'microsoft'}).json()
        self.assertTrue(result['fallback']);self.assertNotIn('secret',json.dumps(result))
        self.assertEqual(client.post('/matches/demo/ask',content='x'*20000).status_code,413)
    def test_logs_are_structured_and_do_not_contain_question_or_keys(self):
        client=TestClient(create_app({'demo':self.path}))
        with self.assertLogs('matchmind.requests',level='INFO') as log:
            r=client.post('/matches/demo/ask',json={'question':'private-secret coach instructions'})
        text=' '.join(log.output);self.assertNotIn('private-secret',text);self.assertIn('unknown_question_count',text);self.assertIn('provider_latency_ms',text)
        self.assertTrue(r.headers['X-Request-ID'])
    def test_report_injects_only_public_service_configuration(self):
        (self.path/'report.html').write_text('<script>const x=1;</script>')
        client=TestClient(create_app({'demo':self.path}))
        text=client.get('/matches/demo/report').text
        self.assertIn('MATCHMIND_API_BASE',text);self.assertNotIn('API_KEY',text)
        (self.path/'synthetic.mp4').write_bytes(b'video')
        (self.path/'report.html').write_text('<video id="video" src="../../old.mp4"></video><script>const x=1;</script>')
        client=TestClient(create_app({'demo':self.path},media_root=self.path))
        self.assertIn('src="video.mp4"',client.get('/matches/demo/report').text)
        self.assertEqual(client.get('/matches/demo/video.mp4').content,b'video')
    def test_api_followup_contract(self):
        client=TestClient(create_app({'demo':self.path}));r=client.post('/matches/demo/ask',json={'question':'Explain this event','event_id':self.event.event_id}).json()
        follow=client.post('/matches/demo/ask',json={'question':'Take me there','conversation':r['conversation']}).json()
        self.assertEqual(follow['conversation']['last_event_id'],self.event.event_id)
    def test_demo_fallback_and_truthful_unknown(self):
        from scripts.demo_matchmind import run
        rows=run(self.tools,'demo');self.assertEqual(len(rows),8);self.assertEqual(rows[-1]['response']['answer']['status'],'unsupported')
    def test_container_static_boundary(self):
        text=Path('Dockerfile.service').read_text();self.assertIn('USER 10001',text);self.assertIn('EXPOSE 8000',text)
        self.assertNotIn('COPY data',text);self.assertNotIn('COPY src ./src',text)
    def test_azure_deployment_static_boundary(self):
        text=Path('deploy/azure/container-app.bicep').read_text();self.assertIn('/ready',text);self.assertIn('keyVaultUrl',text);self.assertIn('minReplicas: 1',text)

    def test_optional_framework_runtime_uses_dispatch_and_discards_prose(self):
        from src.providers.agent_framework import FrameworkProvider
        class FakeAgent:
            def __init__(self,**kwargs):self.tools={f.__name__:f for f in kwargs['tools']}
            async def run(self,question):
                self.tools['get_fastest_players'](limit=1)
                return 'Invented speed 9999; this prose must be discarded.'
        with patch.dict('sys.modules',{'agent_framework':SimpleNamespace(Agent=FakeAgent)}):
            provider=FrameworkProvider(self.tools,self.provider(),client=object())
            result=ask(self.tools,'Who was the fastest player?',provider=provider)
        self.assertFalse(result.fallback);self.assertEqual(result.answer,query_question(self.tools,'Who was the fastest player?'))
        self.assertEqual(result.tools,['get_fastest_players'])
