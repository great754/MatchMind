import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.query import MatchMindTools,query_question,dispatch
from src.query.schema import Selection,QueryResult
from src.providers.agent import ask
from src.providers.agent_framework import function_tools,contract
from src.providers.microsoft import MicrosoftFoundryProvider,MicrosoftCommentaryProvider
from src.providers.base import ProviderError,ToolPlan
from src.serving.app import create_app
from tests.helpers import write_query_artifacts

class ProviderServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        self.event=write_query_artifacts(self.path);self.tools=MatchMindTools(self.path)
        self.client=TestClient(create_app({'synthetic':self.path}))
    def provider(self):
        p=Mock();p.configuration={'provider':'test'}
        p.plan.return_value=ToolPlan(calls=[dict(name='get_possession_summary',arguments={})]);p.select.return_value=Selection(statement_ids=[0])
        return p
    def test_same_facts_across_provider_boundary(self):
        p=self.provider();r=ask(self.tools,'Which team had more possession?',provider=p)
        facts=p.select.call_args.args[2][0].facts
        self.assertEqual(facts,self.tools.get_possession_summary().facts)
        self.assertEqual(r.answer.facts,query_question(self.tools,'Which team had more possession?').facts)
        self.assertTrue(r.answer.statements[0].fact_ids)
    def test_provider_failure_preserves_deterministic_answer(self):
        p=self.provider();p.plan.side_effect=RuntimeError('secret-key')
        result=ask(self.tools,'Which team had more possession?',provider=p)
        self.assertTrue(result.fallback);self.assertEqual(result.answer,query_question(self.tools,'Which team had more possession?'))
        self.assertNotIn('secret-key',result.model_dump_json())
    def test_invented_statement_rejected(self):
        p=self.provider();p.select.return_value={'statement_ids':[1000],'score':'2-1'}
        self.assertTrue(ask(self.tools,'Which team had more possession?',provider=p).fallback)
    def test_unsupported_question_avoids_provider(self):
        p=self.provider();r=ask(self.tools,'What is the official final score?',provider=p)
        self.assertEqual(r.answer.status,'unsupported');p.plan.assert_not_called()
    def test_closed_tool_contract(self):
        for spec in contract()['tools']:self.assertFalse(spec['input_schema']['additionalProperties'])
        self.assertFalse(contract()['response_schema']['additionalProperties'])
        with self.assertRaises(ValidationError):dispatch(self.tools,'get_player_summary',{'player_id':-1})
        with self.assertRaises(ValidationError):dispatch(self.tools,'get_tactical_state',{'timestamp':float('nan')})
        with self.assertRaises(ValidationError):dispatch(self.tools,'get_match_summary',{'api_key':'x'})
    def test_framework_functions_reuse_dispatch_and_provenance(self):
        functions={f.__name__:f for f in function_tools(self.tools)}
        result=QueryResult.model_validate(functions['get_possession_summary']())
        self.assertEqual(result,self.tools.get_possession_summary())
        with self.assertRaises(ValidationError):functions['get_player_summary'](player_id='101')
    def test_microsoft_request_contract_and_auth_secrecy(self):
        session=Mock();session.post.return_value=SimpleNamespace(status_code=200,content=b'{}',json=lambda:{'choices':[{'message':{'tool_calls':[{'function':{'name':'get_match_summary','arguments':'{}'}}]}}]})
        provider=MicrosoftFoundryProvider('https://test.openai.azure.com','deployment','secret-key',session=session)
        self.assertEqual(provider.plan('summary',contract()['tools']).calls[0].name,'get_match_summary')
        kwargs=session.post.call_args.kwargs
        self.assertFalse(kwargs['allow_redirects']);self.assertEqual(kwargs['timeout'],(5,30));self.assertTrue(kwargs['json']['parallel_tool_calls'])
        self.assertNotIn('secret-key',json.dumps(provider.configuration))
        session.post.return_value=SimpleNamespace(status_code=200,content=b'{}',json=lambda:{'choices':[{'message':{'content':'{\"statement_ids\":[0]}'}}]})
        local=query_question(self.tools,'Which team had more possession?')
        provider.select('Which team had more possession?','fan',[self.tools.get_possession_summary()],local.statements)
        request=json.loads(session.post.call_args.kwargs['json']['messages'][1]['content'])
        self.assertEqual(request['facts'][0],self.tools.get_possession_summary().model_dump(mode='json'))
        self.assertTrue(session.post.call_args.kwargs['json']['response_format']['json_schema']['strict'])
    def test_microsoft_safe_network_failure(self):
        session=Mock();session.post.side_effect=RuntimeError('secret-key upstream body')
        provider=MicrosoftFoundryProvider('https://test.openai.azure.com','deployment','secret-key',session=session)
        with self.assertRaises(ProviderError) as e:provider.plan('summary',contract()['tools'])
        self.assertNotIn('secret-key',str(e.exception))
    def test_microsoft_endpoint_validation(self):
        for endpoint in ['http://test.openai.azure.com','https://attacker.example','https://test.openai.azure.com/?key=secret']:
            with self.assertRaises(ProviderError):MicrosoftFoundryProvider(endpoint,'d','k')
    def test_commentary_same_catalogue_and_rejects_invention(self):
        p=MicrosoftCommentaryProvider('https://test.openai.azure.com','d','k');p._request=Mock(return_value={'content':'{"sentence_ids":["s1"]}'})
        request={'sentences':{'s1':'Known fact'},'instruction':'Select only'}
        self.assertEqual(json.loads(p.generate(request)),{'sentence_ids':['s1']})
        p._request.return_value={'content':'{"sentence_ids":["invented"]}'}
        with self.assertRaises(ProviderError):p.generate(request)
    def test_health_and_summary(self):
        self.assertEqual(self.client.get('/health').status_code,200)
        r=self.client.get('/matches/synthetic/summary');self.assertEqual(r.status_code,200);self.assertIn('provenance',r.json())
    def test_event_and_player_lookup(self):
        self.assertEqual(self.client.get('/matches/synthetic/events').status_code,200)
        self.assertEqual(self.client.get('/matches/synthetic/players/101').json()['status'],'ok')
        self.assertEqual(self.client.get('/matches/synthetic/players/999').json()['status'],'unknown')
        self.assertEqual(self.client.get('/matches/synthetic/players/-1').status_code,422)
    def test_explain_and_unknown_event(self):
        self.assertEqual(self.client.get(f'/matches/synthetic/events/{self.event.event_id}/explain').status_code,200)
        self.assertEqual(self.client.get('/matches/synthetic/events/missing/explain').status_code,404)
    def test_ask_and_malformed_query(self):
        self.assertEqual(self.client.post('/matches/synthetic/ask',json={'question':'Which team had more possession?'}).status_code,200)
        for body in [{'question':''},{'question':'summary','api_key':'secret'},{'question':'summary','timestamp':-1}]:
            self.assertEqual(self.client.post('/matches/synthetic/ask',json=body).status_code,422)
    def test_unknown_match(self):self.assertEqual(self.client.get('/matches/unknown/summary').status_code,404)
    def test_authentication_and_no_key_leakage(self):
        client=TestClient(create_app({'synthetic':self.path},token='server-secret'))
        self.assertEqual(client.get('/matches/synthetic/summary').status_code,401)
        result=client.get('/matches/synthetic/summary',headers={'Authorization':'Bearer server-secret'})
        self.assertEqual(result.status_code,200);self.assertNotIn('server-secret',result.text)
    def test_report_allowlisted(self):
        self.assertEqual(self.client.get('/matches/synthetic/report').status_code,404)
        (self.path/'report.html').write_text('<p>Cached report</p>')
        self.assertIn('Cached report',self.client.get('/matches/synthetic/report').text)
        (self.path/'motion.png').write_bytes(b'png')
        self.assertEqual(self.client.get('/matches/synthetic/motion.png').content,b'png')
        self.assertEqual(self.client.get('/matches/synthetic/summary.json').status_code,404)
