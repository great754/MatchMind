"""Provider hardening tests are entirely offline; no real Azure credentials required."""
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import certifi
import requests
from src.providers.agent import ask
from src.providers.base import ToolPlan
from src.providers.diagnostics import MicrosoftFailure as F,MicrosoftProviderError
from src.providers.microsoft import MicrosoftFoundryProvider
from src.providers.conversation import ConversationState
from src.query import MatchMindTools,query_question
from src.query.schema import QueryResult,Fact
from scripts.verify_microsoft import verify
from tests.helpers import write_query_artifacts

SECRET='must-never-appear-secret'

def response(status=200,message=None):
    return SimpleNamespace(status_code=status,content=SECRET.encode(),json=lambda:{'choices':[{'message':message or {'tool_calls':[{'function':{'name':'get_match_summary','arguments':'{}'}}]}}]})

class MicrosoftHardeningTests(unittest.TestCase):
    def provider(self,session=None):return MicrosoftFoundryProvider('https://test.openai.azure.com/openai/v1/','test-deployment',SECRET,session=session if session is not None else Mock())
    def test_explicit_portable_certifi_verification(self):
        session=Mock();session.verify=False;session.post.return_value=response()
        with patch.dict(os.environ,{'SSL_CERT_FILE':'invalid-file','REQUESTS_CA_BUNDLE':'invalid-file','CURL_CA_BUNDLE':'invalid-file'}):
            plan=self.provider(session).plan('summary',[])
        self.assertEqual(plan.calls[0].name,'get_match_summary')
        args=session.post.call_args.kwargs
        self.assertEqual(args['verify'],certifi.where());self.assertTrue(Path(args['verify']).is_file())
        self.assertFalse(args['allow_redirects']);self.assertEqual(args['timeout'],(5,30))
    def test_ssl_cert_file_not_required(self):
        session=Mock();session.post.return_value=response()
        with patch.dict(os.environ,{},clear=True):self.provider(session).plan('summary',[])
        self.assertEqual(session.post.call_args.kwargs['verify'],certifi.where())
    def test_session_injection_preserved(self):
        session=Mock();session.__bool__=Mock(return_value=False);session.post.return_value=response()
        self.provider(session).plan('summary',[]);session.post.assert_called_once()
    def assert_failure(self,item,code,status=None):
        session=Mock()
        if isinstance(item,Exception):session.post.side_effect=item
        else:session.post.return_value=item
        with self.assertLogs('tactiqai.providers',level='WARNING') as logs:
            with self.assertRaises(MicrosoftProviderError) as caught:self.provider(session).plan('private request '+SECRET,[])
        self.assertEqual(caught.exception.code,code);self.assertEqual(caught.exception.http_status,status)
        combined=str(caught.exception)+' '.join(logs.output)+json.dumps(caught.exception.diagnostic())
        self.assertNotIn(SECRET,combined);self.assertNotIn('private request',combined);self.assertNotIn('Authorization',combined)
    def test_http_400(self):self.assert_failure(response(400),F.BAD_REQUEST,400)
    def test_http_401(self):self.assert_failure(response(401),F.UNAUTHORIZED,401)
    def test_http_403(self):self.assert_failure(response(403),F.FORBIDDEN,403)
    def test_http_404(self):self.assert_failure(response(404),F.NOT_FOUND,404)
    def test_http_429(self):self.assert_failure(response(429),F.RATE_LIMITED,429)
    def test_http_500(self):self.assert_failure(response(500),F.SERVER_ERROR,500)
    def test_http_503(self):self.assert_failure(response(503),F.SERVER_ERROR,503)
    def test_redirect_never_followed(self):self.assert_failure(response(302),F.HTTP_ERROR,302)
    def test_tls_failure_sanitized(self):self.assert_failure(requests.exceptions.SSLError(SECRET),F.TLS)
    def test_network_failure_sanitized(self):self.assert_failure(requests.exceptions.ConnectionError(SECRET),F.NETWORK)
    def test_timeout_sanitized(self):self.assert_failure(requests.exceptions.Timeout(SECRET),F.TIMEOUT)
    def test_internal_error_sanitized(self):self.assert_failure(RuntimeError(SECRET),F.INTERNAL)
    def test_malformed_json_response(self):
        r=response();r.json=lambda:json.loads(SECRET)
        self.assert_failure(r,F.MALFORMED_RESPONSE)
    def test_unexpected_response_shape(self):
        for r in [None,SimpleNamespace(status_code='200'),SimpleNamespace(status_code=200,content=b'{}',json=lambda:{'choices':[]})]:
            with self.subTest(response_type=type(r).__name__):self.assert_failure(r,F.MALFORMED_RESPONSE)
    def test_refusal_not_accepted(self):self.assert_failure(response(message={'refusal':SECRET}),F.MALFORMED_RESPONSE)
    def test_oversized_response(self):
        r=response();r.content=b'x'*1_000_001;self.assert_failure(r,F.MALFORMED_RESPONSE)
    def test_invalid_tool_plan(self):self.assert_failure(response(message={'tool_calls':[{'function':{'name':'get_match_summary','arguments':SECRET}}]}),F.INVALID_TOOL_PLAN)
    def test_invalid_selection(self):
        session=Mock();session.post.return_value=response(message={'content':'{"statement_ids":[0],"invented":99}'})
        with self.assertRaises(MicrosoftProviderError) as error:self.provider(session).select('question','fan',[],[])
        self.assertEqual(error.exception.code,F.INVALID_SELECTION)
    def test_invalid_configuration_sanitized(self):
        for endpoint in ['https://test.openai.azure.com/openai/v1/responses','https://test.openai.azure.com:invalid',SECRET]:
            with self.subTest(endpoint_shape=endpoint.startswith('https')):
                with self.assertRaises(MicrosoftProviderError) as error:MicrosoftFoundryProvider(endpoint,'d',SECRET)
                self.assertEqual(error.exception.code,F.CONFIGURATION);self.assertNotIn(SECRET,str(error.exception))
    def test_request_bound_sanitized(self):
        session=Mock()
        with self.assertRaises(MicrosoftProviderError) as error:self.provider(session)._request({'messages':['x'*200001]})
        self.assertEqual(error.exception.code,F.REQUEST_INVALID);session.post.assert_not_called()
    def test_bearer_token_not_logged(self):
        session=Mock();session.post.side_effect=requests.Timeout(SECRET)
        p=MicrosoftFoundryProvider('https://test.openai.azure.com','d',session=session,token_provider=lambda:SECRET)
        with self.assertLogs('tactiqai.providers',level='WARNING') as logs:
            with self.assertRaises(MicrosoftProviderError):p.plan('question',[])
        self.assertNotIn(SECRET,' '.join(logs.output))
    def fixture(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name);event=write_query_artifacts(path)
        return MatchMindTools(path),event,path
    def test_provider_failure_preserves_answer_and_diagnoses_timeout(self):
        tools,_,_=self.fixture();session=Mock();session.post.side_effect=requests.Timeout(SECRET)
        with self.assertLogs('tactiqai.providers',level='WARNING') as logs:r=ask(tools,'Who was the fastest player?',provider=self.provider(session))
        self.assertTrue(r.fallback);self.assertEqual(r.answer,query_question(tools,'Who was the fastest player?'))
        self.assertTrue(any('"code": "timeout"' in row for row in logs.output));self.assertNotIn(SECRET,r.model_dump_json())
    def test_saved_formation_plan_rejected_before_selection(self):
        # Replay the saved live demo's get_match_summary plan. The two fact sets
        # are disjoint, so this is a tool-evidence rejection, not a transport error.
        tools,_,_=self.fixture()
        formation=Fact(fact_id='formation_fact',record_id='left:0',field='estimate',value='unknown',source='tactical_states.json#record=left:0',timestamp=0)
        tools.get_formation_timeline=Mock(return_value=QueryResult(question_type='formation_timeline',facts=[formation]))
        provider=Mock();provider.plan.return_value=ToolPlan(calls=[dict(name='get_match_summary',arguments={})])
        with self.assertLogs('tactiqai.providers',level='WARNING') as logs:r=ask(tools,'What formation did MatchMind detect?',provider=provider)
        self.assertTrue(r.fallback);provider.select.assert_not_called()
        self.assertTrue(any('"code": "unrelated_tool_plan"' in row for row in logs.output));self.assertEqual(r.answer.facts,[formation,formation])
    def test_invalid_arguments_diagnosed_without_logging_values(self):
        tools,_,_=self.fixture();provider=Mock();provider.plan.return_value=ToolPlan(calls=[dict(name='get_fastest_players',arguments={'limit':SECRET})])
        with self.assertLogs('tactiqai.providers',level='WARNING') as logs:r=ask(tools,'Who was the fastest player?',provider=provider)
        self.assertTrue(r.fallback);self.assertNotIn(SECRET,' '.join(logs.output));self.assertTrue(any('tool_arguments_invalid' in row for row in logs.output))
    def test_unknown_tool_name_not_echoed_to_service_logs(self):
        tools,_,_=self.fixture();provider=Mock();provider.plan.return_value=ToolPlan(calls=[dict(name=SECRET,arguments={})])
        with self.assertLogs('tactiqai.providers',level='WARNING') as logs:r=ask(tools,'Who was the fastest player?',provider=provider)
        self.assertEqual(r.tools,[]);self.assertNotIn(SECRET,' '.join(logs.output))
    def test_intentional_followup_not_a_fallback_without_provider(self):
        tools,event,_=self.fixture();state=ConversationState(recording_id='demo',last_event_id=event.event_id)
        r=ask(tools,'Why?',conversation=state,recording_id='demo',requested_provider='microsoft')
        self.assertEqual(r.provider_status,'local_followup');self.assertFalse(r.fallback);self.assertTrue(r.evidence)
    def test_unsupported_question_not_a_fallback_without_provider(self):
        tools,_,_=self.fixture();r=ask(tools,'What did the coach tell the team?',requested_provider='microsoft')
        self.assertEqual(r.answer.status,'unsupported');self.assertFalse(r.fallback)
    def test_verifier_config_check_uses_no_network(self):
        p=self.provider();result,status=verify(provider=p)
        self.assertEqual(status,0);self.assertTrue(result['configuration_valid']);self.assertFalse(result['live_verified']);p._session.post.assert_not_called()
    def test_verifier_missing_credentials(self):
        with patch.dict(os.environ,{},clear=True):result,status=verify()
        self.assertEqual(status,2);self.assertFalse(result['configuration_valid']);self.assertEqual(result['diagnostics'][0]['code'],'configuration_error')
    def test_verifier_live_failure_is_sanitized(self):
        tools,_,path=self.fixture();registry=path/'registry.json';registry.write_text(json.dumps({'demo':str(path)}))
        session=Mock();session.post.side_effect=requests.Timeout(SECRET)
        result,status=verify(True,registry,'demo',provider=self.provider(session))
        self.assertEqual(status,1);self.assertFalse(result['live_verified']);self.assertNotIn(SECRET,json.dumps(result));self.assertEqual(result['diagnostics'][-1]['code'],'timeout')

    def test_verifier_mocked_live_success(self):
        tools,_,path=self.fixture();registry=path/'registry.json';registry.write_text(json.dumps({'demo':str(path)}))
        provider=Mock();provider.configuration={'provider':'microsoft_foundry'}
        provider.plan.return_value=ToolPlan(calls=[dict(name='get_fastest_players',arguments={'limit':1})]);provider.select.return_value={'statement_ids':[0]}
        result,status=verify(True,registry,'demo',provider=provider)
        self.assertEqual(status,0);self.assertTrue(result['live_verified']);self.assertFalse(result['fallback'])
    def test_verifier_flag_removes_only_process_ssl_variable(self):
        from scripts.verify_microsoft import main
        observed=[]
        def check(*args):
            observed.append('SSL_CERT_FILE' in os.environ);return {'configuration_valid':True},0
        with patch.dict(os.environ,{'SSL_CERT_FILE':'legacy-test-path'}),patch('dotenv.load_dotenv'),patch('scripts.verify_microsoft.verify',side_effect=check),patch('sys.argv',['verify_microsoft','--without-ssl-cert-file']),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(),0)
        self.assertEqual(observed,[False])
