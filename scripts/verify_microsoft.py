"""Secret-free Microsoft configuration/live verification; never required for startup."""
import argparse
import json
import logging
import os
from pathlib import Path
import certifi
from src.providers.agent import ask
from src.providers.diagnostics import MicrosoftProviderError
from src.providers.microsoft import MicrosoftFoundryProvider

class DiagnosticCapture(logging.Handler):
    def __init__(self):super().__init__();self.items=[]
    def emit(self,record):
        # Only our allowlisted structured diagnostic fields, never exception text.
        try:
            value=json.loads(record.getMessage())
            if value.get('event') in ('provider_failure','agent_fallback'):
                self.items.append({k:value[k] for k in ('event','provider','stage','code','http_status') if k in value})
        except (ValueError,TypeError):pass

def verify(live=False,registry='configs/serving/local.json',recording='118575_clip',question='Who was the fastest player?',provider=None):
    capture=DiagnosticCapture();logger=logging.getLogger('tactiqai.providers');logger.addHandler(capture)
    result=dict(configuration_valid=False,tls_verification=True,ca_bundle='certifi',ssl_cert_file_required=False,ssl_cert_file_present=bool(os.environ.get('SSL_CERT_FILE')),live_requested=live,live_verified=False)
    try:
        provider=provider or MicrosoftFoundryProvider()
        if not Path(certifi.where()).is_file():raise ValueError('CA bundle missing')
        result['configuration_valid']=True
        if live:
            from src.artifacts import LocalArtifactStore
            from src.query import MatchMindTools
            store=LocalArtifactStore(json.loads(Path(registry).read_text()))
            tools=MatchMindTools(store.resolve(recording))
            response=ask(tools,question,provider=provider,recording_id=recording,requested_provider='microsoft')
            result.update(provider=response.provider,fallback=response.fallback,provider_status=response.provider_status,query_type=response.answer.question_type,tools=response.tools)
            result['live_verified']=response.provider=='microsoft_foundry' and not response.fallback and response.provider_status=='grounded_selection'
        result['diagnostics']=capture.items
        return result,0 if not live or result['live_verified'] else 1
    except MicrosoftProviderError as error:
        result['diagnostics']=capture.items or [error.diagnostic()];return result,2
    except Exception:
        # Artifact/configuration failures must not print local paths or raw input.
        result['diagnostics']=[dict(code='verification_setup_error')];return result,2
    finally:logger.removeHandler(capture)

def main():
    from dotenv import load_dotenv
    load_dotenv(override=False)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--without-ssl-cert-file',action='store_true',help='Remove SSL_CERT_FILE in this verification process only, after loading .env.')
    p.add_argument('--live',action='store_true',help='Make bounded model requests using precomputed evidence. Default checks configuration only.')
    p.add_argument('--registry',default='configs/serving/local.json');p.add_argument('--recording',default='118575_clip');p.add_argument('--question',default='Who was the fastest player?')
    a=p.parse_args()
    if a.without_ssl_cert_file:os.environ.pop('SSL_CERT_FILE',None)
    result,status=verify(a.live,a.registry,a.recording,a.question);print(json.dumps(result,indent=2));return status
if __name__=='__main__':raise SystemExit(main())
