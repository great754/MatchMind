"""Actual HTTP smoke checks against a running MatchMind artifact service."""
import argparse
import json
import requests

def smoke(base,recording,hosted=True):
    session=requests.Session();session.trust_env=False
    def get(path,status=200):
        r=session.get(base+path,timeout=20);assert r.status_code==status,(path,r.status_code);return r
    def post(body,status=200):
        if not hosted:body={**body,'provider':'local'}
        r=session.post(base+'/matches/'+recording+'/ask',json=body,timeout=60);assert r.status_code==status,r.status_code;return r
    prefix='/matches/'+recording
    get('/health');get('/ready');get(prefix+'/summary');events=get(prefix+'/events').json()
    get('/matches/missing/summary',404)
    facts=events['facts'];event=next((f['event_id'] for f in facts if f['event_id']),None)
    assert event,'Smoke recording needs at least one canonical event'
    get(prefix+'/events/'+event+'/explain');get(prefix+'/events/unknown/explain',404)
    fastest=post(dict(question='Who was the fastest player?',provider='local')).json()
    assert fastest['answer']['status']=='ok' and fastest['evidence']
    actor=next(f['value'] for f in fastest['evidence'] if f['field']=='player_id')
    get(prefix+'/players/'+str(actor));assert get(prefix+'/players/999999999').json()['status']=='unknown'
    assert post(dict(question='What did the coach tell the team?')).json()['answer']['status']=='unsupported'
    post(dict(question='',secret='must-not-echo'),422)
    result=post(dict(question='Who was the fastest player?',provider='microsoft' if hosted else 'local')).json()
    assert result['answer']['facts']==fastest['answer']['facts']
    assert 'must-not-echo' not in post(dict(question='',secret='must-not-echo'),422).text
    report=get(prefix+'/report')
    assert 'MATCHMIND_API_BASE' in report.text and 'ask-provider' in report.text
    assert report.headers.get('X-Request-ID')
    if 'src="video.mp4"' in report.text:
        media=session.get(base+prefix+'/video.mp4',headers={'Range':'bytes=0-63'},timeout=20)
        assert media.status_code==206 and len(media.content)==64

    return dict(status='passed',recording=recording,event_lookup=True,player_lookup=True,provider=result['provider'],fallback=result['fallback'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8000');p.add_argument('--recording',default='118575_clip');p.add_argument('--local-only',action='store_true',help='Exercise local HTTP routes without any external provider call.');a=p.parse_args();print(json.dumps(smoke(a.base,a.recording,not a.local_only),indent=2))
if __name__=='__main__':main()
