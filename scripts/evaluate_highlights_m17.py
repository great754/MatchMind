"""M17 measured language/retrieval/render/live evidence, without changing analytics."""
import json,time,uuid,argparse,hashlib,threading
from pathlib import Path
import requests
ROOT=Path('outputs/m17_evaluation');ROOT.mkdir(parents=True,exist_ok=True)
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def language():
 from src.highlights.agent import HighlightAgent
 a=HighlightAgent();cases=json.loads(Path('configs/highlights/m17_language_test.json').read_text());results=[]
 for q,expected in cases:
  expected={**{'intent':'create'},**expected}
  try:p=a.parse(q).model_dump();checks={k:p[k]==v for k,v in expected.items()};error=None
  except Exception as e:p={};checks={k:False for k in expected};error=type(e).__name__
  results.append({'question':q,'expected':expected,'actual':p,'correct':all(checks.values()),'checks':checks,'error':error})
 dimensions={}
 for key in ['intent','event_types','metric','team','player','player_namespace','start','end','duration','ordinal']:
  tested=[r['checks'][key] for r in results if key in r['checks']];dimensions[key]={'correct':sum(tested),'total':len(tested)}
 follow=[]
 for q,prev,expected in [('Just show the second one','Show all goals',{'ordinal':2,'event_types':['goal']}),('Who made that pass?','What was the longest pass?',{'intent':'identity','event_types':['pass']}),('More buildup','Show first goal',{'buildup':'more','ordinal':1}),('Shorter clips','Show all shots',{'buildup':'short','event_types':['shot']})]:
  p=a.parse(q,a.parse(prev)).model_dump();follow.append({'question':q,'previous':prev,'correct':all(p[k]==v for k,v in expected.items()),'actual':p,'expected':expected})
 save('language.json',{'protocol':'50 hand-authored component-labelled English requests plus4followups; local interpreter only, not a population estimate or live hosted-model accuracy','suite_sha256':hashlib.sha256(Path('configs/highlights/m17_language_test.json').read_bytes()).hexdigest(),'correct':sum(r['correct'] for r in results),'total':len(results),'dimensions':dimensions,'cases':results,'followup':follow});print('LANGUAGE',sum(r['correct'] for r in results),len(results),[r['question'] for r in results if not r['correct']],flush=True)

def live(base='http://127.0.0.1:8000'):
 s=requests.Session();session=str(uuid.uuid4());results=[];caps=[]
 catalog=s.get(base+'/recordings',timeout=30).json()['recordings']
 for r in catalog:
  response=s.get(base+f"/matches/{r['recording_id']}/highlights/capabilities",timeout=30);response.raise_for_status();caps.append(response.json())
 save('capabilities.json',caps)
 def ask(recording,question,previous=None,provider='local'):
  started=time.monotonic();r=s.post(base+f'/matches/{recording}/highlights',json={'question':question,'session_id':session,'previous_job':previous,'provider':provider},timeout=60);r.raise_for_status();job=r.json()
  while job['state'] in ['QUEUED','PROCESSING']:
   if time.monotonic()-started>600:raise TimeoutError('Highlight generation exceeded evaluation budget')
   time.sleep(.5);job=s.get(base+f"/matches/{recording}/highlights/{job['job_id']}",params={'session_id':session},timeout=30).json()
  record={'recording':recording,'question':question,'previous_job':previous,'job':job,'wall_seconds':time.monotonic()-started}
  if job['video_url']:
   video=s.get(base+job['video_url'],headers={'Range':'bytes=0-1023'},timeout=30);record['range_status']=video.status_code;record['range_bytes']=len(video.content)
   manifest=job['manifest'];record['all_intervals_in_bounds']=all(0<=v['start']<v['end']<=manifest['timebase']['frames']/manifest['timebase']['fps'] for v in manifest['segments']);record['all_events_have_provenance']=all(e['evidence_source'] and e['event_id'] for e in manifest['events'])
  results.append(record);save('live.json',{'results':results});print('DEMO',recording,question,job['state'],job['message'],flush=True);return job
 goals=ask('118575_clip','Show me all the goals.')
 ask('118575_clip','Show me the buildup to the first goal.')
 longest=ask('118575_clip','What was the longest pass?')
 ask('117093_clip','Show me all the shots.')
 ask('118575_clip','Make a one-minute highlight video.')
 ask('118575_clip','Show me the second one.',goals['job_id'])
 ask('118575_clip','Who made that pass?',longest['job_id'])
 ask('118575_clip','Show me all the goals by the home team.')
 ask('test_match','Show me all the goals.')
 ask('118575_clip','Show me the fastest shot.')
 ask('118575_clip',"Show me player 10's highlights.")
 ask('128058_clip','Review unconfirmed shots')
 ask('118575_clip','Show me all the goals.') # cache demonstration
 hosted=ask('118575_clip','Show me all the goals.',provider='auto')
 wrong=s.post(base+'/matches/128058_clip/highlights',json={'question':'Show the second one','session_id':session,'previous_job':goals['job_id'],'provider':'local'},timeout=30)
 other=s.get(base+f"/matches/118575_clip/highlights/{goals['job_id']}",params={'session_id':str(uuid.uuid4())},timeout=30)
 health={name:s.get(base+'/'+name,timeout=30).status_code for name in ['health','ready','recordings']};answer=s.post(base+'/matches/118575_clip/ask',json={'question':'Which player ran the fastest?','provider':'local'},timeout=60).json()
 save('integration.json',{'wrong_match_context_status':wrong.status_code,'wrong_session_status':other.status_code,'endpoints':health,'ready':s.get(base+'/ready').json(),'ask':{'status':answer['answer']['status'],'evidence_rows':len(answer['evidence'])},'hosted_provider_actual':hosted['provider'],'hosted_provider_diagnostic':hosted.get('provider_diagnostic')})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['language','live']);p.add_argument('--base',default='http://127.0.0.1:8000');a=p.parse_args();language() if a.mode=='language' else live(a.base)
