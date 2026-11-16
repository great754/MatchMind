"""M18 evidence checks. Development labels are not independent human evaluation."""
import argparse,csv,json,re,time,uuid,threading,math
from pathlib import Path
ROOT=Path('outputs/m18_evaluation');ROOT.mkdir(parents=True,exist_ok=True)
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda v:v.item())+'\n')
def language():
 from src.assistant.planner import IntentAndQueryPlanner
 cases=[
 ('Show me all the goals.',{'operation':'actions','actions':['goal'],'video':True}),
 ('How many shots did player 7 take?',{'operation':'statistic','actions':['shot']}),
 ('Show me the fastest sprint.',{'operation':'sprint','video':True,'overlay':True}),
 ('Highlight the player who made the fastest sprint.',{'operation':'sprint','overlay':True}),
 ('Show me everything player 10 did.',{'operation':'actions','video':True}),
 ('What was the most difficult pass?',{'operation':'pass_rank','metric':'difficulty'}),
 ('Who made the longest pass, why was it difficult, and can I see it?',{'operation':'pass_rank','metric':'distance','video':True,'explain':True}),
 ('What was the most difficult pass, why was it difficult, and show me the clip?',{'operation':'pass_rank','metric':'difficulty','video':True,'explain':True}),
 ('Why was that pass difficult?',{'operation':'explain','reference':'event'}),
 ('Show me that pass.',{'operation':'followup','reference':'event','video':True}),
 ('Who had the most possession?',{'operation':'statistic','metric':'possession'}),
 ('Why did the first goal happen?',{'operation':'explain','actions':['goal'],'ordinal':1}),
 ('Make a highlight video of the best attacking plays.',{'operation':'recap','video':True}),
 ('Show me all the tackles by player 5.',{'operation':'actions','actions':['tackle']}),
 ('Who ran the farthest?',{'operation':'statistic','metric':'distance'}),
 ('Compare player 7 and player 10.',{'operation':'compare'}),
 ('Show me all his passes.',{'operation':'actions','actions':['pass']}),
 ('Only show his shots.',{'operation':'actions','actions':['shot']}),
 ('Show me his best sprint.',{'operation':'sprint'}),
 ('Show me what he did in the first half.',{'operation':'actions','period':1}),
 ('How many tackles did he make?',{'operation':'statistic','actions':['tackle']}),
 ('Show me his attacking actions.',{'operation':'actions'}),
 ('Compare his actions with player 10.',{'operation':'compare'}),
 ('Who made the most passes?',{'operation':'statistic','metric':'passes'}),
 ('Who took the most shots?',{'operation':'statistic','metric':'shots'}),
 ('Who made the most tackles?',{'operation':'statistic','metric':'tackles'}),
 ('Who had the highest recorded speed?',{'operation':'statistic','metric':'speed'}),
 ('How many sprints did Player A make?',{'operation':'statistic','actions':['sprint']}),
 ('Show me that pass again.',{'operation':'followup','reference':'event'}),
 ('Who was the player in that clip?',{'operation':'identity'}),
 ('Show me his other actions.',{'operation':'actions'}),
 ("Compare the two teams’ shots.",{'operation':'compare','metric':'shots'}),
 ('Show me all the dribbles.',{'operation':'unsupported'}),
 ('Show me the fastest shot.',{'operation':'unsupported'}),
 ('Show me the fastest pass.',{'operation':'unsupported'}),
 ('Show me all ball recoveries.',{'operation':'unsupported'}),
 ('Show Team A shots.',{'operation':'actions','team':'left','actions':['shot']}),
 ('Show the left team passes.',{'operation':'actions','team':'left'}),
 ('Show Team B goals.',{'operation':'actions','team':'right'}),
 ('Show passes between 1:00 and 2:00.',{'operation':'actions','start':60,'end':120}),
 ('Make a 60-second highlight video.',{'operation':'recap','duration':60}),
 ('Make a one-minute recap.',{'operation':'recap','duration':60}),
 ('More buildup.',{'operation':'followup','buildup':'more'}),
 ('Shorter clips.',{'operation':'followup','buildup':'short'}),
 ('Show the second one.',{'operation':'followup','ordinal':2}),
 ('Show me the defenders around him.',{'operation':'explain','reference':'event'}),
 ('Show me pass receptions.',{'operation':'actions','actions':['reception']}),
 ('Show me headers.',{'operation':'actions','actions':['header']}),
 ('Show me carries.',{'operation':'actions','actions':['carry']}),
 ('How many crosses did this player make?',{'operation':'statistic','actions':['cross']})]
 previous={'players':['actor:1'],'selected_player':'actor:1','plan':{}}
 out=[]
 for q,expected in cases:
  p=IntentAndQueryPlanner().plan(q,previous=previous,provider='local')[0].model_dump();out.append({'question':q,'expected':expected,'actual':p,'correct':all(p[k]==v for k,v in expected.items())})
 save('language.json',{'protocol':'50 hand-authored development request labels; not independently labelled or a held-out population estimate. Expected labels fixed before this evaluation run.','correct':sum(c['correct'] for c in out),'total':len(out),'cases':out});print('language',sum(c['correct'] for c in out),len(out),[c['question'] for c in out if not c['correct']],flush=True)
def sources():
 from src.query import MatchMindTools
 from src.assistant.catalog import PlayerCatalog
 from src.assistant.evidence import AssistantEventIndex,PlayerIntelligenceService
 from src.assistant.sprints import SprintService
 results=[];counts=[];attribution=[];ranks=[];sprints=[];overlay=[]
 for p in sorted(Path('outputs/cache/azure_m17').iterdir()):
  if not (p/'summary.json').exists():continue
  t=MatchMindTools(p);cat=PlayerCatalog(p.name,t);idx=AssistantEventIndex(p.name,t,cat);stats=PlayerIntelligenceService(idx,t,cat);s=t.summary;match=str(s['match_id']);bas=json.loads(Path(f'data/soccertrack/bas/{match}/{match}_12_class_events.json').read_text())['actions'];sync=json.loads(Path(f'data/soccertrack/mot/{match}_sync.json').read_text());raw=list(csv.DictReader((p/'transfers.csv').open()))
  for kind,label in [('goal','GOAL'),('shot','SHOT'),('cross','CROSS'),('tackle','PLAYER SUCCESSFUL TACKLE')]:
   expected={i:a for i,a in enumerate(bas) if a['label']==label and int(a['gameTime'].split('-')[0])==s['half'] and 0<=int(a['frame'])-1-s['source_offset_frames']<s['frames']};actual=idx.select([kind]);ids={e['annotation_id'] for e in actual};results.append({'recording':p.name,'type':kind,'tp':len(ids&expected.keys()),'fp':len(ids-expected.keys()),'fn':len(expected.keys()-ids)})
   for e in actual:
    a=expected.get(e['annotation_id']);who=a.get('player') if a else None
    # The original BAS schema is inspected separately below; unknown attribution is not treated as correct.
    attribution.append({'recording':p.name,'annotation_id':e['annotation_id'],'actor_id':e['actor_id'],'source_annotation':a,'mapping':sync['players'].get(str(e['player'])),'kind':kind,'correct':a is not None and str(e['actor_id'])==str(a.get('player_id'))})
  for player in cat.players:
   ms={m['name']:m['value'] for m in stats.statistics(player['player_ref'])};actor=player['actor_id'];eligible=[r for r in raw if int(r.get('sender_id') or -1)==actor and float(r['observed_step_fraction'])>=.9];complete=[r for r in eligible if r['outcome']=='completed'];counts.append({'recording':p.name,'player_ref':player['player_ref'],'attempts':len(eligible),'completed':len(complete),'attempts_correct':ms['Recorded pass attempts']==len(eligible),'completed_correct':ms['Completed pass estimates']==len(complete)})
  eligible=[r for r in raw if r['outcome']=='completed' and float(r['observed_step_fraction'])>=.9]
  for field in ['distance_m','difficulty_score_0_100']:
   candidates=[r for r in eligible if r.get(field) and math.isfinite(float(r[field]))];actual=[e for e in idx.select(['pass']) if field in e['metrics']];expected=max(candidates,key=lambda r:float(r[field])) if candidates else None;best=max(actual,key=lambda e:e['metrics'][field]) if actual else None;ranks.append({'recording':p.name,'metric':field,'correct':bool(expected and best and abs(float(expected[field])-best['metrics'][field])<.500001e-6),'expected':float(expected[field]) if expected else None,'actual':best['metrics'][field] if best else None})
  if p.name=='118575_clip':
   sprint=SprintService(cat);sprints.append({'recording':p.name,'intervals':sprint.intervals,'smoothing_sensitivity':sprint.sensitivity})
   original=json.loads((ROOT/'first_sprint_job.json').read_text())['manifest'];v=original['visual_annotation'];scale=min(1280/3840,720/2160)
   # Use the actual source dimensions reported in the authorized video metadata.
   from src.highlights.render import probe
   source=next(Path('outputs/highlights_m18/sources').glob('*.mp4'));info=probe(source);stream=next(st for st in info['streams'] if st['codec_type']=='video');w,h=stream['width'],stream['height'];scale=min(1280/w,720/h);px=(1280-w*scale)/2;py=(720-h*scale)/2
   for mark in v['samples']:
    box=cat.at(v['player_ref'],mark['source_frame']);x,y,bw,bh=box;expect=[round(x*scale+px),round(y*scale+py),round((x+bw)*scale+px)-round(x*scale+px),round((y+bh)*scale+py)-round(y*scale+py)];overlay.append({'output_frame':mark['output_frame'],'source_frame':mark['source_frame'],'correct':mark['box']==expect})
 save('source_evidence.json',{'protocol':'Raw BAS annotation entries define event references; raw transfer CSV defines pass counts and rank consistency. This checks attribution/source consistency, not independent real-world event/speed truth.','retrieval':results,'player_counts':counts,'attribution':attribution,'rankings':ranks,'sprints':sprints,'overlay':overlay});print('source',len(results),len(counts),len(ranks),'rankcorrect',sum(r['correct'] for r in ranks),'overlay',sum(r['correct'] for r in overlay),len(overlay),flush=True)
def live(base):
 import requests
 session=str(uuid.uuid4());http=requests.Session();out=[]
 def ask(q,previous=None,recording='118575_clip',provider='local'):
  start=time.monotonic();r=http.post(base+f'/matches/{recording}/assistant',json={'question':q,'session_id':session,'previous_turn':previous,'provider':provider},timeout=100);r.raise_for_status();turn=r.json();job=turn['highlight']
  while job and job['state'] in ['QUEUED','PROCESSING']:
   if time.monotonic()-start>500:raise TimeoutError('Render exceeded evaluation limit')
   time.sleep(.5);r=http.get(base+f"/matches/{recording}/highlights/{job['job_id']}",params={'session_id':session},timeout=30);r.raise_for_status();job=r.json()
  if job:turn['highlight']=job
  row={'question':q,'recording':recording,'turn':turn,'wall_seconds':time.monotonic()-start,'plain_text_code_free':not bool(re.search(r'\b(?:BAS|MOT|GSR)\b|evt_|actor:|TRACK_',turn['text']))}
  if job and job['video_url']:
   v=http.get(base+job['video_url'],headers={'Range':'bytes=0-1023'},timeout=30);row['video_range_status']=v.status_code;row['video_range_bytes']=len(v.content)
  out.append(row);save('live.json',{'protocol':'Actual local API with validated live Azure bundles and registered Azure footage. No evaluation jersey assignment is asserted as real identity.','results':out});print('live',q,turn['text'],job['state'] if job else None,flush=True);return turn
 goal=ask('Show me all the goals.');sprint=ask('Show me the fastest sprint and highlight the player.');ask('Who was that?',sprint['turn_id']);ask('Show me all the actions of player 7.');ask('How many shots did player 7 take?');passed=ask('Show me the most difficult pass and explain why.');again=ask('Show me that pass again.',passed['turn_id']);who=ask('Who was the player in that clip?',again['turn_id']);ask('Show me his other actions.',who['turn_id']);ask("Compare the two teams’ shots.");ask('Who ran the farthest?');ask('Who had the most possession?');ask('Who made the longest pass and show me it.');ask('Make a 60-second highlight video.');ask('Show me all the goals.',recording='test_match');ask('Show me the fastest shot.')
 bad=http.post(base+'/matches/128058_clip/assistant',json={'question':'Who was that?','session_id':session,'previous_turn':goal['turn_id'],'provider':'local'},timeout=30)
 other=http.post(base+'/matches/118575_clip/assistant',json={'question':'Who was that?','session_id':str(uuid.uuid4()),'previous_turn':goal['turn_id'],'provider':'local'},timeout=30)
 endpoints={}
 for name in ['health','ready','recordings']:
  r=http.get(base+'/'+name,timeout=30);endpoints[name]={'status':r.status_code,'body':r.json()}
 save('integration.json',{'endpoints':endpoints,'cross_match_status':bad.status_code,'cross_session_status':other.status_code,'session':session})
 r=http.get(base+'/matches/118575_clip/players/roster',params={'session_id':session},timeout=60);r.raise_for_status();http.post(base+'/matches/118575_clip/players/pictures',timeout=60).raise_for_status()
 for _ in range(120):
  roster=http.get(base+'/matches/118575_clip/players/roster',params={'session_id':session},timeout=60).json()
  if roster['pictures']['state'] not in ['PROCESSING','QUEUED']:break
  time.sleep(.5)
 save('portraits_roster.json',roster)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['language','sources','live']);p.add_argument('--base',default='http://127.0.0.1:8000');a=p.parse_args();language() if a.mode=='language' else sources() if a.mode=='sources' else live(a.base)
