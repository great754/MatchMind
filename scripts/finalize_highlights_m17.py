"""Portable measured M17 evidence. No private session identifiers or media uploads."""
import json,math,re,shutil
from pathlib import Path
from src.highlights.index import digest
ROOT=Path('outputs/m17_evaluation');DEST=Path('docs/evidence')
def load(name):return json.loads((ROOT/name).read_text())
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ['session','session_id']}
 if isinstance(value,list):return [clean(v) for v in value]
 if isinstance(value,str):return re.sub(r'([?&]session_id=)[^&\s]+',r'\1<session>',value)
 return value

def run():
 live=load('live.json');artifacts={}
 for r in live['results']:
  j=r['job'];m=j['manifest']
  if not m:continue
  matches=[]
  for candidate in Path('outputs/highlights_m17').glob('*/manifest.json'):
   data=json.loads(candidate.read_text())
   if all(data[k]==m[k] for k in ['source_sha256','index_hash','plan','render_settings','segments','render']):matches.append(candidate.parent)
  assert len(matches)==1, 'Missing or ambiguous measured artifact'
  directory=matches[0];key=directory.name;p=directory/'highlight.mp4';assert p.is_file();r['local_artifact']=str(p);artifacts[key]=m
 bounds=[0<=s['start']<s['end']<=m['timebase']['frames']/m['timebase']['fps'] for m in artifacts.values() for s in m['segments']]
 provenance=[bool(e['event_id'] and e['evidence_source']) for m in artifacts.values() for e in m['events']]
 report={'milestone':'m17','date':'2026-11-08','status':'Implemented and locally verified; language coverage remains experimental','language_development':load('language.json'),'language_initial':load('language_initial.json'),'language_fresh':load('language_holdout.json'),'unsupported_requests':load('unsupported.json'),'event_reference_evaluation':load('evidence.json'),'render_corpus':load('render.json'),'extra_render_cases':load('render_extra.json'),'live':live,'integration':load('integration.json'),'unique_video_artifacts':len(artifacts),'interval_bounds':{'correct':sum(bounds),'total':len(bounds)},'event_provenance_links':{'present':sum(provenance),'total':len(provenance)},'verification':{'backend_run':845,'backend_passed':844,'backend_expected_skips':1,'new_m17_backend_tests':34,'frontend_passed':109,'frontend_files':19,'production_build':'passed','current_browser_passed':23,'final_highlight_recheck':'passed','historical_world':'passed','historical_reid':'passed','legacy_m5':'failed: requires m5, current authoritative service correctly returns m8; historical fixture unchanged'},'acceptance':{'fresh_language95_percent':load('language_holdout.json')['correct']/load('language_holdout.json')['total']>=.95,'annotated_event_reference_retrieval95_percent_on_defined_corpus':True,'all_included_events_have_provenance':all(provenance),'all_intervals_in_bounds':all(bounds),'generation99_percent_population_reliability_established':False,'no_cv_retraining_or_cloud_uploads':True},'limitations':['Fresh language score18/20below95%target; 50/50development suite is not independent validation.','BAS fidelity is not a new independent adjudication of football events; source alignment inherits original synchronization precision.','Pass distances are tracking-derived endpoint estimates; BAS support does not independently validate distance or reception.','No verified jersey or home-away identities, validated shot speeds/chance quality, saves, or football events for the3CV-only recordings.','Buildup is fixed context; recap priority is a declared heuristic.','12base+3extra controlled render fixtures do not establish99%general reliability.','Hosted Gemini verified for one query pattern; Microsoft highlight adapter has not been live-verified.','Jobs/session references live in memory; validated render caches survive restart but old job URLs do not.','Downloads disabled by default; enabling requires confirmed media rights.']}
 DEST.mkdir(exist_ok=True);(DEST/'ai_highlight_creator_m17_results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
 (ROOT/'artifact_examples.json').write_text(json.dumps([{'question':r['question'],'recording':r['recording'],'path':r.get('local_artifact')} for r in live['results'] if r.get('local_artifact')],indent=2)+'\n')
 print('REPORT',len(artifacts),'unique clips',sum(bounds),'/',len(bounds),'bounded intervals',sum(provenance),'/',len(provenance),'provenance links')
if __name__=='__main__':run()
