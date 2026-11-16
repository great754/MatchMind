"""Compare indexed retrieval to original BAS files and raw exported transfer rows."""
import json,csv,math
from pathlib import Path
from types import SimpleNamespace
from src.intelligence.schema import CanonicalEvent
from src.highlights.index import MatchEventIndex
from src.highlights.schema import HighlightQueryPlan as Plan
ROOT=Path('outputs/m17_evaluation')
def run():
 matrix=[];references=[];rankings=[];alignment=[];tp=fp=fn=0
 for p in sorted(Path('outputs/cache/azure_m17').iterdir()):
  if not (p/'summary.json').is_file():continue
  summary=json.loads((p/'summary.json').read_text());recording=p.name
  export=json.loads((p/'events.json').read_text());idx=MatchEventIndex(recording,SimpleNamespace(summary=summary,events=[CanonicalEvent.model_validate(e) for e in export['events']]))
  matrix.append(idx.capabilities());match=str(summary['match_id']);fps=summary['fps'];offset=summary['source_offset_frames'];half=summary['half'];count=summary['frames']
  bas=json.loads(Path(f'data/soccertrack/bas/{match}/{match}_12_class_events.json').read_text())['actions']
  for kind,label in [('goal','GOAL'),('shot','SHOT'),('cross','CROSS'),('tackle','PLAYER SUCCESSFUL TACKLE')]:
   expected={i:a for i,a in enumerate(bas) if a['label']==label and int(a['gameTime'].split('-')[0])==half and 0<=int(a['frame'])-1-offset<count};actual,_=idx.retrieve(Plan(event_types=[kind]));ids={e['annotation_id'] for e in actual};t=len(ids&expected.keys());f=len(ids-expected.keys());n=len(expected.keys()-ids);tp+=t;fp+=f;fn+=n
   references.append({'recording':recording,'type':kind,'tp':t,'fp':f,'fn':n,'expected_annotation_ids':sorted(expected),'actual_annotation_ids':sorted(ids)})
   for e in actual:
    a=expected.get(e['annotation_id'])
    if a:
     alignment.append({'recording':recording,'event_id':e['event_id'],'annotation_id':e['annotation_id'],'frame_error_seconds':abs(e['start']-(int(a['frame'])-1-offset)/fps),'period_clock_roundtrip_error':abs(idx.timebase.period_to_clip(idx.timebase.period_seconds(e['start']))-e['start']),'annotation_position_minus_frame_seconds':float(a['position'])/1000-(int(a['frame'])-1)/fps})
  rows=list(csv.DictReader((p/'transfers.csv').open()));eligible=[r for r in rows if r['outcome']=='completed' and float(r['observed_step_fraction'])>=.9 and math.isfinite(float(r['distance_m']))]
  actual,_=idx.retrieve(Plan(event_types=['pass'],selection='maximum',metric='distance_m'))
  expected=max(eligible,key=lambda r:float(r['distance_m'])) if eligible else None
  rankings.append({'recording':recording,'correct':bool(actual and expected and abs(actual[0]['metrics']['distance_m']-float(expected['distance_m']))<=.500001e-6 and abs(actual[0]['start']-int(expected['start_frame'])/fps)<1e-8),'expected_distance_m':float(expected['distance_m']) if expected else None,'actual':actual[0] if actual else None})
 result={'protocol':'Original source BAS annotations, separate from event export, define goal/shot/cross/tackle references; frame conversion follows analytics loader. Pass ranking checked against exported transfers, not independent surveyed distance truth.','annotated_retrieval':{'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None},'reference_queries':references,'pass_rankings':rankings,'distance_comparison_tolerance_m':.500001e-6,'distance_tolerance_reason':'CSV export rounds to six decimal places; half a last decimal place, not a relaxed event/ranking criterion.','alignment':alignment,'capabilities':matrix}
 (ROOT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n');print('EVIDENCE',result['annotated_retrieval'],'RANK',sum(r['correct'] for r in rankings),len(rankings),'ALIGN',max(r['frame_error_seconds'] for r in alignment))
if __name__=='__main__':run()
