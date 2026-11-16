"""Explicit supported/unsupported cases; no model-produced reference labels."""
import json
from pathlib import Path
from types import SimpleNamespace
from src.highlights.agent import HighlightAgent
from src.highlights.index import MatchEventIndex
from src.intelligence.schema import CanonicalEvent
CASES=[('118575_clip','Show all goals',True),('118575_clip','Show the first goal',True),('118575_clip','Show the second goal',False),('118575_clip','What was the longest pass?',True),('118575_clip','Show all shots',False),('118575_clip','Show the fastest shot',False),('118575_clip','Show goals by the home team',False),('118575_clip',"Show player 10's highlights",False),('118575_clip','Show passes by tracked player 15',True),('118575_clip','Show goals by the left side',False),('118575_clip','Show goals by the right side',True),('117093_clip','Show all shots',True),('117093_clip','Show all goals',False),('117093_clip','Show all saves',False),('128058_clip','Show all shots',False),('128058_clip','Review unconfirmed shots',True),('test_match','Show all goals',False),('team_goal','What was the longest pass?',False),('southampton',"Show player 10's highlights",False),('118575_clip','Show the biggest chances',False)]
def run():
 indexes={};results=[];a=HighlightAgent()
 for rec,q,expected in CASES:
  if rec not in indexes:
   p=Path('outputs/cache/azure_m17')/rec
   if p.exists():t=SimpleNamespace(summary=json.loads((p/'summary.json').read_text()),events=[CanonicalEvent.model_validate(e) for e in json.loads((p/'events.json').read_text())['events']]);indexes[rec]=MatchEventIndex(rec,t)
   else:indexes[rec]=MatchEventIndex(rec)
  plan=a.parse(q);rows,message=indexes[rec].retrieve(plan);actual=bool(rows) and plan.intent!='unsupported';results.append({'recording':rec,'question':q,'expected_available':expected,'actual_available':actual,'correct':actual==expected,'review_only':plan.review,'message':message})
 r={'protocol':'20 hand-labelled evidence availability/unsupported requests from available BAS references and explicit metric/identity limits. Review availability does not certify event truth.','correct':sum(r['correct'] for r in results),'total':len(results),'cases':results};Path('outputs/m17_evaluation/unsupported.json').write_text(json.dumps(r,indent=2));print('LIMITS',r['correct'],r['total'],[r['question'] for r in results if not r['correct']])
if __name__=='__main__':run()
