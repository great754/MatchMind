"""Development identity-resolution fixtures; not independently verified jerseys."""
import json,tempfile,uuid
from pathlib import Path
from types import SimpleNamespace
from src.assistant.identity import ConfirmedIdentityStore
from src.assistant.schema import IdentityConfirmation

def main():
 players=[{'player_ref':'actor:1','label':'Player A','actor_id':1,'mot_id':7,'team':'left','start':0.,'end':30.,'source_signature':'fixture'},{'player_ref':'actor:2','label':'Player B','actor_id':2,'mot_id':10,'team':'right','start':0.,'end':30.,'source_signature':'fixture'}];cat=SimpleNamespace(players=players,by_ref={p['player_ref']:p for p in players});session=uuid.uuid4();rows=[]
 cases=[('Show player 7',None,['actor:1'],False),('Show #7 passes',None,['actor:1'],False),('How many goals did Alex score?',None,['actor:1'],False),('Compare Alex and Sam',None,['actor:1','actor:2'],False),('Compare player 7 and player 10',None,['actor:1','actor:2'],False),('Player A passes',None,['actor:1'],False),('Player B shots',None,['actor:2'],False),('Show this player actions','actor:1',['actor:1'],False),('How many shots did they take?','actor:2',['actor:2'],False),('Show his passes','actor:1',['actor:1'],False),('Show her shots','actor:2',['actor:2'],False),('Show their goals','actor:1',['actor:1'],False),('Show the selected player','actor:2',['actor:2'],False),('How many shots did player 99 take?',None,[],True),('Show player 1',None,[],True),('Show player 2',None,[],True),('Show all goals',None,[],False),('Compare the teams shots',None,[],False),('Show Alex and Player A',None,['actor:1'],False),('Show SAM passes',None,['actor:2'],False)]
 with tempfile.TemporaryDirectory() as d:
  store=ConfirmedIdentityStore(d)
  for ref,jersey,name,team in [('actor:1',7,'Alex','left'),('actor:2',10,'Sam','right')]:store.confirm('r',IdentityConfirmation(session_id=session,player_ref=ref,jersey=jersey,name=name,team=team,confirmed=True),cat)
  data=store.load('r',session,cat)
  for q,selected,expected,error in cases:
   refs,message=store.resolve(q,data,cat,selected);rows.append({'question':q,'expected':expected,'expected_clarification':error,'actual':refs,'clarification':bool(message),'correct':refs==expected and bool(message)==error})
  data=store.confirm('r',IdentityConfirmation(session_id=session,player_ref='actor:2',jersey=7,name='Sam',team='right',confirmed=True),cat)
  for q in ['Show player 7','Show #7','How many shots did player 7 take?']:
   refs,message=store.resolve(q,data,cat);rows.append({'question':q,'fixture':'both teams number7','correct':not refs and bool(message)})
  store.remove('r',session,'actor:1',cat);refs,message=store.resolve('Show player 7',store.load('r',session,cat),cat);rows.append({'question':'Show player 7 after removing first confirmation','correct':refs==['actor:2'] and message is None})
  refs,message=store.resolve('Show player 7',store.load('r',uuid.uuid4(),cat),cat);rows.append({'question':'Show player7 in another session','correct':not refs and bool(message)})
 result={'protocol':'25 labelled development fixtures with explicitly simulated manual identities; not independent real-player jersey/name ground truth.','correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows};Path('outputs/m18_evaluation/identity_resolution.json').write_text(json.dumps(result,indent=2)+'\n');print(result['correct'],result['total'])
if __name__=='__main__':main()
