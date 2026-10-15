import argparse
import csv
import hashlib
import json
from pathlib import Path
from src.artifacts import LocalArtifactStore
from src.query import MatchMindTools

SCHEMAS={
 'Match':['recording_id','match_id','scope','half','frames','duration_seconds','source_files_json','analytics_config_json','limitations_json'],
 'Player':['recording_id','actor_id','tracker_id','team','max_speed_kmh','measured_distance_m','peak_speed_seconds'],
 'Event':['recording_id','event_id','timestamp','frame','type','actor_id','tracker_id','team','validation_state','metrics_json'],
 'Transfer':['recording_id','transfer_id','start_seconds','end_seconds','sender_actor_id','receiver_actor_id','sender_team','outcome','distance_m','difficulty_score_0_100'],
 'TacticalState':['recording_id','frame','timestamp','team','state_json'],
 'Evaluation':['recording_id','split','detector','detections','reference_events','matched','false_positives','false_negatives','precision','recall','f1']}
KEYS={'Match':['recording_id'],'Player':['recording_id','actor_id'],'Event':['recording_id','event_id'],'Transfer':['recording_id','transfer_id'],'TacticalState':['recording_id','frame','team'],'Evaluation':['recording_id','detector']}

def export(registry,output,evaluation=None):
    store=LocalArtifactStore(registry);tables={name:[] for name in SCHEMAS};aliases={};output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for key in store.ids():
        tools=MatchMindTools(store.resolve(key));s=tools.summary
        aliases[(str(s.get('match_id')),s.get('scope'),s.get('half'))]=key
        tables['Match'].append(dict(recording_id=key,**{f:s.get(f) for f in SCHEMAS['Match'][1:] if not f.endswith('_json')},source_files_json=json.dumps(s.get('source_files',[]),sort_keys=True),analytics_config_json=json.dumps(s.get('config',{}),sort_keys=True),limitations_json=json.dumps(s.get('limitations',[]))))
        for p in tools.players:tables['Player'].append(dict(recording_id=key,actor_id=p['player_id'],tracker_id=p.get('mot_id'),**{f:p.get(f) for f in ('team','max_speed_kmh','measured_distance_m','peak_speed_seconds')}))
        for e in tools.events:tables['Event'].append(dict(recording_id=key,event_id=e.event_id,timestamp=e.timestamp,frame=e.frame,type=e.type.value,actor_id=e.actor_id,tracker_id=e.player_id,team=e.team,validation_state=e.validation_state,metrics_json=json.dumps(e.metrics.model_dump(mode='json'),sort_keys=True,allow_nan=False)))
        for t in tools.transfers:tables['Transfer'].append(dict(recording_id=key,transfer_id=t.get('transfer_id',t.get('pass_id')),sender_actor_id=t.get('sender_id'),receiver_actor_id=t.get('receiver_id'),**{f:t.get(f) for f in ('start_seconds','end_seconds','sender_team','outcome','distance_m','difficulty_score_0_100')}))
        for state in tools.states:
            for team in ('left','right'):tables['TacticalState'].append(dict(recording_id=key,frame=state.frame,timestamp=state.timestamp,team=team,state_json=json.dumps(getattr(state,team).model_dump(mode='json'),sort_keys=True,allow_nan=False)))
    if evaluation:
        for r in json.loads(Path(evaluation).read_text())['recordings']:
            alias=aliases.get((str(r.get('match_id')),r.get('scope'),r.get('half')))
            if alias is None:continue
            for kind in ('transfers','shots'):tables['Evaluation'].append(dict(recording_id=alias,split=r['split'],detector=kind,**{f:r[kind].get(f) for f in SCHEMAS['Evaluation'][3:]}))
    manifest=dict(schema_version='1.0',tables={},identity_policy='actor_id is GSR actor namespace; tracker_id is MOT/display identity, never verified jersey number.',foreign_keys='Every table.recording_id references Match.recording_id. Event actor_id may be null/unmapped; never infer a name or jersey.',timestamp_policy='Seconds from recording start, not wall-clock UTC. Join with recording_id and half/scope, never timestamp alone.',limitations='Development metrics only; no held-out match is installed. CSV fields ending _json retain nested typed source evidence. No calculations are performed here.')
    for name,fields in SCHEMAS.items():
        rows=tables[name];keys=[tuple(row.get(f) for f in KEYS[name]) for row in rows]
        if len(keys)!=len(set(keys)) or any(None in k for k in keys):raise ValueError('Duplicate or missing analytical primary key: '+name)
        path=output/(name+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        manifest['tables'][name]=dict(columns=fields,column_types={f:('int64' if f in ('half','frames','frame','actor_id','tracker_id','transfer_id','sender_actor_id','receiver_actor_id','detections','reference_events','matched','false_positives','false_negatives') else 'float64' if f in ('timestamp','start_seconds','end_seconds','duration_seconds','max_speed_kmh','measured_distance_m','peak_speed_seconds','distance_m','difficulty_score_0_100','precision','recall','f1') else 'string') for f in fields},primary_key=KEYS[name],rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nullable_columns=[f for f in fields if any(r.get(f) is None for r in rows)])
    (output/'schema.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,default=Path('configs/serving/local.json'));p.add_argument('--output',type=Path,default=Path('outputs/fabric'));p.add_argument('--evaluation',type=Path,default=Path('outputs/evaluation/evaluation_summary.json'));a=p.parse_args();result=export(json.loads(a.registry.read_text()),a.output,a.evaluation);print(json.dumps({k:v['rows'] for k,v in result['tables'].items()},indent=2))
if __name__=='__main__':main()
