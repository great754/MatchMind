"""Frozen-artifact cross-recording diagnostics. Does not run or tune analytics.

Canonical bundles must already have passed ArtifactStore validation. Optional local
exports are accepted only when their summary and event bytes match the bundle.
"""
import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from .aggregate import aggregate_events
from .event_evaluation import evaluate_events,evaluate_transfers
from src.query.tools import MatchMindTools
from src.query.intelligence import build_intelligence

CATEGORIES=['OBSERVED','INFERRED','HEURISTIC','REFERENCE-SUPPORTED','DEPENDENT VALIDATION','UNAVAILABLE']

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def number(value):
    try:return float(value) if value not in ('',None) else None
    except (ValueError,TypeError):return None

def guard_intervals(rows):
    for i,a in enumerate(rows):
        for b in rows[:i]:
            if a['recording_id']==b['recording_id']:raise ValueError('Duplicate recording')
            if (a['match_id'],a['half'])==(b['match_id'],b['half']) and max(a['source_offset_frames'],b['source_offset_frames'])<min(a['source_offset_frames']+a['frames'],b['source_offset_frames']+b['frames']):
                raise ValueError('Overlapping recordings cannot be aggregated')

def frozen_exports(bundle,exports):
    if not exports:return None
    for name in ('summary.json','events.json'):
        p=exports/name
        if not p.is_file() or p.read_bytes()!=(bundle/name).read_bytes():return None
    return exports

def row_for(bundle,exports=None):
    bundle=Path(bundle)
    from src.artifacts.manifest import validate_snapshot
    validate_snapshot(bundle)
    tools=MatchMindTools(bundle);s=tools.summary
    row={k:s.get(k) for k in ('match_id','half','frames','source_offset_frames','duration_seconds','ball_position_fraction','transfer_outcomes')}
    row['source_offset_frames']=row['source_offset_frames'] or 0
    row['recording_id']=bundle.name
    row['canonical_sha256']={name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in ('summary.json','events.json')}
    players=tools.players
    coverage=[p.get('observed_position_fraction') for p in players if p.get('observed_position_fraction') is not None]
    row['player_observation_coverage']=sum(coverage)/len(coverage) if coverage else None
    states=s.get('possession_state_frames',{})
    row['confirmed_control_fraction']=states.get('controlled',0)/s['frames'] if states else None
    ownership=tools.ownership
    row['attribution_coverage']=sum(o['attribution']!='unknown' for o in ownership)/len(ownership) if ownership else None
    row['unknown_attribution_fraction']=1-row['attribution_coverage'] if ownership else None
    row['formation_unknown_fraction']=sum(f['estimate']=='unknown' for f in tools.formations)/len(tools.formations) if tools.formations else None
    row['pressure_evidence_samples']=sum(st.pressure.defenders_within_local is not None for st in tools.states)
    row['alignment_diagnostics']=None
    # Expose local alignment only when its bytes match canonical source provenance.
    for source in s.get('source_files',[]):
        path=Path(source['path'])
        if path.name.endswith('_sync.json') and path.is_file():
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()==source.get('sha256'):
                sync=json.loads(raw)
                row['alignment_diagnostics']={'mean_assignment_error_px':sync.get('mean_assignment_error_px'),'source_sha256':source['sha256'],'classification':'HEURISTIC','limitation':'Tracking/image assignment residual, not independent pitch-position accuracy.'}
    row['projection_diagnostics']=None
    row['shot_candidates']=s.get('trajectory_shot_candidates')
    row['classifications']={'player_observation_coverage':'OBSERVED','ball_position_fraction':'INFERRED','transfers':'DEPENDENT VALIDATION','shots':'DEPENDENT VALIDATION','formation_unknown_fraction':'HEURISTIC','pressure_evidence_samples':'HEURISTIC','alignment_diagnostics':'UNAVAILABLE','projection_diagnostics':'UNAVAILABLE'}
    if row['alignment_diagnostics']:row['classifications']['alignment_diagnostics']='HEURISTIC'
    ext=frozen_exports(bundle,exports)
    row['diagnostic_exports_status']='byte-matched canonical summary/events' if ext else 'UNAVAILABLE: matching local diagnostic exports absent'
    row['diagnostics']=[]
    if ext:
        row['diagnostic_export_sha256']={name:hashlib.sha256((ext/name).read_bytes()).hexdigest() for name in ('reference_events.csv','shots.csv','bas_shot_windows.csv','ball_frames.csv')}
        refs=read_csv(ext/'reference_events.csv')
        for r in refs:
            for k in ('annotation_id','frame','player_id'):r[k]=int(float(r[k])) if r.get(k) else None
        transfers=read_csv(bundle/'transfers.csv')
        for t in transfers:
            for k in ('start_frame','sender_id'):t[k]=int(float(t[k])) if t.get(k) else None
        shots=read_csv(ext/'shots.csv')
        for shot in shots:shot['start_frame']=int(shot['start_frame'])
        fps=s.get('fps',25)
        row['transfers']=evaluate_transfers(transfers,refs,fps)
        row['shots']=evaluate_events(shots,[r for r in refs if r['label']=='SHOT'],fps)
        # Reject drift instead of silently reporting a different frozen baseline.
        for key,summary_key in [('transfers','pass_validation'),('shots','shot_validation')]:
            for metric in ('matched','reference_events','detections'):
                if metric in s.get(summary_key,{}) and row[key][metric]!=s[summary_key][metric]:
                    raise ValueError(f'{bundle.name}: frozen {key} {metric} disagrees with canonical baseline')
        from .shot_diagnostics import diagnose_exported_shots
        row['diagnostics']=diagnose_exported_shots(ext,row['shots'],refs,fps)
    else:
        for key,summary_key in [('transfers','pass_validation'),('shots','shot_validation')]:
            row[key]=dict(s.get(summary_key,{}),matching=[],timing_error_seconds={'mean':None,'median':None,'p95':None})
    insights=build_intelligence(tools,bundle.name)
    row['demo_insights']=[{'title':i.title,'timestamp':i.timestamp,'actor_id':i.actor_id,'event_id':i.event_id} for i in insights.insights if i.timestamp is not None]
    # Ranking is explanatory evidence density, not scientific accuracy or a quality probability.
    row['demo_score']=len(row['demo_insights'])*10+min(s.get('transfer_outcomes',{}).get('completed',0),50)/10+(s.get('ball_position_fraction') or 0)*5+(row['transfers'].get('f1') or 0)*5
    return row

def build_report(bundles,exports_root=None):
    rows=[row_for(Path(p),Path(exports_root)/('analytics_'+Path(p).name.removesuffix('_clip')+'_tactics') if exports_root else None) for p in sorted(bundles)]
    guard_intervals(rows)
    if not rows:raise ValueError('No recording bundles')
    chosen=sorted(rows,key=lambda r:(-r['demo_score'],r['recording_id']))[0]
    return dict(schema_version='1.0',categories=CATEGORIES,recordings=rows,
        aggregate={k:aggregate_events([r[k] for r in rows]) for k in ('transfers','shots')},
        demo={'recording_id':chosen['recording_id'],'score':chosen['demo_score'],'ranking':'10 x seekable insight categories + capped completed transfers / 10 + 5 x supplied ball coverage + 5 x dependent transfer F1','moments':chosen['demo_insights'],
        'sequence':['0–10s: select this recording; select Local provider.','10–25s: play video and tactical pitch together; inspect player movement.','25–40s: open fastest tracked player insight and Review its actual timestamp.','40–55s: ask Which player was fastest?; open Analyst evidence.','55–65s: switch to Beginner; facts remain the same.','65–80s: open difficult completed transfer insight; seek and inspect evidence.','80–90s: show canonical provenance and geometric/annotation limitations.']},
        limitations=['BAS anchors also reconstruct the ball: all reference agreement is DEPENDENT VALIDATION, never independent accuracy.','Supplied ball-position coverage is not independent visual detection coverage.','Micro event counts exclude zero-reference clips from precision claims. No per-clip F1 averaging.','No thresholds, alignment, possession or event inference changed.','Alignment uses hash-matched local sync provenance when available. Independent projection audits unavailable; no historical constants reused.','Optional local CSV exports must match canonical summary/events bytes; rounded CSV geometry is diagnostic only.'])

def write_report(report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'quality.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['recording_id'],r['duration_seconds'],r['ball_position_fraction'],r['player_observation_coverage'],r['unknown_attribution_fraction'],r['transfers'].get('f1'),r['shots'].get('reference_events'),r['shots'].get('f1'),r['formation_unknown_fraction']])+'</tr>' for r in report['recordings'])
    diagnostics=''.join('<details><summary>'+html.escape(r['recording_id'])+' — '+str(len(r['diagnostics']))+' frozen shot errors</summary>'+''.join('<details><summary>'+html.escape(d['match_status']+' at '+str(d['timestamp'])+'s: '+', '.join(d['hypotheses']))+'</summary><pre>'+html.escape(json.dumps(d,indent=2))+'</pre></details>' for d in r['diagnostics'])+'</details>' for r in report['recordings'] if r['diagnostics'])
    (output/'quality.html').write_text('<!doctype html><meta charset="utf-8"><title>TactiqAI frozen artifact quality</title><style>body{font:16px system-ui;margin:2rem;background:#101926;color:#eee}td,th{padding:.6rem;border:1px solid #667}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{padding:.6rem;border:1px solid #456}a{color:#8dcaff}</style><h1>TactiqAI cross-clip quality</h1><p>DEPENDENT VALIDATION — BAS-derived ball anchors prevent independent accuracy claims. Null means unavailable. Coverage values are fractions.</p><p>OBSERVED · INFERRED · HEURISTIC · REFERENCE-SUPPORTED · DEPENDENT VALIDATION · UNAVAILABLE</p><p><a href="quality.json">Full machine-readable evidence</a> · <a href="demo.md">90-second demo</a></p><table><tr>'+''.join('<th>'+x+'</th>' for x in ['Recording','Seconds','Ball supplied','Player observed','Unknown attribution','Transfer F1','Shot refs','Shot F1','Unknown formation'])+'</tr>'+rows+'</table><h2>Micro aggregation</h2><pre>'+html.escape(json.dumps(report['aggregate'],indent=2))+'</pre><h2>Frozen shot errors: inspect evidence</h2>'+diagnostics+'<h2>Evidence-ranked demo</h2><pre>'+html.escape(json.dumps(report['demo'],indent=2))+'</pre><h2>Limits</h2><pre>'+html.escape(json.dumps(report['limitations'],indent=2))+'</pre>')
    (output/'demo.md').write_text('# TactiqAI 90-second demo\n\nRecording: '+report['demo']['recording_id']+'\n\n'+'\n'.join(report['demo']['sequence'])+'\n\nVerified artifact moments:\n'+json.dumps(report['demo']['moments'],indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundles',required=True,type=Path);p.add_argument('--exports',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    bundles=[p.parent for p in a.bundles.glob('*_clip/bundle.json')]
    report=build_report(bundles,a.exports);write_report(report,a.output)
    print(json.dumps({'recordings':len(report['recordings']),'aggregate':report['aggregate'],'demo':report['demo']['recording_id']},indent=2))
if __name__=='__main__':main()
