"""Evaluate a manifest of non-overlapping recordings with frozen detector configs."""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
from src.analytics.config import AnalyticsConfig
from src.analytics.data import load_recording
from src.analytics.pipeline import analyze_recording,write_outputs
from src.vision.data_loading import DatasetPaths
from src.tactics.config import TacticsConfig
from .event_evaluation import evaluate_events,evaluate_transfers
from .possession_evaluation import evaluate_possession
from .aggregate import aggregate_events,aggregate_transfer_classes
from .tactical_evaluation import sensitivity,implementation_signature


def run(manifest,output,data_root=Path('data/soccertrack'),run_sensitivity=True,reuse_sensitivity=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];keys=set();intervals={}
    # Reject collisions/overlaps before creating per-recording outputs or aggregate counts.
    recordings=[]
    for item in manifest['recordings']:
        if item.get('split','evaluation') not in ('development','evaluation'):raise ValueError('split must be development/evaluation')
        match=str(item['match_id']);scope=item.get('scope','clip');half=item.get('half')
        recording=load_recording(DatasetPaths(match,Path(data_root)),scope,'gsr',half)
        key=f'{match}_{scope}_h{recording.half}'
        if key in keys:raise ValueError(f'Duplicate recording: {key}')
        keys.add(key)
        group=(match,recording.half);lo=recording.source_offset;hi=lo+len(recording.ball)
        if any(lo<b and hi>a for a,b in intervals.get(group,[])):raise ValueError('Overlapping recordings would double-count aggregate events; evaluate clip separately from its half.')
        intervals.setdefault(group,[]).append((lo,hi));recordings.append((key,item,recording))
    for key,item,recording in recordings:
        print(f'Evaluating {key}: {len(recording.ball)} frames',flush=True)
        values=dict(manifest.get('analytics_config',{}))
        values.setdefault('fps',recording.fps);values.setdefault('left_attack_direction',1 if recording.half==1 else -1)
        config=AnalyticsConfig(**values);tconfig=TacticsConfig(**manifest.get('tactics_config',{}))
        result=analyze_recording(recording,config)
        dest=output/'per_match'/key
        cached_path=dest/'evaluation.json'
        cached=json.loads(cached_path.read_text()) if reuse_sensitivity and cached_path.exists() else None
        summary=write_outputs(recording,result,config,dest,True,tactics_config=tconfig)
        reuse=cached is not None and cached['tactics'].get('implementation_sha256')==implementation_signature() and cached['source_files']==summary['source_files'] and cached['config']==asdict(config) and cached['tactics']['baseline_config']==json.loads(json.dumps(asdict(tconfig)))
        measured=dict(recording_id=key,match_id=recording.sync['match_id'],scope=recording.scope,half=recording.half,
            split=item.get('split','evaluation'),frames=len(recording.ball),duration_seconds=len(recording.ball)/recording.fps,
            config=asdict(config),source_files=summary['source_files'],
            transfers=evaluate_transfers(result['passes'],recording.events,config.fps,config.validation_tolerance_seconds),
            shots=evaluate_events(result['shots'],[e for e in recording.events if e['label']=='SHOT'],config.fps,config.validation_tolerance_seconds),
            goals=[e for e in recording.events if e['label']=='GOAL'],
            possession=evaluate_possession(result['possession'],recording.events,config.fps),
            tactics=cached['tactics'] if reuse else sensitivity(recording,result,config,tconfig,run_sensitivity),limitations=summary['limitations'])
        (dest/'evaluation.json').write_text(json.dumps(measured,indent=2,allow_nan=False)+'\n');rows.append(measured)
    # Development data are visible but excluded from evaluation aggregates.
    evaluated=[r for r in rows if r['split']=='evaluation']
    payload=dict(schema_version='1.0',recordings=rows,aggregate={k:aggregate_events([r[k] for r in evaluated]) for k in ('transfers','shots')},
        split_policy='Development recordings excluded from evaluation aggregates. Existing thresholds developed using 118575; held-out matches are unavailable.',
        independence='BAS checks are not independent: reconstructed ball trajectories use BAS anchors.',
        projection_audit=dict(match_id='118575',scope='clip',mean_disagreement_m=11.6831065,median_m=10.7398110,p95_m=24.9560484,rmse_m=14.1924825,finite_pairs=130782,coverage_percent=99.0772727,policy='GSR first; fallback disagreement unchanged; not surveyed ground-contact accuracy.'),
        missing_data='Only downloaded recordings are evaluated; no multi-match generalization claim.')
    payload['aggregate']['transfers']['by_bas_class']=aggregate_transfer_classes([r['transfers'] for r in evaluated])
    (output/'evaluation_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    write_report(payload,output)
    return payload


def write_report(payload,output):
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>MatchMind evaluation</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap}td,th{padding:10px;border-bottom:1px solid #ddd}</style><h1>MatchMind evaluation</h1>'
    page+='<p>Observed: GSR player locations. Inferred: possession, transfers and trajectory shot candidates. Heuristic: difficulty, pressure, block and formation. BAS support checks release timing and sender, not intent or outcomes. Ball interpolation uses BAS anchors, so these checks are not independent validation.</p><p>The 118575 baseline projection disagreement remains 11.6831065 m; GSR-first coordinates are preserved. Geometry tests and threshold sensitivity establish implementation behavior, not tactical ground-truth accuracy. Multiple recordings of one match do not establish cross-match generalization.</p>'
    page+='<h2>Aggregate</h2><pre>'+html.escape(json.dumps(payload['aggregate'],indent=2))+'</pre><p>'+html.escape(payload['split_policy'])+'</p>'
    rows=payload['recordings']
    page+='<h2>Measured results</h2><table><tr><th>Recording</th><th>Detector</th><th>Candidates</th><th>BAS refs</th><th>TP / FP / FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>Median timing error</th></tr>'
    number=lambda x: 'unavailable' if x is None else f'{x:.3f}'
    for row in rows:
        for kind in ('transfers','shots'):
            r=row[kind]
            page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (row['recording_id'],kind,r['detections'],r['reference_events'],f"{r['matched']} / {r['false_positives']} / {r['false_negatives']}",number(r['precision']),number(r['recall']),number(r['f1']),number(r['timing_error_seconds']['median'])+' s'))+'</tr>'
    page+='</table><h2>Major failure modes</h2><p>Shot candidates miss many BAS shots and include unmatched movements toward goal. Transfer candidates omit some annotated passes and include movements that may be tackles or clearances. Unknown possession occupies substantial recording time. The formation estimator requires a fixed ten-player outfield roster; full-half substitutions can therefore leave formation unknown even when ten players are observed. Tracking noise affects maximum speeds and formation fits; persistent unknown formations are not replaced with guessed labels. No authoritative possession or tactical labels exist. Threshold sensitivity is descriptive; detector accuracy has not been improved or independently established.</p>'
    for row in rows:
        page+=f'<h2>{html.escape(row["recording_id"])}</h2><a href="per_match/{html.escape(row["recording_id"])}/report.html">Review recording</a>'
        for key in ('transfers','shots','goals','possession','tactics'):
            page+='<details><summary>'+key+'</summary><pre>'+html.escape(json.dumps(row[key],indent=2))+'</pre></details>'
    page+='</html>';(output/'evaluation_report.html').write_text(page)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    parser.add_argument('--data-root',type=Path,default=Path('data/soccertrack'))
    parser.add_argument('--skip-sensitivity',action='store_true')
    parser.add_argument('--reuse-sensitivity',action='store_true',help='Reuse prior variants only if source hashes and both configs match')
    args=parser.parse_args()
    run(json.loads(args.manifest.read_text()),args.output,args.data_root,not args.skip_sensitivity,args.reuse_sensitivity)

if __name__=='__main__':main()
