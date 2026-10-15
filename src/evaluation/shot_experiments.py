"""Frozen one-factor development experiments; never selects thresholds by held-out F1."""
import argparse
from dataclasses import asdict,replace
import hashlib
import html
import json
from pathlib import Path
from pydantic import Field
from src.query.schema import Closed
from src.analytics.shots import detect_shots
from src.analytics.shot_evidence import ShotOptions
from .artifacts import load_evaluation_recording
from .event_evaluation import evaluate_events
from .aggregate import aggregate_events

class Experiment(Closed):
    name:str=Field(pattern=r'^[a-z0-9_]+$')
    description:str
    analytics_overrides:dict[str,float|int|bool]=Field(default_factory=dict)
    shot_options:dict[str,float|int|bool|str]=Field(default_factory=dict)


def load_experiment(path):
    spec=Experiment.model_validate_json(Path(path).read_text())
    allowed={'shot_min_speed_mps','shot_max_goal_distance_m','shot_goal_margin_m','shot_max_time_to_goal_seconds','shot_persistence_frames','shot_window_seconds'}
    if set(spec.analytics_overrides)-allowed:raise ValueError('Experiments may only override shot-specific fields')
    ShotOptions(**spec.shot_options)
    return spec


def run(evaluation,configs,output):
    evaluation=Path(evaluation);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((evaluation/'evaluation_summary.json').read_text())
    specs=[load_experiment(p) for p in sorted(Path(configs).glob('*.json'))]
    if len({s.name for s in specs})!=len(specs):raise ValueError('Duplicate experiment names')
    all_results={s.name:dict(name=s.name,description=s.description,per_recording=[]) for s in specs}
    for item in metadata['recordings']:
        print('Shot experiments:',item['recording_id'],flush=True)
        recording,result,baseline=load_evaluation_recording(evaluation/'per_match'/item['recording_id'])
        # Experiments are diagnostic development runs on 118575, never held-out optimization.
        if str(recording.sync['match_id'])!='118575' and item['split']!='development':
            raise ValueError('Controlled experiments may not run on a held-out evaluation match; use development split')
        refs=[e for e in recording.events if e['label']=='SHOT']
        for spec in specs:
            config=replace(baseline,**spec.analytics_overrides);options=ShotOptions(**spec.shot_options)
            predicted=detect_shots(result['possession'],result['ball'],config,options)
            if spec.name=='baseline':
                if predicted.start_frame.tolist()!=result['shots'].start_frame.tolist():raise AssertionError('Original baseline shot start frames did not reproduce')
            measured=evaluate_events(predicted,refs,config.fps,config.validation_tolerance_seconds)
            all_results[spec.name]['per_recording'].append(dict(recording_id=item['recording_id'],split='development',
                metrics=measured,analytics_config=asdict(config),shot_options=asdict(options),
                candidate_start_frames=predicted.start_frame.tolist(),source_files=item['source_files']))
    for experiment in all_results.values():experiment['aggregate']=aggregate_events([r['metrics'] for r in experiment['per_recording']])
    baseline=all_results.get('baseline',{}).get('aggregate')
    payload=dict(schema_version='1.0',experiments=list(all_results.values()),production_selection='baseline',
        selection_reason='No production change: one development match and dependent BAS-anchored ball data cannot justify generalization. Recent-possession gating is geometrically defensible but discards shots with missing context.',
        baseline=baseline,config_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(configs).glob('*.json'))},
        limitations=['Development-only component ablations; not held-out validation or an F1 optimization sweep.','BAS labels enter evaluation only. Predictions receive possession, ball motion and configuration.','Ball positions already depend on BAS anchors; acceleration and 2D trajectory extrapolation are uncertain.'])
    (output/'shot_experiments.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    page='<html lang="en"><meta charset="utf-8"><title>Shot experiments</title><h1>Shot component experiments</h1><p>'+html.escape(payload['selection_reason'])+'</p><table><tr><th>Experiment</th><th>TP / FP / FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>Change</th></tr>'
    for e in payload['experiments']:
        m=e['aggregate'];page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (e['name'],f"{m['matched']} / {m['false_positives']} / {m['false_negatives']}",m['precision'],m['recall'],m['f1'],e['description']))+'</tr>'
    (output/'shot_experiments.html').write_text(page+'</table></html>')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=Path('outputs/evaluation'))
    parser.add_argument('--configs',type=Path,default=Path('configs/shots'))
    parser.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    args=parser.parse_args();run(args.evaluation,args.configs,args.output)

if __name__=='__main__':main()
