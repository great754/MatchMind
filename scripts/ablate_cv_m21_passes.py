"""Isolate second-stage verification on unchanged M20 ball and participant evidence."""
import argparse
import json
from pathlib import Path
from src.evaluation.cv_m19 import evaluate
from src.serving.cv import digest
from src.vision.advanced21.experiment import NAMES,fresh_directory,write_json,source_worlds,frozen_m20
from src.vision.advanced21.passes import verify_events
from src.vision.events.passes_m19 import detect_m19


def run(output,m20_reference,audit_root):
    root=fresh_directory(output);audit_root=Path(audit_root)
    dataset=json.loads((audit_root/'development_tasks.json').read_text());labels=json.loads((audit_root/'development_labels.json').read_text())['annotations']
    protocol={**dataset['protocol'],'sources':dataset['sources']};balls={};base={};verified={};clips={}
    for name in NAMES:
        metadata,worlds,_=source_worlds(name);balls[name],_=frozen_m20(name,m20_reference)
        base[name],_,evidence=detect_m19(name,worlds,balls[name],metadata['fps'])
        verified[name],diagnostics=verify_events(base[name],evidence,metadata['fps'])
        write_json(root/(name+'_second_stage.json'),diagnostics)
        clips[name]={'before_estimates':sum(e['status']=='estimated' for e in base[name]),
                     'after_estimates':sum(e['status']=='estimated' for e in verified[name]),
                     'base_candidates_never_promoted':all(e['status']=='candidate' for e,b in zip(verified[name],base[name],strict=True) if b['status']=='candidate'),
                     'diagnostics_sha256':digest(root/(name+'_second_stage.json'))}
    tasks=[t for t in dataset['tasks'] if t['kind']=='passes'];ids={t['task_id'] for t in tasks};refs=[r for r in labels if r['task_id'] in ids]
    report={'reference_role':'development_only','annotations_sha256':digest(audit_root/'development_labels.json'),
        'ball_and_participant_evidence_changed':False,'uses_future':True,'learned_event_model_trained':False,
        'reason_no_event_model_training':'Only seven completed human-reference passes in six usable windows.',
        'base_gates':evaluate(tasks,refs,balls,base,protocol)['passes'],
        'base_gates_plus_second_stage':evaluate(tasks,refs,balls,verified,protocol)['passes'],'clips':clips}
    write_json(root/'report.json',report);print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--m20-reference',required=True);parser.add_argument('--audit',required=True)
    args=parser.parse_args();run(args.output,args.m20_reference,args.audit)
