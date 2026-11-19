"""Aggregate development failure categories without editing labels or reference rules."""
import argparse
from collections import Counter
import json
from pathlib import Path
from src.evaluation.cv_m19 import iou
from src.serving.cv import digest
from src.vision.advanced21.experiment import NAMES,fresh_directory,write_json


def run(source,analysis,audit_root,output):
    source,analysis,audit_root=Path(source),Path(analysis),Path(audit_root);root=fresh_directory(output)
    dataset=json.loads((audit_root/'development_tasks.json').read_text());labels={r['task_id']:r for r in json.loads((audit_root/'development_labels.json').read_text())['annotations']}
    report={'reference_role':'development_only','annotations_sha256':digest(audit_root/'development_labels.json'),
        'box_iou_threshold_unchanged':.25,'center_in_box_is_diagnostic_not_accuracy':True,'clips':{}}
    for name in NAMES:
        records=[json.loads(line) for line in (source/name/'inputs.jsonl').open()]
        results={}
        for variant,group in [('m20_control','m20_candidates'),('offline_dag','m20_candidates'),('temporal_offline','multi_frame'),('imm_multiframe','multi_frame')]:
            rows=[json.loads(line) for line in (analysis/name/variant/'ball_frames.jsonl').open()];counts=Counter()
            for task in dataset['tasks']:
                if task['recording_id']!=name or task['kind']!='ball':continue
                ref=labels[task['task_id']];row=rows[task['frame']]
                if ref['state']=='uncertain':counts['unknown_excluded']+=1;continue
                if ref['state']=='not_visible':
                    counts['detection_on_known_not_visible_frame' if row['directly_observed'] else 'known_not_visible_no_detection']+=1;continue
                candidates=records[task['frame']]['candidate_sets'][group]
                if row['directly_observed'] and iou(row['image_bbox'],ref['bbox'])>=.25:counts['correct_selected_box']+=1;continue
                if row['directly_observed']:counts['selected_box_iou_miss']+=1
                else:counts['no_direct_selected_observation']+=1
                if any(c['confidence']>=.35 and iou(c['bbox'],ref['bbox'])>=.25 for c in candidates):counts['above_gate_reference_candidate_available_but_not_selected']+=1
                elif any(iou(c['bbox'],ref['bbox'])>=.25 for c in candidates):counts['reference_candidate_only_below_visual_gate']+=1
                else:counts['no_neural_reference_box_match']+=1
                x,y,w,h=ref['bbox']
                def inside(box):return x<=box[0]+box[2]/2<=x+w and y<=box[1]+box[3]/2<=y+h
                if row['directly_observed'] and inside(row['image_bbox']):counts['selected_center_inside_reference_but_box_iou_miss']+=1
            results[variant]=dict(counts)
        report['clips'][name]=results
    write_json(root/'report.json',report);print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--analysis',required=True);parser.add_argument('--audit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source,args.analysis,args.audit,args.output)
