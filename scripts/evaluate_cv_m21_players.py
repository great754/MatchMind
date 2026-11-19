"""Compare M9 with IoU, camera/jersey and multi-observation deep-appearance tracking."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from src.serving.cv import digest
from src.evaluation.cv_m19 import evaluate
from src.vision.advanced21.experiment import NAMES, fresh_directory, source_worlds, write_json
from src.vision.advanced21.players import PlayerTracker
from src.vision.events.passes import visible_players


def run(output,audit_root):
    root=fresh_directory(output);audit_root=Path(audit_root)
    dataset=json.loads((audit_root/'development_tasks.json').read_text());labels=json.loads((audit_root/'development_labels.json').read_text())['annotations']
    config=json.loads(Path('configs/vision/cv_m21.json').read_text())['players']
    report={'reference_role':'development_only','annotations_sha256':digest(audit_root/'development_labels.json'),
        'identity_negative_pairs':sum(r['state']=='different' for r in labels), 'manual_identity_changes':0,
        'incorrect_associations':None,'source':'M9 current observed boxes and existing source-verified OSNet body embeddings; no new training or extraction.', 'clips':{}}
    for name in NAMES:
        metadata,worlds,people=source_worlds(name);cache=Path('outputs/world_model/m9_cache')/name
        m=json.loads((cache/'manifest.json').read_text())
        if m['key']['source_sha256']!=metadata['video_sha256'] or m['key']['baseline_sha256']!=metadata['source_m8_frames_sha256'] or digest(cache/'features.json')!=m['features_sha256'] or digest(cache/'embeddings.npz')!=m['embeddings_sha256']:
            raise ValueError('M9 appearance source/hash mismatch')
        features=json.loads((cache/'features.json').read_text())
        with np.load(cache/'embeddings.npz',allow_pickle=False) as archive:embeddings={k:archive[k] for k in archive.files}
        for i,rows in enumerate(people):
            lookup={p['track_id']:p for p in features[i]}
            for d in rows:
                original=lookup.get(d['track_id']);key=original.get('embedding_key') if original else None
                d['embedding']=embeddings[key].tolist() if key in embeddings else None
        tasks=[t for t in dataset['tasks'] if t['recording_id']==name and t['kind']=='identity'];ids={t['task_id'] for t in tasks};refs=[r for r in labels if r['task_id'] in ids]
        protocol={**dataset['protocol'],'sources':dataset['sources']}
        variants={};source_players={i:list(visible_players(w).values()) for i,w in enumerate(worlds)}
        variants['m9']=evaluate(tasks,refs,{}, {},protocol,{name:source_players})['identity']
        for variant,compensated,appearance,deep in [('iou_two_stage',False,False,False),('camera_jersey',True,True,False),('camera_deep_multiview',True,True,True)]:
            start=time.perf_counter();tracker=PlayerTracker(metadata['fps'],config,compensated,appearance,deep);rows=[]
            with (root/(name+'_'+variant+'.jsonl')).open('w') as stream:
                for i,observations in enumerate(people):
                    current=tracker.step(i,worlds[i]['camera'],observations);rows.append(current)
                    stream.write(json.dumps({'frame':i,'players':current},separators=(',',':'))+'\n')
            result=evaluate(tasks,refs,{}, {},protocol,{name:dict(enumerate(rows))})['identity']
            result.update(processing_seconds=time.perf_counter()-start,output_sha256=digest(root/(name+'_'+variant+'.jsonl')),
                multi_observation_tracks=max((p['embedding_observations'] for row in rows for p in row),default=0),
                false_merge_accuracy=None,manual_identity_propagated=False)
            variants[variant]=result
        report['clips'][name]={'appearance_cache':m,'variants':variants}
        write_json(root/'report.json',report)
        print('M21 PLAYER COMPARISON',name,{k:(v['correct_pairs'],v['reviewed_matched_pairs']) for k,v in variants.items()},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--audit',required=True)
    args=parser.parse_args();run(args.output,args.audit)
