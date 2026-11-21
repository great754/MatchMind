"""Prepare and publish a source-inspected Giroud development review, never labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from src.serving.cv import CVReview,digest
from src.serving.ball import BallReview
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write
from src.evaluation.cv_m19 import iou


def build(analysis,output):
    analysis,output=Path(analysis),Path(output)
    if output.exists():raise ValueError('Review tasks and labels are immutable; choose a fresh output')
    report=json.loads((analysis/'report.json').read_text())
    if report['schema_version']!='cv-m23-unlabelled-source-v1' or report['recording_id']!='giroud':
        raise ValueError('This review is authorized for Giroud only')
    world=CVReview(report['world_directory']);m=world.manifest
    ball=BallReview(analysis/'giroud/full_m23',m)
    records=[json.loads(line) for line in Path(ball.manifest['perception_records_path']).open()]
    if len(records)!=m['frames']:raise ValueError('Source proposal clock mismatch')
    groups=[list(range(round(i*(m['frames']-4)/7),round(i*(m['frames']-4)/7)+4)) for i in range(8)]
    selected={f:('uniform_source_sequence',group) for group,frames in enumerate(groups) for f in frames}
    for f in [29,60,102,179,240,303,386,432]:
        if not 0<=f<m['frames'] or f in selected:raise ValueError('Unexpected Giroud source clock or duplicate task')
        selected[f]=('post_hoc_diagnostic',f)
    if len(selected)!=40:raise ValueError('Expected forty distinct source questions')
    tasks=[];tiles=[];capture=cv2.VideoCapture(str(world.video));output.mkdir(parents=True)
    try:
        for f,(kind,group) in sorted(selected.items()):
            record=records[f];state=ball.frame(f)
            pool=sorted(record['methods']['full_m23']['candidates'],
                key=lambda c:(not(state.directly_observed and tuple(c['bbox'])==state.image_bbox),-c['confidence'],c['candidate_id']))
            proposals=[]
            for c in pool:
                if c['confidence']<.05 or any(iou(c['bbox'],p['bbox'])>.5 for p in proposals):continue
                proposals.append({'proposal_id':'m23_'+signature([f,c['bbox'],c['confidence'],c['origin']])[:20],
                    'bbox':c['bbox'],'confidence':c['confidence'],'source_pipeline':'m23_'+c['origin']})
                if len(proposals)==2:break
            tasks.append({'task_id':f'm23:giroud:ball:{f}','recording_id':'giroud','frame':f,'timestamp':f/m['fps'],
                'sequence_id':f'giroud:{kind}:{group}',
                'sequence_frames':groups[group] if kind=='uniform_source_sequence' else [max(0,f-1),f,min(m['frames']-1,f+1)],
                'proposals':proposals,'selection_reason':('Source-spread four-frame sequence' if kind=='uniform_source_sequence' else 'Post-hoc diagnostic example from the existing model run'),
                'reference_role':'development_only','model_proposals_are_truth':False,'selection_kind':kind})
            capture.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=capture.read()
            if not ok:raise ValueError('Selected source frame cannot be decoded')
            tile=cv2.copyMakeBorder(cv2.resize(image,(320,180)),0,24,0,0,cv2.BORDER_CONSTANT,value=(15,20,25))
            cv2.putText(tile,f'giroud frame {f} / {f/m["fps"]:.2f}s',(5,197),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            tiles.append(tile)
    finally:capture.release()
    previews={}
    for page in range(2):
        chunk=tiles[page*20:(page+1)*20];path=output/f'giroud_selected_{page}.jpg'
        if not cv2.imwrite(str(path),np.vstack([np.hstack(chunk[i:i+4]) for i in range(0,20,4)])):raise ValueError('Preview encoding failed')
        previews[str(path)]={'recording_id':'giroud','model_overlays':False,'sha256':digest(path)}
    source={k:m[k] for k in ['recording_id','video_sha256','frames','fps','width','height']}
    document={'schema_version':'cv-m23-source-review-v1','reference_role':'development_only','model_proposals_are_truth':False,
        'sources':{'giroud':source},'tasks':tasks,'preview_sheets':previews,
        'source_suitability':{'source_video_sha256':m['video_sha256'],'authorization':'User explicitly requested the Giroud model run and then preparation of this ball review.',
            'original_match_overlap_audited':False,'human_ball_labels_created':False},
        'selection':{'method':'32 source-spread consecutive frames plus eight post-hoc diagnostic frames',
            'uniform_frames':32,'diagnostic_frames':8,'maximum_proposals_per_frame':2,'independent_final_test':False,
            'proposals':'Actual pinned M23 candidates; selected candidate first when present; confidence unchanged; IoU deduplication',
            'analysis_report_sha256':digest(analysis/'report.json'),'ball_frames_sha256':ball.manifest['ball_frames_sha256'],
            'perception_records_sha256':ball.manifest['perception_records_sha256']},
        'publication_status':'pending_selected_sequence_visual_inspection'}
    document['fingerprint']=signature(document);write(output/'tasks_draft.json',document)
    print(json.dumps({'tasks':len(tasks),'preview_sheets':list(previews),'human_labels_created':0}),flush=True)
    return document


def publish(draft,target,curator,sheets):
    draft,target=Path(draft),Path(target)
    if target.exists():raise ValueError('Never overwrite published tasks or existing review labels')
    data=json.loads(draft.read_text());fingerprint=data.pop('fingerprint')
    if signature(data)!=fingerprint or data['publication_status']!='pending_selected_sequence_visual_inspection':raise ValueError('Review draft changed')
    if not curator.strip() or {str(Path(p).resolve()) for p in sheets}!={str(Path(p).resolve()) for p in data['preview_sheets']}:
        raise ValueError('Inspect both source-only sheets explicitly before publishing')
    for p,meta in data['preview_sheets'].items():
        if meta['model_overlays'] or digest(p)!=meta['sha256']:raise ValueError('Source preview changed')
    data['publication_status']='source_sequences_visually_inspected'
    data['selected_sequence_inspection']={'curator':curator,'active_football_players_and_pitch_seen':True,
        'human_ball_labels_created':False,'independent_test':False,'inspected_sheets':sheets}
    data['fingerprint']=signature(data);write(target,data)
    from src.vision.football22.review import DevelopmentReviewStore
    DevelopmentReviewStore(target,draft.parent/'human_review',approved_sources={'giroud'})
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=Path('outputs/m23/giroud_v1/analysis'))
    parser.add_argument('--output',type=Path,default=Path('outputs/m23/giroud_v1/review_v1'))
    parser.add_argument('--publish-draft',type=Path);parser.add_argument('--curator')
    parser.add_argument('--inspected-sheet',action='append',default=[])
    parser.add_argument('--target',type=Path,default=Path('configs/training/cv_giroud_review.json'))
    args=parser.parse_args()
    if args.publish_draft:
        if not args.curator:parser.error('Supply the source-inspection curator')
        publish(args.publish_draft,args.target,args.curator,args.inspected_sheet)
    else:build(args.analysis,args.output)
