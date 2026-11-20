"""Select proposal-assisted active-learning sequences only in inspected approved footage."""
import argparse,json,math
from pathlib import Path
import cv2
import numpy as np
from src.vision.football22.data import NAMES,write,frame
from src.vision.events.passes import signature
from src.serving.cv import digest
from src.evaluation.cv_m19 import iou


def build(data_root,output):
    data_root=Path(data_root);dataset=json.loads((data_root/'dataset.json').read_text());audit=json.loads((data_root/'source_suitability.json').read_text())
    if audit['base_dataset_sha256']!=digest(data_root/'dataset.json'):raise ValueError('Source suitability audit belongs to other data')
    target=Path('configs/training/cv_m22_tasks.json');root=Path(output)
    if target.exists() or root.exists():raise ValueError('Use a new version; never overwrite review tasks or saved labels')
    root.mkdir(parents=True);tasks=[];ledger=[];sources=dataset['sources'];preview_paths={}
    originals={(r['recording_id'],r['frame']) for r in dataset['rows']}
    from src.vision.advanced21.experiment import frozen_m20
    for name in NAMES:
        if not audit['sources'][name]['active_football_play_seen']:raise ValueError('Visually inspect source before task creation')
        source=sources[name];control,_=frozen_m20(name,'outputs/m20/reproduce_v1')
        directory=Path('outputs/m21/analysis_final_v1')/name/'temporal_offline'
        m=json.loads((directory/'manifest.json').read_text());p=directory/'ball_frames.jsonl'
        if digest(p)!=m['ball_frames_sha256'] or m['video_sha256']!=source['video_sha256']:raise ValueError('Proposal source mismatch')
        alternative=[json.loads(line) for line in p.open()]
        candidates=[]
        for i in range(source['frames']):
            pool=[]
            for origin,row in [('m20',control[i]),('m21',alternative[i])]:
                for c in sorted(row['candidates'],key=lambda c:-c['confidence']):
                    if c['confidence']<.05 or any(iou(c['bbox'],a['bbox'])>.6 for a in pool):continue
                    pool.append({'bbox':c['bbox'],'confidence':c['confidence'],'source_pipeline':origin,
                        'proposal_id':origin+'_'+signature([i,c['bbox'],c['confidence']])[:16]})
            pool=sorted(pool,key=lambda c:-c['confidence'])[:5]
            a,b=control[i].get('image_bbox'),alternative[i].get('image_bbox')
            disagree=1. if (a is None)!=(b is None) else 1-iou(a,b) if a is not None else .5
            uncertain=1-abs((pool[0]['confidence'] if pool else .5)-.5)*2
            candidates.append({'frame':i,'proposals':pool,'priority':float(disagree+.5*uncertain),
                'reason':'detector disagreement and uncertain candidates' if pool else 'missing model candidates during inspected active-play footage'})
        # Ten temporal bins, each one new five-frame sequence: useful context,
        # broad source coverage and no repetition of already-reviewed frames.
        selected=[];used=set();previews=[]
        for group in range(10):
            start=round(3+group*(source['frames']-12)/10);end=round(3+(group+1)*(source['frames']-12)/10)
            centers=sorted(range(start,max(start+1,end)),key=lambda i:(-candidates[i]['priority'],i))
            for center in centers:
                ids=list(range(center,min(center+7,source['frames'])))
                ids=[i for i in ids if (name,i) not in originals and i not in used][:5]
                if len(ids)!=5:continue
                pixels=[frame(source,i) for i in ids]
                metrics=[]
                for image in pixels:
                    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                    green=float(np.mean((hsv[:,:,0]>25)&(hsv[:,:,0]<95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)))
                    metrics.append({'green_fraction_proxy':green,'pixel_std':float(image.std())})
                if any(r['green_fraction_proxy']<.08 or r['pixel_std']<12 for r in metrics):
                    ledger.append({'recording_id':name,'frames':ids,'selected':False,'reason':'automatic blank/non-pitch content gate; not a semantic replay classifier'});continue
                # Every selected sequence gets a source-only sheet, inspected by
                # the curator before publication; image gates alone are insufficient.
                for index,image,quality in zip(ids,pixels,metrics,strict=True):
                    record=candidates[index]
                    task={'task_id':f'm22:{name}:ball:{index}','recording_id':name,'frame':index,'timestamp':index/source['fps'],
                        'sequence_id':f'{name}:{group}','sequence_frames':ids,'proposals':record['proposals'],'selection_reason':record['reason'],
                        'quality_proxies':quality,'reference_role':'development_only','model_proposals_are_truth':False}
                    selected.append(task);used.add(index)
                    tile=cv2.resize(image,(320,180));cv2.putText(tile,f'{name} f{index} {index/source["fps"]:.3f}s',(4,174),0,.4,(255,255,255),1);previews.append(tile)
                ledger.append({'recording_id':name,'frames':ids,'selected':True,'selection_reason':candidates[center]['reason']});break
        if len(selected)<40:raise ValueError('Not enough useful nonduplicate source frames for '+name)
        for page in range(0,len(previews),25):
            chunk=previews[page:page+25]
            while len(chunk)%5:chunk.append(np.zeros_like(chunk[0]))
            path=root/f'{name}_selected_{page//25}.jpg';cv2.imwrite(str(path),np.vstack([np.hstack(chunk[i:i+5]) for i in range(0,len(chunk),5)]))
            preview_paths[str(path)]={'recording_id':name,'model_overlays':False,'sha256':digest(path)}
        tasks.extend(selected)
    document={'schema_version':'cv-m22-development-review-v1','reference_role':'development_only','sources':sources,'tasks':tasks,
        'model_proposals_are_truth':False,'source_suitability':audit,'source_suitability_sha256':digest(data_root/'source_suitability.json'),
        'selection':{'method':'ten temporal bins per clip, five new frames per selected sequence; disagreement/uncertainty priority',
            'original_reviewed_frames_repeated':False,'excluded_withdrawn_m21_set':True,'independent_final_test':False,
            'automatic_quality_gates':'pitch-color and image-variance proxies; not verified action/replay recognition'},
        'preview_sheets':preview_paths,'selection_ledger':ledger,'publication_status':'pending_selected_sequence_visual_inspection'}
    document['fingerprint']=signature(document)
    write(root/'tasks_draft.json',document)
    print('Prepared',len(tasks),'development proposals. Inspect all source-only preview sheets before publication:',root,flush=True)
    return document


def publish(draft, target, curator, inspected_sheets):
    """Record an explicit source-only inspection; never create human ball labels."""
    document=json.loads(Path(draft).read_text());fingerprint=document.pop('fingerprint')
    if signature(document)!=fingerprint:raise ValueError('Draft fingerprint changed')
    if document['publication_status']!='pending_selected_sequence_visual_inspection':raise ValueError('Expected unpublished draft')
    if not curator.strip():raise ValueError('Name the source-inspection curator')
    expected={str(Path(p).resolve()) for p in document['preview_sheets']}
    if {str(Path(p).resolve()) for p in inspected_sheets}!=expected:raise ValueError('Explicitly inspect every selected source-only sheet before publication')
    for path,meta in document['preview_sheets'].items():
        if meta['model_overlays'] or digest(path)!=meta['sha256']:raise ValueError('Source-only preview changed')
    if set(document['sources'])!=set(NAMES):raise ValueError('Review limited to the three approved clips')
    if Path(target).exists():raise ValueError('Never overwrite published review tasks or their saved-label fingerprint')
    document['publication_status']='source_sequences_visually_inspected'
    document['selected_sequence_inspection']={'curator':curator.strip(),
        'all_six_preview_sheets_inspected':len(expected)==6,
        'active_football_players_and_pitch_seen_on_selected_sheets':True,
        'human_ball_labels_created':False,'source_scope':'User-approved test_match/team_goal/southampton only','independent_test':False}
    document['fingerprint']=signature(document);write(target,document)
    return document

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--output');p.add_argument('--publish-draft');p.add_argument('--curator');p.add_argument('--inspected-sheet',action='append',default=[]);p.add_argument('--target',default='configs/training/cv_m22_tasks.json');a=p.parse_args()
    if a.publish_draft:
        if not a.curator:p.error('--curator and explicit --inspected-sheet entries are required after visually checking the source-only sheets')
        publish(a.publish_draft,a.target,a.curator,a.inspected_sheet)
    elif a.data and a.output:build(a.data,a.output)
    else:p.error('Supply --data/--output to prepare, or --publish-draft to publish inspected selections')
