"""Source-separated temporal learning; uncertainty never becomes negative supervision."""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.serving.cv import digest
from .data import NAMES,write,frame,crop_sample
from .model import preprocess,targets,context_indices


def prepare_temporal(data_root,output,weak_local=False):
    base=Path(data_root)/'dataset.json';dataset=json.loads(base.read_text());root=Path(output)
    if root.exists():raise ValueError('Use a fresh temporal-data directory')
    root.mkdir(parents=True)
    epochs={};items=[]
    for name,source in dataset['sources'].items():
        world=Path('outputs/world_model')/(name+'_m9_ready');meta=json.loads((world/'manifest.json').read_text())
        if digest(world/'frames.jsonl')!=meta['frames_sha256']:raise ValueError('World/camera-cut snapshot changed')
        cuts=[];previous=None;epoch=0
        for line in (world/'frames.jsonl').open():
            camera=json.loads(line)['camera']
            if previous is not None and (camera['epoch']!=previous or camera.get('continuity')=='cut'):epoch+=1
            cuts.append(epoch);previous=camera['epoch']
        if len(cuts)!=source['frames']:raise ValueError('Source/cut clock mismatch')
        epochs[name]=cuts
        for row in dataset['rows']:
            if row['recording_id']!=name:continue
            ids=context_indices(row['frame'],source['fps'],cuts);paths=[]
            for i in ids:
                p=root/'context'/name/(str(i)+'.png')
                if not p.exists():
                    p.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(p),frame(source,i)):raise ValueError('Temporal source write failed')
                paths.append({'frame':i,'path':str(p),'sha256':digest(p)})
            items.append({**row,'context':paths,'weight':1.,'ball_loss_known':True})
    weak_info=None
    if weak_local:
        weak_path=Path('outputs/ml/ball_m15/dataset.json');external=json.loads(weak_path.read_text())
        train=[r for r in external['samples'] if r['split']=='train'];lookup={(r['recording_id'],r['frame']):r for r in train}
        candidates=[r for r in train if len(r['balls'])==1 and all((r['recording_id'],r['frame']-d) in lookup for d in [1,2])]
        chosen=[candidates[i] for i in np.unique(np.linspace(0,len(candidates)-1,min(120,len(candidates)),dtype=int))]
        for r in chosen:
            context=[]
            for index in [r['frame']-2,r['frame']-1,r['frame']]:
                old=lookup[(r['recording_id'],index)]
                if digest(old['path'])!=old['source_sha256']:raise ValueError('Existing weak source image changed')
                context.append({'frame':index,'path':old['path'],'sha256':old['source_sha256']})
            items.append({'recording_id':r['recording_id'],'match_id':r['match_id'],'frame':r['frame'],
                'timestamp':r['timestamp'],'state':'weak_positive','bbox':r['balls'][0]['bbox'],'context':context,
                'weight':.15,'weak_supervision':True,'origin':'organizer_box_unverified_visibility_and_in_play_identity',
                'annotation_sha256':r['annotation_sha256'],'source_provenance':r['provenance'],
                'reference_role':'weak_training_only','ball_loss_known':False})
        weak_info={'source_dataset_sha256':digest(weak_path),'frames':len(chosen),'source_matches':sorted({r['match_id'] for r in chosen}),
            'weight':.15,'negative_loss_outside_explicit_labels':False,'human_confirmed':False,
            'complete_original_match_overlap_audit_with_approved_clips':'not_established',
            'limits':'Existing organizer object boxes are weak ball-object supervision, not certified visibility or in-play identity; no historical validation/test images used.'}
    output_manifest={'schema_version':'cv-m22-temporal-data-v1','base_dataset_sha256':digest(base),
        'original_annotations_sha256':dataset['original_annotations_sha256'],'sources':dataset['sources'],
        'source_cut_epochs':epochs,'rows':items,'reference_role':'development_only','weak_or_pseudo_labels_used':weak_local,'weak_training':weak_info,
        'withdrawn_m21_set_used':False,'independent_final_test_available':False}
    write(root/'dataset.json',output_manifest)
    return output_manifest


class TemporalSamples(Dataset):
    def __init__(self, rows, training, repeats=6, seed=20261119):
        self.rows=rows;self.training=training;self.repeats=repeats if training else 1;self.seed=seed;self.epoch=0
        self.indices=[i for repeat in range(self.repeats) for i,r in enumerate(rows) if not r.get('weak_supervision') or repeat==0]
        for row in rows:
            for image in row['context']:
                if digest(image['path'])!=image['sha256']:raise ValueError('Temporal source pixels changed')

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        row=self.rows[self.indices[index]];rng=np.random.default_rng(self.seed+self.epoch*100003+index)
        images=[cv2.imread(c['path']) for c in row['context']]
        if any(i is None for i in images):raise ValueError('Source sequence unavailable')
        bbox=row['bbox'] if row['state'] in ['visible','weak_positive'] else None
        negatives=[list(r) for r in row.get('negative_regions',[])]
        region=None;output_hw=(384,640)
        if self.training:
            # Broadcast organizer balls are much larger in their 1080p source.
            # Zoom OUT those weak examples so their network-pixel sizes cover
            # the small footballs in our 360p/720p development footage.
            sizes=[384,640,960,1080] if row.get('weak_supervision') else [192,256,320,384]
            _,_,region=crop_sample(images[-1],bbox or (negatives[0] if negatives else None),sizes[index%4],rng)
            output_hw=(384,384)
            if rng.random()<.5:
                images=[np.ascontiguousarray(i[:,::-1]) for i in images]
                image_width=images[-1].shape[1];region[0]=image_width-region[0]-region[2]
                if bbox is not None:bbox=[image_width-bbox[0]-bbox[2],bbox[1],bbox[2],bbox[3]]
                negatives=[[image_width-r[0]-r[2],r[1],r[2],r[3]] for r in negatives]
            exposure=rng.uniform(.8,1.2);quality=int(rng.integers(50,96));blur=rng.random()<.25
            augmented=[]
            for image in images:
                im=np.clip(image.astype(np.float32)*exposure,0,255).astype(np.uint8)
                if blur:
                    kernel=np.zeros((3,3),np.float32);kernel[1]=1/3;im=cv2.filter2D(im,-1,kernel)
                ok,encoded=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,quality])
                if not ok:raise ValueError('Augmentation encoding failed')
                augmented.append(cv2.imdecode(encoded,cv2.IMREAD_COLOR))
            images=augmented
            # Train occasional history absence so a same-model current-only check
            # is meaningful; missing history is never invented visual motion.
            if rng.random()<.2:images=[images[-1]]*3
        x,mapping=preprocess(images,output_hw,region)
        if row.get('weak_supervision'):
            target=targets('visible',bbox,mapping,row['weight'])
            target['valid']=target['mask'].clone()  # positive-only; unknown pixels are NOT negative labels
            return x,target
        return x,targets(row['state'],bbox,mapping,row.get('weight',1.),negatives)


def split_sources(rows,excluded):
    if excluded not in NAMES:raise ValueError('Unknown target recording')
    remaining=[name for name in NAMES if name!=excluded]
    train=[row for row in rows if row['recording_id']==remaining[0] or row.get('weak_supervision')]
    val=[row for row in rows if row['recording_id']==remaining[1]]
    if not train or not val:raise ValueError('Source-separated training and validation required')
    return train,val,{'train_sources':[remaining[0]],'validation_sources':[remaining[1]],'excluded_development_source':excluded,
        'validation_is_independent_final_test':False,'target_labels_used_for_fitting_or_early_stopping':False}


def matched_validation(model,rows,config,device):
    """Validation uses the actual whole-frame + tiled + current-only detector.

    It never crops at a ground-truth location. Human boxes are used only AFTER
    source-independent discovery and neural selection, for the fixed IoU score.
    """
    from .inference import FootballDetector
    from src.evaluation.cv_m19 import iou,counts
    detector=object.__new__(FootballDetector);detector.model=model;detector.device=device;detector.config=config;detector.batch_size=4
    detector.checkpoint_sha256='in_memory_validation_not_a_frozen_checkpoint'
    tp=fp=fn=0
    for row in rows:
        if row.get('weak_supervision') or row['state']=='uncertain':raise ValueError('Validation requires strong, known human references')
        images=[cv2.imread(c['path']) for c in row['context']]
        if any(image is None for image in images):raise ValueError('Validation source unavailable')
        result=detector.infer(images,row['frame']);eligible=[c for c in result['candidates'] if c['observation_eligible']]
        selected=eligible[0] if eligible else None
        if row['state']=='visible':
            match=selected is not None and iou(selected['bbox'],row['bbox'])>=.25
            tp+=bool(match);fn+=not match;fp+=selected is not None and not match
        else:fp+=selected is not None
    return {**counts(tp,fp,fn),'reviewed_frames':len(rows),'mode':'actual_full_frame_tiled_current_verified_detector_before_association',
        'source_selection_uses_ground_truth':False,'reference_role':'development_validation_only'}
