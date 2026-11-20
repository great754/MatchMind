"""Source-aligned development data. Model proposals never become training truth."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from src.serving.cv import digest
from src.serving.cv_review_labels import HumanReviewStore
from src.vision.events.passes import signature

NAMES = ('test_match', 'team_goal', 'southampton')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sources():
    result = {}
    for name in NAMES:
        path = Path('outputs/world_model')/(name+'_m9_ready')/'manifest.json'
        meta = json.loads(path.read_text())
        if digest(meta['source_video']) != meta['video_sha256']:
            raise ValueError('Approved source hash changed: '+name)
        result[name] = {k: meta[k] for k in ['source_video','video_sha256','frames','fps','width','height']}
    return result


def frame(source, index):
    if not 0 <= index < source['frames']: raise ValueError('Frame outside approved source')
    capture = cv2.VideoCapture(source['source_video'])
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, pixels = capture.read()
    finally: capture.release()
    if not ok or pixels.shape[:2] != (source['height'],source['width']):
        raise ValueError('Source decode/geometry mismatch')
    return pixels


def original_references():
    store = HumanReviewStore()
    labels = store.labels()
    for label in labels:
        from src.serving.cv_review_labels import HumanAnnotation
        item = HumanAnnotation.model_validate({k:v for k,v in label.items() if k!='source_video_sha256'})
        task = store.tasks[item.task_id]
        if json.loads(json.dumps(store.validate(task['recording_id'],item))) != label:
            raise ValueError('Human reference changed')
    return store.dataset, labels, digest(store.root/'labels.json')


def build(output):
    root = Path(output)
    if root.exists(): raise ValueError('Use a fresh data directory; original labels are immutable')
    source = sources(); dataset, labels, label_sha = original_references()
    root.mkdir(parents=True)
    write(root/'source_inventory.json',source)
    tasks = {t['task_id']:t for t in dataset['tasks']}
    rows = []
    for label in labels:
        task = tasks[label['task_id']]
        if task['kind'] != 'ball' or label['state'] == 'uncertain': continue
        name = task['recording_id']; index = task['frame']; image = frame(source[name],index)
        path = root/'verified_frames'/name/(str(index)+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),image): raise ValueError('Frame write failed')
        rows.append({'recording_id':name,'frame':index,'timestamp':index/source[name]['fps'],
            'source_video_sha256':source[name]['video_sha256'],'path':str(path),'image_sha256':digest(path),
            'state':label['state'],'bbox':label['bbox'],'origin':'existing_human_source_only',
            'model_assisted':False,'reference_role':'development_only'})
    sheets = {}
    for name, meta in source.items():
        tiles=[];indices=sorted(set(np.linspace(0,meta['frames']-1,12,dtype=int).tolist()))
        for i in indices:
            image=frame(meta,i); h,w=image.shape[:2]
            tile=np.zeros((210,320,3),np.uint8);scale=min(320/w,180/h)
            resized=cv2.resize(image,(round(w*scale),round(h*scale)))
            tile[:resized.shape[0],:resized.shape[1]]=resized
            cv2.putText(tile,f'{name} frame {i} | {i/meta["fps"]:.3f}s',(5,201),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
            tiles.append(tile)
        sheet=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,12,3)])
        path=root/(name+'_source_sheet.jpg');cv2.imwrite(str(path),sheet)
        sheets[name]={'path':str(path),'frames':indices,'model_overlays':False,'suitability_status':'pending_source_only_visual_inspection'}
    external=json.loads(Path('outputs/ml/ball_m16/dataset.json').read_text())
    audit={'schema_version':'cv-m22-data-v1','sources':source,'original_annotations_sha256':label_sha,
        'task_fingerprint':dataset['fingerprint'],'reference_role':'development_only','rows':rows,'source_contact_sheets':sheets,
        'external_inventory':external['summary'],
        'external_training_policy':'Existing organizer boxes are not visibility/in-play identity truth; not used as verified supervision in this human-label pilot.',
        'soccertrack_policy':'Pitch/GSR coordinates, reconstructed ball NPZ and BAS times are not verified visible source-image ball labels.',
        'withdrawn_m21_set_used':False,'full_match_used':False,'independent_test_available':False}
    audit['fingerprint']=signature(audit);write(root/'dataset.json',audit)
    return audit


def crop_sample(image, bbox, size, rng):
    """A labelled in-play ball is known visible; only transform its actual source box."""
    h,w=image.shape[:2];size=min(size,w,h)
    if bbox is None:
        x=int(rng.integers(0,w-size+1));y=int(rng.integers(0,h-size+1));label=None
    else:
        a,b,bw,bh=bbox
        if bw>size or bh>size: raise ValueError('Crop would truncate annotated ball')
        x=int(np.clip(a+bw/2-rng.uniform(.25,.75)*size,0,w-size))
        y=int(np.clip(b+bh/2-rng.uniform(.25,.75)*size,0,h-size))
        if not (x<=a and a+bw<=x+size): raise ValueError('Ball box lost from crop')
        if not (y<=b and b+bh<=y+size): raise ValueError('Ball box lost from crop')
        label=[(a+bw/2-x)/size,(b+bh/2-y)/size,bw/size,bh/size]
    return image[y:y+size,x:x+size].copy(),label,[x,y,size,size]
