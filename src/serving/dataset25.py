"""Optional M25 adapter for the existing human-review component; isolated revisions."""
import json,math,threading,os
from pathlib import Path
from datetime import datetime,timezone
from typing import Literal
from pydantic import BaseModel,Field
from fastapi import APIRouter,Depends,HTTPException
from fastapi.responses import FileResponse
from src.serving.cv import digest
from src.vision.football22.review import DevelopmentLabel
from src.vision.intelligent23.inputs import write


class ShotReview(BaseModel):
    shot_type:Literal['broadcast_gameplay','close_up','replay','celebration','crowd_or_non_gameplay','transition','uncertain']
    reviewer_id:str=Field(min_length=2,max_length=80)
    reviewed_from_source:Literal[True]


def create_router(authorize,root='outputs/m25/dataset_v1'):
    root=Path(root);lock=threading.RLock();router=APIRouter(prefix='/cv/m25',dependencies=[Depends(authorize)])
    def batch():
        try:
            value=json.loads((root/'pilot_20.json').read_text())
            if value['schema_version']!='cv-m25-pilot-v1' or len(value['frames'])>20:raise ValueError()
            shot_file=root/'shot_revisions.json'
            if shot_file.exists():
                latest={r['frame_id']:r for r in json.loads(shot_file.read_text())}
                for row in value['frames']:
                    if row['frame_id'] in latest:row['shot_type']=latest[row['frame_id']]['shot_type']
            return value
        except (OSError,ValueError,KeyError):raise HTTPException(503,'M25 pilot is not prepared or valid') from None
    def durable(path,value):
        temp=path.with_suffix('.tmp')
        with temp.open('w') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        temp.chmod(0o600);os.replace(temp,path)
    def labels():
        p=root/'human_revisions.json';return json.loads(p.read_text()) if p.exists() else []
    def find(fid):
        value=batch();row=next((r for r in value['frames'] if r['frame_id']==fid),None)
        if row is None:raise HTTPException(404,'Unknown dataset frame')
        image=Path(row['image'])
        if not image.resolve().is_relative_to(root.resolve()) or digest(image)!=row['image_sha256']:raise HTTPException(503,'Dataset source image changed')
        return value,row
    @router.get('/batch')
    def public():
        value=batch();value['revisions']=labels();value['dataset_fingerprint']=digest(root/'pilot_20.json');return value
    @router.get('/frames/{fid}/frame.jpg')
    def image(fid:str):
        _,row=find(fid);return FileResponse(row['image'],media_type='image/png')
    @router.post('/frames/{fid}/shot')
    def shot(fid:str,body:ShotReview):
        _,row=find(fid)
        with lock:
            path=root/'shot_revisions.json';history=json.loads(path.read_text()) if path.exists() else []
            history.append({'frame_id':fid,'image_sha256':row['image_sha256'],'saved_at_utc':datetime.now(timezone.utc).isoformat(),'revision':len(history)+1,**body.model_dump()});durable(path,history)
        return {'saved':True,'shot_type':body.shot_type}

    @router.post('/frames/{fid}/labels')
    def save(fid:str,body:DevelopmentLabel):
        value,row=find(fid)
        if row.get('shot_type','broadcast_gameplay')!='broadcast_gameplay':raise HTTPException(422,'Only eligible broadcast gameplay frames may receive dataset labels')
        if body.task_id!=fid:raise HTTPException(422,'Frame identity mismatch')
        if body.bbox:
            x,y,w,h=body.bbox
            if not all(math.isfinite(v) for v in body.bbox) or x+w>row['width'] or y+h>row['height']:raise HTTPException(422,'Box outside source image')
        proposals={p['proposal_id']:p for p in row['proposals']}
        if set(body.reviewed_proposals)!=set(proposals) or len(body.reviewed_proposals)!=len(proposals):raise HTTPException(422,'Review every proposal first')
        if not set(body.rejected_proposals)<=set(proposals) or body.accepted_proposal in body.rejected_proposals:raise HTTPException(422,'Invalid proposal rejection')
        if body.accepted_proposal and (body.accepted_proposal not in proposals or list(body.bbox or [])!=proposals[body.accepted_proposal]['bbox']):raise HTTPException(422,'Accepted proposal does not match source box')
        with lock:
            revisions=labels();revision={'revision':len(revisions)+1,'saved_at_utc':datetime.now(timezone.utc).isoformat(),'image_sha256':row['image_sha256'],'source_frame_id':fid,'original_reference':row['existing_reference'],'batch_sha256':digest(root/'pilot_20.json'),**body.model_dump(mode='json')};revisions.append(revision)
            durable(root/'human_revisions.json',revisions)
        return {'saved':True,'task_id':fid,'revision':revision['revision']}
    return router
