"""Model-blind human annotation workflow, isolated from predictions and manual player identities."""
import json,os,threading,math
from contextlib import nullcontext
from pathlib import Path
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException,Response
from pydantic import Field
from src.query.schema import Closed
from src.vision.events.passes import signature

class ReferencePass(Closed):
    release_frame:int=Field(ge=0)
    receipt_frame:int=Field(ge=0)
    outcome:Literal['completed','uncertain']='completed'
    sender_description:str=Field(default='',max_length=100)
    receiver_description:str=Field(default='',max_length=100)

class HumanAnnotation(Closed):
    task_id:str=Field(max_length=100)
    reviewer_id:str=Field(min_length=2,max_length=80)
    origin:Literal['human']='human'
    reviewed_from_source:Literal[True]
    model_assisted:Literal[False]
    state:Literal['visible','not_visible','uncertain','same','different','reviewed']
    bbox:tuple[float,float,float,float]|None=None
    other_bbox:tuple[float,float,float,float]|None=None
    exhaustive:bool=False
    passes:list[ReferencePass]=Field(default_factory=list,max_length=30)
    notes:str=Field(default='',max_length=500)

class HumanReviewStore:
    def __init__(self,task_path='configs/evaluation/cv_m19_tasks.json',root=None):
        self.dataset=json.loads(Path(task_path).read_text());copy={k:v for k,v in self.dataset.items() if k!='fingerprint'}
        if signature(copy)!=self.dataset['fingerprint']:raise ValueError('Review protocol/task fingerprint mismatch')
        if self.dataset.get('selection_uses_model_candidates') is not False:raise ValueError('Model-independent source selection required')
        self.tasks={t['task_id']:t for t in self.dataset['tasks']};self.root=Path(root or os.environ.get('MATCHMIND_CV_REVIEW_ROOT','outputs/m19/human_review'));self.lock=threading.Lock()
    def labels(self):
        path=self.root/'labels.json'
        if not path.exists():return []
        data=json.loads(path.read_text())
        if data['dataset_fingerprint']!=self.dataset['fingerprint']:raise ValueError('Labels belong to another sealed dataset')
        return data['annotations']
    def validate(self,recording,label):
        t=self.tasks.get(label.task_id)
        if t is None or t['recording_id']!=recording:raise HTTPException(404,'Review task unavailable')
        source=self.dataset['sources'][recording]
        def box(value):
            if value is None:return
            x,y,w,h=value
            if not all(map(math.isfinite,value)) or x<0 or y<0 or w<=0 or h<=0 or x+w>source['width'] or y+h>source['height']:raise HTTPException(422,'Draw a valid box inside the source image')
        box(label.bbox);box(label.other_bbox)
        if t['kind']=='ball':
            if label.state not in ['visible','not_visible','uncertain'] or label.state=='visible' and label.bbox is None or label.state!='visible' and label.bbox is not None:raise HTTPException(422,'Visible ball needs a box; hidden/uncertain ball stays unboxed')
        elif t['kind']=='identity':
            if label.state not in ['same','different','uncertain'] or label.state!='uncertain' and (label.bbox is None or label.other_bbox is None):raise HTTPException(422,'Draw the two players and review whether they are the same person')
        else:
            if label.state not in ['reviewed','uncertain']:raise HTTPException(422,'Review the full source window or mark it uncertain')
            for p in label.passes:
                if not t['start_frame']<=p.release_frame<p.receipt_frame<=t['end_frame']:raise HTTPException(422,'Pass endpoints must be inside this review window')
        row=label.model_dump();row['source_video_sha256']=source['video_sha256']
        return row
    def save(self,recording,label):
        row=self.validate(recording,label)
        with self.lock:
            labels=[r for r in self.labels() if not (r['task_id']==row['task_id'] and r['reviewer_id']==row['reviewer_id'])];labels.append(row);self.root.mkdir(parents=True,exist_ok=True);temp=self.root/'labels.tmp';temp.write_text(json.dumps({'dataset_fingerprint':self.dataset['fingerprint'],'annotations':labels},indent=2)+'\n');os.replace(temp,self.root/'labels.json')
        return {'saved':True,'independent_label_count':len(labels),'task_id':label.task_id}
    def public(self,recording):
        return {'dataset_fingerprint':self.dataset['fingerprint'],'selection_uses_model_candidates':False,'source':self.dataset['sources'][recording],'tasks':[t for t in self.dataset['tasks'] if t['recording_id']==recording],'labels':[r for r in self.labels() if self.tasks[r['task_id']]['recording_id']==recording],'instructions':'Review original source footage without model overlays. These labels never change detections or confirmed player identities.'}


def review_router(authorize,reviews,*,store=None,prefix=''):
    router=APIRouter(prefix=prefix,dependencies=[Depends(authorize)]);store=store or HumanReviewStore()
    def source(recording):
        review=reviews.get(recording)
        if review is None or recording not in store.dataset['sources']:raise HTTPException(404,'Source review unavailable')
        if review.manifest['video_sha256']!=store.dataset['sources'][recording]['video_sha256']:raise HTTPException(503,'Review source changed')
        return review
    @router.get('/matches/{recording}/cv/review/tasks')
    def tasks(recording:str):source(recording);return store.public(recording)
    @router.post('/matches/{recording}/cv/review/labels')
    def save(recording:str,body:HumanAnnotation):source(recording);return store.save(recording,body)
    @router.get('/matches/{recording}/cv/review/frame.jpg')
    def image(recording:str,frame:int):
        import cv2
        review=source(recording)
        if not 0<=frame<review.manifest['frames']:raise HTTPException(422,'Frame outside source')
        review.frame((frame+.001)/review.manifest['fps'])
        with review.video_path() if hasattr(review,'video_path') else nullcontext(review.video) as video:
            cap=cv2.VideoCapture(str(video))
            try:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=cap.read()
            finally:cap.release()
        if not ok:raise HTTPException(503,'Source frame unavailable')
        ok,encoded=cv2.imencode('.jpg',pixels)
        if not ok:raise HTTPException(503,'Source frame encoding failed')
        return Response(encoded.tobytes(),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=60'})
    @router.get('/cv/review/export')
    def export():
        return Response(json.dumps({'dataset_fingerprint':store.dataset['fingerprint'],'sources':store.dataset['sources'],'annotations':store.labels()},indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=m19-human-labels.json'})
    return router,store
