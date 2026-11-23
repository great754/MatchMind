"""Authenticated M22 development review routes; proposal acceptance always requires a human save."""
import json
from pathlib import Path
from fastapi import APIRouter,Depends,HTTPException,Query,Response
from fastapi.responses import FileResponse
from src.vision.football22.review import DevelopmentReviewStore,DevelopmentLabel
from src.serving.cv import digest


def create_review_router(authorize,reviews,manifest='configs/training/cv_m22_tasks.json',root='outputs/m22/human_review',prefix='/cv/m22',approved_sources=None):
    store=DevelopmentReviewStore(manifest,root,approved_sources=approved_sources)
    if store.dataset.get('publication_status')!='source_sequences_visually_inspected':raise ValueError('Review source sequences must be inspected before publication')
    router=APIRouter(prefix=prefix,dependencies=[Depends(authorize)])

    def source(recording):
        review=reviews.get(recording);meta=store.dataset['sources'].get(recording)
        if review is None or meta is None:raise HTTPException(404,'Approved development review source unavailable')
        if review.manifest['video_sha256']!=meta['video_sha256']:raise HTTPException(503,'Review source version changed')
        if approved_sources is not None and any(review.manifest[k]!=meta[k] for k in ['recording_id','frames','fps','width','height']):
            raise HTTPException(503,'Review source clock or dimensions changed')
        return review

    @router.get('/matches/{recording}/review/tasks')
    def tasks(recording:str):
        source(recording)
        try:return store.public(recording)
        except ValueError as e:raise HTTPException(503,str(e)) from None

    @router.post('/matches/{recording}/review/labels')
    def save(recording:str,body:DevelopmentLabel):
        source(recording)
        try:return store.save(recording,body)
        except ValueError as e:raise HTTPException(422,str(e)) from None

    @router.get('/matches/{recording}/review/frame.jpg')
    def image(recording:str,frame:int=Query(ge=0)):
        import cv2
        review=source(recording)
        if frame>=review.manifest['frames']:raise HTTPException(422,'Frame outside source')
        review.frame((frame+.001)/review.manifest['fps'])
        with review.video_path() as video:
            cap=cv2.VideoCapture(str(video))
            try:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=cap.read()
            finally:cap.release()
        if not ok:raise HTTPException(503,'Current source pixels unavailable')
        ok,encoded=cv2.imencode('.jpg',pixels)
        if not ok:raise HTTPException(503,'Image encoding failed')
        return Response(encoded.tobytes(),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=60'})

    @router.get('/matches/{recording}/review/contact-sheet/{page}.jpg')
    def contact_sheet(recording:str,page:int):
        source(recording)
        sheets=[(p,m) for p,m in store.dataset['preview_sheets'].items() if m['recording_id']==recording]
        if not 0<=page<len(sheets):raise HTTPException(404,'Source contact sheet unavailable')
        path,meta=sheets[page]
        if digest(path)!=meta['sha256']:raise HTTPException(503,'Source preview changed')
        return FileResponse(path,media_type='image/jpeg')

    @router.get('/review/export')
    def export():
        return Response(json.dumps({'schema_version':'cv-m23-temporal-supervision-v1' if approved_sources is not None else 'cv-m22-temporal-supervision-v1','dataset_fingerprint':store.dataset['fingerprint'],
            'reference_role':'development_only','model_assisted':True,'sources':store.dataset['sources'],
            'tasks':list(store.tasks.values()),'annotations':store.labels()},indent=2),media_type='application/json',
            headers={'Content-Disposition':f'attachment; filename={prefix.rsplit("/",1)[-1]}-development-labels.json'})
    return router,store
