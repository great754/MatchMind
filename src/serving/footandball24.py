"""Optional, fail-closed, read-only M24 benchmark. Never resolves analytics models."""
import json
from pathlib import Path
from fastapi import APIRouter,Depends,HTTPException,Query,Response
from fastapi.responses import FileResponse
from .cv import digest


class Benchmark:
    def __init__(self,config):
        config=json.loads(Path(config).read_text());self.report_path=Path(config['report'])
        if digest(self.report_path)!=config['report_sha256']:raise ValueError('M24 report hash mismatch')
        self.report=json.loads(self.report_path.read_text());self.inputs_path=Path(config['inputs'])
        if digest(self.inputs_path)!=self.report['inputs_manifest_sha256']:raise ValueError('M24 input snapshot mismatch')
        self.inputs=json.loads(self.inputs_path.read_text());self.versions={};self.files={}
        self.pin(self.report_path,config['report_sha256']);self.pin(self.inputs_path,self.report['inputs_manifest_sha256'])
        for name,source in self.inputs['sources'].items():
            self.pin(Path(source['video']),source['video_sha256']);entry=self.report['frame_files'][name];path=Path(entry['path']);self.pin(path,entry['sha256'])
            self.files[name]=path

    def pin(self,path,sha):
        if digest(path)!=sha:raise ValueError('M24 evidence hash mismatch')
        stat=path.stat();self.versions[path]=(stat.st_size,stat.st_mtime_ns)

    def valid(self,name):
        if name not in self.files:raise HTTPException(404,'Unknown M24 recording')
        try:
            for path,version in self.versions.items():
                stat=path.stat()
                if (stat.st_size,stat.st_mtime_ns)!=version:raise ValueError('M24 evidence changed; reload a validated benchmark')
        except (OSError,ValueError) as exc:raise HTTPException(503,str(exc)) from None
        return self.inputs['sources'][name]

    def frame(self,name,index):
        source=self.valid(name)
        if index>=source['frames']:raise HTTPException(422,'Frame outside M24 source')
        with self.files[name].open('rb') as stream:
            stream.seek(self.report['frame_files'][name]['offsets'][index]);row=json.loads(stream.readline())
        if row['frame']!=index:raise HTTPException(503,'M24 frame index mismatch')
        return row

    def metadata(self,name):
        source=self.valid(name);navigation={g:{k:[] for k in ['reviewed','correct','missed','false','fa_wins','m23_wins','both_fail']} for g in self.inputs['references']}
        for line in self.files[name].read_text().splitlines():
            row=json.loads(line)
            for group,ref in row['references'].items():
                if ref['state']=='uncertain':continue
                nav=navigation[group];nav['reviewed'].append(row['frame']);a=ref['status']['footandball'];b=ref['status']['m23']
                for flag in ['correct','missed','false']:
                    if a[flag]:nav[flag].append(row['frame'])
                if a['correct'] and not b['correct']:nav['fa_wins'].append(row['frame'])
                if b['correct'] and not a['correct']:nav['m23_wins'].append(row['frame'])
                if ref['state']=='visible' and not a['correct'] and not b['correct']:nav['both_fail'].append(row['frame'])
        return {'schema_version':'cv-m24-browser-v1','recording_id':name,'frames':source['frames'],'fps':source['fps'],'width':source['width'],'height':source['height'],
            'recordings':list(self.files),'video_url':f'/cv/m24/matches/{name}/video.mp4','read_only':True,'threshold':.7,
            'tolerance_diagonal_fraction':.01,'navigation':navigation,'groups':{g:self.report['groups'][g] for g in navigation},
            'reference_revision':self.report['inputs_manifest_sha256'],'experimental':True}


def create_router(authorize,config='configs/serving/cv_m24.json'):
    router=APIRouter(prefix='/cv/m24',dependencies=[Depends(authorize)]);benchmark=None;diagnostic='M24 benchmark is not installed'
    try:benchmark=Benchmark(config)
    except (OSError,ValueError,KeyError,TypeError) as exc:diagnostic='M24 benchmark unavailable: '+str(exc)
    def get():
        if benchmark is None:raise HTTPException(503,diagnostic)
        return benchmark

    @router.get('/matches/{name}/metadata')
    def metadata(name:str):return get().metadata(name)

    @router.get('/matches/{name}/frame')
    def frame(name:str,frame:int=Query(ge=0)):return get().frame(name,frame)

    @router.get('/matches/{name}/video.mp4')
    def video(name:str):return FileResponse(get().valid(name)['video'],media_type='video/mp4')

    @router.get('/matches/{name}/source.jpg')
    def image(name:str,frame:int=Query(ge=0)):
        import cv2
        b=get();row=b.frame(name,frame);capture=cv2.VideoCapture(b.valid(name)['video'])
        try:capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=capture.read()
        finally:capture.release()
        if not ok:raise HTTPException(503,'M24 source decode failed')
        if __import__('hashlib').sha256(pixels.tobytes()).hexdigest()!=row['footandball']['source_pixels_sha256']:raise HTTPException(503,'Decoded source pixels differ from benchmark')
        ok,data=cv2.imencode('.jpg',pixels)
        if not ok:raise HTTPException(503,'M24 source encoding failed')
        return Response(data.tobytes(),media_type='image/jpeg')

    @router.get('/matches/{name}/heatmap.png')
    def heatmap(name:str,frame:int=Query(ge=0)):
        import cv2,numpy as np
        row=get().frame(name,frame)['footandball'];path=Path(row['heatmap_file'])
        if digest(path)!=row['heatmap_sha256']:raise HTTPException(503,'M24 heatmap hash mismatch')
        heat=np.load(path)['ball'];m=row['mapping'];yy,xx=np.indices((m['source_height'],m['source_width']),dtype=np.float32)
        # Source pixel -> stride-four cell; padding is cropped, never stretched.
        mapped=cv2.remap(heat,(xx*m['scale']-1.5)/4,(yy*m['scale']-1.5)/4,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        color=cv2.applyColorMap(np.uint8(np.clip(mapped,0,1)*255),cv2.COLORMAP_INFERNO)
        ok,data=cv2.imencode('.png',color)
        if not ok:raise HTTPException(503,'M24 heatmap encoding failed')
        return Response(data.tobytes(),media_type='image/png')
    return router
