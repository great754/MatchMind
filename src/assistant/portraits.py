"""Association-safe crop selection; OCR is an unverified ranking aid, never identity."""
import json,time,threading,subprocess,shutil,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
from src.highlights.index import digest
from src.highlights.render import file_hash

def quality(image,box,others):
    import cv2
    h,w=image.shape[:2];x,y,bw,bh=box
    if min(bw,bh)<12 or bh<35 or x<0 or y<0 or x+bw>w or y+bh>h:return None
    overlap=0.
    for b in others:
        if b==box:continue
        a=max(0,min(x+bw,b[0]+b[2])-max(x,b[0]))*max(0,min(y+bh,b[1]+b[3])-max(y,b[1]));overlap=max(overlap,a/(bw*bh))
    if overlap>.4:return None
    crop=image[int(y):math.ceil(y+bh),int(x):math.ceil(x+bw)];torso=crop[int(bh*.2):max(int(bh*.65),int(bh*.2)+1)]
    if torso.size==0:return None
    grey=cv2.cvtColor(torso,cv2.COLOR_BGR2GRAY);sharp=float(cv2.Laplacian(grey,cv2.CV_64F).var());contrast=float(grey.std());edges=cv2.Canny(grey,60,150);edge_fraction=float((edges>0).mean())
    # This proxy cannot prove back-facing orientation or human number readability.
    resolution=min(1,bh/180);clarity=min(1,sharp/600);number_region=min(1,edge_fraction/.12)*min(1,contrast/50)
    score=.3*resolution+.3*clarity+.25*number_region+.15*(1-overlap)
    rgb=np.median(cv2.cvtColor(torso,cv2.COLOR_BGR2RGB).reshape(-1,3),axis=0).astype(int);swatch='#' + ''.join(f'{c:02x}' for c in rgb)
    return crop,{'resolution_height_px':bh,'sharpness':sharp,'occlusion_fraction_proxy':overlap,'torso_contrast':contrast,'number_region_edge_fraction':edge_fraction,'orientation':'UNKNOWN','readability':'UNVERIFIED','score':score,'swatch':swatch}

def number_hint(crop):
    import cv2
    if not shutil.which('tesseract'):return None
    h=crop.shape[0];torso=crop[int(h*.2):int(h*.7)];ok,b=cv2.imencode('.png',cv2.resize(torso,None,fx=3,fy=3))
    if not ok:return None
    try:
        p=subprocess.run(['tesseract','stdin','stdout','--psm','6','-c','tessedit_char_whitelist=0123456789'],input=b.tobytes(),capture_output=True,timeout=3)
        text=p.stdout.decode().strip();return text if text.isdigit() and len(text)<=2 else None
    except (OSError,subprocess.TimeoutExpired):return None

class PlayerImageService:
    def __init__(self,root,highlights,catalog_provider):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.highlights=highlights;self.catalog_provider=catalog_provider;self.jobs={};self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='player-pictures');self.lock=threading.Lock()
    def start(self,recording):
        with self.lock:
            existing=self.jobs.get(recording)
            if existing and existing['state'] in ['QUEUED','PROCESSING','COMPLETED']:return self.public(recording)
            if sum(j['state'] in ['QUEUED','PROCESSING'] for j in self.jobs.values())>=3:raise ValueError('Please wait for the other player pictures to finish.')
            self.jobs[recording]={'state':'QUEUED','players':{},'message':'Finding clear player pictures…'};self.executor.submit(self.work,recording);return self.public(recording)
    def public(self,recording):
        j=self.jobs.get(recording,{'state':'NOT_STARTED','players':{},'message':'Player pictures have not been prepared yet.'});return {k:v for k,v in j.items() if k!='directory'}
    def work(self,recording):
        started=time.monotonic();j=self.jobs[recording];j['state']='PROCESSING'
        try:
            import cv2
            cat=self.catalog_provider(recording);path,video_hash,_=self.highlights.source.resolve(recording,threading.Event());key=digest(['m18-portraits-1',recording,cat.source_signature,video_hash]);directory=self.root/key;directory.mkdir(exist_ok=True);j['directory']=str(directory);manifest=directory/'manifest.json'
            if manifest.exists():
                data=json.loads(manifest.read_text())
                if all((directory/c['file']).is_file() and file_hash(directory/c['file'])==c['sha256'] for cs in data['players'].values() for c in cs):j.update(**data,directory=str(directory));return
            choices=defaultdict(list);frame_refs=defaultdict(list)
            for player in cat.players:
                ref=player['player_ref'];boxes=cat.boxes.get(ref,{})
                # Pick a good-sized box from ten separate time bins, retaining time diversity.
                frames=sorted(boxes)
                for group in np.array_split(frames,min(10,len(frames))) if frames else []:
                    frame=max(group,key=lambda f:boxes[int(f)][2]*boxes[int(f)][3]);frame_refs[int(frame)].append(ref)
            cap=cv2.VideoCapture(str(path))
            if not cap.isOpened():raise ValueError('Video not available')
            try:
                cursor=0
                for frame,refs in sorted(frame_refs.items()):
                    while cursor<frame:
                        if not cap.grab():break
                        cursor+=1
                    ok,image=cap.read();cursor=frame+1
                    if not ok:continue
                    others=[boxes[frame] for boxes in cat.boxes.values() if frame in boxes]
                    for ref in refs:
                        result=quality(image,cat.boxes[ref][frame],others)
                        if result is not None:
                            crop,factors=result;choices[ref].append((factors['score'],frame,crop,factors))
            finally:cap.release()
            players={}
            for player in cat.players:
                ref=player['player_ref'];ranked=[]
                for score,frame,crop,factors in choices[ref]:
                    hint=number_hint(crop);factors['ocr_hint_unverified']=hint
                    bonus=.08 if hint and factors['sharpness']>=150 and factors['resolution_height_px']>=80 else 0
                    factors['ocr_ranking_bonus']=bonus;ranked.append((score+bonus,frame,crop,factors))
                selected=sorted(ranked,key=lambda row:(-row[0],row[1]))[:3];items=[]
                for rank,(_,frame,crop,factors) in enumerate(selected):
                    image_id=digest([key,ref,frame])[:32];file=image_id+'.jpg';cv2.imwrite(str(directory/file),crop);items.append({'image_id':image_id,'file':file,'sha256':file_hash(directory/file),'frame':frame,'timestamp':frame/cat.fps,'factors':factors,'url':f'/matches/{recording}/players/pictures/{image_id}.jpg','label':'Best available view' if rank==0 else 'Another view','number_confirmed':False})
                players[ref]=items;j['players'][ref]=items
            data={'state':'COMPLETED','players':players,'message':'Choose a view and confirm the number yourself. A clear number is not guaranteed.','video_sha256':video_hash,'source_signature':cat.source_signature,'seconds':time.monotonic()-started,'ranking':'Resolution, torso sharpness, overlap and number-region contrast proxy; orientation/readability unverified; OCR not identity.'};manifest.write_text(json.dumps(data,indent=2));j.update(**data,directory=str(directory))
        except Exception:j.update(state='FAILED',message='I couldn’t prepare player pictures from this footage. You can still enter identities manually.')
    def image(self,recording,image_id):
        j=self.jobs.get(recording,{})
        if j.get('state') not in ['PROCESSING','COMPLETED']:return None
        for cs in j['players'].values():
            for c in cs:
                if c['image_id']==image_id:
                    p=Path(j['directory'])/c['file']
                    return p if p.is_file() and file_hash(p)==c['sha256'] else None
        return None
