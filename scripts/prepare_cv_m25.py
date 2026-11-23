"""Prepare unannotated source frames first; never invokes a detector or Gemini."""
import argparse,json,subprocess,threading,hashlib,random
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from src.highlights.azure_stream import azure_video_source
from src.highlights.render import probe
from src.serving.cv import digest
from src.serving.media import AzureClipMedia
from src.vision.intelligent23.inputs import write


def prepare(root):
    import os
    load_dotenv();root=Path(root);root.mkdir(parents=True,exist_ok=True)
    client=ContainerClient(os.environ['MATCHMIND_BLOB_ACCOUNT_URL'],os.environ['MATCHMIND_MEDIA_BLOB_CONTAINER'],credential=DefaultAzureCredential(),max_single_get_size=1024**2,max_chunk_get_size=1024**2)
    registry=json.loads(Path('configs/serving/cv_media.json').read_text());sources={name:{**entry,'kind':'short_clip','original_match_group':'unresolved_'+name,'split':'protected_evaluation' if name=='giroud' else 'unassigned'} for name,entry in registry['recordings'].items()}
    long=json.loads(Path('docs/evidence/external_video_upload_2026_11_13.json').read_text())['azure'];sources['tottenham_arsenal_long']={'blob':long['blob'],'bytes':long['bytes'],'sha256':long['sha256_metadata'],'etag':long['etag'],'kind':'long_broadcast','original_match_group':'external_vibuxer_z4ny0atm95s8','split':'unassigned'}
    refs=json.loads(Path('outputs/m24/inputs_v1/manifest.json').read_text())['references'];selected={}
    # Exact existing human references, spread in source time, chosen without detector success.
    for name in registry['recordings']:
        candidates=[r for r in refs['giroud_completed' if name=='giroud' else 'original_m19'] if r['recording_id']==name and r['state']=='visible']
        selected[name]=[candidates[0]['frame'],candidates[len(candidates)//2]['frame']]
    source_config=Path('configs/training/cv_m25_sources.json')
    if source_config.exists():sources=json.loads(source_config.read_text())['sources']
    frame_rows=[]
    for name,source in sources.items():
        media=AzureClipMedia(client);media.sources[name]=source['blob'];media.register_half=lambda *a:None
        props=client.get_blob_client(source['blob']).get_blob_properties()
        if props.etag!=source['etag'] or props.size!=source['bytes']:raise ValueError('Pinned Azure source changed')
        with azure_video_source(media,name,{'match_id':name,'half':1},threading.Event(),byte_budget=600*1024**2) as (url,identity,origin,state):
            info=probe(url);fps=float(__import__('fractions').Fraction(info['fps']));source.update(width=info['width'],height=info['height'],fps=fps,duration=info['duration'],frames=int(info['nb_frames']) if info['nb_frames'] not in [None,'N/A'] else round(info['duration']*fps),integrity='ETag-conditioned Azure byte ranges; original archive SHA recorded, not freshly whole-file rehashed')
            frames=selected.get(name,[round(t*fps) for t in np.linspace(120,info['duration']-120,30)])
            print('SOURCE',name,info['duration'],'seconds',info['width'],info['height'],'candidates',len(frames),flush=True)
            for frame in frames:
                fid=hashlib.sha256((source['sha256']+':'+str(frame)).encode()).hexdigest();path=root/'images'/(fid+'.png');path.parent.mkdir(exist_ok=True)
                if name in registry['recordings']:
                    from src.serving.cv_media import AzureCVSource
                    cloud=AzureCVSource(client,name,registry['recordings'][name])
                    if source['sha256']!=cloud.sha256:raise ValueError('Registered source identity changed; use a distinct source ID')
                    with cloud.open() as local:
                        cap=cv2.VideoCapture(str(local));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,pixels=cap.read();cap.release()
                        if not ok:raise ValueError('Exact source decode failed')
                        cv2.imwrite(str(path),pixels)
                elif not path.exists():
                    command=['ffmpeg','-nostdin','-v','error','-ss',str(frame/fps),'-i',url,'-frames:v','1','-threads','1',str(path)]
                    result=subprocess.run(command,capture_output=True,timeout=90)
                    if result.returncode:raise ValueError('Frame extraction failed; private FFmpeg output retained only in process')
                image=cv2.imread(str(path))
                if image is None or image.shape[:2]!=(source['height'],source['width']):raise ValueError('Invalid source frame shape')
                hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);green=cv2.inRange(hsv,np.array([28,25,25]),np.array([95,255,255]));fraction=float(np.mean(green>0));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);contrast=float(gray.std());edges=cv2.Canny(gray,70,160);lines=cv2.HoughLinesP(edges,1,np.pi/180,70,minLineLength=image.shape[1]//12,maxLineGap=12)
                # A geometry screen, not a replay classifier: all survivors require context inspection.
                reason='blank_or_corrupt' if contrast<8 else 'not_pitch_dominant' if fraction<.35 else 'insufficient_field_geometry' if lines is None or len(lines)<4 else 'wide_view_candidate_requires_player_and_replay_context_check'
                small=cv2.resize(gray,(32,32)).astype(np.float32);dct=cv2.dct(small)[:8,:8];bits=(dct>dct.flatten()[1:].mean()).flatten();phash=sum(int(v)<<i for i,v in enumerate(bits))
                near=[r['frame_id'] for r in frame_rows if (phash^int(r['perceptual_hash'],16)).bit_count()<=3]
                if near:reason='near_duplicate_candidate_requires_review'
                ref=next((r for group in refs.values() for r in group if r['recording_id']==name and r['frame']==frame),None)
                if any(r['frame_id']==fid for r in frame_rows):continue
                if ref and source['sha256']!=json.loads(Path('outputs/m24/inputs_v1/manifest.json').read_text())['sources'][name]['video_sha256']:ref=None
                frame_rows.append({'frame_id':fid,'recording_id':name,'frame':frame,'timestamp':frame/fps,'image':str(path),'image_sha256':digest(path),'width':source['width'],'height':source['height'],'green_fraction':fraction,'contrast':contrast,'line_segments':len(lines) if lines is not None else 0,'shot_type':'uncertain' if reason.startswith('wide_view') else 'crowd_or_non_gameplay','filter_reason':reason,'existing_reference':ref,'review_status':'unreviewed','perceptual_hash':hex(phash),'possible_near_duplicates':near,'selection_seed':25,'selection_reason':'source_time_diversity_not_detector_success','original_match_group':source['original_match_group'],'split':source['split']})
            source['range_bytes_read']=state['bytes']
        write(root/'videos.json',sources);write(root/'frames.json',frame_rows)
    # Contact sheet for actual source inspection before choosing annotation frames.
    from PIL import Image,ImageDraw
    for page,start in enumerate(range(0,len(frame_rows),12)):
        sheet=Image.new('RGB',(1200,4*210),'#111820');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(frame_rows[start:start+12]):
            im=Image.open(row['image']);im.thumbnail((390,175));x=(i%3)*400;y=(i//3)*210;sheet.paste(im,(x,y+28));draw.text((x+3,y+3),f'{start+i} {row["recording_id"]} {row["timestamp"]:.1f}s',fill='white')
        sheet.save(root/f'source_sheet_{page}.jpg')
    print('UNANNOTATED',len(frame_rows),'no detection/Gemini calls; inspect source sheets before pilot selection.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/m25/dataset_v1');a=p.parse_args();prepare(a.root)
