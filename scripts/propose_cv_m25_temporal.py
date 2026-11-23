"""Attach unchanged M22 temporal proposals using actual neighboring Azure frames."""
import json,os,subprocess,threading
from pathlib import Path
import cv2,torch
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from src.serving.media import AzureClipMedia
from src.highlights.azure_stream import azure_video_source
from src.vision.football22.inference import FootballDetector
from src.vision.football22.model import spacing
from src.vision.intelligent23.inputs import write
from src.serving.cv import digest


def run():
    root=Path('outputs/m25/dataset_v1');p=root/'pilot_20.json';m=json.loads(p.read_text());load_dotenv('.env');torch.set_num_threads(4)
    client=ContainerClient(os.environ['MATCHMIND_BLOB_ACCOUNT_URL'],os.environ['MATCHMIND_MEDIA_BLOB_CONTAINER'],credential=DefaultAzureCredential(),max_single_get_size=1024**2,max_chunk_get_size=1024**2)
    checkpoint=Path('outputs/m24/inputs_v1/model/m22_giroud.pth');config=json.loads(Path('outputs/m24/inputs_v1/model/m22_config.json').read_text());detector=FootballDetector(checkpoint,digest(checkpoint),config)
    name='tottenham_arsenal_long';source=m['sources'][name];media=AzureClipMedia(client);media.sources[name]=source['blob'];media.register_half=lambda *a:None
    with azure_video_source(media,name,{'match_id':name,'half':1},threading.Event(),byte_budget=400*1024**2) as (url,identity,origin,state):
        for row in m['frames']:
            if row['recording_id']!=name or row['existing_reference']:continue
            context=[]
            for delta in [2*spacing(source['fps']),spacing(source['fps'])]:
                frame=max(0,row['frame']-delta);image=root/'context'/f'{row["frame_id"]}_{frame}.png';image.parent.mkdir(exist_ok=True)
                if not image.exists():
                    r=subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(frame/source['fps']),'-i',url,'-frames:v','1','-threads','1',str(image)],capture_output=True,timeout=90)
                    if r.returncode:raise ValueError('Temporal context decode failed')
                context.append(cv2.imread(str(image)))
            context.append(cv2.imread(row['image']));raw=detector.infer(context,row['frame']);cache=root/'m22_native_context'/f'{row["frame_id"]}.json';write(cache,raw)
            row['proposals']=[v for v in row['proposals'] if not v['source_pipeline'].startswith('M22')]
            for i,c in enumerate(raw['candidates'][:3]):row['proposals'].append({'proposal_id':f'm22:{row["frame_id"][:12]}:{i}','bbox':c['bbox'],'confidence':c['confidence'],'source_pipeline':'M22 temporal native box','native_box':True,'checkpoint_sha256':digest(checkpoint),'context_frames':[max(0,row['frame']-2*spacing(source['fps'])),max(0,row['frame']-spacing(source['fps'])),row['frame']],'current_only_confidence':c['current_only_confidence'],'observation_eligible':c['observation_eligible']})
            print('M22',row['frame'],len(raw['candidates']),flush=True)
    m['summary']['m22_proposal_frames']=sum(any(c['source_pipeline'].startswith('M22') for c in r['proposals']) for r in m['frames']);m['summary']['cv_proposal_frames']=sum(bool(r['proposals']) for r in m['frames']);m['m22_source_range_bytes']=state['bytes'];write(p,m)

if __name__=='__main__':run()
