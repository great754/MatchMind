"""Bounded 20-image pilot: exact references, separate cached CV and Gemini proposals."""
import base64,hashlib,json,os,time
from pathlib import Path
import cv2,numpy as np,requests
from dotenv import load_dotenv
from src.serving.cv import digest
from src.vision.intelligent23.inputs import write
from src.vision.events.passes import signature

ROOT=Path('outputs/m25/dataset_v1')
PROMPT='Examine this entire football broadcast frame. Locate the actual football used in the match. Return a tight box_2d in [ymin,xmin,ymax,xmax], normalized 0..1000. Do not label boots, socks, advertising, spectators or markings. If hidden or uncertain return ball_visible false, box_2d null, uncertain true; do not guess from player gaze or expected trajectory. Return JSON only.'
SCHEMA={'type':'object','properties':{'ball_visible':{'type':'boolean'},'box_2d':{'type':['array','null'],'items':{'type':'number'}},'uncertain':{'type':'boolean'}},'required':['ball_visible','box_2d','uncertain'],'additionalProperties':False}


def convert(value,w,h):
    if set(value)!=set(SCHEMA['required']) or type(value['ball_visible']) is not bool or type(value['uncertain']) is not bool:raise ValueError('Invalid Gemini schema')
    box=value['box_2d']
    if not value['ball_visible'] or value['uncertain']:
        if box is not None:raise ValueError('Uncertain/absent result cannot carry a box')
        return None
    if not isinstance(box,list) or len(box)!=4 or any(type(x) not in [int,float] or not np.isfinite(x) for x in box):raise ValueError('Invalid Gemini coordinates')
    y,x,b,r=box
    if not 0<=x<r<=1000 or not 0<=y<b<=1000:raise ValueError('Gemini box outside image')
    return [x*w/1000,y*h/1000,(r-x)*w/1000,(b-y)*h/1000]


def run():
    load_dotenv('.env');frames=json.loads((ROOT/'frames.json').read_text());videos=json.loads((ROOT/'videos.json').read_text());indices=[*range(8),8,10,12,14,16,18,20,23,26,29,35,37]
    selected=[frames[i] for i in indices];assert len(selected)==20
    existing=ROOT/'pilot_20.json'
    if existing.exists():
        cached=json.loads(existing.read_text())
        if len(cached['frames'])!=20 or any(digest(r['image'])!=r['image_sha256'] for r in cached['frames']):raise ValueError('Cached pilot source changed')
        print('CACHED PILOT: no model/API calls; preserve review progress',cached['summary'],flush=True);return
    manifest_path=ROOT/'pilot_20.json';model='gemini-3.5-flash-lite';proposals=[];gemini_calls=0;cost=0;rawroot=ROOT/'gemini';rawroot.mkdir(exist_ok=True)
    from ultralytics import YOLO
    checkpoint=Path('outputs/ml/ball_m16/training/soccer11s/weights/best.pt');yolo=YOLO(str(checkpoint));print('M16 labels',yolo.names,flush=True)
    for row in selected:
        row['shot_type']='broadcast_gameplay';row['eligibility']={'method':'geometry screen plus actual source contact-sheet inspection; score-overlay continuity where visible','visually_inspected':True,'replay_detection':'no identifiable replay marker; hidden replay ambiguity remains','human_annotation_created':False};row['proposals']=[]
        ref=row['existing_reference'];row['annotation_state']='verified_'+('visible' if ref['state']=='visible' else 'absent' if ref['state']=='not_visible' else 'uncertain') if ref else 'unreviewed'
        if ref:
            # Verify the short-frame decode and exact source identity against frozen perception.
            rows=[json.loads(s) for s in Path('outputs/m24/perception_v1',row['recording_id'],'native/frames.jsonl').read_text().splitlines()];pixel_hash=hashlib.sha256(cv2.imread(row['image']).tobytes()).hexdigest()
            if pixel_hash!=rows[row['frame']]['source_pixels_sha256']:raise ValueError('Exact reviewed source pixel mapping failed')
            row['exact_reference_pixels_verified']=True
            for previous in ['m21','m22','m23']:
                cached=Path('outputs/m24/inputs_v1/baselines')/row['recording_id']/(previous+'.jsonl')
                if not cached.exists():continue
                old=[json.loads(line) for line in cached.read_text().splitlines()][row['frame']]
                for i,c in enumerate(old['candidates'][:2]):row['proposals'].append({'proposal_id':f'{previous}:{row["frame_id"][:12]}:{i}','bbox':c['bbox'],'confidence':c['confidence'],'source_pipeline':previous.upper()+' frozen source candidate','native_box':True,'source_output_sha256':digest(cached),'source_frame_id':row['frame_id'],'model_version':old['version'],'source':'FROZEN_MODEL_OUTPUT','prediction_timestamp_utc':None,'prediction_timestamp_semantics':'Original inference time not recorded in frozen cache'})
            continue
        result=yolo.predict(row['image'],conf=.05,imgsz=1280,device='mps',verbose=False)[0]
        write(ROOT/'m16_raw'/(row['frame_id']+'.json'),{'image_sha256':row['image_sha256'],'model_sha256':digest(checkpoint),'names':yolo.names,'boxes_xyxy_conf_class':result.boxes.data.detach().cpu().tolist(),'configuration':{'imgsz':1280,'conf':.05}})
        for i,b in enumerate(result.boxes):
            if 'ball' not in str(yolo.names[int(b.cls.item())]).lower():continue
            x,y,r,bottom=b.xyxy[0].tolist();row['proposals'].append({'proposal_id':f'm16:{row["frame_id"][:12]}:{i}','bbox':[x,y,r-x,bottom-y],'confidence':float(b.conf.item()),'source_pipeline':'M16 YOLO11s native box','checkpoint_sha256':digest(checkpoint),'native_box':True,'configuration':{'confidence_floor':.05,'imgsz':1280}})
        row['proposals']=row['proposals'][:8]
        if (ROOT/'gemini_quota_check.json').exists() and json.loads((ROOT/'gemini_quota_check.json').read_text()).get('error',{}).get('code')==429:
            row['gemini']={'status':'blocked_project_spending_cap','model':model,'valid':False,'proposal_not_truth':True};continue
        key=signature([row['image_sha256'],model,PROMPT,SCHEMA]);cache=rawroot/(key+'.json')
        if cache.exists():data=json.loads(cache.read_text())
        else:
            # Conservative reservation: <=16000 input and 1024 output tokens at official standard rates.
            reserve=(16000*.30+1024*2.50)/1e6
            if cost+reserve>1.0:raise ValueError('Configured $1 pilot cost cap reached')
            write(rawroot/(key+'.started.json'),{'image_sha256':row['image_sha256'],'model':model,'reserved_usd':reserve})
            body={'contents':[{'parts':[{'text':PROMPT},{'inline_data':{'mime_type':'image/png','data':base64.b64encode(Path(row['image']).read_bytes()).decode()}}]}],'generationConfig':{'temperature':0,'maxOutputTokens':1024,'responseMimeType':'application/json','responseJsonSchema':SCHEMA}}
            response=requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',headers={'x-goog-api-key':os.environ['GEMINI_API_KEY']},json=body,timeout=(10,90),allow_redirects=False)
            if response.status_code!=200:raise ValueError('Gemini HTTP '+str(response.status_code))
            data=response.json();write(cache,data);gemini_calls+=1
        usage=data.get('usageMetadata',{});cost+=(usage.get('promptTokenCount',0)*.30+(usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0))*2.50)/1e6
        text=''.join(p.get('text','') for p in data['candidates'][0]['content']['parts'] if not p.get('thought'));row['gemini']={'cache_key':key,'raw_response_sha256':digest(cache),'model':model,'usage':usage,'prompt_version':digest(Path(__file__)),'proposal_not_truth':True}
        try:
            value=json.loads(text);box=convert(value,row['width'],row['height']);row['gemini']['parsed']=value;row['gemini']['valid']=True
            if box:row['proposals'].append({'proposal_id':'gemini:'+row['frame_id'][:12],'bbox':box,'confidence':None,'source_pipeline':'Gemini full-frame proposal','native_box':True,'coordinate_convention':'ymin,xmin,ymax,xmax normalized 1000'})
        except (ValueError,TypeError,KeyError):row['gemini']['valid']=False;row['gemini']['diagnostic']='Invalid response preserved; manual review required'
        print('PREPARED',row['recording_id'],row['frame'],'CV',len(row['proposals']),'Gemini valid',row['gemini']['valid'],flush=True)
    rejected={15:'celebration',19:'close_up',21:'uncertain',22:'uncertain',25:'crowd_or_non_gameplay',28:'close_up',30:'uncertain',31:'close_up',32:'celebration',33:'close_up',34:'close_up',36:'close_up'}
    write(manifest_path,{'schema_version':'cv-m25-pilot-v1','batch_id':'m25_pilot_20','sources':videos,'frames':selected,'candidate_count':len(frames),'not_selected_shot_labels':{frames[i]['frame_id']:v for i,v in rejected.items()},'selection_seed':25,'new_human_labels':0,'training_performed':False,'summary':{'selected':20,'verified_reused':sum(bool(r['existing_reference']) for r in selected),'gemini_frames':sum('gemini' in r and r['gemini'].get('status')!='blocked_project_spending_cap' for r in selected),'gemini_blocked_frames':sum(r.get('gemini',{}).get('status')=='blocked_project_spending_cap' for r in selected),'gemini_attempted_distinct_frames':1 if (ROOT/'gemini_quota_check.json').exists() else gemini_calls,'gemini_new_calls_this_run':gemini_calls,'gemini_valid_boxes':sum(any(p['source_pipeline'].startswith('Gemini') for p in r['proposals']) for r in selected),'cv_proposal_frames':sum(any(p['source_pipeline'].startswith('M16') for p in r['proposals']) for r in selected),'estimated_standard_usd':cost,'max_cost_usd':1.0}})
    print('PILOT',json.loads(manifest_path.read_text())['summary'],flush=True)

if __name__=='__main__':run()
