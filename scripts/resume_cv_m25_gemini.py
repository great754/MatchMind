"""Explicit, cached Gemini-only completion of the frozen M25 pilot after quota repair."""
import argparse,base64,json,os
from datetime import datetime,timezone
from pathlib import Path
import requests
from dotenv import load_dotenv
from scripts.annotate_cv_m25 import ROOT,PROMPT,SCHEMA,convert
from src.serving.cv import digest
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write


def run(max_cost=1.0):
    if not 0<max_cost<=1.0:raise ValueError('M25 pilot cap must be positive and at most $1')
    load_dotenv('.env');path=ROOT/'pilot_20.json';batch=json.loads(path.read_text())
    if len(batch['frames'])!=20:raise ValueError('Only the frozen twenty-image pilot may run')
    revisions=json.loads((ROOT/'human_revisions.json').read_text()) if (ROOT/'human_revisions.json').exists() else []
    latest={r['task_id']:r for r in revisions};model='gemini-3.5-flash-lite';cache_root=ROOT/'gemini';cost=0.;calls=0
    write(ROOT/'pilot_history'/(digest(path)+'.json'),batch)
    for row in batch['frames']:
        if digest(row['image'])!=row['image_sha256']:raise ValueError('Pilot image changed')
        if row['existing_reference'] and row['existing_reference']['state'] in ['visible','not_visible']:continue
        if latest.get(row['frame_id'],{}).get('state') in ['visible','not_visible']:continue
        key=signature([row['image_sha256'],model,PROMPT,SCHEMA]);cache=cache_root/(key+'.json')
        if cache.exists():response=json.loads(cache.read_text())
        else:
            reserve=(16000*.30+1024*2.50)/1e6
            if cost+reserve>max_cost:raise ValueError('Pilot API budget exhausted')
            body={'contents':[{'parts':[{'text':PROMPT},{'inline_data':{'mime_type':'image/png','data':base64.b64encode(Path(row['image']).read_bytes()).decode()}}]}],'generationConfig':{'temperature':0,'maxOutputTokens':1024,'responseMimeType':'application/json','responseJsonSchema':SCHEMA}}
            raw=requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',headers={'x-goog-api-key':os.environ['GEMINI_API_KEY']},json=body,timeout=(10,90),allow_redirects=False)
            response=raw.json();calls+=1
            if raw.status_code!=200:
                write(cache_root/(key+'.http_error.json'),response)
                raise RuntimeError('Gemini HTTP '+str(raw.status_code)+'; private diagnostic saved. No automatic quota retries.')
            write(cache,response)
        usage=response.get('usageMetadata',{});cost+=(usage.get('promptTokenCount',0)*.30+(usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0))*2.50)/1e6
        row['proposals']=[p for p in row['proposals'] if not p['source_pipeline'].startswith('Gemini')]
        row['gemini']={'model':model,'model_version':response.get('modelVersion','unreported'),'cache_key':key,'raw_response_sha256':digest(cache),'usage':usage,'valid':False,'proposal_not_truth':True}
        try:
            text=''.join(p.get('text','') for p in response['candidates'][0]['content']['parts'] if not p.get('thought'));value=json.loads(text);box=convert(value,row['width'],row['height']);row['gemini'].update(parsed=value,valid=True)
            if box:row['proposals'].append({'proposal_id':'gemini:'+row['frame_id'][:12],'bbox':box,'confidence':None,'source_pipeline':'Gemini full-frame proposal','native_box':True,'coordinate_convention':'ymin,xmin,ymax,xmax normalized 1000','source_frame_id':row['frame_id'],'model_version':response.get('modelVersion','unreported'),'configuration':{'cache_key':key,'temperature':0,'max_output_tokens':1024},'source':'GEMINI_RESPONSE','prediction_timestamp_utc':datetime.fromtimestamp(cache.stat().st_mtime,timezone.utc).isoformat(),'prediction_timestamp_semantics':'Local response recording time, not remote inference start'})
        except (ValueError,KeyError,TypeError):row['gemini']['diagnostic']='Invalid output preserved; human review required'
        write(path,batch)
    batch['summary'].update(gemini_frames=sum('usage' in r.get('gemini',{}) for r in batch['frames']),gemini_blocked_frames=sum(r.get('gemini',{}).get('status')=='blocked_project_spending_cap' for r in batch['frames']),gemini_valid_boxes=sum(any(p['source_pipeline'].startswith('Gemini') for p in r['proposals']) for r in batch['frames']),estimated_standard_usd=cost,gemini_new_calls_this_run=calls)
    write(path,batch);print(batch['summary'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-cost-usd',type=float,default=1.0);args=parser.parse_args();run(args.max_cost_usd)
