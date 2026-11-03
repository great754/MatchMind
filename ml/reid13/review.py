"""Visual changed-edge and held-out failure evidence; no fitting or selection."""
import json,math
from pathlib import Path
import cv2,numpy as np
from .common import ROOT,load,save
from .evaluate import vectors
from .protocol import retrieval

def fit_image(image,w,h):
 canvas=np.full((h,w,3),235,np.uint8)
 if image is None or not image.size:return canvas
 ratio=min(w/image.shape[1],h/image.shape[0]);image=cv2.resize(image,(max(1,round(image.shape[1]*ratio)),max(1,round(image.shape[0]*ratio))));y=(h-image.shape[0])//2;x=(w-image.shape[1])//2;canvas[y:y+image.shape[0],x:x+image.shape[1]]=image;return canvas

def changed_edges():
 shadow=load('shadow_results.json')['sources'];review=[]
 for source,values in shadow.items():
  base=Path('outputs/world_model')/(source+'_m9_ready');manifest=json.loads((base/'manifest.json').read_text());segments={s['segment_id']:s for s in json.loads((base/'segments.json').read_text())};features=json.loads((Path('outputs/world_model/m9_cache')/source/'features.json').read_text());a={tuple(e) for e in values['m9']['edges']};b={tuple(e) for e in values['m10']['edges']};new={tuple(e) for e in values['m13_selected_pool']['edges']};changed=(new^a)|(new^b);cap=cv2.VideoCapture(manifest['source_video']);directory=ROOT/'changed_connections'/source;directory.mkdir(parents=True,exist_ok=True)
  for index,(left,right) in enumerate(sorted(changed)):
   sa,sb=segments[left],segments[right];frames=[(sa,(sa['start']+sa['end'])//2),(sa,sa['end']),(sb,sb['start']),(sb,(sb['start']+sb['end'])//2)];tiles=[];context=[]
   for segment,frame in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
    if not ok:raise ValueError('Missing changed-connection frame')
    person=next((p for p in features[frame] if p['track_id']==segment['track_id']),None)
    if person is None:
     candidates=[(abs(i-frame),i,p) for i in range(segment['start'],segment['end']+1) for p in features[i] if p['track_id']==segment['track_id']]
     if not candidates:raise ValueError('Missing endpoint observation')
     _,frame,person=min(candidates,key=lambda x:x[0]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
    x,y,w,h=map(int,person['bbox']);crop=image[max(0,y):min(image.shape[0],y+h),max(0,x):min(image.shape[1],x+w)].copy()
    if crop.size:crop[:int(.18*crop.shape[0])]=127
    tile=fit_image(crop,180,230);cv2.putText(tile,f'{segment["track_id"]} frame{frame}',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,0),1);tiles.append(tile);ctx=image.copy();cv2.rectangle(ctx,(x,y),(x+w,y+h),(0,0,255),3);context.append(fit_image(ctx,360,230))
   canvas=np.vstack([np.hstack(tiles),np.hstack([context[1],context[2]])]);filename=f'{index:02d}.jpg';cv2.imwrite(str(directory/filename),canvas);review.append({'source':source,'from_segment':left,'to_segment':right,'m9_linked':(left,right) in a,'m10_linked':(left,right) in b,'m13_linked':(left,right) in new,'evidence':str(directory/filename),'assessment':'REVIEW_PENDING','independent_identity_ground_truth':False})
  cap.release()
 save('changed_connection_review.json',review);print('Changed connections',len(review),flush=True)

def failures():
 lock=load('selection_lock.json');rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];q,g,_,_=retrieval(rows);by={r['sample_id']:i for i,r in enumerate(rows)};candidate=next(v for v in lock['variants'] if v['name']==lock['visual_variant']);zs={'m10':vectors(rows,'m10'),'m13':vectors(rows,lock['visual_variant'],candidate['selected']['checkpoint'])};cases=[]
 for a in q:
  eligible=[b for b in g if b['team_identity']==a['team_identity'] and b['match_id']==a['match_id']];rankings={k:sorted(eligible,key=lambda b:-float(v[by[a['sample_id']]]@v[by[b['sample_id']]])) for k,v in zs.items()};correct={k:r[0]['player_identity']==a['player_identity'] for k,r in rankings.items()}
  if not correct['m13']:cases.append({'query':a,'m10_correct':correct['m10'],'m13_correct':False,'rankings':{k:[{'sample':b,'similarity':float(zs[k][by[a['sample_id']]]@zs[k][by[b['sample_id']]])} for b in bs[:3]] for k,bs in rankings.items()}})
 cases.sort(key=lambda c:not c['m10_correct']);save('failure_analysis.json',{'query_count':len(q),'m13_rank1_errors':len(cases),'m10_correct_m13_wrong':sum(c['m10_correct'] for c in cases),'cases':cases,'interpretation':'annotation-defined same-team errors; crop size/pose/occlusion/lighting observations are qualitative, not independently labelled causes'})
 for page in range(min(2,math.ceil(len(cases)/4))):
  panels=[]
  for case in cases[page*4:page*4+4]:
   records=[('QUERY',case['query'])]+[(kind,entry['sample']) for kind,ranked in case['rankings'].items() for entry in ranked];tiles=[]
   for title,r in records:
    tile=fit_image(cv2.imread(r['crop_path']),145,200);cv2.putText(tile,title,(3,12),cv2.FONT_HERSHEY_SIMPLEX,.33,(0,0,0),1);cv2.putText(tile,r['player_identity'][-20:],(3,26),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1);tiles.append(tile)
   panels.append(np.hstack(tiles))
  if panels:cv2.imwrite(str(ROOT/f'failures_{page}.jpg'),np.vstack(panels))
if __name__=='__main__':changed_edges();failures()
