import json,math
import numpy as np
from ml.ball15.common import load,save
from ml.ball15.motion import samples,HORIZONS
from src.vision.ball15.motion import extrapolate
from src.vision.ball.geometry import camera_motion,warp_point

def evaluate(split):
 rows=[r for r in load('dataset.json')['samples'] if r['split']==split];data=load(split+'_cameras.json');assert [r['sample_id'] for r in rows]==data['sample_ids'];lookup={(r['recording_id'],r['frame']):c for r,c in zip(rows,data['cameras'])};es=[];used=0
 for sample in samples(rows):
  seq=sample['recording_id'];frame=sample['frame'];points=sample['points'][:8].copy();matrix=None;okay=True
  for f in range(frame-6,frame+1):
   a=lookup.get((seq,f-1));b=lookup.get((seq,f));m,_=camera_motion(a,b) if a and b else (None,'missing')
   if m is None:okay=False;break
   for j in range(f-(frame-7)):
    p=warp_point(m,points[j]);
    if p is None:okay=False;break
    points[j]=p
   matrix=m
  predictions=[]
  for j,h in enumerate(HORIZONS):
   p=extrapolate(points if okay else sample['points'][:8],sample['times'][:8],sample['times'][8+j])
   if okay:
    for _ in range(h):
     warped=warp_point(matrix,p)
     if warped is not None:p=warped
   predictions.append(p)
  used+=okay;es.append(np.linalg.norm(np.array(predictions)-sample['points'][8:],axis=1))
 a=np.array(es);return {'samples':len(es),'accepted_past_camera_chain_samples':used,'horizons':HORIZONS,'mean_error_px':a.mean(0).tolist(),'median_error_px':np.median(a,axis=0).tolist(),'p90_error_px':np.quantile(a,.9,axis=0).tolist(),'protocol':'Past-only accepted M8 image warps align history, constant last warp extrapolated to future; unsupported chains fall back to CV. No future images/GT used for forecasting. Diagnostic comparison, no post-TEST selection.'}
def run():
 report={'validation':evaluate('validation'),'test':evaluate('test')}
 save('camera_motion_prediction_diagnostic.json',report);print(report)
if __name__=='__main__':run()
