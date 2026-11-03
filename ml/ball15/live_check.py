"""Explicit local live verification; no credentials or signed Azure media URLs serialized."""
import requests
from .common import save

def run():
 base='http://127.0.0.1:8000';out={}
 for endpoint in ['health','ready','recordings']:
  r=requests.get(base+'/'+endpoint,timeout=60);assert r.status_code==200;out[endpoint]={'status':r.status_code,'body':r.json()}
 for name in ['test_match','team_goal','southampton']:
  variants={}
  for version in ['m14','m15']:
   r=requests.get(base+f'/matches/{name}/cv',params={'ball_pipeline':version},timeout=30);b=r.json();assert r.status_code==200 and b['ball_model_version']==version and b['camera_model_version']=='m8' and b['player_model_version']=='m9';variants[version]={'status':r.status_code,'camera':'m8','player':'m9','ball':version,'frames':b['frames']}
   frame=requests.get(base+f'/matches/{name}/cv/frame',params={'timestamp':50/b['fps'],'ball_pipeline':version},timeout=30);s=frame.json();assert frame.status_code==200 and s['ball_world']['frame']==50 and s['ball_world']['version']==version
  out[name]=variants
 r=requests.get(base+'/matches/118575_clip/view',timeout=60);assert r.status_code==200;out['azure_view']={'status':200}
 r=requests.get(base+'/matches/118575_clip/video.mp4',headers={'Range':'bytes=0-1023'},timeout=60);assert r.status_code==206;out['azure_range']={'status':206,'bytes':len(r.content)}
 r=requests.post(base+'/matches/118575_clip/ask',json={'question':'Which player ran the fastest?','provider':'local'},timeout=60);b=r.json();assert r.status_code==200 and b['answer']['status']=='ok' and len(b['evidence'])>0;out['ask']={'status':200,'answer_status':b['answer']['status'],'evidence_rows':len(b['evidence'])}
 r=requests.post(base+'/matches/test_match/ask',json={'question':'Who scored?','provider':'local'},timeout=30);assert r.status_code==503;out['cv_ask']={'status':503,'grounding_still_unavailable':True}
 save('live_regression.json',out);print('LIVE',out['ready'],out['ask'],out['azure_range']);return out
if __name__=='__main__':run()
