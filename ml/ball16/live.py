"""Explicit local API/Azure/grounding regression checks after service restart."""
import requests
from .common import save

def run():
 base='http://127.0.0.1:8000';out={}
 for endpoint in ['health','ready','recordings']:
  r=requests.get(base+'/'+endpoint,timeout=60);assert r.status_code==200;out[endpoint]={'status':r.status_code,'body':r.json()}
 for name in ['test_match','team_goal','southampton']:
  out[name]={}
  for version in ['m14','m15','m16']:
   r=requests.get(base+f'/matches/{name}/cv',params={'ball_pipeline':version},timeout=30);b=r.json();assert r.status_code==200 and b['ball_model_version']==version and b['camera_model_version']=='m8' and b['player_model_version']=='m9';f=requests.get(base+f'/matches/{name}/cv/frame',params={'timestamp':50/b['fps'],'ball_pipeline':version},timeout=30);assert f.status_code==200 and f.json()['ball_world']['frame']==50;out[name][version]={'metadata':200,'frame50':200}
 r=requests.get(base+'/matches/118575_clip/view',timeout=60);assert r.status_code==200;out['azure_view']=200;r=requests.get(base+'/matches/118575_clip/video.mp4',headers={'Range':'bytes=0-1023'},timeout=60);assert r.status_code==206;out['azure_video']={'status':206,'bytes':len(r.content)};r=requests.post(base+'/matches/118575_clip/ask',json={'question':'Which player ran the fastest?','provider':'local'},timeout=60);b=r.json();assert r.status_code==200 and b['answer']['status']=='ok';out['ask']={'status':200,'answer_status':'ok','evidence_rows':len(b['evidence'])};save('live_regression.json',out);print('LIVE',out['ready'],out['ask'])
if __name__=='__main__':run()
