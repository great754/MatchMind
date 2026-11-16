"""Independent pixel/frame/MOT export consistency; not a human readability review."""
import json,hashlib,math
from pathlib import Path
import cv2,numpy as np
from src.highlights.render import probe
from src.assistant.catalog import PlayerCatalog
from src.assistant.sprints import SprintService
from src.query import MatchMindTools
ROOT=Path('outputs/m18_evaluation')
def main():
 roster=json.loads((ROOT/'portraits_roster.json').read_text());pictures=roster['pictures'];source=next(Path('outputs/highlights_m18/sources').glob('*.mp4'));match='118575';sync=json.loads(Path(f'data/soccertrack/mot/{match}_sync.json').read_text());tracks={int(v['player_id']):int(k) for k,v in sync['players'].items()};mot=np.loadtxt(f'data/soccertrack/mot/{match}.txt',delimiter=',',usecols=range(6));boxes={(int(f),int(t)):(x,y,w,h) for f,t,x,y,w,h in mot};directory=next(Path('outputs/assistant_m18/pictures').glob('*/manifest.json')).parent;frames={};out=[]
 for ref,items in pictures['players'].items():
  for item in items:frames.setdefault(item['frame'],[]).append((ref,item))
 cap=cv2.VideoCapture(str(source));cursor=0
 try:
  for frame,items in sorted(frames.items()):
   while cursor<frame:
    if not cap.grab():raise ValueError('Source video truncated')
    cursor+=1
   ok,image=cap.read();cursor=frame+1
   if not ok:raise ValueError('Source frame unavailable')
   for ref,item in items:
    actor=int(ref.split(':')[1]);track=tracks[actor];x,y,w,h=boxes[(frame,track)];crop=image[int(y):math.ceil(y+h),int(x):math.ceil(x+w)];encoded=cv2.imencode('.jpg',crop)[1].tobytes();actual=(directory/item['file']).read_bytes();out.append({'player_ref':ref,'source_frame':frame,'source_track':track,'timestamp_correct':abs(item['timestamp']-frame/roster['fps'])<1e-8,'encoded_crop_matches_raw_mot_source_pixels':hashlib.sha256(encoded).hexdigest()==hashlib.sha256(actual).hexdigest(),'height_px':crop.shape[0],'width_px':crop.shape[1],'rank':pictures['players'][ref].index(item)+1})
 finally:cap.release()
 cat=PlayerCatalog('118575_clip',MatchMindTools(Path('outputs/cache/azure_m17/118575_clip')));cfg=json.loads(Path('configs/assistant/m18.json').read_text());sensitivity=[]
 for window in [15,25,35]:
  service=SprintService(cat,{**cfg,'smoothing_frames':window});sensitivity.append({'smoothing_frames':window,'qualifying_runs':len(service.intervals),'fastest':service.best(),'diagnostic':service.diagnostic})
 data={'protocol':'Decode the original registered source sequentially, crop using independently re-read raw MOT rows and explicit sync mapping, and compare encoded JPEG hashes; no catalogue box lookup used. Does not independently adjudicate identity switches or human jersey readability.','crop_checks':out,'correct_crops':sum(r['encoded_crop_matches_raw_mot_source_pixels'] for r in out),'total_crops':len(out),'players':len(pictures['players']),'source_sha256':pictures['video_sha256'],'sprint_smoothing_sensitivity':sensitivity,'human_readability_review':None,'manual_identification_completion_rate':None,'independent_speed_error':None,'independent_sprint_interval_error':None}
 (ROOT/'image_sprint_evidence.json').write_text(json.dumps(data,indent=2,default=lambda v:v.item())+'\n');print('CROPS',data['correct_crops'],len(out),'SMOOTHING',[(r['smoothing_frames'],r['qualifying_runs'],r['fastest']['speed_mps'] if r['fastest'] else None) for r in sensitivity],flush=True)
if __name__=='__main__':main()
