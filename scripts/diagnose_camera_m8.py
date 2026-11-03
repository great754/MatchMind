"""Freeze M7 evidence over every unknown frame, before M8 algorithms."""
import json
from pathlib import Path
from collections import Counter
import cv2,numpy as np
import ast,hashlib
CLASSES=next(ast.literal_eval(n.value) for n in ast.parse(Path('src/vision/world/semantic_pitch.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CLASSES' for t in n.targets))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def groups(mask,confidence):
 result={}
 for i,name in enumerate(CLASSES,1):
  yy,xx=np.where((mask==i)&(confidence>=.5))
  if len(xx)>=12:result[name]=np.column_stack([(xx+.5)/mask.shape[1],(yy+.5)/mask.shape[0]])
 return result
SOURCES={'test_match':('outputs/world_model/test_match_m7_current','outputs/world_model/test_match_m2_pose'),'team_goal':('outputs/world_model/team_goal_m7_current','outputs/world_model/team_goal/poses'),'southampton':('outputs/world_model/southampton/current','outputs/world_model/southampton/poses')}
def run():
 for name,(directory,poses) in SOURCES.items():
  root=Path(directory);out=Path('outputs/world_model/m8_verification')/name/'before';out.mkdir(parents=True,exist_ok=True);m=json.loads((root/'manifest.json').read_text());assert digest(root/'frames.jsonl')==m['frames_sha256'];rows=[json.loads(l) for l in (root/'frames.jsonl').open()];cache=np.load(Path(poses)/'semantic_masks.npz');masks=cache['mask'];confidence=cache['confidence'];stats=Counter();report=[]
  for i,r in enumerate(rows):
   c=r['camera'];g=groups(masks[i],confidence[i]);d=c['motion_evidence']['temporal_camera'];ellipse=None
   if len(g.get('Circle central',[]))>=12:
    try:
     raw=np.asarray(g['Circle central'])*[m['width'],m['height']];fit=cv2.fitEllipseAMS(raw.astype(np.float32));ellipse={'center_image':list(fit[0]),'axes':list(fit[1]),'rotation_degrees':float(fit[2]),'warning':'diagnostic ellipse fit, not independent camera pose'}
    except cv2.error:pass
   row={'frame':i,'timestamp':r['timestamp'],'accepted_m7':c['calibration_valid'],'state':c['motion_evidence']['localization']['status'],'reason':c['source'],'epoch':c['epoch'],'named_landmarks':list(g),'circle_pixel_count':len(g.get('Circle central',[])),'halfway_pixel_count':len(g.get('Middle line',[])),'diagnostic_ellipse':ellipse,'field_motion':d['field_motion'],'world_fields':d.get('world_field'),'previous_accepted':i>0 and rows[i-1]['camera']['calibration_valid'],'next_accepted':i+1<len(rows) and rows[i+1]['camera']['calibration_valid'],'prior_camera':d.get('last_accepted_frame'),'predicted_region':d.get('predicted_region'),'uncertainty':d.get('trajectory_uncertainty_px')};report.append(row)
   if not c['calibration_valid']:
    stats['unknown']+=1;stats['circle']+='Circle central' in g;stats['halfway']+='Middle line' in g;stats['circle_halfway_top']+=all(k in g for k in ['Circle central','Middle line','Side line top']);stats['measured_motion']+=d['field_motion'].get('accepted',False);stats['world_fit']+=d.get('world_field',{}).get('accepted',False)
  cache.close();(out/'frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in report));(out/'summary.json').write_text(json.dumps({'source':directory,'frames_sha256':m['frames_sha256'],'video_sha256':m['video_sha256'],'frames':len(rows),'unknown_diagnosis':dict(stats),'center216':report[min(216,len(rows)-1)]},indent=2));print(name,dict(stats),flush=True)
if __name__=='__main__':run()
