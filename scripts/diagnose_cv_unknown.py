"""Immutable M6 UNKNOWN-frame diagnosis before M7 estimator changes."""
import json,argparse
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from src.vision.world.review import digest
from scripts.diagnose_cv_temporal import line_intersections,line_deltas

def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False);m=json.loads((root/'manifest.json').read_text());assert digest(root/'frames.jsonl')==m['frames_sha256'];rows=[json.loads(l) for l in (root/'frames.jsonl').open()];reports=[];counts=Counter();cap=cv2.VideoCapture(m['video_path'] if 'video_path' in m else 'data/raw/'+m['recording_id']+'.mp4');prev=None;changes=[]
 for r in rows:
  ok,img=cap.read();assert ok
  gray=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(160,90));changes.append(float(np.mean(np.abs(gray.astype(float)-prev))) if prev is not None else None);prev=gray
 cap.release()
 for i,r in enumerate(rows):
  c=r['camera']
  if c['calibration_valid']:continue
  t=c['motion_evidence'].get('temporal_camera',{});loc=c['motion_evidence']['localization'];f=t.get('field_motion',{});names=loc['semantic_classes'];reason=loc['outcome'];inter=line_intersections(line_deltas(rows[i-1]['camera'],c,m['width'],m['height'])) if i else [];categories=[]
  checks={'insufficient_absolute_geometry':not c['motion_evidence'].get('pose',{}).get('accepted',False),'insufficient_independent_geometry':'INDEPENDENT' in reason,'missing_weak_semantics':len(names)<2,'failed_field_motion':not f.get('accepted',False),'lost_texture_anchors':t.get('reobserved_anchor',{}).get('reason')=='NO_REOBSERVED_FIELD_ANCHOR','visible_region_failure':'REGION' in reason,'temporal_inconsistency':bool(t.get('geometry_inconsistent') or t.get('baseline_geometry_inconsistent') or 'INCONSISTENT' in reason),'scale_zoom_rejection':'ZOOM' in f.get('reason',''),'line_disagreement':f.get('reason')=='STATIC_LINES_CONTRADICT_MOTION','circle_ambiguity':bool(any(n.startswith('Circle') for n in names) and ('AMBIGU' in reason or 'INDEPENDENT' in reason)),'no_previous_camera':i==0 or not rows[i-1]['camera']['calibration_valid'],'cannot_reconnect_after_gap':bool(i and not rows[i-1]['camera']['calibration_valid'] and not t.get('reobserved_anchor',{}).get('reason')=='FRESH_REOBSERVED_FIELD_ANCHOR')}
  for k,v in checks.items():
   counts[k]+=int(v)
  reports.append({'frame':i,'timestamp':r['timestamp'],'reason':reason,'previous_localized':i>0 and rows[i-1]['camera']['calibration_valid'],'next_localized':i+1<len(rows) and rows[i+1]['camera']['calibration_valid'],'reliable_measured_field_motion':f.get('accepted',False),'motion':f,'named_lines':[n for n in names if not n.startswith('Circle')],'top_touchline_visible':'Side line top' in names,'intersections':inter,'center_circle_visible':'Circle central' in names,'box_markings_visible':any('rect.' in n for n in names),'image_mean_absolute_change_160x90':changes[i],'large_image_change_proxy_mean_abs_gt15':changes[i] is not None and changes[i]>15,'previous_camera_propagation_plausible':bool(i and rows[i-1]['camera']['calibration_valid'] and f.get('accepted')),'failure_categories':checks,'existing_diagnostic':t})
 (out/'unknown_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reports));summary={'baseline':str(root),'frames_sha256':m['frames_sha256'],'unknown':len(reports),'overlapping_failure_counts':dict(counts),'outcomes':dict(Counter(r['reason'] for r in reports)),'categories_are_overlapping_proxies':'Insufficient absolute geometry means no accepted cached single-frame fit, not proof fitting is impossible. Circle ambiguity counted only when rejection explicitly identifies ambiguity/independent-geometry failure; unresolved circle visibility alone is not counted. Image-change threshold is a diagnostic proxy, not a cut detector.'};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--baseline');p.add_argument('--output');a=p.parse_args();run(a.baseline,a.output)
