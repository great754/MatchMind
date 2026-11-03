"""M7 coverage, matched-denominator proxies and full long-gap timeline."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.evaluate_cv_temporal import evaluate
from scripts.evaluate_camera_placement import polygon_iou
from src.vision.world.temporal_camera import transform,disagreement
from src.vision.world.schema import WorldState

def distribution(values):
 values=[v for v in values if v is not None and np.isfinite(v)]
 return {'count':len(values),'median':float(np.median(values)) if values else None,'p90':float(np.quantile(values,.9)) if values else None,'max':max(values) if values else None}

def run(before,after,output):
 output=Path(output);meta=json.load(open(Path(after)/'manifest.json'));report=evaluate(before,after,output);rows=[WorldState.model_validate_json(l) for l in (Path(after)/'frames.jsonl').open()];timelines=[];world_count=[];lifetimes=[];survival=[];heldout=[];line=[];circle=[];corrections=[];agreement=[];camera_delta=[];compensated=[];iou=[];reasons=Counter();states=Counter();quarantines=[];unknown_created=[]
 for i,r in enumerate(rows):
  c=r.camera;d=c.motion_evidence['temporal_camera'];w=d.get('world_field',{});f=d['field_motion'];states[c.motion_evidence['localization']['status']]+=1
  if not c.calibration_valid:reasons[c.source]+=1
  if not c.calibration_valid and d.get('new_world_anchors',0):unknown_created.append(i)
  world_count.append(w.get('active_world_anchors'));lifetimes.append(w.get('mean_track_lifetime_frames'));survival.append(d.get('feature_survival_fraction'));heldout.append(w.get('heldout_p90_small_px'));line.append(d.get('line_validation',{}).get('p90') if d.get('line_validation') else None);circle.append(d.get('circle_validation',{}).get('p90') if d.get('circle_validation') else None);corrections.append(d.get('anchor_correction_p90_m'));agreement.append(d.get('forward_backward_agreement_p90_m'))
  if d.get('semantic_line_quarantine'):quarantines.append({'frame':i,'excluded_labels':d['semantic_line_quarantine']})
  if i and c.calibration_valid and rows[i-1].camera.calibration_valid and c.epoch==rows[i-1].camera.epoch:
   camera_delta.append(disagreement(transform(c),transform(rows[i-1].camera),meta['width'],meta['height']))
   if f.get('accepted'):
    predicted=transform(rows[i-1].camera)@np.linalg.inv(np.array(f['homography']));compensated.append(disagreement(transform(c),predicted,meta['width'],meta['height']))
   iou.append(polygon_iou(c.visible_pitch_region,rows[i-1].camera.visible_pitch_region))
  timelines.append({'frame':i,'timestamp':r.timestamp,'state':c.motion_evidence['localization']['status'],'epoch':c.epoch,'active_world_field_tracks':w.get('active_world_anchors'),'top_touchline':f.get('top_touchline_tracked'),'named_lines':c.motion_evidence['localization']['semantic_classes'],'intersections':d.get('tracked_intersections'),'circle':d.get('circle_validation'),'field_motion_reason':f.get('reason'),'heldout_world_error':w.get('heldout_p90_small_px'),'predicted_region':d.get('predicted_region'),'semantic_validation':d.get('landmark_validation'),'uncertainty_px':d.get('trajectory_uncertainty_px'),'reason':c.source,'directions':d.get('m7_trajectory_review',{}).get('directions'),'route':d.get('measurement_route'),'quarantine':d.get('semantic_line_quarantine')})
 assert not unknown_created
 meta=json.load(open(Path(after)/'manifest.json'));diag=json.load(open(Path(after)/'diagnostics.json'))
 result={'recording_id':meta['recording_id'],'frames':len(rows),'camera_accepted':sum(c.camera.calibration_valid for c in rows),'camera_coverage_percent':diag['camera_coverage_percent'],'player_placement_percent':diag['player_placement_percent'],'states':dict(states),'newly_recovered':len(report['previous_unknown_recovered']),'previously_accepted_lost':len(report['previously_accepted_now_rejected']),'longest_supported':report['after_runs']['longest_supported'],'longest_unknown':report['after_runs']['longest_unknown'],'active_world_anchor_count':distribution(world_count),'mean_anchor_lifetime_frames_across_current_fits':distribution(lifetimes),'feature_survival_fraction':distribution(survival),'heldout_world_fit_p90_working_px':distribution(heldout),'line_p90_source_px':distribution(line),'circle_p90_source_px':distribution(circle),'absolute_correction_p90_m':distribution(corrections),'forward_backward_agreement_p90_m':distribution(agreement),'camera_adjacent_transform_change_p90_m':distribution(camera_delta),'camera_motion_compensated_change_p90_m':distribution(compensated),'visible_region_iou':distribution(iou),'unknown_reasons':dict(reasons),'semantic_label_quarantines':quarantines,'cuts':sum(r.camera.continuity=='cut' for r in rows),'epochs':sorted({r.camera.epoch for r in rows}),'route_timings':{k:{'timing':v['timing'],'wall_seconds':v['wall_seconds']} for k,v in diag['routes'].items()},'total_wall_seconds':diag['wall_seconds'],'teams_frozen':True,'unknown_mints_world_anchors':False,'accuracy':'dependent proxies only; independent real metric accuracy unavailable'}
 (output/'coverage.json').write_text(json.dumps(result,indent=2,allow_nan=False));(output/'camera_failure_timeline.jsonl').write_text(''.join(json.dumps(t,allow_nan=False)+'\n' for t in timelines));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.before,a.after,a.output)
