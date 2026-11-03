"""Identical IoU/center protocols; no false visibility or metric ground truth claims."""
import math
from collections import Counter
import numpy as np
from ml.ball14.evaluate import detection_metrics,iou

def wilson(hits,n):
 if n<=0:return None
 z=1.96;p=hits/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]

def tracking(rows,states):
 counts=Counter();errors=[];fragments=0;last_rec=None;previous_match=False;had_match=False;gaps=0;reacquired=correct_reacquired=gap_opportunities=0;unsupported=longest=0;last_frame=None;metric=[]
 for r,s in zip(rows,states,strict=True):
  d=s.model_dump() if hasattr(s,'model_dump') else s;counts[d['tracking_status']]+=1;observed=d['directly_observed'];counts['observed']+=observed;counts['predicted']+=d['predicted_image_position'] is not None
  if r['recording_id']!=last_rec:previous_match=False;had_match=False;gaps=0;last_rec=r['recording_id'];unsupported=0
  good=False
  if observed:
   best=max([iou(d['image_bbox'],b['bbox']) for b in r['balls']],default=0);good=best>=.25;counts['observed_iou_matched']+=good;counts['observed_iou_unmatched']+=not good;point=d['image_position'];err=min([math.dist(point,(b['bbox'][0]+b['bbox'][2]/2,b['bbox'][1]+b['bbox'][3]/2)) for b in r['balls']],default=float('inf'));counts['observed_center_matched']+=err<=12;counts['observed_center_unmatched']+=err>12
   if math.isfinite(err):errors.append(err)
   if gaps and had_match:reacquired+=1;correct_reacquired+=good
   if good and not previous_match and had_match:fragments+=1
  if good:
   if gaps and had_match:gap_opportunities+=1
   had_match=True;gaps=0
  elif had_match:gaps+=1
  previous_match=good
  if d['predicted_image_position'] is not None:
   counts['prediction_gt_comparable']+=bool(r['balls']);err=min([math.dist(d['predicted_image_position'],(b['bbox'][0]+b['bbox'][2]/2,b['bbox'][1]+b['bbox'][3]/2)) for b in r['balls']],default=float('inf'));counts['prediction_gt_inconsistent']+=bool(r['balls']) and err>12;unsupported+=1/r['fps'];longest=max(longest,unsupported)
  else:unsupported=0
  counts['metric_supported']+=d['pitch_position'] is not None
 observed=counts['observed'];pred=counts['prediction_gt_comparable'];return {'frames':len(rows),'counts':dict(counts),'observed_coverage':observed/len(rows),'predicted_coverage':counts['predicted']/len(rows),'accepted_iou_precision':counts['observed_iou_matched']/max(1,observed),'accepted_center_consistency':counts['observed_center_matched']/max(1,observed),'accepted_center_wilson95':wilson(counts['observed_center_matched'],observed),'prediction_inconsistency_rate':counts['prediction_gt_inconsistent']/pred if pred else None,'prediction_inconsistency_wilson95':wilson(counts['prediction_gt_inconsistent'],pred),'localization_median_px':float(np.median(errors)) if errors else None,'localization_p90_px':float(np.quantile(errors,.9)) if errors else None,'longest_predicted_continuation_seconds':longest,'matched_track_fragmentations':fragments,'reacquisition_attempts_after_matched_gap':reacquired,'reacquisition_correct_iou':correct_reacquired,'reacquisition_precision':correct_reacquired/reacquired if reacquired else None,'reacquisition_recall':None,'metric_error_m':None,'invalid_ground_acceptance_rate':None,'limits':'Organizer visibility/true occlusion/active-ball and independent metric GT absent; reacquisition is matched tracker-gap recovery, not visibility-conditioned recall. Wilson intervals descriptive only: adjacent frames correlated, two test clips insufficient for broad uncertainty claims.'}
