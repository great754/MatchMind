"""Detector metrics on original GT boxes; no assumption that missing annotation means invisible."""
import math
import numpy as np

def iou(a,b):
 ax,ay,aw,ah=a;bx,by,bw,bh=b;inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by));return inter/max(aw*ah+bw*bh-inter,1e-9)

def detection_metrics(rows,predictions,threshold=.1,overlap=.25):
 tp=fp=fn=0;errors=[];frames_with_tp=0;strata={k:{'gt':0,'tp':0} for k in ['small_under12px','medium12_24px','large_over24px','low_sharpness_proxy','higher_sharpness_proxy']}
 for row,ps in zip(rows,predictions,strict=True):
  gt=row['balls'];matched=set();hits=0
  for p in sorted([p for p in ps if p['confidence']>=threshold],key=lambda p:-p['confidence']):
   choices=[(iou(p['bbox'],g['bbox']),j) for j,g in enumerate(gt) if j not in matched]
   best=max(choices,default=(0,-1))
   if best[0]>=overlap:
    j=best[1];matched.add(j);tp+=1;hits+=1;a=p['bbox'];b=gt[j]['bbox'];errors.append(math.hypot(a[0]+a[2]/2-b[0]-b[2]/2,a[1]+a[3]/2-b[1]-b[3]/2))
   else:fp+=1
  fn+=len(gt)-len(matched);frames_with_tp+=hits>0
  for j,g in enumerate(gt):
   diameter=max(g['bbox'][2:]);size='small_under12px' if diameter<12 else 'medium12_24px' if diameter<=24 else 'large_over24px';blur='low_sharpness_proxy' if (g.get('sharpness_proxy') or 0)<50 else 'higher_sharpness_proxy'
   for key in [size,blur]:strata[key]['gt']+=1;strata[key]['tp']+=j in matched
 precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
 return {'frames':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),'false_positives_per_frame':fp/max(len(rows),1),'localization_median_px':float(np.median(errors)) if errors else None,'localization_p90_px':float(np.quantile(errors,.9)) if errors else None,'detected_gt_frame_coverage':frames_with_tp/max(len(rows),1),'threshold':threshold,'iou_match_threshold':overlap,'strata':{k:{**v,'recall':v['tp']/v['gt'] if v['gt'] else None} for k,v in strata.items()},'visibility_or_occlusion_gt_available':False}
