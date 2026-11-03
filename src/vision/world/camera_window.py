"""Offline robust pitch-only anchor refinement over a measured temporal window.

One 8-DOF image-to-world camera is optimized against fixed canonical lines and
circles in up to eleven current/neighbor frames. Neighbor transforms are measured
field motion, never coefficient interpolation. Pixel residual gates stay M5/M6.
"""
import cv2,numpy as np
from scipy.optimize import least_squares
from .persistent_camera import corrected,safe_region
from .temporal_camera import disagreement

def refine_window(index,cameras,groups,forward,backward,validator,initial_h=None,extended=False,fps=25):
 raw=cameras[index];h=np.array(initial_h if initial_h is not None else raw.calibration.image_to_pitch).reshape(3,3);width,height=validator.width,validator.height
 evidence=[(index,np.eye(3))]
 for step,motions in ((1,forward),(-1,backward)):
  transport=np.eye(3)
  budget=0.
  for distance in range(1,(round(fps)+1 if extended and step==-1 else 6)):
   j=index+step*distance
   if not 0<=j<len(cameras) or cameras[j].epoch!=raw.epoch or cameras[max(j,j-step)].continuity=='cut':break
   motion=motions[j]
   if not motion.get('accepted'):break
   budget=float(np.hypot(budget,motion['heldout_p90_small_px']*motion.get('scale_to_source',1)))
   if budget>20:break
   transport=np.array(motion['homography'])@transport;evidence.append((j,transport.copy()))
 names=set(name for j,t in evidence for name in groups[j] if name in validator.geometry.segments)
 families=[sum(validator.geometry.segments[name][0][axis]==validator.geometry.segments[name][1][axis] for name in names) for axis in (0,1)]
 if extended and min(families)<2:return h,{'accepted':False,'reason':'INSUFFICIENT_MULTIFRAME_INDEPENDENT_LINE_GEOMETRY','frames':[j for j,t in evidence],'line_families':families}
 def objective(p,heldout=False):
  candidate=corrected(h,p,width,height);values=[]
  for j,t in evidence:
   residual,names=validator.geometry.residuals(candidate@np.linalg.inv(t),groups[j],width,height,heldout)
   if len(residual):values.extend(residual/np.sqrt(len(residual)))
  return np.r_[values,p/.2]
 starts=[np.zeros(8),np.array([.001,0,.001,0,-.001,-.001,0,0]),np.array([-.001,0,-.001,0,.001,.001,0,0])];fits=[]
 for initial in starts:
  try:
   fit=least_squares(objective,initial,bounds=(-.2,.2),loss='soft_l1',f_scale=2,max_nfev=100);candidate=corrected(h,fit.x,width,height);stats=validator.statistics(candidate,groups[index]);window_stats=[validator.statistics(candidate@np.linalg.inv(t),groups[j]) for j,t in evidence if len(groups[j])>=2]
   if stats['accepted'] and safe_region(candidate,width,height) is not None and all(s['accepted'] for s in window_stats):fits.append((candidate,stats,float(np.sum(np.square(objective(fit.x,True))))))
  except (ValueError,cv2.error,np.linalg.LinAlgError):pass
 debug={'frames':[j for j,t in evidence],'source':'fixed pitch line/circle constraints + measured relative field motion; correlated semantic pixels','valid_starts':len(fits),'accepted':False,'line_families':families,'future_evidence':any(j>index for j,t in evidence)}
 if len(fits)<2:return h,debug
 spread=max(disagreement(fits[0][0],a[0],width,height) for a in fits);debug['hypothesis_spread_p90_m']=spread
 if spread>4:return h,debug
 best=min(fits,key=lambda f:f[2]);debug.update(accepted=True,absolute_adjustment_p90_m=disagreement(best[0],h,width,height),heldout=best[1]);return best[0],debug
