"""Perspective-dependent camera viewing motion on the fixed metric pitch.

The tracked ground point stays fixed. A shift of a fixed image ray's ground
intersection describes changing framing, not a solved physical camera centre.
"""
import cv2,numpy as np

def project(h,p):return cv2.perspectiveTransform(np.asarray(p,float)[None],np.asarray(h,float))[0]

def jacobians(h,p):
 p=np.asarray(p,float);q=project(h,p);return np.stack([project(h,p+[1,0])-q,project(h,p+[0,1])-q],axis=-1)

def metric_update(previous,current,width,height,motion=None,pixel_budget=0.):
 previous=np.asarray(previous,float);current=np.asarray(current,float);reference=np.array([[width/2,height/2]],float);old=project(previous,reference)[0];new=project(current,reference)[0]
 relative=np.linalg.inv(current)@previous;derivative=jacobians(relative,reference)[0];u,s,vt=np.linalg.svd(derivative);rot=u@vt;angle=float(np.degrees(np.arctan2(rot[1,0],rot[0,0])));shift=new-old
 corners=np.array([(width*x,height*y) for y in (.3,.6,.9) for x in (.2,.5,.8)],float);j=jacobians(current,corners);sizes=np.linalg.svd(j,compute_uv=False);uncertainty=float(np.quantile(sizes[:,0],.9)*pixel_budget)
 field=[]
 for v in (motion or {}).get('vectors',[]):
  if not v.get('inlier'):continue
  a,b=np.array(v['previous']),np.array(v['current']);pa=project(previous,[a])[0];pb=project(current,[b])[0]
  field.append({'image_before':a.tolist(),'image_current':b.tolist(),'pitch_before':pa.tolist(),'pitch_current':pb.tolist(),'fixed_point_residual_m':float(np.linalg.norm(pa-pb)),'local_metres_per_pixel_singular_values':np.linalg.svd(jacobians(previous,[a])[0],compute_uv=False).tolist()})
 return {'reference_image_point':reference[0].tolist(),'previous_viewing_point_m':old.tolist(),'current_viewing_point_m':new.tolist(),'incremental_view_shift_m':shift.tolist(),'pan_direction':'right' if shift[0]>.001 else 'left' if shift[0]<-.001 else 'negligible','metric_pan_magnitude_m':float(abs(shift[0])),'vertical_framing_shift_m':float(shift[1]),'image_displacement_at_reference_px':(project(relative,reference)[0]-reference[0]).tolist(),'zoom_singular_values':s.tolist(),'rotation_degrees':angle,'projective_component':relative[2,:2].tolist(),'uncertainty_pitch_m_proxy':uncertainty,'uncertainty_pixel_budget':pixel_budget,'metric_field_correspondences':field[:32],'interpretation':'fixed image-ray ground-intersection shift; NOT physical 3D camera translation; local scale varies over image/pitch'}

class MetricTrajectory:
 def __init__(self):self.previous=None;self.epoch=None;self.origin=None;self.accumulated=np.zeros(2);self.history=[]
 def update(self,frame,epoch,h,width,height,motion=None,budget=0.,anchor=None,cut=False):
  if cut or self.epoch!=epoch:self.previous=None;self.origin=None;self.accumulated=np.zeros(2);self.history=[]
  self.epoch=epoch
  if h is None:
   self.previous=None;self.origin=None;self.accumulated=np.zeros(2);self.history=[]
   return {'frame':frame,'epoch':epoch,'accepted':False,'current_transform':None,'incremental_view_shift_m':None,'accumulated_view_shift_m':None,'reason':'NO_ACCEPTED_CURRENT_CAMERA','history':self.history[-12:]}
  h=np.asarray(h,float);d=metric_update(self.previous if self.previous is not None else h,h,width,height,motion,budget)
  if self.previous is None:d.update(previous_viewing_point_m=None,incremental_view_shift_m=[0.,0.],interpretation='new accepted trajectory segment; no unmeasured gap displacement')
  if self.origin is None:self.origin=frame
  self.accumulated+=d['incremental_view_shift_m'];d.update(frame=frame,epoch=epoch,accepted=True,current_transform=h.tolist(),segment_origin_frame=self.origin,independent_anchor_frame=anchor,accumulated_view_shift_m=self.accumulated.tolist())
  self.history.append({'frame':frame,'viewing_point_m':project(h,[[width/2,height/2]])[0].tolist(),'transform':h.tolist()});d['history']=self.history[-12:];self.previous=h.copy();return d
