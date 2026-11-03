"""M8 pitch conic/line relocalization with explicit reflection ambiguity.

An image ellipse is a projective image of the known circle, not its metric
centre. Current named lines and held-out conic pixels determine hypotheses.
No physical camera translation or global pixels-to-metres scale is asserted.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from .pitch_geometry import PitchGeometry

G=PitchGeometry()

def points(h,p):return cv2.perspectiveTransform(np.asarray(p,float)[None],np.asarray(h,float))[0]

def handedness(h,p):
 h=np.asarray(h);den=float(h[2]@[*p,1]);return int(np.sign(np.linalg.det(h)/den**3)) if abs(den)>1e-9 else 0

def conic_fit(raw,width,height):
 raw=np.asarray(raw,float);record={'accepted':False,'reason':'NO_CIRCLE_SUPPORT','points':len(raw)}
 if len(raw)<40:return None,record
 image=raw*[width,height];train=image[::2].astype(np.float32)
 try:ellipse=cv2.fitEllipseAMS(train)
 except cv2.error:return None,{**record,'reason':'ELLIPSE_FIT_FAILED'}
 center,axes,angle=ellipse;axes=np.asarray(axes)/2
 if not np.isfinite([*center,*axes,angle]).all() or min(axes)<3 or max(axes)>max(width,height)*4:return None,{**record,'reason':'DEGENERATE_CONIC'}
 theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);a=np.linalg.inv(rot@np.diag(axes));affine=np.eye(3);affine[:2,:2]=a;affine[:2,2]=-a@center
 unit=points(affine,image);angles=np.sort(np.mod(np.arctan2(unit[:,1],unit[:,0]),2*np.pi));span=float(np.degrees(2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()))
 q=affine.T@np.diag([1,1,-1])@affine;p=np.column_stack([image[1::2],np.ones(len(image[1::2]))]);gradient=2*(p@q)[:,:2];error=np.abs(np.sum((p@q)*p,axis=1))/np.maximum(np.linalg.norm(gradient,axis=1),1e-9)
 accepted=bool(span>=120 and np.median(error)<=10 and np.quantile(error,.9)<=20)
 record.update(accepted=accepted,reason='BROAD_HELDOUT_CONIC' if accepted else 'SHORT_ARC_OR_CONIC_RESIDUAL_REJECTED',matrix=q.tolist(),ellipse_center_image=list(map(float,center)),ellipse_axes=list(map(float,axes*2)),ellipse_angle_degrees=float(angle),arc_span_degrees=span,heldout_median_px=float(np.median(error)),heldout_p90_px=float(np.quantile(error,.9)))
 return affine if accepted else None,record

def pixel_statistics(h,groups,width,height):
 residual,names=G.residuals(h,groups,width,height,True);stats={}
 for kind in ['lines','circles']:
  values=np.abs(residual[[n.startswith('Circle')==(kind=='circles') for n in names]])
  if len(values):stats[kind]={'mean':float(values.mean()),'p90':float(np.quantile(values,.9))}
 return bool(stats and all(np.isfinite(list(s.values())).all() and s['mean']<=10 and s['p90']<=20 for s in stats.values())),stats

def solve_midfield(groups,width,height,orientation=None,initial_h=None):
 affine,conic=conic_fit(groups.get('Circle central',[]),width,height);record={'kind':'CENTER_CIRCLE','accepted':False,'conic':conic,'candidates':[],'orientation_constraint':orientation,'reason':'INSUFFICIENT_CURRENT_LANDMARKS'}
 if affine is None or 'Middle line' not in groups or 'Side line top' not in groups:return None,record
 # Align the ellipse-normalized halfway direction with canonical y. The
 # affine initialization is only a starting guess; projective fitting follows.
 mid=points(affine,np.asarray(groups['Middle line'])*[width,height]);vx,vy,*_=cv2.fitLine(mid.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();rotation=np.eye(3);rotation[:2,:2]=[[vy,-vx],[vx,vy]]
 normalizer=np.array([[width,0,width/2],[0,height,height/2],[0,0,1.]])
 world=np.array([[9.15,0,52.5],[0,9.15,34],[0,0,1.]])
 candidates=[];all_groups={k:v for k,v in groups.items() if k in G.segments or k in G.circle_centers}
 inv_normalizer=np.linalg.inv(normalizer)
 _,names=G.residuals(np.eye(3),all_groups,width,height);weights=np.array([1/np.sqrt(max(1,names.count(name))) for name in names])
 _,held_names=G.residuals(np.eye(3),all_groups,width,height,True);held_weights=np.array([1/np.sqrt(max(1,held_names.count(name))) for name in held_names])
 starts=[]
 for sx,sy,jitter in [(sx,sy,j) for sx,sy in [(1,1),(-1,1),(1,-1),(-1,-1)] for j in [0,.001]]:
  initial=world@np.diag([sx,sy,1])@rotation@affine@normalizer;initial/=initial[2,2];x0=initial.flat[:8].copy();x0[2]+=jitter;starts.append((x0,'CURRENT_ELLIPSE'))
 if initial_h is not None:
  initial=np.asarray(initial_h)@normalizer;initial/=initial[2,2]
  for jitter in [0,.001]:
   x0=initial.flat[:8].copy();x0[2]+=jitter;starts.append((x0,'MEASURED_MOTION_INITIALIZATION'))
 for x0,initialization in starts:
  def objective(parameters,heldout=False):
   h=np.r_[parameters,1.].reshape(3,3)@inv_normalizer;residual,_=G.residuals(h,all_groups,width,height,heldout);return residual*(held_weights if heldout else weights)
  try:
   fit=least_squares(objective,x0,loss='soft_l1',f_scale=2,max_nfev=180);h=np.r_[fit.x,1].reshape(3,3)@np.linalg.inv(normalizer);valid,stats=pixel_statistics(h,all_groups,width,height);sign=handedness(h,np.mean(np.asarray(groups['Circle central'])*[width,height],axis=0));image_pitch=points(np.linalg.inv(h),[[0,0],[105,0],[105,68],[0,68]])
   inv=np.linalg.inv(h);depth=np.array([[0,0,1],[105,0,1],[105,68,1],[0,68,1]])@inv[2]
   safe=bool(np.isfinite(image_pitch).all() and np.all(np.sign(depth)==np.sign(depth[0])) and abs(cv2.contourArea(image_pitch.astype(np.float32)))>100)
   valid=bool(valid and safe and (orientation is None or sign==orientation));cost=float(np.mean(np.square(objective(fit.x,True))))
   entry={'initialization':initialization,'accepted':valid,'transform':h.tolist(),'handedness':sign,'heldout':stats,'score':cost,'reason':'HELDOUT_LANDMARKS_AGREE' if valid else 'RESIDUAL_OR_HORIZON_OR_ORIENTATION_REJECTED'};record['candidates'].append(entry)
   if valid:candidates.append((h,cost,sign))
  except (ValueError,np.linalg.LinAlgError,cv2.error):record['candidates'].append({'accepted':False,'reason':'OPTIMIZATION_FAILED'})
 if not candidates:record['reason']='NO_VALID_LANDMARK_HYPOTHESIS';return None,record
 if orientation is None and len({c[2] for c in candidates})>1:
  record['reason']='UNRESOLVED_LEFT_RIGHT_REFLECTION';return None,record
 # Agree on field-interior queries, not arbitrary off-pitch sky pixels.
 query=np.asarray(groups['Circle central'])*[width,height];query=query[np.linspace(0,len(query)-1,min(40,len(query)),dtype=int)]
 best=min(candidates,key=lambda c:c[1]);mapped=np.array([points(c[0],query) for c in candidates]);spread=float(np.quantile(np.linalg.norm(mapped-mapped[0],axis=2),.9));record['hypothesis_spread_p90_m']=spread
 if len(candidates)<2 or spread>1:record['reason']='UNSTABLE_LANDMARK_HYPOTHESES';return None,record
 record.update(accepted=True,reason='CURRENT_CONIC_HALFWAY_TOUCHLINE_RELOCALIZED',selected_transform=best[0].tolist(),score=best[1]);return best[0],record


def solve_penalty(groups,width,height,orientation=None):
 """Absolute named-line resection; intersections are correlated line evidence."""
 names=[n for n in groups if n in G.segments];families=[[n for n in names if G.segments[n][0][axis]==G.segments[n][1][axis]] for axis in [0,1]];record={'accepted':False,'kind':'PENALTY_LINES','reason':'INSUFFICIENT_INDEPENDENT_LINE_FAMILIES','candidates':[],'line_families':list(map(len,families))}
 if min(map(len,families))<2 or not any('rect.' in n for n in names):return None,record
 solutions=[]
 for parity in [0,1]:
  lines={}
  for name in names:
   p=np.asarray(groups[name])[parity::2]*[width,height];vx,vy,x,y=cv2.fitLine(p.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();lines[name]=np.array([-vy,vx,vy*x-vx*y])
  image=[];world=[]
  for a in families[0]:
   for b in families[1]:
    v=np.cross(lines[a],lines[b]);xy=np.array([G.segments[a][0][0],G.segments[b][0][1]])
    if abs(v[2])<1e-8:continue
    p=v[:2]/v[2]
    if not (-width*.25<=p[0]<=width*1.25 and -height*.25<=p[1]<=height*1.25):continue
    image.append(p);world.append(xy)
  if len(image)<4:continue
  image=np.array(image,float);world=np.array(world,float)
  if cv2.contourArea(cv2.convexHull(image.astype(np.float32)))<width*height*.02:continue
  h,_=cv2.findHomography(image,world,0)
  if h is None:continue
  valid,stats=pixel_statistics(h,groups,width,height);sign=handedness(h,np.mean(image,axis=0));inv=np.linalg.inv(h);depth=np.array([[0,0,1],[105,0,1],[105,68,1],[0,68,1]])@inv[2]
  valid=bool(valid and np.all(np.sign(depth)==np.sign(depth[0])) and (orientation is None or orientation==sign));record['candidates'].append({'accepted':valid,'transform':h.tolist(),'heldout':stats,'derived_intersections':len(image),'reason':'CURRENT_FIXED_FIELD_LINES_VALIDATED' if valid else 'HELDOUT_OR_HORIZON_OR_ORIENTATION_REJECTED'})
  if valid:solutions.append(h)
 if len(solutions)<2:record['reason']='LINE_HYPOTHESES_REJECTED';return None,record
 sample=np.vstack([np.asarray(groups[n])*[width,height] for n in names]);sample=sample[np.linspace(0,len(sample)-1,40,dtype=int)];spread=float(np.quantile(np.linalg.norm(points(solutions[0],sample)-points(solutions[1],sample),axis=1),.9))
 if spread>1:record['reason']='UNSTABLE_LINE_RESECTION';return None,record
 record.update(accepted=True,reason='ABSOLUTE_NAMED_PENALTY_LINE_RELOCALIZATION',hypothesis_spread_p90_m=spread,selected_transform=solutions[0].tolist());return solutions[0],record
