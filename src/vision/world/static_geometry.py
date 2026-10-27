"""Current-to-current pitch-only measurements, including named field lines.

Line endpoints are NOT point correspondences: the visible line segment changes.
We compare transported infinite lines and sample actual field points with LK.
"""
import cv2
import numpy as np
from .field_motion import validate_motion
from .pitch_geometry import PitchGeometry


def line_measurements(before,after,h,width,height,scale=1.):
    result=[];geometry=PitchGeometry()
    for name in sorted(before.keys() & after.keys() & geometry.segments.keys()):
        a=np.asarray(before[name])*[width,height];b=np.asarray(after[name])*[width,height]
        if len(a)<12 or len(b)<12:continue
        def fit(points):
            vx,vy,x,y=cv2.fitLine(points.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();return np.array([-vy,vx,vy*x-vx*y]),np.array([vx,vy]),np.array([x,y])
        old,_,_=fit(a);current,direction,center=fit(b)
        predicted=np.linalg.inv(h).T@old;predicted/=max(np.linalg.norm(predicted[:2]),1e-12)
        normal=current[:2]/np.linalg.norm(current[:2]);distance=(np.column_stack([b,np.ones(len(b))])@current)/np.linalg.norm(current[:2]);centerline=b-distance[:,None]*normal
        errors=np.abs(np.column_stack([centerline,np.ones(len(b))])@predicted)*scale
        before_direction=np.array([-predicted[1],predicted[0]]);angle=float(np.degrees(np.arccos(np.clip(abs(before_direction@direction),0,1))))
        # Project segment extremes onto each fitted line; draw measured support.
        def segment(points,line):
            normal=line[:2]/max(np.linalg.norm(line[:2]),1e-12);tangent=np.array([-normal[1],normal[0]]);middle=points.mean(axis=0);middle-=normal*(middle@normal+line[2]/max(np.linalg.norm(line[:2]),1e-12));positions=points@tangent;return [list(middle+tangent*(np.quantile(positions,q)-middle@tangent)) for q in (.05,.95)]
        result.append({'kind':name,'previous':segment(a,old),'current':segment(b,current),'median_normal_error_small_px':float(np.median(errors)),'p90_normal_error_small_px':float(np.quantile(errors,.9)),'rotation_error_degrees':angle,'line_normal_translation_source_px':float((np.r_[center,1]@old)/max(np.linalg.norm(old[:2]),1e-9)),'delta_y_at_center_x_source_px':float(-(current[0]*width/2+current[2])/current[1]+(old[0]*width/2+old[2])/old[1]) if abs(current[1])>.2 and abs(old[1])>.2 else None,'accepted':bool(np.median(errors)<=2 and np.quantile(errors,.9)<=4 and angle<=2)})
    return result


def measure(previous,current,before_groups=None,after_groups=None,width=None,height=None):
    before,mask,scale=previous;after,current_mask,next_scale=current
    result={'accepted':False,'reason':'INSUFFICIENT_FIELD_FEATURES','matches':0,'scale_to_source':1/scale,'vectors':[],'lines':[]}
    if before.shape!=after.shape or scale!=next_scale:return result
    pts=cv2.goodFeaturesToTrack(before,800,.001,5,mask=mask)
    if pts is None:return result
    nxt,status,_=cv2.calcOpticalFlowPyrLK(before,after,pts,None,winSize=(21,21),maxLevel=3,minEigThreshold=.001)
    if nxt is None:return result
    back,reverse,_=cv2.calcOpticalFlowPyrLK(after,before,nxt,None,winSize=(21,21),maxLevel=3,minEigThreshold=.001)
    if back is None:return result
    fb=np.linalg.norm(back[:,0]-pts[:,0],axis=1);xy=np.round(nxt[:,0]).astype(int);inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<after.shape[1])&(xy[:,1]<after.shape[0]);supported=np.zeros(len(pts),bool);supported[inside]=current_mask[xy[inside,1],xy[inside,0]]>0
    good=(status[:,0]>0)&(reverse[:,0]>0)&(fb<1)&supported;a=pts[good,0];b=nxt[good,0]
    measured=validate_motion(a,b,previous,current);result.update(measured)
    if len(a):result['lk_forward_backward_p90_small_px']=float(np.quantile(fb[good],.9))
    if 'homography' in result:
        h=np.array(result['homography']);pred=cv2.perspectiveTransform((a/scale)[None].astype(float),h)[0];errors=np.linalg.norm(pred-b/scale,axis=1)*scale
        ids=np.linspace(0,len(a)-1,min(64,len(a)),dtype=int)
        result['vectors']=[{'previous':(a[i]/scale).tolist(),'current':(b[i]/scale).tolist(),'inlier':bool(errors[i]<=1),'heldout':bool(i%4==0),'kind':'static_pitch'} for i in ids]
        if before_groups is not None and after_groups is not None:
            result['lines']=line_measurements(before_groups,after_groups,h,width,height,scale)
            good_lines=sum(l['accepted'] for l in result['lines']);result['tracked_line_count']=good_lines;result['top_touchline_tracked']=any(l['kind']=='Side line top' and l['accepted'] for l in result['lines'])
            # Named field markings independently validate the point fit. A single
            # line never determines full pose or tangential image translation.
            if len(result['lines'])>=2 and good_lines<len(result['lines'])/2:result.update(accepted=False,reason='STATIC_LINES_CONTRADICT_MOTION')
    result['ransac_inliers']=round(result.get('inlier_ratio',0)*max(0,len(a)-(len(a)+3)//4))
    result['working_width']=before.shape[1];return result


def with_lines(motion,before_groups,after_groups,width,height):
    result=dict(motion)
    if 'homography' not in result:return result
    result['lines']=line_measurements(before_groups,after_groups,np.array(result['homography']),width,height,1/result.get('scale_to_source',1))
    count=sum(l['accepted'] for l in result['lines']);result['tracked_line_count']=count;result['top_touchline_tracked']=any(l['kind']=='Side line top' and l['accepted'] for l in result['lines'])
    if len(result['lines'])>=2 and count<len(result['lines'])/2:result.update(accepted=False,reason='STATIC_LINES_CONTRADICT_MOTION')
    return result
