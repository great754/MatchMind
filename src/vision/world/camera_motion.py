"""Measured projective inter-frame motion; held-out feature residuals are not accuracy."""
import cv2
import numpy as np

def projective_motion(previous, current):
    result={'accepted':False,'reason':'INSUFFICIENT_FEATURE_MATCHES','matches':0}
    points=cv2.goodFeaturesToTrack(previous,250,.02,8)
    if points is None:return result
    nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None)
    if nxt is None:return result
    back,reverse,_=cv2.calcOpticalFlowPyrLK(current,previous,nxt,None)
    if back is None:return result
    good=(status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.)
    a=points[good,0];b=nxt[good,0];result['matches']=len(a)
    if len(a)<50:return result
    # Deterministic disjoint spatial feature samples; held-out points never fit H.
    hold=np.arange(len(a))%4==0
    h,inliers=cv2.findHomography(a[~hold],b[~hold],cv2.RANSAC,1.)
    if h is None or inliers is None:result['reason']='MOTION_OPTIMIZATION_FAILED';return result
    ratio=float(inliers.mean());coverage=float(cv2.contourArea(cv2.convexHull(a[~hold][inliers[:,0]>0]))/(previous.shape[0]*previous.shape[1]))
    predicted=cv2.perspectiveTransform(a[hold][None].astype(float),h)[0]
    errors=np.linalg.norm(predicted-b[hold],axis=1)
    result.update(inlier_ratio=ratio,feature_hull_fraction=coverage,heldout_median_small_px=float(np.median(errors)),heldout_p90_small_px=float(np.quantile(errors,.9)))
    if ratio<.8 or coverage<.15 or not np.isfinite(errors).all() or np.median(errors)>1 or np.quantile(errors,.9)>2:
        result['reason']='TEMPORAL_TRACKING_FAILED';return result
    # A projective model can explain pan; substantial local scale changes are
    # deliberately unsupported until focal/zoom behavior is independently tested.
    ys,xs=previous.shape;grid=np.array([(x,y) for y in (.2*ys,.5*ys,.8*ys) for x in (.2*xs,.5*xs,.8*xs)],float)
    corners=np.array([[0,0],[xs,0],[xs,ys],[0,ys]],float);v=np.column_stack([corners,np.ones(4)])@h.T
    if not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-10 or np.any(np.abs(v[:,2])<1e-8) or np.any(np.sign(v[:,2])!=np.sign(v[0,2])):
        result['reason']='IMPOSSIBLE_MOTION_TRANSFORM';return result
    base=cv2.perspectiveTransform(grid[None],h)[0]
    dx=cv2.perspectiveTransform((grid+[1,0])[None],h)[0]-base
    dy=cv2.perspectiveTransform((grid+[0,1])[None],h)[0]-base
    scales=np.linalg.svd(np.stack([dx,dy],axis=-1),compute_uv=False)
    result.update(min_local_scale=float(scales.min()),max_local_scale=float(scales.max()))
    if not np.isfinite(scales).all() or scales.min()<.98 or scales.max()>1.02:
        result['reason']='ZOOM_OR_SCALE_CHANGE_UNSUPPORTED';return result
    result.update(accepted=True,reason='MEASURED_PROJECTIVE_MOTION',homography=h.tolist())
    return result
