"""Pitch-only inter-image motion. People mask excludes moving pixels, not metric priors."""
import cv2,numpy as np

def field_image(image,boxes=()):
    height,width=image.shape[:2];scale=min(1.,640/width);small=cv2.resize(image,None,fx=scale,fy=scale);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);grass=cv2.inRange(hsv,(25,35,25),(95,255,255));components,labels,stats,_=cv2.connectedComponentsWithStats(grass)
    keep=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=grass.size*.01);keep=keep[keep!=0]
    grass=np.isin(labels,keep).astype(np.uint8)*255
    mask=cv2.morphologyEx(grass,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8));mask=cv2.erode(mask,np.ones((5,5),np.uint8))
    for x,y,w,h in boxes:
        a=(max(0,int((x-.1*w)*scale)),max(0,int((y-.1*h)*scale)));b=(min(gray.shape[1]-1,int((x+1.1*w)*scale)),min(gray.shape[0]-1,int((y+1.1*h)*scale)));cv2.rectangle(mask,a,b,0,-1)
    return gray,mask,scale

def measure_field_motion(previous,current):
    before,mask,scale=previous;after,current_mask,current_scale=current
    result={'accepted':False,'reason':'INSUFFICIENT_FIELD_FEATURES','matches':0,'scale_to_source':1/scale}
    if before.shape!=after.shape or scale!=current_scale:return result
    points=cv2.goodFeaturesToTrack(before,800,.001,5,mask=mask)
    if points is None:return result
    nxt,status,_=cv2.calcOpticalFlowPyrLK(before,after,points,None,winSize=(21,21),maxLevel=3,minEigThreshold=.001)
    if nxt is None:return result
    back,reverse,_=cv2.calcOpticalFlowPyrLK(after,before,nxt,None,winSize=(21,21),maxLevel=3,minEigThreshold=.001)
    if back is None:return result
    good=(status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1)
    xy=np.round(nxt[:,0]).astype(int);inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<after.shape[1])&(xy[:,1]<after.shape[0]);supported=np.zeros(len(points),bool);supported[inside]=current_mask[xy[inside,1],xy[inside,0]]>0;good &= supported
    a=points[good,0];b=nxt[good,0]
    return validate_motion(a,b,previous,current)

def validate_motion(a,b,previous,current,hold=None,interval_frames=1):
    before,mask,scale=previous;after,current_mask,current_scale=current
    result={'accepted':False,'reason':'INSUFFICIENT_FIELD_FEATURES','matches':0,'scale_to_source':1/scale}
    result['matches']=len(a)
    if len(a)<40:return result
    hold=np.arange(len(a))%4==0 if hold is None else hold
    if hold.sum()<10 or (~hold).sum()<30:return result
    candidates=[]
    for kind in ('similarity','affine','projective'):
        if kind!='similarity' and len(a)<50:continue
        if kind=='projective':h,inliers=cv2.findHomography(a[~hold],b[~hold],cv2.RANSAC,1.)
        else:
            estimator=cv2.estimateAffinePartial2D if kind=='similarity' else cv2.estimateAffine2D
            affine,inliers=estimator(a[~hold],b[~hold],method=cv2.RANSAC,ransacReprojThreshold=1.)
            h=np.vstack([affine,[0,0,1]]) if affine is not None else None
        candidate=assess_motion(h,inliers,a,b,hold,previous,current,kind,interval_frames);candidate['model']=kind;candidates.append(candidate)
    accepted=[r for r in candidates if r['accepted']]
    # Smallest supported model, all subject to identical validation gates.
    chosen=(accepted or candidates)[0]
    chosen['models']=[{k:v for k,v in c.items() if k not in ('homography','models')} for c in candidates]
    return chosen

def assess_motion(h,inliers,a,b,hold,previous,current,model="projective",interval_frames=1):
    before,mask,scale=previous;after,current_mask,current_scale=current
    result={'accepted':False,'reason':'FIELD_MOTION_FIT_FAILED','matches':len(a),'scale_to_source':1/scale}
    if h is None or inliers is None:result['reason']='FIELD_MOTION_FIT_FAILED';return result
    if not np.isfinite(h).all() or abs(h[2,2])<1e-10 or np.linalg.det(h/h[2,2])<=1e-10:result['reason']='TRANSFORM_FOLDING';return result
    residual=np.linalg.norm(cv2.perspectiveTransform(a[hold][None].astype(float),h)[0]-b[hold],axis=1)
    if not np.isfinite(residual).all():result['reason']='FIELD_MOTION_NONFINITE';return result
    ratio=float(inliers.mean());hull=float(cv2.contourArea(cv2.convexHull(a[~hold][inliers[:,0]>0]))/before.size)
    result.update(inlier_ratio=ratio,feature_hull_fraction=hull,heldout_median_small_px=float(np.median(residual)),heldout_p90_small_px=float(np.quantile(residual,.9)))
    spread_valid=hull>=.15
    if model in ('similarity','affine'):
        # A line of widely separated field points can constrain similarity (4
        # DOF), but cannot determine projective perspective (8 DOF). Validate
        # actual model rank and extrapolation rather than reusing a hull rule.
        height,width=before.shape;normalizer=max(width,height);train=(a[~hold][inliers[:,0]>0]-[width/2,height/2])/normalizer
        def design(points):
            x,y=points[:,0],points[:,1];one=np.ones(len(points));zero=np.zeros(len(points))
            if model=='similarity':return np.stack([np.column_stack([x,-y,one,zero]),np.column_stack([y,x,zero,one])],axis=1).reshape(-1,4)
            return np.stack([np.column_stack([x,y,one,zero,zero,zero]),np.column_stack([zero,zero,zero,x,y,one])],axis=1).reshape(-1,6)
        matrix=design(train);singular=np.linalg.svd(matrix,compute_uv=False);condition=float(singular[0]/max(singular[-1],1e-12))
        queries=np.array([(x,y) for y in (-height/2,0,height/2) for x in (-width/2,0,width/2)])/normalizer
        amplification=float(np.abs(design(queries)@np.linalg.pinv(matrix)).sum(axis=1).max());extrapolation=float(amplification*np.quantile(residual,.9))
        result.update(design_condition=condition,extrapolated_pixel_error_diagnostic=extrapolation)
        spread_valid=condition<=100 and extrapolation<=2
    if ratio<.8 or not spread_valid or np.median(residual)>1 or np.quantile(residual,.9)>2:result['reason']='FIELD_MOTION_VALIDATION_FAILED';return result
    height,width=before.shape;corners=np.array([[0,0],[width,0],[width,height],[0,height]],float);v=np.column_stack([corners,np.ones(4)])@h.T
    if np.any(np.abs(v[:,2])<1e-8) or np.any(np.sign(v[:,2])!=np.sign(v[0,2])):result['reason']='TRANSFORM_FOLDING';return result
    grid=np.array([(x,y) for y in (.2*height,.5*height,.8*height) for x in (.2*width,.5*width,.8*width)],float);p=cv2.perspectiveTransform(grid[None],h)[0];dx=cv2.perspectiveTransform((grid+[1,0])[None],h)[0]-p;dy=cv2.perspectiveTransform((grid+[0,1])[None],h)[0]-p;scales=np.linalg.svd(np.stack([dx,dy],axis=-1),compute_uv=False)
    result.update(min_local_scale=float(scales.min()),max_local_scale=float(scales.max()),pan_small_px=np.median(p-grid,axis=0).tolist())
    per_frame=scales**(1/max(1,interval_frames));result['interval_frames']=interval_frames
    if not np.isfinite(scales).all() or per_frame.min()<.98 or per_frame.max()>1.02:result['reason']='ZOOM_UNSUPPORTED';return result
    s=np.diag([1/scale,1/scale,1]);source=s@h@np.linalg.inv(s);warped=cv2.warpPerspective(mask,h,(width,height));overlap=np.count_nonzero((warped>0)&(current_mask>0))/max(1,min(np.count_nonzero(warped),np.count_nonzero(current_mask)))
    if interval_frames>1:
        # Compare only reobserved/co-visible field support, not a demand that
        # a panned camera retains the same entire viewport.
        region=np.zeros_like(mask);cv2.fillConvexPoly(region,cv2.convexHull(a.astype(np.float32)).astype(int),255)
        visible=cv2.warpPerspective(region,h,(width,height))>0
        overlap=np.count_nonzero((warped>0)&(current_mask>0)&visible)/max(1,np.count_nonzero((warped>0)&visible))
    result.update(accepted=True,reason='VALIDATED_PITCH_FEATURE_MOTION',homography=source.tolist(),field_overlap=float(overlap))
    if overlap<.8:result.update(accepted=False,reason='FIELD_APPEARANCE_CHANGED')
    return result

class FieldMotionTracker:
    """Persistent field features; current held-out IDs were supported BEFORE fitting.

    Past consensus determines static membership, not the current held-out fit.
    New features mature for three observations. No metric player locations used.
    """
    def __init__(self,debug=False):self.debug=debug;self.measurements=None;self.previous=None;self.points=None;self.ids=np.array([],int);self.ages=np.array([],int);self.static=np.array([],bool);self.next_id=0
    def update(self,current):
        self.measurements=None
        result={'accepted':False,'reason':'FIRST_FRAME','matches':0}
        if self.previous is not None and self.points is not None and len(self.points):
            before,_,_=self.previous;after,mask,_=current
            nxt,status,_=cv2.calcOpticalFlowPyrLK(before,after,self.points,None,winSize=(21,21),maxLevel=3,minEigThreshold=.001)
            back,reverse,_=cv2.calcOpticalFlowPyrLK(after,before,nxt,None,winSize=(21,21),maxLevel=3,minEigThreshold=.001) if nxt is not None else (None,None,None)
            if back is not None:
                xy=np.round(nxt[:,0]).astype(int);inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<after.shape[1])&(xy[:,1]<after.shape[0]);supported=np.zeros(len(xy),bool);supported[inside]=mask[xy[inside,1],xy[inside,0]]>0
                good=(status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-self.points[:,0],axis=1)<1)&supported
                a=self.points[good,0];b=nxt[good,0];ids=self.ids[good];ages=self.ages[good]+1;eligible=self.static[good]&(ages>=2)
                self.measurements={'points':b[eligible].copy(),'ids':ids[eligible].copy()}
                result=validate_motion(a[eligible],b[eligible],self.previous,current,hold=(ids[eligible]%4==0));result['mature_static_features']=int(eligible.sum())
                if self.debug and 'homography' in result:
                    source_scale=self.previous[2];aa=a[eligible]/source_scale;bb=b[eligible]/source_scale;pred=cv2.perspectiveTransform(aa[None].astype(float),np.array(result['homography']))[0];errors=np.linalg.norm(pred-bb,axis=1)*source_scale;selected=np.linspace(0,len(aa)-1,min(64,len(aa)),dtype=int)
                    result['vectors']=[{'previous':aa[j].tolist(),'current':bb[j].tolist(),'inlier':bool(errors[j]<=1),'heldout':bool(ids[eligible][j]%4==0),'kind':'mature_static_pitch'} for j in selected]
                    result['lk_forward_backward_p90_small_px']=float(np.quantile(np.linalg.norm(back[good,0]-self.points[good,0],axis=1)[eligible],.9))
                    result['ransac_inliers']=round(result.get('inlier_ratio',0)*int((ids[eligible]%4!=0).sum()));result['working_width']=current[0].shape[1]

                stable=np.zeros(len(a),bool)
                if len(a)>=12:
                    h,inliers=cv2.findHomography(a,b,cv2.RANSAC,1.)
                    if h is not None and np.isfinite(h).all() and abs(np.linalg.det(h))>1e-10:
                        predicted=cv2.perspectiveTransform(a[None].astype(float),h)[0];stable=np.linalg.norm(predicted-b,axis=1)<1
                self.points=b[:,None].astype(np.float32);self.ids=ids;self.ages=ages;self.static=stable
            else:self.points=None;self.ids=np.array([],int);self.ages=np.array([],int);self.static=np.array([],bool)
        # Replenish without resetting stable feature identities or their held-out split.
        mask=current[1].copy()
        if self.points is not None:
            for x,y in self.points[:,0]:cv2.circle(mask,(int(round(x)),int(round(y))),6,0,-1)
        count=len(self.ids);new=cv2.goodFeaturesToTrack(current[0],max(1,800-count),.001,5,mask=mask) if count<700 else None
        if new is not None:
            n=len(new);self.points=np.concatenate([self.points,new]) if self.points is not None else new;self.ids=np.concatenate([self.ids,np.arange(self.next_id,self.next_id+n)]);self.ages=np.concatenate([self.ages,np.zeros(n,int)]);self.static=np.concatenate([self.static,np.zeros(n,bool)]);self.next_id+=n
        self.previous=current
        return result
