"""Reobserve static field-feature IDs against accepted camera keyframes.

These are inherited camera anchors, NOT independent metric landmarks. Unknown
frames never add world anchors; no player world coordinates are used in fitting.
"""
import cv2,numpy as np
from .field_motion import validate_motion
from .pose import visible_region

class TextureAnchors:
    def __init__(self):self.keys=[]
    def add(self,frame,epoch,estimate,field,measurements,direct=False):
        if estimate is None or estimate.epoch!=epoch or measurements is None or len(measurements['ids'])<40:return
        if self.keys and not direct and frame-self.keys[-1]['frame']<8:return
        self.keys.append({'frame':frame,'epoch':epoch,'estimate':estimate,'field':field,'ids':measurements['ids'].copy(),'points':measurements['points'].copy()});self.keys=self.keys[-8:]
    def propose(self,frame,epoch,field,measurements,motion,width,height):
        if measurements is None or motion.get('reason') in ('ZOOM_UNSUPPORTED','TRANSFORM_FOLDING'):return None,{'reason':'NO_CURRENT_TEXTURE_MEASUREMENT'}
        proposals=[];rejected=[]
        for key in reversed(self.keys):
            if key['epoch']!=epoch or frame<=key['frame']:continue
            ids,here,there=np.intersect1d(measurements['ids'],key['ids'],return_indices=True)
            if len(ids)<40:continue
            measured=validate_motion(measurements['points'][here],key['points'][there],field,key['field'],hold=ids%4==0,interval_frames=frame-key['frame'])
            if not measured['accepted']:rejected.append({'anchor_frame':key['frame'],'reason':measured['reason'],'matches':len(ids)});continue
            candidate=np.array(key['estimate'].transform).reshape(3,3)@np.array(measured['homography'])
            if visible_region(candidate,width,height) is None:continue
            proposals.append({'homography':candidate,'motion':measured,'anchor':key['estimate'],'anchor_frame':key['frame'],'current_points':measurements['points'][here]/field[2]})
        diagnostic={'validated_anchor_candidates':len(proposals),'rejected_anchor_candidates':rejected}
        if not proposals:return None,{**diagnostic,'reason':'NO_REOBSERVED_FIELD_ANCHOR'}
        # Distinct anchors must agree on currently reobserved field points.
        points=proposals[0]['current_points'][None].astype(float);mapped=np.array([cv2.perspectiveTransform(points,p['homography'])[0] for p in proposals]);spread=np.linalg.norm(mapped-mapped[0],axis=2)
        disagreement=float(np.quantile(spread,.9)) if np.isfinite(spread).all() else None
        diagnostic['anchor_disagreement_p90_m']=disagreement
        if disagreement is None or disagreement>4:return None,{**diagnostic,'reason':'AMBIGUOUS_TEXTURE_ANCHORS'}
        chosen=proposals[0];return chosen,{**diagnostic,'reason':'FRESH_REOBSERVED_FIELD_ANCHOR','anchor_frame':chosen['anchor_frame'],'anchor_age_frames':frame-chosen['anchor_frame'],'matches':chosen['motion']['matches'],'heldout_p90_small_px':chosen['motion']['heldout_p90_small_px'],'motion_model':chosen['motion']['model']}
