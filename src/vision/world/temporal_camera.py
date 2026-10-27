"""M6 measured prediction/correction and bounded offline two-sided validation.

No coefficient averaging, stale copies, player-derived pose or cross-epoch gaps.
"""
import cv2
import numpy as np
from .persistent_camera import PersistentCamera,safe_region
from .pitch_geometry import PitchGeometry
from .texture_anchors import TextureAnchors
from src.observations.schema import PitchCalibrationObservation,Provenance
from .schema import ImagePitchSegment,CameraMemory,CameraEstimate,CameraFailure

MAX_GAP_FRAMES=2  # observed isolated gaps are one frame; test only one/two, never long gaps
MAX_GAP_SECONDS=.10
MAX_TEMPORAL_DISAGREEMENT_M=2.
MAX_BIDIRECTIONAL_DISAGREEMENT_M=1.


def transform(camera):return np.array(camera.calibration.image_to_pitch).reshape(3,3) if camera.calibration_valid else None


def queries(width,height):return np.array([[(x*width,y*height) for x in (.2,.5,.8) for y in (.2,.5,.8)]],float)


def disagreement(a,b,width,height):
    q=queries(width,height);p=cv2.perspectiveTransform(q,a);r=cv2.perspectiveTransform(q,b);values=np.linalg.norm(p-r,axis=2)
    return float(np.quantile(values,.9)) if np.isfinite(values).all() else float('inf')


def strongly_observed(h,groups,width,height):
    """Independent reanchor requires two line families or broad midfield support.

    A tiny observed circle arc is not permission to reset global field geometry.
    This is an observability safeguard, not a calibrated accuracy guarantee.
    """
    g=PitchGeometry();families=[0,0];spans={}
    for name in groups.keys() & g.segments.keys():
        a,b=g.segments[name];families[0 if a[0]==b[0] else 1]+=1
    for name,center in g.circle_centers.items():
        if name not in groups:continue
        p=cv2.perspectiveTransform((groups[name]*[width,height])[None].astype(float),h)[0]-center;angles=np.sort(np.mod(np.arctan2(p[:,1],p[:,0]),2*np.pi));span=2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max();spans[name]=float(np.degrees(span))
    strong=min(families)>=2 or (spans.get('Circle central',0)>=120 and 'Middle line' in groups and ('Side line top' in groups or 'Side line bottom' in groups))
    return bool(strong),{'line_families':families,'circle_span_degrees':spans}


def compatible_path(cameras,start,end,fps):
    if abs(end-start)>MAX_GAP_FRAMES+1 or abs(end-start)/fps>MAX_GAP_SECONDS+1/fps:return False
    lo,hi=sorted((start,end));epoch=cameras[start].epoch
    return all(cameras[i].epoch==epoch and (i==lo or cameras[i].continuity!='cut') for i in range(lo,hi+1))


class TemporalCameraFilter:
    def __init__(self,width,height,fps):
        self.width=width;self.height=height;self.fps=fps;self.validator=PersistentCamera(width,height);self.geometry=PitchGeometry()
    def candidate(self,h,groups,allow_weak=False):
        if safe_region(h,self.width,self.height) is None:return None,{'reason':'VISIBLE_REGION_INVALID'}
        if len(groups)<2:
            if not allow_weak:return None,{'reason':'INSUFFICIENT_CURRENT_CONSTRAINTS'}
            if groups and not self.validator.statistics(h,groups)['accepted']:return None,{'reason':'CURRENT_WEAK_GEOMETRY_CONTRADICTS_PREDICTION'}
            return h,{'reason':'MEASURED_FIELD_ONLY_REQUIRES_TWO_DIRECTIONS','current_semantic_support':bool(groups)}
        return self.validator.refine(h,groups)
    def publish(self,base,h,state,reason,debug,age=None):
        loc=dict(base.motion_evidence['localization']);valid=h is not None
        stats=self.validator.statistics(h,debug['groups']) if valid else {};groups=debug.pop('groups')
        loc.update(relative_motion_error_budget_px=debug.get('relative_motion_error_budget_px') if valid else None,status=state,outcome=reason,transition='relocalization' if state=='RELOCALIZED' else 'tracking' if state.startswith('TEMPORAL') else 'partial_constraint' if state in ('PARTIAL_CONSTRAINED','BIDIRECTIONAL_INTERPOLATED') else 'acquisition' if state=='DIRECT' else 'loss',tracking_blockers=[] if valid else [reason],last_absolute_age_seconds=age,current_semantic_support=bool(groups) and valid,support_score=float(np.clip(1-(stats.get('p90') or 20)/40,0,1)) if valid else None,landmark_validation=stats,quality='M6 current semantics + measured temporal evidence; heuristic' if valid else 'unavailable')
        motion={**base.motion_evidence,'localization':loc,'temporal_camera':debug};common={'motion_evidence':motion,'memory':None,'projected_pitch_geometry':[]}
        if not valid:return base.model_copy(update={**common,'calibration_valid':False,'calibration':None,'visible_pitch_region':None,'calibration_status':'lost','source':reason})
        region=safe_region(h,self.width,self.height);debug['accepted_region']=region[1];source='direct_calibrated' if state in ('DIRECT','RELOCALIZED') else 'bidirectional_interpolated' if state=='BIDIRECTIONAL_INTERPOLATED' else 'temporal_corrected' if state=='TEMPORAL_CORRECTED' else 'temporal_predicted' if state=='TEMPORAL_PREDICTED' else 'partial_constrained'
        calibration=PitchCalibrationObservation(start_frame=base.frame,end_frame=base.frame,image_to_pitch=tuple(h.flat),validity_region=region[0],provenance=Provenance(provider='cv-camera-m6',source=reason,quality='inferred' if source=='direct_calibrated' else 'propagated'))
        return base.model_copy(update={**common,'calibration':calibration,'calibration_valid':True,'calibration_status':'valid' if source=='direct_calibrated' else 'partial','calibration_source_kind':source,'visible_pitch_region':[tuple(p) for p in region[1]],'projected_pitch_geometry':[ImagePitchSegment(**s) for s in self.geometry.overlay(h,self.width,self.height)],'source':reason})
    def run(self,cameras,groups,forward,backward,fields=None,feature_measurements=None):
        result=[];last=None;last_direct=None;budget=0.;flags=[];anchors=TextureAnchors();anchor_origins={}
        for i,raw in enumerate(cameras):
            hraw=transform(raw);current=groups[i];motion=forward[i];h=None;reason='NO_CURRENT_GEOMETRY';state='UNKNOWN';debug={'groups':current,'direction':'forward','field_motion':motion,'previous_timestamp':last.timestamp if last is not None else None,'current_timestamp':raw.timestamp,'geometry_inconsistent':False};prediction=None
            if i>0 and hraw is not None and transform(cameras[i-1]) is not None and motion.get('accepted') and raw.epoch==cameras[i-1].epoch and raw.continuity!='cut':
                baseline_prediction=transform(cameras[i-1])@np.linalg.inv(np.array(motion['homography']))
                baseline_delta=disagreement(hraw,baseline_prediction,self.width,self.height);debug['baseline_raw_motion_difference_p90_m']=baseline_delta
                if baseline_delta>MAX_TEMPORAL_DISAGREEMENT_M:debug['baseline_geometry_inconsistent']=True;flags.append(i)
            epoch_reset=raw.continuity=='cut' or (i>0 and cameras[i-1].epoch!=raw.epoch)
            if epoch_reset:last=None;last_direct=None;budget=0.;anchors=TextureAnchors();anchor_origins={}
            if last is not None and raw.frame-last.frame==1 and motion.get('accepted'):
                prediction=transform(last)@np.linalg.inv(np.array(motion['homography']));debug['previous_region']=last.visible_pitch_region;debug['predicted_region']=(safe_region(prediction,self.width,self.height) or (None,None))[1]
                budget=float(np.hypot(budget,motion['heldout_p90_small_px']*motion.get('scale_to_source',1)))
                corrected,stats=self.candidate(prediction,current);debug['correction']=stats
                if hraw is not None:
                    delta=disagreement(hraw,prediction,self.width,self.height);debug['raw_vs_predicted_p90_m']=delta
                    if delta>MAX_TEMPORAL_DISAGREEMENT_M:debug['geometry_inconsistent']=True;flags.append(i)
                if corrected is not None and budget<=20:
                    h=corrected;state='TEMPORAL_CORRECTED' if raw.calibration_valid and raw.calibration_source_kind=='direct_calibrated' else 'PARTIAL_CONSTRAINED';reason='MEASURED_MOTION_CURRENT_GEOMETRY_CORRECTED'
                    # Keep current independent anchor separately; never call a
                    # corrected/propagated coordinate independently provided.
                    if hraw is not None and raw.calibration_source_kind=='direct_calibrated' and disagreement(hraw,h,self.width,self.height)<=.5:last_direct=raw.timestamp;budget=0.
                elif debug['geometry_inconsistent']:reason='TEMPORAL_GEOMETRY_INCONSISTENT'
                else:reason=stats.get('reason','CURRENT_GEOMETRY_REJECTED')
            if h is None and fields is not None and feature_measurements is not None and not epoch_reset:
                anchor,anchor_diagnostic=anchors.propose(i,raw.epoch,fields[i],feature_measurements[i],motion,self.width,self.height);debug['reobserved_anchor']=anchor_diagnostic
                if anchor is not None:
                    candidate,stats=self.candidate(anchor['homography'],current)
                    new_budget=float(np.hypot(anchor['anchor'].motion_error_budget_px,anchor['motion']['heldout_p90_small_px']*anchor['motion'].get('scale_to_source',1)))
                    if candidate is not None and new_budget<=20:
                        h=candidate;state='PARTIAL_CONSTRAINED';reason='FRESH_REOBSERVED_M6_ANCHOR_CURRENT_GEOMETRY';budget=new_budget;last_direct=anchor_origins.get(anchor['anchor_frame'],last_direct);debug['predicted_region']=safe_region(anchor['homography'],self.width,self.height)[1];debug['correction']=stats
            if h is None and hraw is not None and raw.calibration_source_kind=='direct_calibrated':
                strong,observability=strongly_observed(hraw,current,self.width,self.height);debug['observability']=observability
                # Do not reset to a contradictory absolute fit after a measured
                # temporal failure. Reacquisition needs independently rich geometry.
                if strong and not debug['geometry_inconsistent']:
                    h=hraw;state='RELOCALIZED' if last_direct is not None else 'DIRECT';reason='STRONG_CURRENT_GEOMETRY_ANCHOR';last_direct=raw.timestamp;budget=0.
                elif reason=='NO_CURRENT_GEOMETRY':reason='TEMPORAL_GEOMETRY_INCONSISTENT' if debug.get('baseline_geometry_inconsistent') else 'INSUFFICIENT_INDEPENDENT_GEOMETRY'
            if epoch_reset and h is None:reason='HARD_CUT'
            debug['relative_motion_error_budget_px']=budget if h is not None else None;debug['absolute_anchor_frame']=round(last_direct*self.fps) if last_direct is not None else None
            age=raw.timestamp-last_direct if last_direct is not None else None;out=self.publish(raw,h,state,reason,debug,age);result.append(out)
            last=out if out.calibration_valid else None
            if out.calibration_valid and fields is not None and feature_measurements is not None:
                estimate=CameraEstimate(frame=i,timestamp=out.timestamp,epoch=out.epoch,transform=out.calibration.image_to_pitch,source=out.calibration_source_kind,visible_region=out.visible_pitch_region,landmark_ids=out.motion_evidence['localization']['semantic_classes'],support=out.motion_evidence['localization'].get('support_score') or 0,motion_error_budget_px=budget)
                anchors.add(i,out.epoch,estimate,fields[i],feature_measurements[i],out.calibration_source_kind=='direct_calibrated');anchor_origins[i]=last_direct

        result,attempts=self.fill_short_gaps(cameras,groups,forward,backward,result)
        result=self.attach_memory(result,cameras)
        return result,{'geometry_inconsistent_frames':sorted(set(flags)),'gap_attempts':attempts,'max_gap_frames':MAX_GAP_FRAMES,'max_gap_seconds':MAX_GAP_SECONDS}

    def fill_short_gaps(self,cameras,groups,forward,backward,result):
        # Only an entire unknown block of at most two frames can be filled. No
        # recursive reuse of newly filled gaps as future independent anchors.
        frozen=list(result);attempts=[];i=0
        while i<len(result):
            if frozen[i].calibration_valid:i+=1;continue
            start=i
            while i<len(result) and not frozen[i].calibration_valid:i+=1
            end=i-1;left=start-1;right=i;size=end-start+1
            if size>MAX_GAP_FRAMES or size/self.fps>MAX_GAP_SECONDS:continue
            for frame in range(start,end+1):
                estimates=[];directions=[]
                for anchor,direction,measurements in ((left,'forward',forward),(right,'backward',backward)):
                    if anchor<0 or anchor>=len(frozen) or not frozen[anchor].calibration_valid or not compatible_path(cameras,anchor,frame,self.fps):continue
                    h=transform(frozen[anchor]);ok=True;path=[];step=1 if direction=='forward' else -1
                    for target in range(anchor+step,frame+step,step):
                        motion=measurements[target];path.append(motion)
                        if not motion.get('accepted'):ok=False;break
                        h=h@np.linalg.inv(np.array(motion['homography']))
                    if ok:
                        candidate,stats=self.candidate(h,groups[frame],allow_weak=True)
                        if candidate is not None:estimates.append(candidate);directions.append({'anchor_frame':anchor,'direction':direction,'motion':path[-1],'validation':stats,'relative_motion_error_budget_px':float(np.sqrt((frozen[anchor].motion_evidence.get('temporal_camera',{}).get('relative_motion_error_budget_px') or 0)**2+sum((p['heldout_p90_small_px']*p.get('scale_to_source',1))**2 for p in path)))})
                if any(d['relative_motion_error_budget_px']>20 for d in directions):attempts.append({'frame':frame,'accepted':False,'reason':'RELATIVE_MOTION_UNCERTAINTY_EXCEEDED'});continue
                record={'frame':frame,'directions':len(estimates),'accepted':False,'reason':'INSUFFICIENT_MEASURED_DIRECTIONS'}
                if len(estimates)==2:
                    delta=disagreement(estimates[0],estimates[1],self.width,self.height);record['agreement_p90_m']=delta
                    if delta>MAX_BIDIRECTIONAL_DISAGREEMENT_M:record['reason']='BIDIRECTIONAL_DISAGREEMENT';attempts.append(record);continue
                    state='BIDIRECTIONAL_INTERPOLATED';reason='BOTH_MEASURED_DIRECTIONS_AGREE_WITH_CURRENT_GEOMETRY'
                elif len(estimates)==1 and len(groups[frame])>=2 and (left<0 or right>=len(frozen)):
                    state='TEMPORAL_PREDICTED';reason='BOUNDARY_BACKWARD_MEASUREMENT_CURRENT_GEOMETRY' if directions[0]['direction']=='backward' else 'BOUNDARY_FORWARD_MEASUREMENT_CURRENT_GEOMETRY'
                else:attempts.append(record);continue
                anchor_frame=frozen[directions[0]['anchor_frame']].motion_evidence.get('temporal_camera',{}).get('absolute_anchor_frame',directions[0]['anchor_frame'])
                debug={'groups':groups[frame],'relative_motion_error_budget_px':max(d['relative_motion_error_budget_px'] for d in directions),'absolute_anchor_frame':anchor_frame,'direction':directions[0]['direction'] if len(estimates)==1 else 'bidirectional','estimates':directions,'field_motion':directions[0]['motion'],'forward_backward_agreement_p90_m':record.get('agreement_p90_m'),'geometry_inconsistent':False,'predicted_region':safe_region(estimates[0],self.width,self.height)[1]}
                result[frame]=self.publish(cameras[frame],estimates[0],state,reason,debug,cameras[frame].timestamp-cameras[anchor_frame].timestamp if anchor_frame is not None and anchor_frame<=frame else None);record.update(accepted=True,reason=reason);attempts.append(record)
        return result,attempts

    def attach_memory(self,result,raw):
        memory=CameraMemory()
        for i,c in enumerate(result):
            debug=c.motion_evidence['temporal_camera'];loc=c.motion_evidence['localization'];anchor=debug.get('absolute_anchor_frame');updates={'epoch':c.epoch}
            if memory.epoch!=c.epoch:memory=CameraMemory(epoch=c.epoch)
            if anchor is not None and 0<=anchor<len(raw) and raw[anchor].epoch==c.epoch and raw[anchor].calibration_valid:
                a=raw[anchor];updates['last_direct']=CameraEstimate(frame=anchor,timestamp=a.timestamp,epoch=a.epoch,transform=tuple(transform(a).flat),source='independent geometry anchor; future evidence' if anchor>i else 'independent geometry anchor',visible_region=a.visible_pitch_region,landmark_ids=a.motion_evidence['localization']['semantic_classes'],support=a.motion_evidence['localization'].get('support_score') or 0,motion_error_budget_px=0)
                debug['absolute_anchor_timestamp']=a.timestamp;debug['anchor_age_seconds']=c.timestamp-a.timestamp
            if c.calibration_valid:
                estimate=CameraEstimate(frame=i,timestamp=c.timestamp,epoch=c.epoch,transform=c.calibration.image_to_pitch,source=c.calibration_source_kind,visible_region=c.visible_pitch_region,landmark_ids=loc['semantic_classes'],support=loc.get('support_score') or 0,motion_error_budget_px=debug.get('relative_motion_error_budget_px') or 0)
                updates.update(last_accepted=estimate,accepted_history=(memory.accepted_history+[estimate])[-12:],last_constraint_timestamp=c.timestamp)
            else:updates['failure_history']=(memory.failure_history+[CameraFailure(frame=i,timestamp=c.timestamp,reason=loc['outcome'])])[-12:]
            memory=memory.model_copy(update=updates);result[i]=c.model_copy(update={'memory':memory,'last_reliable_timestamp':memory.last_direct.timestamp if memory.last_direct else None,'last_reliable_transform':memory.last_direct.transform if memory.last_direct else None})
        return result
