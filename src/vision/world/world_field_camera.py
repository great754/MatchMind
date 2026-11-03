"""M7 persistent world-field localization. All current poses require measured pixels.

World coordinates inherit an accepted independent camera anchor, not ground truth.
Fixed world coordinates never move with players or optimization. Held-out IDs are
chosen before fitting and remain held out for the lifetime of a feature.
"""
from collections import deque
import cv2
import numpy as np
from .temporal_camera import TemporalCameraFilter,transform,disagreement
from .persistent_camera import safe_region
from .field_motion import validate_motion
from .camera_window import refine_window
from .static_geometry import semantic_line_migrations

MAX_UNCERTAINTY_PX=20.
MAX_ANCHOR_AGE_SECONDS=10.  # backstop only; evidence budgets usually stop earlier

class WorldFieldTracks:
    def __init__(self,width,height):
        self.width=width;self.height=height;self.tracks={};self.expired=[];self.reference=None
    def observe(self,frame,measurements,scale):
        ids=measurements['ids'] if measurements else [];points=measurements['points']/scale if measurements else []
        current={int(k):np.asarray(p,float) for k,p in zip(ids,points)}
        for k in list(self.tracks):
            t=self.tracks[k]
            if k not in current:
                self.expired.append(abs(t['last_seen']-t['first_seen'])+1);del self.tracks[k]
            else:
                t['last_seen']=frame;t['image_history'].append(current[k].tolist());t['image_history']=t['image_history'][-10:];t['observations']+=1
        return current
    def seed(self,frame,current,h,epoch,origin,uncertainty,field=None):
        if uncertainty>MAX_UNCERTAINTY_PX:return 0
        if self.reference is None and field is not None:self.reference=(h.copy(),field,frame)
        added=0
        for k,p in current.items():
            if k in self.tracks:continue
            world=cv2.perspectiveTransform(p[None,None].astype(float),h)[0,0]
            if not np.isfinite(world).all() or not (0<=world[0]<=105 and 0<=world[1]<=68):continue
            self.tracks[k]={'id':f'FIELD_FEATURE_{k:06d}','first_seen':frame,'last_seen':frame,'world_coordinate':world.tolist(),'image_history':[p.tolist()],'observations':1,'camera_epoch':epoch,'independent_anchor_frame':origin,'origin_uncertainty_px':uncertainty,'tracking_quality':'masked static LK, past RANSAC membership','provenance':'inherited world coordinate; not independent surveyed landmark'};added+=1
        return added
    def fit(self,current,field):
        ids=np.array(sorted(self.tracks.keys()&current.keys()),int)
        diagnostic={'active_world_anchors':len(ids),'reason':'INSUFFICIENT_WORLD_FIELD_ANCHORS','accepted':False}
        if len(ids)<40:return None,diagnostic
        image=np.array([current[k] for k in ids]);world=np.array([self.tracks[k]['world_coordinate'] for k in ids]);hold=ids%4==0
        if hold.sum()<10 or (~hold).sum()<30:return None,diagnostic
        scale=field[2];inverse,inliers=cv2.findHomography(world[~hold],image[~hold],cv2.RANSAC,1/scale)
        if inverse is None or inliers is None:return None,diagnostic
        try:h=np.linalg.inv(inverse)
        except np.linalg.LinAlgError:return None,diagnostic
        pred=cv2.perspectiveTransform(world[None],inverse)[0];residual=np.linalg.norm(pred-image,axis=1)*scale
        hull=float(cv2.contourArea(cv2.convexHull((image[~hold][inliers[:,0]>0]*scale).astype(np.float32)))/field[0].size)
        diagnostic.update(ransac_inliers=int(inliers.sum()),inlier_ratio=float(inliers.mean()),heldout_median_small_px=float(np.median(residual[hold])),heldout_p90_small_px=float(np.quantile(residual[hold],.9)),feature_hull_fraction=hull,median_track_lifetime_frames=float(np.median([self.tracks[k]['observations'] for k in ids])),mean_track_lifetime_frames=float(np.mean([self.tracks[k]['observations'] for k in ids])),spatial_cells=len(set((int(p[0]*scale//(field[0].shape[1]/4)),int(p[1]*scale//(field[0].shape[0]/4))) for p in image)),origin_uncertainty_px=max(self.tracks[k]['origin_uncertainty_px'] for k in ids))
        valid=bool(np.isfinite(residual).all() and inliers.mean()>=.8 and hull>=.15 and np.median(residual[hold])<=1 and np.quantile(residual[hold],.9)<=2 and safe_region(h,self.width,self.height) is not None)
        if not valid and self.reference is not None:
            reference_h,reference_field,reference_frame=self.reference
            reference_image=cv2.perspectiveTransform(world[None],np.linalg.inv(reference_h))[0]
            relative=validate_motion((reference_image*scale).astype(np.float32),(image*scale).astype(np.float32),reference_field,field,hold=hold,interval_frames=max(1,abs(next(iter(self.tracks.values()))['last_seen']-reference_frame)))
            diagnostic['reference_fit']=relative
            if relative.get('accepted'):
                h=reference_h@np.linalg.inv(np.array(relative['homography']));valid=safe_region(h,self.width,self.height) is not None
                diagnostic.update(heldout_median_small_px=relative['heldout_median_small_px'],heldout_p90_small_px=relative['heldout_p90_small_px'],model=relative['model'],inlier_ratio=relative['inlier_ratio'])
        diagnostic.update(accepted=valid,reason='HELDOUT_WORLD_FIELD_ANCHORS_VALIDATED' if valid else 'WORLD_FIELD_ANCHOR_VALIDATION_FAILED')
        # Current inconsistent features are removed, never repositioned in world.
        for k,error in zip(ids,residual):
            if error>2:
                self.expired.append(self.tracks[k]['observations']);del self.tracks[k]
        return h if valid else None,diagnostic
    def samples(self):return list(self.tracks.values())[:32]

class WorldFieldCamera(TemporalCameraFilter):
    def publish(self,base,h,state,reason,debug,age=None):
        result=super().publish(base,h,state,reason,debug,age)
        result.motion_evidence['localization']['quality']='M7 measured persistent field trajectory; inherited metric anchor; heuristic' if h is not None else 'unavailable'
        if result.calibration is not None:
            result=result.model_copy(update={'calibration':result.calibration.model_copy(update={'provenance':result.calibration.provenance.model_copy(update={'provider':'cv-camera-m7'})})})
        return result
    def attach_memory(self,result,raw):
        result=super().attach_memory(result,raw)
        for i,c in enumerate(result):
            d=c.motion_evidence['temporal_camera'];anchor=d.get('absolute_anchor_frame')
            if anchor is None or c.memory is None or c.memory.last_direct is None:continue
            origin=raw[anchor];independent=c.memory.last_direct
            window=origin.motion_evidence.get('temporal_camera',{}).get('absolute_window',{})
            if window.get('accepted'):
                kind='M7_MULTIFRAME_ABSOLUTE_WINDOW'
            elif origin.calibration_source_kind=='direct_calibrated':kind='M6_CURRENT_INDEPENDENT'
            elif origin.memory is not None and origin.memory.last_direct is not None and origin.memory.last_direct.frame==anchor:
                # M6 current pose may be TEMPORAL_CORRECTED while its separate
                # independent M5 anchor is genuine. Preserve that exact anchor;
                # never promote the current partial transform to an absolute fit.
                independent=origin.memory.last_direct;kind='M6_PRESERVED_INDEPENDENT'
            else:
                raise ValueError('Unproven independent camera origin')
            d['independent_anchor_proof']={'kind':kind,'frame':anchor,'transform':list(independent.transform),'source':independent.source,'future_evidence':anchor>i}
            result[i]=c.model_copy(update={'memory':c.memory.model_copy(update={'last_direct':independent}),'last_reliable_transform':independent.transform,'last_reliable_timestamp':independent.timestamp})
        return result

    def directional(self,cameras,groups,motions,fields,measurements,reverse=False):
        indices=list(range(len(cameras)-1,-1,-1) if reverse else range(len(cameras)));out={};last=None;origin=None;uncertainty=0.;tracks=WorldFieldTracks(self.width,self.height);history=deque(maxlen=10);previous_epoch=None;quarantine={};previous_index=None
        for i in indices:
            raw=cameras[i];motion=motions[i];current_group=groups[i];direction='backward' if reverse else 'forward';epoch_reset=previous_epoch is not None and (raw.epoch!=previous_epoch or (cameras[i+1].continuity=='cut' if reverse and i+1<len(cameras) else raw.continuity=='cut'))
            if epoch_reset or motion.get('reason') in ('ZOOM_UNSUPPORTED','TRANSFORM_FOLDING'):
                last=None;origin=None;uncertainty=0.;tracks=WorldFieldTracks(self.width,self.height);history.clear();quarantine={}
            previous_epoch=raw.epoch
            if previous_index is not None and not epoch_reset:
                migrations=semantic_line_migrations(groups[previous_index],current_group,motion,self.width,self.height,fields[i][2])
            else:migrations={}
            previous_index=i
            previous_track_count=len(tracks.tracks);current=tracks.observe(i,measurements[i],fields[i][2]);h=None;state='UNKNOWN';reason='NO_USABLE_WORLD_CAMERA';prediction=None;world_h,world=tracks.fit(current,fields[i]);stats={};anchor_correction=None
            debug={'groups':current_group,'direction':direction,'field_motion':motion,'previous_timestamp':last.timestamp if last is not None else None,'current_timestamp':raw.timestamp,'world_field':world,'geometry_inconsistent':False,'last_accepted_frame':last.frame if last is not None else None}
            if last is not None and motion.get('accepted'):
                prediction=transform(last)@np.linalg.inv(np.array(motion['homography']));uncertainty=float(np.hypot(uncertainty,motion['heldout_p90_small_px']*motion.get('scale_to_source',1)));reason='RELATIVE_MOTION_UNCERTAINTY_EXCEEDED' if uncertainty>MAX_UNCERTAINTY_PX else 'CURRENT_GEOMETRY_CONTRADICTS_TRAJECTORY'
                debug['previous_region']=last.visible_pitch_region;debug['predicted_region']=(safe_region(prediction,self.width,self.height) or (None,None))[1]
                history.append({'frame':i,'pan':motion.get('pan_small_px'),'scale':[motion.get('min_local_scale'),motion.get('max_local_scale')],'error':motion.get('heldout_p90_small_px')})
            if world_h is not None and last is not None:
                relative=np.linalg.inv(world_h)@transform(last)
                grid=np.array([[(x*self.width,y*self.height) for x in (.2,.5,.8) for y in (.2,.5,.8)]],float)
                projected=cv2.perspectiveTransform(grid,relative)
                dx=cv2.perspectiveTransform(grid+[1,0],relative)-projected;dy=cv2.perspectiveTransform(grid+[0,1],relative)-projected
                scales=np.linalg.svd(np.stack([dx[0],dy[0]],axis=-1),compute_uv=False)
                world['relative_scale_min']=float(scales.min());world['relative_scale_max']=float(scales.max())
                if not np.isfinite(scales).all() or scales.min()<.98 or scales.max()>1.02:
                    world_h=None;world.update(accepted=False,reason='WORLD_ANCHOR_ZOOM_UNSUPPORTED')
            if world_h is not None:
                # Multiframe fixed-world remeasurement bounds inherited uncertainty;
                # no accumulating pairwise homography product is used for this fit.
                candidate_uncertainty=float(np.hypot(world['origin_uncertainty_px'],world['heldout_p90_small_px']/fields[i][2]))
                if prediction is None or disagreement(world_h,prediction,self.width,self.height)<=2:
                    prediction=world_h;uncertainty=candidate_uncertainty;debug['world_fit_used']=True
                else:debug['world_fit_rejected']='WORLD_TRACK_VS_MOTION_DISAGREEMENT'
            if prediction is not None and origin is not None:
                # A demonstrated class migration may be quarantined while fixed-world
                # tracks and measured motion remain strong. Contradictory pixels
                # remain explicit, never become independent anchor evidence.
                for name,record in migrations.items():
                    current_stats=self.validator.statistics(prediction,{name:current_group[name]})
                    previous_stats=self.validator.statistics(prediction,{record['previous_name']:current_group[name]})
                    if not current_stats['accepted'] and previous_stats['accepted'] and motion.get('inlier_ratio',0)>=.8:quarantine[name]=record
                for name in list(quarantine):
                    if name in current_group and self.validator.statistics(prediction,{name:current_group[name]})['accepted']:del quarantine[name]
                usable={k:v for k,v in current_group.items() if k not in quarantine}
                debug['semantic_line_quarantine']={k:v for k,v in quarantine.items() if k in current_group}
                debug['all_current_semantic_validation']=self.validator.statistics(prediction,current_group) if current_group else None
                stats=self.validator.statistics(prediction,usable) if usable else {'accepted':True,'reason':'NO_CURRENT_SEMANTICS_TRACKS_REQUIRED'}
                age=abs(raw.timestamp-cameras[origin].timestamp)
                # Semantic disappearance needs real retained world tracks. No stale H.
                evidence=bool(world_h is not None or motion.get('accepted'))
                if stats['accepted'] and evidence and uncertainty<=MAX_UNCERTAINTY_PX and age<=MAX_ANCHOR_AGE_SECONDS and safe_region(prediction,self.width,self.height) is not None:
                    h=prediction;state='TEMPORAL_PREDICTED';reason='PERSISTENT_WORLD_FIELD_MEASURED_CONTINUATION'
                elif not stats['accepted']:
                    candidate,correction=self.validator.refine(prediction,usable) if len(usable)>=2 else (None,{})
                    debug['semantic_correction']=correction
                    if candidate is not None and disagreement(candidate,prediction,self.width,self.height)<=1 and uncertainty<=MAX_UNCERTAINTY_PX and age<=MAX_ANCHOR_AGE_SECONDS:
                        h=candidate;state='TEMPORAL_CORRECTED';reason='WORLD_TRAJECTORY_BOUNDED_SEMANTIC_CORRECTION';stats=self.validator.statistics(candidate,usable)
                    else:reason='CURRENT_SEMANTICS_CONTRADICT_WORLD_TRAJECTORY'
            # Independently validated M6 anchors remain immutable. A conflict does
            # not automatically choose the higher-coverage trajectory.
            if raw.calibration_valid and raw.calibration_source_kind=='direct_calibrated':
                anchor=transform(raw);delta=disagreement(anchor,prediction,self.width,self.height) if prediction is not None else None;debug['absolute_reacquisition_difference_p90_m']=delta
                if delta is None or delta<=2:
                    h=anchor;state='RELOCALIZED' if origin is not None else 'DIRECT';reason='PRESERVED_INDEPENDENT_ANCHOR';origin=i;uncertainty=0.;anchor_correction=delta;tracks=WorldFieldTracks(self.width,self.height)
                elif h is None:reason='ABSOLUTE_VS_TRAJECTORY_DISAGREEMENT';debug['geometry_inconsistent']=True
            # Preserve M6 accepted temporally validated poses as fallback ONLY.
            # They cannot mint independent origin evidence or hide disagreement.
            if h is None and raw.calibration_valid:
                anchor=raw.motion_evidence.get('temporal_camera',{}).get('absolute_anchor_frame')
                if prediction is None or disagreement(transform(raw),prediction,self.width,self.height)<=2:
                    h=transform(raw);state=raw.motion_evidence['localization']['status'];reason='PRESERVED_M6_VALIDATED_CAMERA';origin=anchor;uncertainty=raw.motion_evidence['temporal_camera'].get('relative_motion_error_budget_px') or 0.
            debug['accepted_semantic_classes']=[name for name in current_group if name not in quarantine]
            debug.update(absolute_anchor_frame=origin,relative_motion_error_budget_px=uncertainty if h is not None else None,trajectory_uncertainty_px=uncertainty,window=list(history),anchor_correction_p90_m=anchor_correction,landmark_validation=stats,feature_survival_fraction=min(1.,len(tracks.tracks)/max(1,previous_track_count)),motion_model={'pan_history':[v['pan'] for v in history],'scale_history':[v['scale'] for v in history]},circle_validation=stats.get('types',{}).get('circles'),line_validation=stats.get('types',{}).get('lines'))
            if prediction is not None:
                residual,names=self.geometry.residuals(prediction,current_group,self.width,self.height,True)
                debug['named_semantic_residuals']={name:{'mean_source_px':float(np.abs(residual[np.array(names)==name]).mean()),'p90_source_px':float(np.quantile(np.abs(residual[np.array(names)==name]),.9))} for name in set(names)}
                debug['top_touchline_constraint']=debug['named_semantic_residuals'].get('Side line top')
                from .static_geometry import tracked_line_intersections
                debug['tracked_intersections']=tracked_line_intersections(motion.get('lines',[]))
                debug['intersection_evidence']='derived from same named line fits; not independent of line constraints'
            if h is not None:
                # New world constraints inherit the complete bounded source uncertainty.
                # Mint only on independent anchors or a held-out validated world
                # remeasurement; adjacent image motion alone cannot renew anchors.
                may_seed=(state in ('DIRECT','RELOCALIZED') and origin==i) or world.get('accepted',False)
                added=tracks.seed(i,current,h,raw.epoch,origin,uncertainty,fields[i]) if may_seed else 0;debug['new_world_anchors']=added;debug['persistent_field_anchors']=tracks.samples();debug['world_field']['active_world_anchors_after_seed']=len(tracks.tracks)
            else:debug['new_world_anchors']=0
            age=raw.timestamp-cameras[origin].timestamp if origin is not None else None
            debug['groups']={k:v for k,v in current_group.items() if k not in quarantine} if h is not None else current_group
            camera=self.publish(raw,h,state,reason,debug,age);out[i]=camera;last=camera if h is not None else None
        return out
    def run(self,cameras,groups,forward,backward,fields=None,feature_measurements=None,backward_measurements=None,absolute_candidates=None):
        refined=[];anchor_refinements={}
        for i,c in enumerate(cameras):
            if not c.calibration_valid and absolute_candidates and absolute_candidates[i] is not None:
                initial=np.array(absolute_candidates[i]['homography'])
                h,record=refine_window(i,cameras,groups,forward,backward,self.validator,initial_h=initial,extended=True,fps=self.fps)
                anchor_refinements[i]=record
                if record['accepted']:
                    d={'groups':groups[i],'field_motion':forward[i],'direction':'offline-window','absolute_anchor_frame':i,'absolute_window':record,'relative_motion_error_budget_px':0.}
                    c=self.publish(c,h,'RELOCALIZED','MULTIFRAME_ABSOLUTE_GEOMETRY_REACQUIRED',d,0.)
            elif c.calibration_valid and c.calibration_source_kind=='direct_calibrated':
                h,record=refine_window(i,cameras,groups,forward,backward,self.validator);anchor_refinements[i]=record
                if record['accepted']:
                    d={**c.motion_evidence.get('temporal_camera',{}),'groups':groups[i],'absolute_window':record};c=self.publish(c,h,c.motion_evidence['localization']['status'],'MULTIFRAME_PITCH_ANCHOR_REFINED',d,0.)
            refined.append(c)
        original=cameras;cameras=refined
        a=self.directional(cameras,groups,forward,fields,feature_measurements);b=self.directional(cameras,groups,backward,fields,backward_measurements,True);result=[];attempts=[]
        for i,raw in enumerate(cameras):
            left,right=a[i],b[i];valid_l=left.calibration_valid;valid_r=right.calibration_valid;chosen=left if valid_l else right if valid_r else left
            record={'frame':i,'forward_valid':valid_l,'backward_valid':valid_r,'accepted':chosen.calibration_valid,'directions':{direction:{'reason':c.source,'state':c.motion_evidence['localization']['status'],'last_accepted_frame':c.motion_evidence['temporal_camera'].get('last_accepted_frame'),'world_field':c.motion_evidence['temporal_camera'].get('world_field'),'uncertainty_px':c.motion_evidence['temporal_camera'].get('trajectory_uncertainty_px'),'semantic_validation':c.motion_evidence['temporal_camera'].get('landmark_validation'),'motion_reason':c.motion_evidence['temporal_camera']['field_motion'].get('reason')} for direction,c in [('forward',left),('backward',right)]}}
            if valid_l and valid_r:
                delta=disagreement(transform(left),transform(right),self.width,self.height);record['agreement_p90_m']=delta
                if delta<=1:
                    # Select an actual measured estimate, never average H.
                    chosen=min((left,right),key=lambda c:c.motion_evidence['temporal_camera'].get('trajectory_uncertainty_px',20));d=chosen.motion_evidence['temporal_camera'];d['forward_backward_agreement_p90_m']=delta;d['both_directions_validated']=True
                    if not raw.calibration_valid:
                        debug={**d,'groups':groups[i],'direction':'bidirectional'};chosen=self.publish(raw,transform(chosen),'BIDIRECTIONAL_INTERPOLATED','BOTH_PERSISTENT_WORLD_TRAJECTORIES_AGREE',debug,chosen.motion_evidence['localization'].get('last_absolute_age_seconds'))
                elif not raw.calibration_valid:
                    record.update(accepted=False,reason='BIDIRECTIONAL_DISAGREEMENT');chosen=self.publish(raw,None,'UNKNOWN','BIDIRECTIONAL_DISAGREEMENT',{'groups':groups[i],'direction':'bidirectional',**left.motion_evidence['temporal_camera'],'field_motion':forward[i],'forward_backward_agreement_p90_m':delta})
            # Preserve already validated M6 on incompatible M7 proposals.
            if raw.calibration_valid and not chosen.calibration_valid:chosen=raw
            chosen.motion_evidence['temporal_camera']['m7_trajectory_review']=record
            if i in anchor_refinements:chosen.motion_evidence['temporal_camera']['absolute_window']=anchor_refinements[i]
            result.append(chosen);attempts.append(record)
        result=self.attach_memory(result,cameras)
        return result,{'algorithm':'persistent world-field tracks, ten-frame history, measured forward/backward reconstruction','frames':attempts,'absolute_window_refinements':anchor_refinements,'geometry_inconsistent_frames':[i for i,c in enumerate(result) if c.motion_evidence['temporal_camera'].get('geometry_inconsistent')],'max_support_age_backstop_seconds':MAX_ANCHOR_AGE_SECONDS,'maximum_uncertainty_source_px':MAX_UNCERTAINTY_PX,'independent_metric_accuracy':'unavailable'}
