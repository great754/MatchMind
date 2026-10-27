"""Motion-predicted, landmark-constrained camera filtering over a fixed pitch.

Never seeds from an underdetermined landmark; no stale transform is published.
Support and error budgets are heuristic diagnostics, not absolute accuracy.
"""
import cv2,numpy as np
from scipy.optimize import least_squares
from .schema import CameraMemory,CameraEstimate,CameraHypothesis,CameraFailure,ImagePitchSegment
from .pose import visible_region
from .pitch_geometry import PitchGeometry
from .field_motion import measure_field_motion
from .diagnosis import absolute_outcome
from src.observations.schema import PitchCalibrationObservation,Provenance

def safe_region(h,width,height):
    try:
        h=np.asarray(h,float)
        if h.shape!=(3,3) or not np.isfinite(h).all() or abs(h[2,2])<1e-10:return None
        return visible_region(h,width,height)
    except (ValueError,TypeError,np.linalg.LinAlgError,cv2.error):return None

def corrected(h,parameters,width,height):
    if len(parameters)==8:
        a,b,dx,c,d,dy,g,k=parameters;normal=np.array([[1+a,b,dx],[c,1+d,dy],[g,k,1.]])
        scale_matrix=np.array([[width,0,width/2],[0,height,height/2],[0,0,1.]])
        return h@np.linalg.inv(scale_matrix@normal@np.linalg.inv(scale_matrix))
    dx,dy,angle,logscale=parameters;scale=np.exp(logscale);a=cv2.getRotationMatrix2D((width/2,height/2),angle*180/np.pi,scale);a[:,2]+=[dx,dy]
    return h@np.linalg.inv(np.vstack([a,[0,0,1]]))

class PersistentCamera:
    def __init__(self,width,height,geometry=None):
        self.width=width;self.height=height;self.geometry=geometry or PitchGeometry();self.memory=CameraMemory();self.accepted_field=None
    def statistics(self,h,groups,heldout=True):
        residual,names=self.geometry.residuals(h,groups,self.width,self.height,heldout)
        if not len(residual) or not np.isfinite(residual).all():return {'accepted':False,'mean':None,'p90':None,'types':{},'reason':'NO_CURRENT_LANDMARK_SUPPORT'}
        errors=np.abs(residual);categories=np.array(['circles' if name.startswith('Circle') else 'lines' for name in names]);types={}
        for kind in set(categories):
            e=errors[categories==kind];types[str(kind)]={'mean':float(e.mean()),'p90':float(np.quantile(e,.9))}
        valid=all(r['mean']<=10 and r['p90']<=20 for r in types.values())
        return {'accepted':valid,'mean':float(errors.mean()),'p90':float(np.quantile(errors,.9)),'types':types,'reason':'CURRENT_LANDMARKS_AGREE' if valid else 'LANDMARK_RESIDUAL_TOO_HIGH'}
    def refine(self,h,groups,frame=None,epoch=None):
        """Robust motion-prior measurement update; simpler correction first.

        Lines and circles are balanced, as in the absolute solver. An optional
        bounded projective correction handles shear/tilt that similarity cannot.
        Every candidate uses disjoint held-out semantics and the same gates.
        """
        _,names=self.geometry.residuals(h,groups,self.width,self.height)
        types=np.array(['circles' if n.startswith('Circle') else 'lines' for n in names])
        weights=np.ones(len(types))
        for kind in set(types):weights[types==kind]=np.sqrt(len(types)/(len(set(types))*sum(types==kind)))
        proposals=[]
        for family in ('similarity','projective'):
            if family=='similarity':
                priors=np.array([4.,4.,.002,.004]);bounds=np.array([8.,8.,.004,.008]);initials=(np.zeros(4),np.array([2.,-2.,.001,.001]),np.array([-2.,2.,-.001,-.001]))
            else:
                priors=np.array([.002,.002,4/self.width,.002,.002,4/self.height,.001,.001]);bounds=priors*2
                initials=(np.zeros(8),np.array([0,0,2/self.width,0,0,-2/self.height,0,0]),np.array([0,0,-2/self.width,0,0,2/self.height,0,0]))
            def objective(p):
                candidate=corrected(h,p,self.width,self.height);residual,_=self.geometry.residuals(candidate,groups,self.width,self.height)
                return np.concatenate([weights*residual/10,p/priors])
            results=[]
            for initial in initials:
                try:
                    opt=least_squares(objective,initial,bounds=(-bounds,bounds),loss='soft_l1',f_scale=1,max_nfev=35)
                    candidate=corrected(h,opt.x,self.width,self.height);stats=self.statistics(candidate,groups)
                    grid=np.array([[(x*self.width,y*self.height) for y in (.2,.5,.8) for x in (.2,.5,.8)]],float)
                    adjustment=cv2.perspectiveTransform(grid,np.linalg.inv(candidate)@h)-grid
                    correction=float(np.quantile(np.linalg.norm(adjustment,axis=2),.9))
                    valid=stats['accepted'] and correction<=12 and safe_region(candidate,self.width,self.height) is not None
                    hypothesis=CameraHypothesis(frame=frame,epoch=epoch,transform=tuple(candidate.flat) if np.isfinite(candidate).all() else None,source='bounded '+family+' semantic correction',accepted=valid,heldout_mean_px=stats['mean'],heldout_p90_px=stats['p90'],correction_norm_px=correction,reason=stats['reason'] if not stats['accepted'] else 'CORRECTION_EXCEEDS_PRIOR' if correction>12 else 'HELDOUT_AND_PRIOR_VALIDATED' if valid else 'VISIBLE_REGION_INVALID')
                    results.append((candidate,stats,hypothesis))
                except (ValueError,np.linalg.LinAlgError,cv2.error):results.append((None,{'accepted':False},CameraHypothesis(frame=frame,epoch=epoch,source='bounded '+family+' semantic correction',accepted=False,reason='SEMANTIC_UPDATE_FAILED')))
            proposals.extend(r[2] for r in results);self.memory=self.memory.model_copy(update={'candidates':proposals})
            good=[r for r in results if r[2].accepted]
            if len(good)<2:continue
            positions=np.array([cv2.perspectiveTransform(grid,r[0])[0] for r in good]);spread=np.linalg.norm(positions-positions[0],axis=2);p90=float(np.quantile(spread,.9)) if np.isfinite(spread).all() else None
            if p90 is None or p90>4:return None,{'reason':'AMBIGUOUS_GEOMETRY','hypothesis_spread_p90_m':p90}
            best=min(good,key=lambda r:r[1]['mean']);return best[0],{**best[1],'hypothesis_spread_p90_m':p90,'hypotheses_passed':len(good),'filter':'robust motion-prior '+family+' update','correction_p90_source_px':best[2].correction_norm_px}
        return None,{'reason':'LANDMARK_RESIDUAL_TOO_HIGH','hypotheses_passed':0}
    def update(self,base,groups,fit,diagnostic,motion,current_field=None,anchor_measurement=None,anchor_diagnostic=None):
        m=self.memory;previous=m.last_accepted;epoch_reset=base.continuity=='cut' or base.epoch!=m.epoch
        if epoch_reset:
            m=m.model_copy(update={'epoch':base.epoch,'last_accepted':None,'last_constraint_timestamp':None,'candidates':[]});previous=None;self.accepted_field=None
        self.memory=m;classes=sorted(k for k in groups if k in self.geometry.segments or k in self.geometry.circle_centers);absolute=absolute_outcome(fit,diagnostic,classes);accepted=None;source='direct_calibrated';reason='NO_PRIOR';stats={};budget=0.;event=None;direct=False;semantic=False
        if fit is not None:
            h=np.array(fit['homography'],float);region=safe_region(h,self.width,self.height);stats=self.statistics(h,groups) if region is not None else {'accepted':False}
            if region is not None and stats['accepted']:
                accepted=h;direct=True;semantic=True;reason='DIRECT_ACCEPTED';event='relocalization' if m.last_direct is not None and m.last_direct.epoch==base.epoch and (previous is None or previous.source!='direct_calibrated' or previous.frame!=base.frame-1) else 'acquisition'
            elif region is not None:
                candidate,refined=self.refine(h,groups,base.frame,base.epoch)
                if candidate is not None:accepted=candidate;stats=refined;direct=True;semantic=True;reason='DIRECT_ACCEPTED';event='relocalization' if m.last_direct is not None and m.last_direct.epoch==base.epoch and (previous is None or previous.source!='direct_calibrated' or previous.frame!=base.frame-1) else 'acquisition'
                else:absolute='DIRECT_LANDMARK_REJECTED'
            else:absolute='VISIBLE_REGION_INVALID'
        if accepted is None and previous is not None and previous.epoch==base.epoch:
            age=base.timestamp-previous.timestamp
            # An unavailable intermediate frame may be crossed only by a NEW
            # direct image measurement from the last accepted image to this one.
            if base.frame-previous.frame>1 and age<=.25 and current_field is not None and self.accepted_field is not None:
                rescue=measure_field_motion(self.accepted_field,current_field)
                if rescue['accepted']:motion={**rescue,'gap_remeasurement_seconds':age}
            if not 0<age<=.25:reason='NO_RECENT_MEASURED_PRIOR'
            elif not motion.get('accepted'):reason=motion.get('reason','TEMPORAL_TRACK_FAILED')
            else:
                try:
                    image_motion=np.array(motion['homography'],float)
                    if image_motion.shape!=(3,3) or not np.isfinite(image_motion).all() or abs(image_motion[2,2])<1e-10 or np.linalg.det(image_motion/image_motion[2,2])<=0:raise ValueError('Invalid image motion')
                    prediction=np.array(previous.transform).reshape(3,3)@np.linalg.inv(image_motion)
                except (np.linalg.LinAlgError,ValueError):prediction=None
                budget=float(np.hypot(previous.motion_error_budget_px,motion['heldout_p90_small_px']*motion.get('scale_to_source',1)))
                if prediction is None or safe_region(prediction,self.width,self.height) is None:reason='VISIBLE_REGION_INVALID'
                elif budget>20:reason='RELATIVE_MOTION_UNCERTAINTY_EXCEEDED'
                else:
                    stats=self.statistics(prediction,groups)
                    if len(classes)>=2:
                        candidate,refined=self.refine(prediction,groups,base.frame,base.epoch)
                        if candidate is not None:accepted=candidate;stats=refined;source='partial_constrained';semantic=True;reason='PARTIAL_CONSTRAINED_ACCEPTED'
                        else:
                            reason=refined['reason'];stats=refined
                            # One temporarily contradictory semantic category does
                            # not invalidate a fresh, measured image transform.
                            # Keep only the supported category; never refresh the
                            # strong-constraint clock with this weak update.
                            prior_stats=self.statistics(prediction,groups)
                            supported=[k for k,v in prior_stats.get('types',{}).items() if v['mean']<=10 and v['p90']<=20]
                            if supported and m.last_constraint_timestamp is not None and base.timestamp-m.last_constraint_timestamp<=.25:
                                accepted=prediction;source='camera_propagated';semantic=True;reason='TEMPORAL_SEMANTIC_DROPOUT_BRIDGED';stats={**prior_stats,'supported_categories':supported,'rejected_categories':[k for k in prior_stats.get('types',{}) if k not in supported]}
                    elif m.last_constraint_timestamp is not None and base.timestamp-m.last_constraint_timestamp<=.25 and (not classes or stats['accepted']):
                        accepted=prediction;source='camera_propagated';reason='TEMPORAL_TRACKED_ACCEPTED';semantic=bool(classes)
                    else:reason='INSUFFICIENT_CURRENT_CONSTRAINTS'
        if accepted is None and not epoch_reset and anchor_measurement is not None and len(classes)>=2:
            anchor=anchor_measurement['anchor'];prediction=anchor_measurement['homography'];measured=anchor_measurement['motion']
            if anchor.epoch==base.epoch:
                candidate,refined=self.refine(prediction,groups,base.frame,base.epoch)
                if candidate is not None:
                    new_budget=float(np.hypot(anchor.motion_error_budget_px,measured['heldout_p90_small_px']*measured.get('scale_to_source',1)))
                    if new_budget<=20:
                        accepted=candidate;source='partial_constrained';semantic=True;reason='REOBSERVED_ANCHOR_PARTIAL_ACCEPTED';stats=refined;budget=new_budget
                else:reason=refined['reason'];stats=refined
        if epoch_reset and accepted is None:reason='HARD_CUT'
        self.memory=self.memory.model_copy(update={'pan_direction':'right' if motion.get('pan_small_px',[0])[0]>0 else 'left' if motion.get('pan_small_px',[0])[0]<0 else 'stationary_or_unknown','scale_range':(motion['min_local_scale'],motion['max_local_scale']) if 'min_local_scale' in motion else None})
        status='RELOCALIZED' if direct and event=='relocalization' else 'DIRECT' if direct else 'PARTIAL_CONSTRAINED' if accepted is not None and source=='partial_constrained' else 'TEMPORAL_TRACKED' if accepted is not None else 'UNKNOWN'
        if accepted is None:
            failures=(self.memory.failure_history+[CameraFailure(frame=base.frame,timestamp=base.timestamp,reason=reason)])[-12:];self.memory=self.memory.model_copy(update={'failure_history':failures})
            event='cut_reset' if epoch_reset else 'loss' if previous is not None and previous.frame==base.frame-1 else None
        else:
            accepted=accepted/accepted[2,2];region=safe_region(accepted,self.width,self.height);support=float(np.clip(1-(stats.get('p90') or motion.get('heldout_p90_small_px',2))/40,0,1))
            estimate=CameraEstimate(frame=base.frame,timestamp=base.timestamp,epoch=base.epoch,transform=tuple(accepted.flat),source=source,visible_region=[tuple(p) for p in region[1]],landmark_ids=classes,support=support,motion_error_budget_px=budget)
            updates={'last_accepted':estimate,'accepted_history':(self.memory.accepted_history+[estimate])[-12:]}
            if direct:updates.update(last_direct=estimate,last_constraint_timestamp=base.timestamp)
            elif source=='partial_constrained':updates['last_constraint_timestamp']=base.timestamp
            self.memory=self.memory.model_copy(update=updates);self.accepted_field=current_field
            if not direct:event='partial_constraint' if source=='partial_constrained' else 'tracking'
        last_direct=self.memory.last_direct;age_direct=base.timestamp-last_direct.timestamp if last_direct is not None and last_direct.epoch==base.epoch else None
        loc={'status':status,'outcome':reason,'absolute_outcome':absolute,'absolute_attempted':diagnostic.get('solver_attempted','heldout_mean_px' in diagnostic),'tracking_blockers':[] if accepted is not None else [reason],'semantic_classes':classes,'transition':event,'quality':'heuristic direct fit' if direct else 'current partial landmarks + measured camera motion' if source=='partial_constrained' and accepted is not None else 'fresh pitch-feature motion; limited semantic dropout' if accepted is not None else 'unavailable','last_absolute_age_seconds':age_direct,'camera_epoch':base.epoch,'current_semantic_support':semantic,'support_score':self.memory.last_accepted.support if accepted is not None else None,'relative_motion_error_budget_px':budget if accepted is not None else None,'landmark_validation':stats,'semantic_constraint_upper_bound':min(8,2*sum(k in self.geometry.segments for k in classes)+5*sum(k in self.geometry.circle_centers for k in classes))}
        evidence={**base.motion_evidence,'pose':diagnostic,'field_motion':motion,'anchor_measurement':anchor_diagnostic or {},'localization':loc}
        common={'motion_evidence':evidence,'memory':self.memory,'last_reliable_timestamp':last_direct.timestamp if last_direct else None,'last_reliable_transform':last_direct.transform if last_direct else None}
        if accepted is None:return base.model_copy(update={**common,'calibration':None,'calibration_valid':False,'calibration_status':'lost' if last_direct else 'unknown','visible_pitch_region':None,'projected_pitch_geometry':[],'source':reason})
        calibration=PitchCalibrationObservation(start_frame=base.frame,end_frame=base.frame,image_to_pitch=tuple(accepted.flat),validity_region=region[0],provenance=Provenance(provider='cv-camera-m5',source=loc['quality'],quality='inferred' if direct else 'propagated'))
        return base.model_copy(update={**common,'calibration':calibration,'calibration_valid':True,'calibration_status':'valid' if direct else 'partial','calibration_source_kind':source,'visible_pitch_region':[tuple(p) for p in region[1]],'projected_pitch_geometry':[ImagePitchSegment(**s) for s in self.geometry.overlay(accepted,self.width,self.height)],'source':loc['quality']})
