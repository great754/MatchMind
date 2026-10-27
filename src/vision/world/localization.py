"""Per-frame camera fits and strictly bounded, explicitly propagated transforms."""
import cv2,numpy as np
from src.observations.schema import PitchCalibrationObservation,Provenance
from .pitch import TemporalCamera
from .pose import visible_region
from .semantic_pitch import GEOMETRY
from .diagnosis import absolute_outcome
from .camera_motion import projective_motion

class CameraLocalizer:
    def __init__(self,width,height,max_propagation_seconds=.25):
        self.camera=TemporalCamera();self.width=width;self.height=height;self.max_age=max_propagation_seconds
        self.transform=None;self.direct_time=None;self.last_direct_transform=None;self.last_direct_time=None
        self.previous_gray=None;self.previous_valid=False;self.previous_epoch=None;self.previous_kind="unknown"
    def state(self,image,frame,timestamp,evidence,fit=None,diagnostic=None):
        c=self.camera.update(image,frame,timestamp,evidence)
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(320,180))
        measured=projective_motion(self.previous_gray,gray) if self.previous_gray is not None else {'accepted':False,'reason':'NO_PREVIOUS_FRAME','matches':0}
        self.previous_gray=gray
        c=c.model_copy(update={'motion_evidence':{**c.motion_evidence,'projective_motion':measured}})
        return self.apply(c,evidence,fit,diagnostic or {})
    def apply(self,c,evidence,fit=None,diagnostic=None):
        was_valid=self.previous_valid;had_absolute=self.last_direct_time is not None
        epoch_reset=c.continuity=='cut' or (self.previous_epoch is not None and c.epoch!=self.previous_epoch)
        if epoch_reset:self.transform=None;self.direct_time=None
        diagnostic=diagnostic or {};source_kind='direct_calibrated';accepted=None
        motion=dict(c.motion_evidence);motion['pose']=diagnostic
        classes=sorted({l.kind for l in evidence.lines}|{v.kind for v in evidence.curves})
        absolute=absolute_outcome(fit,diagnostic,classes);blockers=[];prediction=None
        # Recheck finite geometry at the state boundary, including externally
        # supplied fits. Never accept a caller-provided polygon as validation.
        if fit is not None:
            try:
                matrix=np.asarray(fit['homography'],float)
                region=visible_region(matrix,self.width,self.height) if matrix.shape==(3,3) and np.isfinite(matrix).all() and abs(matrix[2,2])>1e-12 else None
            except (ValueError,TypeError,KeyError):region=None
            if region is None:absolute='INVALID_VISIBLE_REGION';fit=None
            else:fit={**fit,'image_region':region[0],'pitch_region':region[1]}
        if self.transform is not None and self.direct_time is not None:
            age=c.timestamp-self.direct_time;motion['propagation_age_seconds']=age
            flow=motion.get('image_affine');matches=motion.get('feature_matches',0);inliers=motion.get('affine_inliers',0);error=motion.get('affine_median_error_small_px')
            if not 0<age<=self.max_age:blockers.append('PROPAGATION_EXPIRED')
            if c.continuity!='continuous' or flow is None:blockers.append('TEMPORAL_TRACKING_FAILED')
            if matches<50:blockers.append('INSUFFICIENT_FEATURE_MATCHES')
            if inliers/max(matches,1)<.8:blockers.append('INSUFFICIENT_MOTION_INLIERS')
            if error is None or error>1:blockers.append('MOTION_RESIDUAL_TOO_HIGH')
            measured=motion.get('projective_motion')
            if measured is not None:
                if not measured['accepted']:blockers.append(measured['reason'])
                small=np.asarray(measured.get('homography',np.eye(3)),float)
            else:
                # Legacy/synthetic callers retain their measured affine contract.
                small=np.vstack([flow,[0,0,1]]) if flow is not None else np.eye(3)
                scales=np.linalg.svd(small[:2,:2],compute_uv=False)
                if not np.isfinite(scales).all() or scales.min()<.98 or scales.max()>1.02:blockers.append('ZOOM_OR_SCALE_CHANGE_UNSUPPORTED')
            if not blockers:
                scale=np.diag([self.width/320,self.height/180,1]);image_motion=scale@small@np.linalg.inv(scale)
                try:prediction=self.transform@np.linalg.inv(image_motion)
                except np.linalg.LinAlgError:blockers.append('IMPOSSIBLE_MOTION_TRANSFORM')
                if prediction is not None:
                    checks=line_agreement(prediction,evidence.lines);motion['propagated_line_agreement_px']=checks
                    if checks is None:blockers.append('INSUFFICIENT_GEOMETRIC_CONSTRAINTS')
                    elif checks['mean']>12 or checks['p90']>24:blockers.append('TRACKED_PITCH_RESIDUAL_TOO_HIGH')
                    region=visible_region(prediction,self.width,self.height)
                    if region is None:blockers.append('INVALID_VISIBLE_REGION')
                    if not blockers and fit is None:
                        accepted={'homography':prediction.tolist(),'image_region':region[0],'pitch_region':region[1]};self.transform=prediction;source_kind='camera_propagated'
        elif fit is None:blockers.append('NO_ACTIVE_TRACKING_ANCHOR')
        if epoch_reset:blockers.append('CAMERA_CUT')
        if fit is not None:
            accepted=fit
            if prediction is not None:
                # Dependent drift diagnostic, never a ground-truth error.
                points=np.array([[(self.width*x,self.height*y) for y in (.3,.6,.9) for x in (.2,.5,.8)]],float)
                delta=np.linalg.norm(cv2.perspectiveTransform(points,prediction)-cv2.perspectiveTransform(points,np.asarray(fit['homography'],float)),axis=2)
                motion['relocalization_correction_p90_m']=float(np.quantile(delta,.9)) if np.isfinite(delta).all() else None
            self.transform=np.array(fit['homography'],float);self.direct_time=c.timestamp;self.last_direct_transform=self.transform.copy();self.last_direct_time=c.timestamp
        outcome='DIRECT_ACCEPTED' if fit else 'TEMPORALLY_TRACKED' if accepted else ('CAMERA_CUT' if epoch_reset else blockers[0] if blockers and blockers[0]!='NO_ACTIVE_TRACKING_ANCHOR' else absolute)
        transition=('cut_reset' if epoch_reset else 'relocalization' if fit and had_absolute and self.previous_epoch==c.epoch and (not was_valid or self.previous_kind=='camera_propagated') else 'acquisition' if fit and not was_valid else 'tracking' if accepted and source_kind=='camera_propagated' and self.previous_kind!='camera_propagated' else 'loss' if not accepted and was_valid else None)
        status='RELOCALIZED' if transition=='relocalization' else 'DIRECT' if fit else 'TRACKED' if accepted else 'UNKNOWN'
        motion['localization']={'status':status,'outcome':outcome,'absolute_outcome':absolute,'absolute_attempted':diagnostic.get('solver_attempted','heldout_mean_px' in diagnostic),
            'tracking_blockers':blockers,'semantic_classes':classes,'transition':transition,'quality':'heuristic direct fit' if fit else 'measured motion; increased uncertainty' if accepted else 'unavailable',
            'last_absolute_age_seconds':c.timestamp-self.last_direct_time if self.last_direct_time is not None else None,'camera_epoch':c.epoch}
        self.previous_valid=accepted is not None;self.previous_epoch=c.epoch;self.previous_kind=source_kind if accepted else 'unknown'
        if accepted:
            h=np.array(accepted['homography']);h=h/h[2,2]
            source='Experimental direct semantic line/curve camera fit' if source_kind=='direct_calibrated' else 'Bounded optical-motion camera propagation; increased uncertainty; not a direct fit'
            calibration=PitchCalibrationObservation(start_frame=c.frame,end_frame=c.frame,landmarks=[l.start for l in evidence.lines],
                image_to_pitch=tuple(h.flat),validity_region=accepted['image_region'],provenance=Provenance(provider='cv-camera-m3',source=source,quality='inferred' if source_kind=='direct_calibrated' else 'propagated'))
            return c.model_copy(update={'calibration':calibration,'calibration_valid':True,'calibration_status':'valid' if source_kind=='direct_calibrated' else 'partial',
                'calibration_source_kind':source_kind,'visible_pitch_region':[tuple(p) for p in accepted['pitch_region']],'source':source,'motion_evidence':motion,
                'last_reliable_transform':tuple(self.last_direct_transform.flat),'last_reliable_timestamp':self.last_direct_time})
        # Never carry a stale current position or unmodified transform forward.
        self.transform=None
        return c.model_copy(update={'calibration':None,'calibration_valid':False,'visible_pitch_region':None,
            'calibration_status':'lost' if self.direct_time is not None else c.calibration_status,'motion_evidence':motion,
            'last_reliable_timestamp':self.last_direct_time,'last_reliable_transform':tuple(self.last_direct_transform.flat) if self.last_direct_transform is not None else None})

def line_agreement(h,lines):
    try:inverse=np.linalg.inv(h)
    except np.linalg.LinAlgError:return None
    errors=[];classes=set()
    for line in lines:
        if line.kind not in GEOMETRY:continue
        axis,value=GEOMETRY[line.kind];points=np.array([[value,0],[value,68]] if axis==0 else [[0,value],[105,value]],np.float64)
        pixels=cv2.perspectiveTransform(points[None],inverse)[0]
        equation=np.cross([*pixels[0],1],[*pixels[1],1]);norm=np.linalg.norm(equation[:2])
        if not np.isfinite(equation).all() or norm<1e-8:continue
        classes.add(line.kind)
        errors.extend(abs(np.dot(equation,[*p,1]))/norm for p in (line.start,line.end))
    if len(classes)<2 or not errors or not np.isfinite(errors).all():return None
    return {'mean':float(np.mean(errors)),'p90':float(np.quantile(errors,.9)),'classes':len(classes)}
