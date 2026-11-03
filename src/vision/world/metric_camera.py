"""M8 motion predicts / independently observed metric pitch landmarks correct."""
import numpy as np
from .world_field_camera import WorldFieldCamera
from .temporal_camera import transform,disagreement
from .metric_landmarks import solve_midfield,solve_penalty,handedness
from .metric_motion import MetricTrajectory
from .static_geometry import measure
from .persistent_camera import safe_region
from .schema import CameraEstimate,CameraMemory
from .temporal_camera import TemporalCameraFilter

class MetricCamera(WorldFieldCamera):
 def publish(self,base,h,state,reason,debug,age=None):
  c=super().publish(base,h,state,reason,debug,age)
  if c.calibration is not None:c=c.model_copy(update={'calibration':c.calibration.model_copy(update={'provenance':c.calibration.provenance.model_copy(update={'provider':'cv-camera-m8'})})})
  return c
 def attach_memory(self,result,raw):
  # M7 may preserve an anchor hypothesis that differs from the camera selected
  # at that anchor's own frame. Resolve the stored certificate, not that image's
  # current partial pose. Ambiguous missing lineage cannot authorize recovery.
  result=TemporalCameraFilter.attach_memory(self,result,raw)
  registry={}
  for old in self._baseline:
   proof=old.motion_evidence['temporal_camera'].get('independent_anchor_proof')
   if proof:registry.setdefault((old.epoch,proof['frame']),[]).append(proof)
  for i,c in enumerate(result):
   d=c.motion_evidence['temporal_camera'];origin=d.get('absolute_anchor_frame')
   if self._baseline[i].calibration_valid:
    old=self._baseline[i];result[i]=c.model_copy(update={'memory':old.memory,'last_reliable_transform':old.last_reliable_transform,'last_reliable_timestamp':old.last_reliable_timestamp})
    if old.motion_evidence['temporal_camera'].get('independent_anchor_proof'):d['independent_anchor_proof']=old.motion_evidence['temporal_camera']['independent_anchor_proof']
    continue
   if not c.calibration_valid or origin is None:continue
   a=raw[origin];certificate=a.motion_evidence.get('temporal_camera',{}).get('landmark_m8',{});proof=None
   if d.get('keyframe_origin_proof') is not None and d['keyframe_origin_proof']['frame']==origin:proof=d['keyframe_origin_proof']
   elif certificate.get('accepted'):
    proof={'kind':'M8_CURRENT_METRIC_LANDMARK','frame':origin,'transform':list(a.calibration.image_to_pitch),'source':certificate['reason']}
   elif a.memory is not None and a.memory.last_direct is not None and a.memory.last_direct.frame==origin:
    proof={'kind':'M7_PRESERVED_INDEPENDENT','frame':origin,'transform':list(a.memory.last_direct.transform),'source':a.memory.last_direct.source}
   else:
    options=registry.get((c.epoch,origin),[]);unique={tuple(np.round(p['transform'],7)):p for p in options}
    if len(unique)==1:proof=next(iter(unique.values()))
   if proof is None:
    result[i]=self.publish(c,None,'UNKNOWN','M8_UNPROVEN_INDEPENDENT_ORIGIN',{'groups':self._groups[i],'field_motion':d['field_motion'],'direction':d['direction'],'m8_origin_rejection':origin});continue
   proof={**proof,'future_evidence':origin>i};d['independent_anchor_proof']=proof
   estimate=CameraEstimate(frame=origin,timestamp=a.timestamp,epoch=c.epoch,transform=tuple(proof['transform']),source=proof['source'],visible_region=a.visible_pitch_region or [],landmark_ids=a.motion_evidence['localization']['semantic_classes'],support=0.,motion_error_budget_px=0.)
   memory=c.memory or CameraMemory(epoch=c.epoch);result[i]=c.model_copy(update={'memory':memory.model_copy(update={'last_direct':estimate}),'last_reliable_transform':estimate.transform,'last_reliable_timestamp':estimate.timestamp})
  return result
 def recover_keyframes(self,result,groups,fields):
  proposals={};attempts=[]
  for direction,indices in [('forward',range(len(result))),('backward',range(len(result)-1,-1,-1))]:
   anchor=None
   for i in indices:
    raw=result[i]
    if raw.calibration_valid:anchor=i;continue
    if anchor is None or abs(i-anchor)/self.fps>1 or raw.epoch!=result[anchor].epoch:continue
    lo,hi=sorted((i,anchor))
    if any(result[j].continuity=='cut' for j in range(lo+1,hi+1)):continue
    measured=measure(fields[anchor],fields[i],groups[anchor],groups[i],self.width,self.height,interval_frames=abs(i-anchor))
    record={'frame':i,'anchor_frame':anchor,'direction':direction,'motion':measured,'accepted':False}
    if not measured.get('accepted'):attempts.append(record);continue
    prior=result[anchor];prediction=transform(prior)@np.linalg.inv(np.array(measured['homography']));budget=float(np.hypot(prior.motion_evidence['temporal_camera'].get('relative_motion_error_budget_px') or 0,measured['heldout_p90_small_px']*measured['scale_to_source']))
    valid,stats=(None,{}) if safe_region(prediction,self.width,self.height) is None else (prediction,self.validator.statistics(prediction,groups[i]))
    if not stats.get('accepted') or budget>20:record['reason']='CURRENT_GEOMETRY_OR_UNCERTAINTY_REJECTED';attempts.append(record);continue
    origin=prior.motion_evidence['temporal_camera'].get('absolute_anchor_frame');age=abs(raw.timestamp-result[origin].timestamp) if origin is not None else float('inf')
    if age>10:record['reason']='ABSOLUTE_SUPPORT_BACKSTOP';attempts.append(record);continue
    debug={'groups':groups[i],'field_motion':measured,'direction':direction,'keyframe_remeasurement':record,'keyframe_origin_proof':prior.motion_evidence['temporal_camera'].get('independent_anchor_proof'),'absolute_anchor_frame':origin,'relative_motion_error_budget_px':budget,'previous_region':prior.visible_pitch_region,'predicted_region':safe_region(prediction,self.width,self.height)[1],'previous_timestamp':prior.timestamp,'current_timestamp':raw.timestamp,'landmark_m8':raw.motion_evidence['temporal_camera'].get('landmark_m8',{})}
    # Avoid a self-reference: keep the immutable attempt separate from debug.
    record['accepted']=True;debug['keyframe_remeasurement']={k:v for k,v in record.items() if k!='motion'}
    candidate=self.publish(raw,prediction,'TEMPORAL_PREDICTED','M8_FRESH_KEYFRAME_FIELD_REMEASUREMENT',debug,raw.timestamp-result[origin].timestamp);proposals.setdefault(i,[]).append(candidate);attempts.append(record)
  for i,options in proposals.items():
   if len(options)==2 and disagreement(transform(options[0]),transform(options[1]),self.width,self.height)>1:continue
   result[i]=min(options,key=lambda c:c.motion_evidence['temporal_camera']['relative_motion_error_budget_px'])
  return result,attempts
 def run(self,cameras,groups,forward,backward,fields=None,feature_measurements=None,backward_measurements=None,absolute_candidates=None):
  self._baseline=cameras;self._groups=groups
  prepared=[];landmarks=[];orientation=None;epoch=None;orientation_origin=None
  for i,raw in enumerate(cameras):
   if epoch!=raw.epoch or raw.continuity=='cut':orientation=None;orientation_origin=None
   epoch=raw.epoch
   # Handedness is a topological epoch property, never a stale metric pose.
   if raw.calibration_valid and raw.calibration_source_kind=='direct_calibrated':orientation=handedness(transform(raw),(self.width/2,self.height*.65));orientation_origin=i
   if not raw.calibration_valid:
    h,d=solve_midfield(groups[i],self.width,self.height,orientation);circle_d=d
    if h is None:
     h,penalty=solve_penalty(groups[i],self.width,self.height,orientation)
     if h is not None:d={**penalty,'circle_attempt':circle_d}
     else:d={**circle_d,'penalty_attempt':penalty}
    d['orientation_origin_frame']=orientation_origin
    raw=raw.model_copy(update={'motion_evidence':{**raw.motion_evidence,'temporal_camera':{**raw.motion_evidence['temporal_camera'],'landmark_m8':d}}})
    if h is not None and safe_region(h,self.width,self.height) is not None:
     debug={'groups':groups[i],'field_motion':forward[i],'direction':'absolute-landmarks','absolute_anchor_frame':i,'absolute_anchor_timestamp':raw.timestamp,'anchor_age_seconds':0.,'landmark_m8':d,'relative_motion_error_budget_px':0.}
     raw=self.publish(raw,h,'RELOCALIZED','M8_ABSOLUTE_METRIC_'+d['kind'],debug,0.)
    landmarks.append({'frame':i,**d})
   prepared.append(raw)
   if i%100==0:print('M8 landmark hypotheses',i,'/',len(cameras),flush=True)
  # Motion can initialize a fresh absolute solve when ellipse-only starts
  # fall into a wrong local minimum. All current conic/line gates still apply.
  fused_attempts=[]
  for step,motions in [(1,forward),(-1,backward)]:
   indices=range(len(prepared)) if step==1 else range(len(prepared)-1,-1,-1);anchor=None
   for i in indices:
    if prepared[i].calibration_valid:anchor=i;continue
    if anchor is None or abs(i-anchor)/self.fps>1 or prepared[i].epoch!=prepared[anchor].epoch or not all(n in groups[i] for n in ['Circle central','Middle line','Side line top']):continue
    lo,hi=sorted((i,anchor))
    if any(prepared[j].continuity=='cut' for j in range(lo+1,hi+1)):continue
    measured=measure(fields[anchor],fields[i],groups[anchor],groups[i],self.width,self.height,interval_frames=abs(i-anchor))
    if not measured.get('accepted'):continue
    prediction=transform(prepared[anchor])@np.linalg.inv(np.array(measured['homography']));orientation=handedness(prediction,(self.width/2,self.height*.65));h,d=solve_midfield(groups[i],self.width,self.height,orientation,initial_h=prediction);d['motion_prior_frame']=anchor;d['motion_measurement']=measured;d['orientation_origin_frame']=anchor
    delta=disagreement(h,prediction,self.width,self.height) if h is not None else None;d['metric_correction_p90_m']=delta
    if h is not None and delta<=2 and safe_region(h,self.width,self.height) is not None:
     debug={'groups':groups[i],'field_motion':measured,'direction':'motion-landmark-fused','absolute_anchor_frame':i,'absolute_anchor_timestamp':prepared[i].timestamp,'anchor_age_seconds':0.,'landmark_m8':d,'relative_motion_error_budget_px':0.,'predicted_region':safe_region(prediction,self.width,self.height)[1],'previous_region':prepared[anchor].visible_pitch_region}
     prepared[i]=self.publish(prepared[i],h,'RELOCALIZED','M8_MOTION_INITIALIZED_ABSOLUTE_LANDMARK',debug,0.)
     for j,record in enumerate(landmarks):
      if record['frame']==i:landmarks[j]={'frame':i,'cold_attempt':record,**d};break
    elif h is not None:d.update(accepted=False,reason='MOTION_LANDMARK_CORRECTION_INCONSISTENT')
    fused_attempts.append({'frame':i,**d})
  a=self.directional(prepared,groups,forward,fields,feature_measurements);b=self.directional(prepared,groups,backward,fields,backward_measurements,True);result=[];decisions=[]
  for i,baseline in enumerate(cameras):
   left,right=a[i],b[i];choice=left if left.calibration_valid else right
   record={'frame':i,'forward_accepted':left.calibration_valid,'backward_accepted':right.calibration_valid}
   if not baseline.calibration_valid and left.calibration_valid and right.calibration_valid:
    delta=disagreement(transform(left),transform(right),self.width,self.height);record['agreement_p90_m']=delta
    if delta>1:
     choice=self.publish(baseline,None,'UNKNOWN','M8_FORWARD_BACKWARD_DISAGREEMENT',{'groups':groups[i],'field_motion':forward[i],'direction':'bidirectional','forward_backward_agreement_p90_m':delta},None)
    else:
     choice=min((left,right),key=lambda c:c.motion_evidence['temporal_camera'].get('trajectory_uncertainty_px',20));choice.motion_evidence['temporal_camera']['forward_backward_agreement_p90_m']=delta
   # Existing validated M7 poses are immutable regression references. M8
   # extends unknown intervals; it does not improve coverage by retagging them.
   if baseline.calibration_valid:choice=baseline.model_copy(deep=True)
   d=choice.motion_evidence['temporal_camera'];d['m8_decision']=record
   origin=d.get('absolute_anchor_frame')
   if origin is not None and prepared[origin].motion_evidence.get('temporal_camera',{}).get('landmark_m8',{}).get('accepted'):
    d['m8_independent_anchor']=prepared[origin].motion_evidence['temporal_camera']['landmark_m8']
   if prepared[i].motion_evidence.get('temporal_camera',{}).get('landmark_m8') is not None:d['landmark_m8']=prepared[i].motion_evidence['temporal_camera']['landmark_m8']
   result.append(choice);decisions.append(record)
  # Parent origin attachment preserves M7/M6 proofs; new origins have their
  # own current conic/line certificate, never inherited camera coordinates.
  result=self.attach_memory(result,prepared)
  result,gap_attempts=self.recover_keyframes(result,groups,fields)
  result=self.attach_memory(result,prepared)
  for c in result:
   d=c.motion_evidence['temporal_camera']
   if d.get('m8_independent_anchor') and d.get('independent_anchor_proof'):d['independent_anchor_proof']['kind']='M8_CURRENT_METRIC_LANDMARK'
  trajectory=MetricTrajectory()
  for i,c in enumerate(result):
   d=c.motion_evidence['temporal_camera'];certificate=d.get('landmark_m8',{});metric_state='UNKNOWN' if not c.calibration_valid else 'LANDMARK_RELOCALIZED' if certificate.get('accepted') else 'DIRECT' if c.calibration_source_kind=='direct_calibrated' else 'MOTION_LANDMARK_FUSED' if c.source=='WORLD_TRAJECTORY_BOUNDED_SEMANTIC_CORRECTION' else 'PARTIAL' if c.calibration_source_kind=='partial_constrained' else 'MOTION_PROPAGATED'
   source='LANDMARK_RELOCALIZED_PROJECTED' if metric_state=='LANDMARK_RELOCALIZED' else 'DIRECT_PROJECTED' if metric_state=='DIRECT' else 'MOTION_LANDMARK_FUSED_PROJECTED' if metric_state=='MOTION_LANDMARK_FUSED' else 'PARTIAL_CONSTRAINED_PROJECTED' if metric_state=='PARTIAL' else 'MOTION_PROPAGATED_PROJECTED' if c.calibration_valid else 'UNKNOWN'
   relative_budget=d.get('relative_motion_error_budget_px') or 0
   semantic_error=c.motion_evidence['localization'].get('landmark_validation',{}).get('p90') or 0
   budget=float(np.hypot(relative_budget,semantic_error))
   metric=trajectory.update(i,c.epoch,transform(c),self.width,self.height,d.get('field_motion'),budget,d.get('absolute_anchor_frame'),c.continuity=='cut');metric.update(state=metric_state,placement_provenance=source,reason=c.source,current_visible_region=c.visible_pitch_region,predicted_visible_region=d.get('predicted_region'),previous_visible_region=d.get('previous_region'),landmarks=c.motion_evidence['localization']['semantic_classes'],independent_anchor_age_seconds=d.get('anchor_age_seconds'),landmark_certificate=certificate if certificate else None)
   c.motion_evidence['metric_camera']=metric
   if c.memory is not None:c=c.model_copy(update={'memory':c.memory.model_copy(update={'metric_trajectory':metric})})
   result[i]=c
  return result,{'algorithm':'metric viewing trajectory + current conic/line absolute relocalization; frozen M7 accepted poses and people','motion_landmark_attempts':fused_attempts,'keyframe_remeasurement_attempts':gap_attempts,'landmark_candidates':landmarks,'decisions':decisions,'geometry_inconsistent_frames':[],'independent_physical_camera_translation':'not estimated'}
