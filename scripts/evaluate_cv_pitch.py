"""Reference-supported line diagnostics; no reference-to-automatic label leakage.

Accept already decoded frames. Does not download or modify Azure artifacts.
"""
import argparse,ast,json,time
from pathlib import Path
import cv2
import numpy as np
from src.vision.world.pitch import PitchPerception,TemporalCamera
from src.vision.coordinate_transform import PitchCoordinateTransformer

def evaluate(keypoints,frames,output,semantic_model=None,model_sha256=None):
    raw=json.loads(Path(keypoints).read_text());pairs=[(ast.literal_eval(k),v) for k,v in raw.items()]
    segments=[]
    for axis in (0,1):
        for value in sorted({p[0][axis] for p in pairs}):
            group=sorted([p for p in pairs if p[0][axis]==value],key=lambda p:p[0][1-axis])
            # Only sampled pitch grid lines represented in supplied calibration.
            for a,b in zip(group,group[1:]):segments.append((np.array(a[1],float),np.array(b[1],float)))
    camera=TemporalCamera();perception=PitchPerception();result=[];all_errors=[]
    semantic=None
    if semantic_model:
        from src.vision.world.semantic_pitch import SemanticPitch
        semantic=SemanticPitch(semantic_model,model_sha256)
    for index,path in enumerate(frames):
        image=cv2.imread(str(path));start=time.perf_counter();points=None
        if semantic:e,points=semantic.detect(image)
        else:e=perception.detect(image)
        c=camera.update(image,index,index/25,e,points,'Automatic semantic model; reference points excluded from fitting');elapsed=time.perf_counter()-start
        errors=[]
        for pitch,image_point in pairs:
            projected=camera.project(image_point,c)
            if projected is not None:errors.append(float(np.linalg.norm(np.array(projected)-pitch)))
        all_errors.extend(errors)
        distances=[]
        for line in e.lines:
            for p in (line.start,line.end):
                p=np.array(p);best=float('inf')
                for a,b in segments:
                    d=b-a;t=np.clip(np.dot(p-a,d)/max(np.dot(d,d),1e-8),0,1)
                    best=min(best,float(np.linalg.norm(p-(a+t*d))))
                distances.append(best)
        result.append(dict(frame_file=str(path),automatic_status=c.calibration_status,lines=len(e.lines),
            projected_reference_points=len(errors),reference_points=len(pairs),
            metric_projection_disagreement_m=errors,median_error_m=float(np.median(errors)) if errors else None,p95_error_m=float(np.quantile(errors,.95)) if errors else None,
            reference_supported_line_distance_px={'median':float(np.median(distances)) if distances else None,'p95':float(np.quantile(distances,.95)) if distances else None,
                'within_8px_fraction':sum(d<=8 for d in distances)/len(distances) if distances else None},processing_ms=elapsed*1000))
    report=dict(recording_id='118575_clip',sampled_timestamps_seconds=[10,60,120],classification='REFERENCE-SUPPORTED / DEPENDENT diagnostic, not independent accuracy',
        method='Anonymous line endpoints compared with segments between supplied SoccerTrack grid points; not all grid lines are painted field markings.',
        automatic_calibrated_frames=sum(r['automatic_status']=='valid' for r in result),projection_disagreement_median=float(np.median(all_errors)) if all_errors else None,projection_disagreement_p95=float(np.quantile(all_errors,.95)) if all_errors else None,
        temporal_stability='UNAVAILABLE: three sparse frames, not consecutive camera evaluation',
        reason='SoccerTrack panorama is nonlinear; a global automatic homography may be unavailable or disagree. Reference calibration is excluded from inference. Coverage is limited to the fitted support hull.',frames=result)
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--keypoints',required=True);p.add_argument('--frames',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--semantic-model');p.add_argument('--model-sha256');a=p.parse_args();evaluate(a.keypoints,a.frames,a.output,a.semantic_model,a.model_sha256)
