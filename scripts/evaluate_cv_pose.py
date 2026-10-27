"""Sparse real SoccerTrack calibration comparison. Reference never enters fitting."""
import argparse,ast,json,time
from pathlib import Path
import cv2,numpy as np
from src.vision.world.semantic_pitch import SemanticPitch
from src.vision.world.pose import PoseFitter

def evaluate(keypoints,images,output,model,model_sha256):
    reference=[(ast.literal_eval(k),p) for k,p in json.loads(Path(keypoints).read_text()).items()]
    semantic=SemanticPitch(model,model_sha256);rows=[];all_errors=[]
    for path in images:
        image=cv2.imread(str(path));height,width=image.shape[:2];tick=time.perf_counter();mask,confidence=semantic.masks(image);groups=PoseFitter.groups(mask,confidence)
        fit=None;diagnostic={'accepted':False,'reason':'Insufficient semantic evidence'}
        if PoseFitter.sufficient(groups):fit,diagnostic=PoseFitter(width,height).batch([groups])[0]
        errors=[]
        if fit:
            h=np.array(fit['homography']);polygon=np.array(fit['image_region'],np.float32)
            for pitch,pixel in reference:
                if cv2.pointPolygonTest(polygon,tuple(map(float,pixel)),False)<0:continue
                position=cv2.perspectiveTransform(np.array([[pixel]],float),h)[0,0]
                if np.isfinite(position).all():errors.append(float(np.linalg.norm(position-pitch)))
        all_errors.extend(errors)
        rows.append({'image':str(path),'image_shape':[width,height],'accepted':fit is not None,'source_classes':list(groups),'diagnostic':diagnostic,
            'projected_reference_points':len(errors),'metric_disagreements_m':errors,'seconds':time.perf_counter()-tick})
    result={'recording_id':'118575_clip','timestamps_seconds':[10,60,120],'classification':'REFERENCE-SUPPORTED calibration comparison; not independent player ground truth',
        'reference_excluded_from_inference':True,'samples':rows,'accepted_frames':sum(r['accepted'] for r in rows),'projected_reference_points':len(all_errors),
        'median_disagreement_m':float(np.median(all_errors)) if all_errors else None,'p90_disagreement_m':float(np.quantile(all_errors,.9)) if all_errors else None,
        'p95_disagreement_m':float(np.quantile(all_errors,.95)) if all_errors else None,
        'temporal_stability':'UNAVAILABLE: sparse frames, not a consecutive camera experiment','independent_accuracy':None,
        'limitations':['SoccerTrack panorama has nonlinear distortion; the candidate perspective camera may be rejected or disagree.','Only existing calibration/grid points are compared; no independently labeled player-foot accuracy.']}
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--keypoints',required=True);p.add_argument('--images',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--model',required=True);p.add_argument('--model-sha256',required=True);a=p.parse_args();evaluate(a.keypoints,a.images,a.output,a.model,a.model_sha256)
