"""Soft local size/overlap diagnostics. Never discard a person for pixel height."""
import cv2
import numpy as np
from .schema import SizeEvidence
from .tracking import iou
from .pitch import TemporalCamera

def project_player(point,camera):
    if not np.isfinite(point).all():return None,'INVALID_FOOT_POINT'
    if not camera.calibration_valid or camera.calibration is None:return None,'NO_CAMERA_GEOMETRY'
    polygon=np.array(camera.calibration.validity_region,np.float32)
    if len(polygon)<3:return None,'NO_CAMERA_GEOMETRY'
    if cv2.pointPolygonTest(polygon,tuple(map(float,point)),False)<0:return None,'OUTSIDE_CALIBRATION_REGION'
    try:
        h=np.array(camera.calibration.image_to_pitch,float).reshape(3,3);v=h@[*point,1.]
        if not np.isfinite(v).all() or abs(v[2])<1e-10:return None,'TRANSFORM_FAILED'
        p=v[:2]/v[2]
        if not np.isfinite(p).all() or np.any(p<[0,0]) or np.any(p>[105,68]):return None,'OUTSIDE_CANONICAL_PITCH'
        return tuple(map(float,p)),'PROJECTED'
    except (ValueError,np.linalg.LinAlgError):return None,'TRANSFORM_FAILED'

def assess_sizes(detections,camera,width,height):
    projected=[project_player(d.foot,camera)[0] for d in detections];result=[]
    for i,d in enumerate(detections):
        flags=[];x,y,w,h=d.bbox
        if x<=1 or y<=1 or x+w>=width-1 or y+h>=height-1:flags.append('EDGE_TRUNCATION')
        if h/w<1 or h/w>6:flags.append('UNUSUAL_ASPECT_RATIO')
        if any(i!=j and iou(d.bbox,o.bbox)>.1 for j,o in enumerate(detections)):flags.append('OVERLAPPING_BOXES')
        for j,o in enumerate(detections):
            if i==j:continue
            ox,oy,ow,oh=o.bbox
            intersection=max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
            if intersection/(w*h)>.9 and h<.7*oh and abs(x+w/2-ox-ow/2)<.25*ow:flags.append('NESTED_BOX_DUPLICATE_CANDIDATE');break
        peers=[o.bbox[3] for j,o in enumerate(detections) if i!=j and abs(o.foot[1]-d.foot[1])<height*.12]
        source='image-space neighboring foot-depth statistics'
        if projected[i] is not None:
            metric=[o.bbox[3] for j,o in enumerate(detections) if i!=j and projected[j] is not None and np.linalg.norm(np.array(projected[i])-projected[j])<18 and abs(o.foot[1]-d.foot[1])<height*.15]
            if len(metric)>=3:peers=metric;source='calibrated ground-neighborhood + image-depth statistics; not physical person height'
        reference=float(np.median(peers)) if len(peers)>=3 else None;ratio=h/reference if reference else None
        if ratio is not None and (ratio>2.2 or ratio<.4):flags.append('LOCAL_SIZE_OUTLIER')
        support=.65 if 'LOCAL_SIZE_OUTLIER' in flags else .85 if 'UNUSUAL_ASPECT_RATIO' in flags else 1.
        result.append(d.model_copy(update={'size_evidence':SizeEvidence(reference_height=reference,height_ratio=ratio,source=source if reference else 'insufficient local support',support=support,flags=flags)}))
    return result
