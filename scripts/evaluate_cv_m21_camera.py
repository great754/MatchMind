"""Raw versus smoothed relative camera fits; optical-flow support is NOT landmark truth."""
import argparse
import json
import time
import cv2
import numpy as np
from src.serving.cv import digest
from src.vision.advanced21.camera import refine_motion,smooth_relative
from src.vision.advanced21.experiment import NAMES,fresh_directory,source_worlds,write_json


def run(output):
    root=fresh_directory(output);report={'pitch_dimensions_m':[105,68],'absolute_mapping_changed':False,
        'independent_landmark_reprojection_accuracy':None,'smoothing_time_constant_seconds':.08,
        'metric_definition':'In-sample bidirectionally tracked field-feature residuals; not independent pitch-landmark accuracy.',
        'clips':{}}
    for name in NAMES:
        metadata,worlds,people=source_worlds(name);capture=cv2.VideoCapture(metadata['source_video'])
        previous=previous_camera=None;old_people=[];smoothed=None;raw_errors=[];smooth_errors=[];rows=[];started=time.perf_counter()
        try:
            for i,world in enumerate(worlds):
                ok,image=capture.read()
                if not ok:raise ValueError('Source video incomplete')
                camera=world['camera']
                if previous_camera is None or previous_camera['epoch']!=camera['epoch'] or camera.get('continuity')=='cut':smoothed=None
                result=refine_motion(previous,image,previous_camera,camera,old_people,include_support=True)
                if result['accepted']:
                    raw=np.asarray(result['homography']);smoothed=smooth_relative(raw,smoothed,metadata['width'],metadata['height'],metadata['fps'])
                    src=np.float32(result.pop('support_source_pixels'));dst=np.asarray(result.pop('support_target_pixels'))
                    raw_error=np.linalg.norm(cv2.perspectiveTransform(src[:,None,:],raw).reshape(-1,2)-dst,axis=1)
                    smooth_error=np.linalg.norm(cv2.perspectiveTransform(src[:,None,:],smoothed).reshape(-1,2)-dst,axis=1)
                    raw_errors.extend(raw_error.tolist());smooth_errors.extend(smooth_error.tolist())
                    result.update(smoothed_homography=smoothed.tolist(),smoothed_internal_median_residual_px=float(np.median(smooth_error)))
                else:smoothed=None
                rows.append({'frame':i,**result});previous,previous_camera,old_people=image,camera,people[i]
        finally:capture.release()
        write_json(root/(name+'.json'),rows)
        report['clips'][name]={'video_sha256':metadata['video_sha256'],'world_sha256':metadata['frames_sha256'],
            'accepted_relative_frames':sum(r['accepted'] for r in rows),'frames':metadata['frames'],'field_feature_observations':len(raw_errors),
            'raw_internal_median_px':float(np.median(raw_errors)) if raw_errors else None,
            'smoothed_internal_median_px':float(np.median(smooth_errors)) if smooth_errors else None,
            'raw_internal_p90_px':float(np.quantile(raw_errors,.9)) if raw_errors else None,
            'smoothed_internal_p90_px':float(np.quantile(smooth_errors,.9)) if smooth_errors else None,
            'seconds':time.perf_counter()-started,'output_sha256':digest(root/(name+'.json')),
            'adopted_for_absolute_mapping':False,'independent_pitch_landmarks':0}
        write_json(root/'report.json',report);print('M21 CAMERA',name,json.dumps(report['clips'][name]),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(parser.parse_args().output)
