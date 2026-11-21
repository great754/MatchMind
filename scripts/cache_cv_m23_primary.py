"""Freeze a primary-protected proposal cache from actual M23 neural measurements.

M22 proposals may bridge bounded primary gaps offline; they cannot replace the
primary visual path or feed themselves back into causal search memory.
No neural output, box, confidence, frame or human reference is changed.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import hashlib
import cv2
import numpy as np
from src.serving.cv import digest
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write, NAMES, Inputs
from src.vision.advanced21.temporal import batch_visual_features, deterministic_verdict


def run(parent, output, inputs_root):
    parent, output = Path(parent), Path(output)
    if output.exists():raise ValueError('Use a new immutable cache root')
    report = json.loads((parent/'report.json').read_text())
    inputs = Inputs(inputs_root)
    if inputs.sha256 != report['input_manifest_sha256']:raise ValueError('Pinned input snapshot changed')
    if tuple(report['clips']) != NAMES:raise ValueError('All approved clips must be complete')
    output.mkdir(parents=True)
    frozen = copy.deepcopy(report)
    frozen['inputs_root'] = str(inputs_root)
    frozen['parent_perception'] = {'root':str(parent),'report_sha256':digest(parent/'report.json')}
    frozen['cache_derivation'] = {'script_sha256':digest(__file__),
        'primary_memory':'guided_verified', 'complement_causal_feedback':False,
        'neural_measurements_changed':False,'new_neural_inference':False,
        'description':'Primary current-pixel search and memory are protected; M22 only supplies bounded offline gap proposals.'}
    frozen['config']['offline']['primary_anchored_complement_only'] = True
    frozen['config']['offline']['m22_dual_current_visual_proposals_require_two_primary_anchors'] = True
    frozen['config_sha256'] = signature(frozen['config'])
    for name in NAMES:
        tick = time.perf_counter()
        source = parent/name/'inputs.jsonl'
        if digest(source) != report['clips'][name]['inputs_sha256']:raise ValueError('Parent measurements changed')
        directory = output/name
        directory.mkdir()
        metadata, _, _ = inputs.worlds(name)
        global_path=inputs.path(Path(inputs.manifest['m22_global_root'])/name/'detections.jsonl')
        global_rows=[json.loads(line) for line in global_path.open()]
        capture=cv2.VideoCapture(metadata['source_video'])
        previous=None
        with source.open() as original, (directory/'inputs.jsonl').open('w') as target:
            for line in original:
                row = json.loads(line)
                ok,image=capture.read()
                if not ok or hashlib.sha256(image.tobytes()).hexdigest()!=row['source_pixels_sha256']:
                    raise ValueError('Cached current pixels changed')
                global_row=global_rows[row['frame']]
                if global_row['current_frame_pixels_sha256']!=row['source_pixels_sha256'] or global_row['checkpoint_sha256']!=row['m22_checkpoint_sha256']:
                    raise ValueError('Complementary visual source/checkpoint changed')
                base = copy.deepcopy(row['methods']['guided_verified'])
                pool={c['candidate_id']:c for c in [*global_row['candidates'],*row['m22_candidates']]
                    if c['observation_eligible'] and c['confidence']>=.35 and c['current_only_confidence']>=.35}
                complements=[{'candidate_id':'m23_gap_'+c['candidate_id'],'bbox':c['bbox'],
                    'confidence':c['confidence'],'current_only_confidence':c['current_only_confidence'],
                    'inference_view':'M22_ACTUAL_CURRENT_PIXELS_PRIMARY_ANCHORED_GAP_PROPOSAL',
                    'source':'VIDEO_CV_DETECTOR','origin':'complement','visual_evidence':'CURRENT_PIXELS',
                    'source_region':None,'source_model_sha256':row['m22_checkpoint_sha256']}
                    for c in pool.values()]
                step=np.asarray(row['relative_camera_homography']) if row['relative_camera_homography'] is not None else None
                features=batch_visual_features(previous,image,complements,step)
                for c,feature in zip(complements,features,strict=True):
                    c.update(temporal_features=feature,deterministic=deterministic_verdict(feature,c['confidence']),
                        support_views=[{'origin':'complement','bbox':c['bbox'],'confidence':c['confidence'],'source_region':None}],
                        observation_eligible=True,decision_reason='DUAL_CURRENT_VISUAL_PROPOSAL_REQUIRES_TWO_SIDED_PRIMARY_ASSOCIATION',
                        joint_memory_evidence=None,blur_supported=False,
                        blur_definition='Not classified by the complementary proposal cache',m22_support=[],association_adjustment=0.,
                        detector_confidence_unchanged=True,confidence_is_calibrated_probability=False)
                base['candidates'].extend(complements)
                row['methods']['full_m23'] = base
                row['config_sha256'] = frozen['config_sha256']
                target.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                previous=image
        if capture.read()[0]:raise ValueError('Unindexed source frames')
        capture.release()
        elapsed = time.perf_counter()-tick
        clip=frozen['clips'][name]
        clip['parent_inputs_sha256']=report['clips'][name]['inputs_sha256']
        clip['inputs_sha256']=digest(directory/'inputs.jsonl')
        clip['cache_derivation_seconds']=elapsed
        clip['recorded_processing_seconds']+=elapsed
        clip['processing_fps']=clip['frames']/clip['recorded_processing_seconds']
    write(output/'report.json',frozen)
    print(json.dumps({'root':str(output),'report_sha256':digest(output/'report.json'),'fresh_neural_inference':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--inputs',required=True,type=Path)
    args=parser.parse_args();run(args.parent,args.output,args.inputs)
