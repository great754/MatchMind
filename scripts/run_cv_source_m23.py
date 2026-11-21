"""Run existing M23 on a new, explicitly selected video without reading labels.

Separate source artifacts never enter the frozen three-clip M23 evaluation.
Preprocessing weights must already exist; this command never downloads weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import time

import cv2
import numpy as np
from src.serving.cv import CVReview, digest
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write

SEMANTIC_SHA = 'ec551dc181692e1758c953e9bffde718664cebd19f39b674918f7a4588ca9066'


def source_world(video, root, models):
    from scripts.process_cv_video import run
    from src.vision.world.temporal_review import run as camera
    from src.vision.world.player_replay import run as players
    video, root, models = Path(video).resolve(), Path(root), Path(models)
    root.mkdir(parents=True, exist_ok=True)
    cached = root/(video.stem+'.mp4')
    if not cached.exists():shutil.copyfile(video, cached)
    if digest(cached) != digest(video):raise ValueError('Cached source differs from selected video')
    if (root/'world/manifest.json').exists():
        review=CVReview(root/'world')
        if review.manifest['video_sha256']!=digest(cached):raise ValueError('Completed world belongs to another source')
        return root/'world'
    stages = root/'world_stages'
    if not (stages/'summary.json').exists():
        run(cached, models/'yolox-s.onnx', models/'tvcalib-train59.pt', SEMANTIC_SHA, stages)
    else:CVReview(stages/'current')
    metric = root/'camera_m8'
    if not metric.exists():camera(cached, stages/'current', stages/'poses', metric, camera_model='m8')
    else:CVReview(metric)
    world = root/'world'
    if not world.exists():
        players(video.stem, world, 'outputs/models/m9/osnet_x0_25_msmt17.pth',
                baseline=metric, cache=root/'appearance_cache', ocr=False)
    CVReview(world)
    return world


def compact(world):
    """Retain the same M23 world fields, without any cross-source lookups."""
    from src.vision.advanced21.players import observations
    review = CVReview(world)
    meta = review.manifest
    worlds, people = [], []
    for line in review.data.open():
        row = json.loads(line);camera = row['camera'];people.append(observations(row))
        worlds.append({'frame':row['frame'],'timestamp':row['timestamp'],
            'camera':{k:camera[k] for k in ['epoch','continuity','calibration_valid','calibration']} | {
                'motion_evidence':{k:camera.get('motion_evidence',{}).get(k,{}) for k in ['field_motion','metric_camera','temporal_camera']}},
            'entities':[{k:e[k] for k in ['track_id','image_bbox','visibility']} for e in row['entities'] if e['visibility']=='visible'],
            'unassigned':[{'observation':{k:u['observation'][k] for k in ['track_id','bbox']}} for u in row['unassigned']],
            'player_world':{'players':[{k:p.get(k) for k in ['entity_id','current_segment_id','current_track_id','visibility_state',
                'identity_status','team','current_position','position_provenance','position_uncertainty_m']}
                for p in row['player_world']['players'] if p['visibility_state']=='VISIBLE']}})
    return meta, worlds, people


def infer_source(root, inputs_root):
    """Fresh M20/M21 search, protected M23 memory and M22 gap proposals."""
    import torch
    from ultralytics import YOLO
    from ml.ball16.detector import infer
    from scripts.run_cv_m20 import timestamps
    from src.vision.ball20.detector import YoloCropAdapter, merge_candidates
    from src.vision.ball20.tracker import TrajectoryBallTracker
    from src.vision.intelligent23.inputs import Inputs
    from src.vision.intelligent23.detection import TemporalPrimary, region_candidates
    from src.vision.intelligent23.memory import BallMemory
    from src.vision.intelligent23.verification import combine, verify
    from src.vision.advanced21.camera import causal_camera
    from src.vision.advanced21.temporal import batch_visual_features, deterministic_verdict
    from src.vision.football22.inference import FootballDetector
    from src.vision.football22.model import context_indices
    root=Path(root);inputs=Inputs(inputs_root)
    meta,worlds,people=compact(root/'world');name=meta['recording_id']
    pts=timestamps(meta['source_video'])
    if len(pts)!=meta['frames'] or any(abs(t-i/meta['fps'])>2e-6 for i,t in enumerate(pts)):
        raise ValueError('Video presentation timestamps differ from the world clock')
    output=root/'perception';output.mkdir(exist_ok=True)
    report_path=output/'report.json'
    config=json.loads(Path('configs/vision/cv_m23.json').read_text())
    config['offline'].update(primary_anchored_complement_only=True,m22_dual_current_visual_proposals_require_two_primary_anchors=True)
    config_sha=signature(config)
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['config_sha256']!=config_sha or report['source_world_sha256']!=meta['frames_sha256'] or digest(output/'inputs.jsonl')!=report['inputs_sha256']:
            raise ValueError('Completed source inference changed')
        return analyze_source(root,report,meta,worlds)
    path=output/'inputs.jsonl'
    if path.exists():raise ValueError('Partial inference retained; choose a fresh run directory')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    m20=inputs.json('configs/vision/cv_m20.json')
    generic_path=inputs.path('outputs/ml/ball_m16/yolo11s.pt')
    soccer_path=inputs.path('outputs/ml/ball_m16/training/soccer11s/weights/best.pt')
    # Fixed deployment choice, declared before seeing this clip's predictions.
    # This is an existing development-trained model, not an independent test.
    trained=inputs.manifest['complement_models']['test_match']
    components=[{'role':'generic_primary','path':str(generic_path),'sha256':digest(generic_path)},
        {'role':'soccer_fresh_global_measurements','path':str(soccer_path),'sha256':digest(soccer_path)},
        {'role':'m22_complementary','path':str(inputs.path(trained['checkpoint'])),'sha256':trained['checkpoint_sha256']}]
    if components[0]['sha256']!='85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5' or components[1]['sha256']!=m20['detector']['sha256']:
        raise ValueError('Primary checkpoint changed')
    primary=TemporalPrimary(YOLO(generic_path),components[0]['sha256'],meta['width'],meta['height'],device)
    soccer=YOLO(soccer_path);adapter=YoloCropAdapter(soccer,640,device)
    control=TrajectoryBallTracker(meta['width'],meta['height'],m20)
    memory=BallMemory(meta['width'],meta['height'],m20,config)
    complement=FootballDetector(components[2]['path'],components[2]['sha256'],inputs.json('configs/vision/cv_m22.json'),device=device)
    epochs=[];epoch=0
    for i,w in enumerate(worlds):
        if i and (w['camera']['epoch']!=worlds[i-1]['camera']['epoch'] or w['camera']['continuity']=='cut'):epoch+=1
        epochs.append(epoch)
    capture=cv2.VideoCapture(meta['source_video']);images={};started=time.perf_counter()
    totals={'extra_crop_views':0,'primary_views':0,'m22_views':0,'soccer_views':0}
    try:
        with path.open('w') as stream:
            for i,w in enumerate(worlds):
                tick=time.perf_counter();ok,image=capture.read()
                if not ok:raise ValueError('Source ended before its verified clock')
                images[i]=image
                for old in list(images):
                    if old<i-6:del images[old]
                regions,_=control.prepare(i,i/meta['fps'],w['camera'],people[i])
                global_candidates,gt=infer(soccer,image,640)
                local_candidates,lt=adapter.infer(image,regions)
                control_ball,_=control.consume(image,merge_candidates(global_candidates,local_candidates))
                refinement,step,_=primary.camera(image,w['camera']);previous=primary.previous
                camera=causal_camera(w['camera'],refinement)
                plan=memory.prepare(i,i/meta['fps'],camera,people[i])
                current,timing=primary.detect(image,w['camera'],people[i],[c.model_dump() for c in control_ball.candidates],refinement,step)
                requested=[tuple(r['bbox']) for r in plan['regions']]
                local,local_timing=region_candidates(primary.detector,image,requested)
                discovery={tuple(r['bbox']) for r in plan['regions'] if r['kind']=='global_discovery'}
                guided=[{**c,'origin':'discovery' if region in discovery else 'guided'} for region in requested for c in local[region]]
                for c,feature in zip(guided,batch_visual_features(previous,image,guided,step),strict=True):
                    c.update(temporal_features=feature,deterministic=deterministic_verdict(feature,c['confidence']))
                checked=[verify(c,memory,image,config,additional_verification=True) for c in combine(current,guided,i)]
                ball,snapshot=memory.consume(image,[c for c in checked if c['observation_eligible']])
                # Complementary network never feeds the protected causal memory.
                ids=context_indices(i,meta['fps'],epochs)
                global_m22=complement.infer([images[j] for j in ids],i)
                m22_regions=requested[:config['verification']['maximum_m22_regions']]
                m22=complement.infer([images[j] for j in ids],i,regions=m22_regions)
                pool={c['candidate_id']:c for c in [*global_m22['candidates'],*m22['candidates']] if c['observation_eligible'] and c['confidence']>=.35 and c['current_only_confidence']>=.35}
                proposals=[{'candidate_id':'m23_gap_'+c['candidate_id'],'bbox':c['bbox'],'confidence':c['confidence'],
                    'current_only_confidence':c['current_only_confidence'],'inference_view':'M22_ACTUAL_CURRENT_PIXELS_PRIMARY_ANCHORED_GAP_PROPOSAL',
                    'source':'VIDEO_CV_DETECTOR','origin':'complement','visual_evidence':'CURRENT_PIXELS','source_region':None,
                    'source_model_sha256':components[2]['sha256']} for c in pool.values()]
                for c,feature in zip(proposals,batch_visual_features(previous,image,proposals,step),strict=True):
                    c.update(temporal_features=feature,deterministic=deterministic_verdict(feature,c['confidence']),
                        support_views=[{'origin':'complement','bbox':c['bbox'],'confidence':c['confidence'],'source_region':None}],
                        observation_eligible=True,decision_reason='DUAL_CURRENT_VISUAL_PROPOSAL_REQUIRES_TWO_SIDED_PRIMARY_ASSOCIATION',
                        joint_memory_evidence=None,blur_supported=False,blur_definition='Not classified by complementary proposals',
                        m22_support=[],association_adjustment=0.,detector_confidence_unchanged=True,confidence_is_calibrated_probability=False)
                method={'candidates':checked+proposals,'memory':snapshot,'requested_crop_views':len(requested)}
                row={'frame':i,'timestamp':i/meta['fps'],'source_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                    'config_sha256':config_sha,'primary_candidates':current,'camera_refinement':refinement,'causal_camera':camera,
                    'methods':{'full_m23':method},'relative_camera_homography':step.tolist() if step is not None else None,
                    'm22_context_frames':ids,'m22_checkpoint_sha256':components[2]['sha256'],
                    'frame_wall_seconds':time.perf_counter()-tick}
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
                for key,value in [('extra_crop_views',local_timing['views']),('primary_views',timing['views']),
                    ('m22_views',global_m22['inference_views']+m22['inference_views']),('soccer_views',gt['views']+lt['views'])]:totals[key]+=value
                if i%50==0:print('SOURCE M23',name,i,'/',meta['frames'],'elapsed',round(time.perf_counter()-started,1),flush=True)
        if capture.read()[0]:raise ValueError('Unindexed source frames')
    finally:capture.release()
    elapsed=time.perf_counter()-started
    report={'schema_version':'cv-m23-source-perception-v1','source_world_sha256':meta['frames_sha256'],
        'source_video_sha256':meta['video_sha256'],'inputs_sha256':digest(path),'config':config,'config_sha256':config_sha,
        'components':components,'checkpoint_policy':'Fixed existing test_match-excluded M22 checkpoint; no Giroud labels or fitting; match overlap not independently audited',
        'labels_read':False,'new_training':False,'device':device,'torch':torch.__version__,'opencv':cv2.__version__,
        'processing_fps':meta['frames']/elapsed,'recorded_processing_seconds':elapsed,'timing':totals,
        'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'source_code_sha256':{str(p):digest(p) for p in [Path(__file__),*Path('src/vision/intelligent23').glob('*.py')]}}
    write(report_path,report)
    return analyze_source(root,report,meta,worlds)


def analyze_source(root,perception,meta,worlds):
    from scripts.analyze_cv_m23 import selected_candidates, write_artifacts
    from src.vision.ball.schema import BallFrame
    from src.vision.ball20.tracker import appearance
    from src.vision.advanced21.offline import dag_link, reconstruct
    from src.vision.advanced21.tracking import offline_rows
    from src.vision.intelligent23.offline import anchored_complements,reconstruct_camera
    analysis=root/'analysis'
    if analysis.exists():raise ValueError('Analysis is immutable; select a fresh root')
    path=root/'perception/inputs.jsonl';records=[json.loads(l) for l in path.open()]
    if len(records)!=meta['frames'] or any(r['frame']!=i or abs(r['timestamp']-i/meta['fps'])>1e-6 for i,r in enumerate(records)):
        raise ValueError('Source inference clock mismatch')
    descriptors={};capture=cv2.VideoCapture(meta['source_video'])
    try:
        for i,row in enumerate(records):
            ok,image=capture.read()
            if not ok or hashlib.sha256(image.tobytes()).hexdigest()!=row['source_pixels_sha256']:raise ValueError('Inference/source pixels changed')
            for c in [*row['primary_candidates'],*row['methods']['full_m23']['candidates']]:descriptors[(i,c['candidate_id'])]=appearance(image,c['bbox'])
    finally:capture.release()
    config=perception['config'];world_cameras=[w['camera'] for w in worlds];cameras=[r['causal_camera'] for r in records]
    diagonal=math.hypot(meta['width'],meta['height']);name=meta['recording_id'];results={}
    for method in ['m21_temporal_offline','full_m23']:
        candidates=selected_candidates(records,method)
        by_object={id(c):descriptors[(i,c['candidate_id'])] for i,row in enumerate(candidates) for c in row}
        def edge(a,b,dt):
            x,y=by_object.get(id(a)),by_object.get(id(b))
            return .1*(1-float(np.clip(np.sqrt(x*y).sum(),0,1))) if x is not None and y is not None else 0.
        selected=dag_link(candidates,world_cameras,meta['fps'],diagonal,maximum_gap_seconds=config['offline']['maximum_gap_seconds'],
            node_adjustment=(lambda c,f:c['association_adjustment']) if method=='full_m23' else None,
            edge_adjustment=edge if method=='full_m23' else None)
        if method=='full_m23':
            primary_sets=[[c for c in row if c['origin']!='complement'] for row in candidates]
            selected=dag_link(primary_sets,world_cameras,meta['fps'],diagonal,maximum_gap_seconds=config['offline']['maximum_gap_seconds'],
                node_adjustment=lambda c,f:c['association_adjustment'],edge_adjustment=edge)
            selected=anchored_complements(selected,candidates,cameras,meta['fps'],diagonal,descriptors,config['offline']['maximum_gap_seconds'])
        inferred=reconstruct(selected,world_cameras,meta['fps']) if method=='m21_temporal_offline' else reconstruct_camera(selected,cameras,meta['fps'],meta['width'],meta['height'],config['offline']['reconstruction_gap_seconds'])
        balls=[]
        for i,b in enumerate(offline_rows(selected,inferred,candidates,worlds,meta)):
            values={**b.model_dump(),'version':'m23','coordinate_provenance':'M23_CURRENT_PIXEL_EVIDENCE; NO_VERIFIED_HEIGHT',
                'motion_model':'M20_IMM_MEMORY_M21_OFFLINE_DAG; OFFLINE_ASSOCIATION','search_stage':'M23_'+method.upper()}
            if not b.directly_observed and b.predicted_image_position is None and len(candidates[i])>1:
                values.update(tracking_status='AMBIGUOUS',ambiguity_reason='UNRESOLVED_CURRENT_CANDIDATES_AFTER_OFFLINE_LINKING')
            balls.append(BallFrame.model_validate(values))
        components=perception['components'] if method=='full_m23' else perception['components'][:2]
        model_sha=signature({'components':components,'config':perception['config_sha256'],'variant':method})
        raw,events,summary,evidence,manifest=write_artifacts(analysis,name,method,{**meta,'_worlds':worlds},balls,selected,inferred,records,cameras,config,components,model_sha,perception['config_sha256'],path)
        results[method]={'engineering':manifest['statistics'],'full_clip_passes':summary,
            'estimated_passes':len([e for e in events if e['safe_for_automatic']]),'all_events':len(events),
            'navigation':[{'kind':'ball_observed','frame':i} for i,b in enumerate(raw) if b['directly_observed'] and (i==0 or not raw[i-1]['directly_observed'])][:30],
            'pass_traces':[],'accuracy':None}
    ref=analysis/name/'m21_temporal_offline'
    report={'schema_version':'cv-m23-unlabelled-source-v1','recording_id':name,'reference_role':'unlabelled_inference',
        'source':{k:meta[k] for k in ['video_sha256','frames_sha256','frames','fps','width','height']},
        'world_directory':str(root/'world'),'source_video':meta['source_video'],
        'independent_final_test':'unavailable: no human labels or independent source-overlap audit',
        'promotion':{'status':'not_promoted','ordinary_default':'m14','reason':'Unlabelled new-source inference; no accuracy or production promotion claim.'},
        'metrics':results['full_m23'],'control':results['m21_temporal_offline'],'runtime':perception,
        'reference':{'path':str(ref),'manifest_sha256':digest(ref/'manifest.json'),'role':'fresh_same_source_m21_control_not_human_truth'},
        'precision':None,'recall':None,'f1':None,'human_labels_read':False,'new_training':False,
        'limitations':['Estimated passes are unverified. Goals, shots, tackles, pass attempts and pass accuracy are unavailable.',
            'Player identities and teams are anonymous hypotheses; filename does not identify a player.',
            'Model coverage is not accuracy. Camera projection, speed and heatmaps may be unavailable where geometry is rejected.']}
    write(analysis/'report.json',report)
    print('SOURCE ANALYSIS',json.dumps(report['metrics']),flush=True)
    return report


def install_source(root):
    from src.serving.intelligent23 import IntelligentRegistry
    root=Path(root);analysis=root/'analysis';report=json.loads((analysis/'report.json').read_text());name=report['recording_id']
    if report['schema_version']!='cv-m23-unlabelled-source-v1':raise ValueError('Wrong source report')
    entry={'root':str(analysis),'report_sha256':digest(analysis/'report.json')}
    pending=root/'pending_registry.json';write(pending,{name:entry})
    registry=IntelligentRegistry({name:CVReview(root/'world')},pending)
    registry.load(name);registry.metadata(name)
    for i in [0,report['source']['frames']//2,report['source']['frames']-1]:registry.reference(name,i)
    for path,value in [(Path('outputs/m23/runtime_cv_registry.json'),str(root/'world')),
        (Path('configs/serving/cv.json'),str(root/'world')),(Path('configs/serving/cv_m23.json'),entry)]:
        entries=json.loads(path.read_text());entries[name]=value;write(path,entries)
    write(root/'installation.json',{'recording_id':name,'report_sha256':entry['report_sha256'],'strict_validation':'passed'})
    print('SOURCE INSTALLED',name,flush=True)


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--models',type=Path,default=Path.home()/'Library/Caches/TactiqAI/models')
    parser.add_argument('--inputs',type=Path,default=Path('outputs/m23/inputs_v2'))
    parser.add_argument('--stage',choices=['world','ball','install','all'],default='all')
    return parser.parse_args()


if __name__=='__main__':
    args=arguments()
    if args.stage in ['world','all']:source_world(args.video,args.output,args.models)
    if args.stage in ['ball','all']:infer_source(args.output,args.inputs)
    if args.stage in ['install','all']:install_source(args.output)
