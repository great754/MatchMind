"""Fixed M21 association + unchanged pass gates; no tracker search or event anchors."""
import argparse,json,time
from pathlib import Path
from src.serving.cv import digest
from src.vision.football22.data import NAMES,write,original_references
from src.vision.football22.interactions import interaction_frames,pass_trace
from src.vision.football22.schema import FootballFrame
from src.vision.advanced21.experiment import source_worlds
from src.vision.advanced21.offline import dag_link,reconstruct
from src.vision.advanced21.tracking import offline_rows
from src.vision.ball.schema import BallFrame,canonical
from src.vision.events.passes import PassConfig,signature
from src.vision.events.passes_m19 import detect_m19
from src.evaluation.cv_m19 import evaluate,iou
from scripts.evaluate_cv_m20 import localization
from scripts.run_cv_m19 import ball_metrics


def candidate_recall(tasks,labels,records):
    lookup={t['task_id']:t for t in tasks};positive=[r for r in labels if lookup[r['task_id']]['kind']=='ball' and r['state']=='visible']
    result={'visible_reference_frames':len(positive),'reference_role':'development_only','minimum_candidate_confidence':.05,'matching_iou':.25}
    for k in [1,3,5]:
        count=sum(any(iou(c['bbox'],r['bbox'])>=.25 for c in records[lookup[r['task_id']]['recording_id']][lookup[r['task_id']]['frame']]['candidates'][:k]) for r in positive)
        result['top_'+str(k)]={'matched':count,'total':len(positive),'recall':count/len(positive) if positive else None}
    return result


def run(perception_root,training_root,output,register=True,completed_review_snapshot=None):
    source=Path(perception_root);root=Path(output)
    if root.exists():raise ValueError('Use a fresh analysis directory')
    perception=json.loads((source/'report.json').read_text())
    if set(perception['clips'])!=set(NAMES):raise ValueError('Finish all three source inference runs before analysis')
    dataset,labels,label_sha=original_references();protocol={**dataset['protocol'],'sources':dataset['sources']};root.mkdir(parents=True)
    all_balls={};all_events={};all_people={};records={};per_clip={};training={};traces={};detector_balls={};detector_metrics={}
    for name in NAMES:
        started=time.perf_counter();meta,worlds,_=source_worlds(name);path=source/name/'detections.jsonl'
        if digest(path)!=perception['clips'][name]['detections_sha256']:raise ValueError('Neural inference cache changed')
        record=[json.loads(line) for line in path.open()];records[name]=record
        if len(record)!=meta['frames'] or any(r['frame']!=i or abs(r['timestamp']-i/meta['fps'])>1e-6 for i,r in enumerate(record)):raise ValueError('Neural input clock mismatch')
        eligible=[[c for c in row['candidates'] if c['observation_eligible']] for row in record]
        detector_balls[name]=[{'directly_observed':bool(candidates),'image_bbox':candidates[0]['bbox'] if candidates else None,
            'image_position':[candidates[0]['bbox'][0]+candidates[0]['bbox'][2]/2,candidates[0]['bbox'][1]+candidates[0]['bbox'][3]/2] if candidates else None} for candidates in eligible]
        selected=dag_link(eligible,[w['camera'] for w in worlds],meta['fps'],(meta['width']**2+meta['height']**2)**.5)
        inferred=reconstruct(selected,[w['camera'] for w in worlds],meta['fps'])
        states=offline_rows(selected,inferred,eligible,worlds,meta)
        balls=[BallFrame.model_validate({**b.model_dump(),'version':'m22','coordinate_provenance':'M22_CURRENT_PIXEL_NEURAL_CANDIDATE; NOT_HUMAN_VERIFIED; NO_VERIFIED_HEIGHT',
            'search_stage':'M22_TEMPORAL_FOOTBALL_DETECTOR_WITH_CURRENT_ONLY_CHECK'}) for b in states]
        raw=[b.model_dump() for b in balls];events,summary,evidence=detect_m19(name,worlds,raw,meta['fps']);contacts=interaction_frames(evidence,events)
        directory=root/name;directory.mkdir();ball_offsets=[];debug_offsets=[]
        trained=json.loads((Path(training_root)/name/'result.json').read_text());training[name]=trained
        with (directory/'ball_frames.jsonl').open('wb') as bf,(directory/'video_ball_observations.jsonl').open('w') as canon,(directory/'football_frames.jsonl').open('wb') as detail:
            for i,(ball,row) in enumerate(zip(balls,record,strict=True)):
                ball_offsets.append(bf.tell());bf.write(ball.model_dump_json().encode()+b'\n');canon.write(canonical(name,meta['fps'],ball,trained['checkpoint_sha256']).model_dump_json()+'\n')
                candidates=[{k:c[k] for k in ['candidate_id','bbox','confidence','current_only_confidence','observation_eligible','decision_reason']}|
                    {'selected':bool(ball.directly_observed and tuple(c['bbox'])==ball.image_bbox and c['confidence']==ball.detection_confidence)} for c in row['candidates']]
                debug=FootballFrame(schema_version='cv-m22-frame-v1',recording_id=name,frame=i,timestamp=ball.timestamp,camera_epoch=ball.camera_epoch,
                    context_frames=row['context_frames'],checkpoint_sha256=row['checkpoint_sha256'],directly_observed=ball.directly_observed,candidates=candidates,
                    heatmap=row['heatmap'],heatmap_is_calibrated=False,association='existing_m21_offline_dag',uses_future_for_association=True,
                    reconstruction=inferred[i],interaction=contacts[i])
                debug_offsets.append(detail.tell());detail.write(debug.model_dump_json().encode()+b'\n')
        manifest={k:meta[k] for k in ['recording_id','frames','fps','width','height','video_sha256']}
        config=perception['config'];manifest.update(version='m22',source_m9_frames_sha256=meta['frames_sha256'],
            detector_sha256=trained['checkpoint_sha256'],trained_checkpoint=trained['checkpoint'],frame_offsets=ball_offsets,
            ball_frames_sha256=digest(directory/'ball_frames.jsonl'),canonical_sha256=digest(directory/'video_ball_observations.jsonl'),
            football_frame_offsets=debug_offsets,football_sha256=digest(directory/'football_frames.jsonl'),
            config=config,config_sha256=signature(config),statistics=ball_metrics(raw))
        write(directory/'manifest.json',manifest)
        event_dir=directory/'passes';event_dir.mkdir();write(event_dir/'events.json',events)
        (event_dir/'evidence_frames.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in evidence))
        cfg=PassConfig().model_dump();event_manifest={k:meta[k] for k in ['recording_id','frames','fps','width','height','video_sha256']}
        event_manifest.update(schema_version='cv-passes-v1',algorithm_version='cv-pass-m22-v1',independent_reference_status='development_only_no_final_test',
            world_frames_sha256=meta['frames_sha256'],ball_version='m22',ball_frames_sha256=manifest['ball_frames_sha256'],ball_detector_sha256=manifest['detector_sha256'],
            events_sha256=digest(event_dir/'events.json'),evidence_sha256=digest(event_dir/'evidence_frames.jsonl'),config=cfg,config_sha256=signature(cfg),statistics=summary,
            unavailable=['verified_touches','verified_possession','goals','shots','tackles','interceptions','pass_attempts','pass_accuracy'])
        write(event_dir/'manifest.json',event_manifest);write(event_dir/'summary.json',summary)
        all_balls[name]=raw;all_events[name]=events
        from src.vision.events.passes import visible_players
        players={t[key]:list(visible_players(worlds[t[key]]).values()) for t in dataset['tasks'] if t['recording_id']==name and t['kind']=='identity' for key in ['frame','other_frame']};all_people[name]=players
        tasks=[t for t in dataset['tasks'] if t['recording_id']==name];task_lookup={t['task_id']:t for t in tasks};refs=[l for l in labels if l['task_id'] in task_lookup]
        metrics=evaluate(tasks,refs,{name:raw},{name:events},protocol,{name:players});metrics['localization']=localization(tasks,refs,{name:raw},protocol)
        metrics['candidate_recall']=candidate_recall(tasks,refs,{name:record});metrics['processing']=manifest['statistics'];metrics['analysis_seconds']=time.perf_counter()-started
        per_clip[name]=metrics;traces[name]=[]
        ball_tasks=[t for t in tasks if t['kind']=='ball'];ball_ids={t['task_id'] for t in ball_tasks};ball_refs=[r for r in refs if r['task_id'] in ball_ids]
        detector_metrics[name]=evaluate(ball_tasks,ball_refs,{name:detector_balls[name]},{},protocol)
        for label in refs:
            task=task_lookup[label['task_id']]
            if task['kind']=='passes' and label['state']=='reviewed' and label['exhaustive'] and all(p['outcome']!='uncertain' for p in label['passes']):
                for ref in label['passes']:
                    if ref['outcome']=='completed':traces[name].append(pass_trace(ref,task,raw,evidence,events,meta['fps']))
        write(directory/'pass_reference_traces.json',traces[name])
    aggregate=evaluate(dataset['tasks'],labels,all_balls,all_events,protocol,all_people)
    aggregate['localization']=localization(dataset['tasks'],labels,all_balls,protocol);aggregate['candidate_recall']=candidate_recall(dataset['tasks'],labels,records)
    ball_tasks=[t for t in dataset['tasks'] if t['kind']=='ball'];ball_ids={t['task_id'] for t in ball_tasks}
    detector_aggregate=evaluate(ball_tasks,[r for r in labels if r['task_id'] in ball_ids],detector_balls,{},protocol)
    old=json.loads(Path('docs/evidence/cv_m21_results.json').read_text())
    baselines={key:{'aggregate':old['variants'][key]['aggregate'],'per_recording':old['variants'][key]['per_recording']} for key in ['m20_control','offline_dag','temporal_offline']}
    report={'schema_version':'cv-m22-analysis-v1','reference_role':'development_only','original_annotations_sha256':label_sha,
        'perception_report_sha256':digest(source/'report.json'),'perception':perception,'training':training,'per_recording':per_clip,'aggregate':aggregate,
        'baselines':baselines,'detector_only_per_recording':detector_metrics,'detector_only_aggregate':detector_aggregate,
        'pass_traces':traces,'pass_reference_count':sum(map(len,traces.values())),
        'promotion':{'status':'not_promoted','ordinary_default':'m14','experimental_control':'m20','independent_final_test':'not_available',
            'reason':'Development-only trained model; independent user-approved final footage/labels required. Do not infer improvement from training completion.'},
        'human_review':{'tasks':150,'source_scope':list(NAMES),'model_proposals_are_truth':False,'withdrawn_set_used':False}}
    if report['pass_reference_count']!=7:raise ValueError('All seven original eligible passes require an evidence trace')
    targets={'ball_precision':.75,'ball_recall':.55,'ball_f1':.60,'completed_passes_matched':3}
    values={'ball_precision':aggregate['ball']['precision'],'ball_recall':aggregate['ball']['recall'],
        'ball_f1':aggregate['ball']['f1'],'completed_passes_matched':aggregate['passes']['tp']}
    achieved={key:values[key] is not None and values[key]>=threshold for key,threshold in targets.items()}
    report['research_targets']={'reference':'original_60_development_reviews','thresholds':targets,'values':values,
        'achieved':achieved,'all_met':all(achieved.values()),'independent_accuracy_acceptance':False}
    if completed_review_snapshot:
        from scripts.evaluate_cv_m22_reviews import run as review_evaluation
        from src.evaluation.cv_m19 import counts
        reviewed=review_evaluation(completed_review_snapshot,root/'completed_review',root)
        original_m22=json.loads(Path('outputs/m22/analysis_final_v1/report.json').read_text())
        originals={'m20_control':baselines['m20_control'],'m21_offline_dag':baselines['offline_dag'],
            'm21_temporal_offline':baselines['temporal_offline'],'m22_initial':original_m22,
            'm22_reviewed':{'aggregate':aggregate,'per_recording':per_clip}}
        def combined(a,b):
            return {**counts(*(a[k]+b[k] for k in ['tp','fp','fn'])),
                'reviewed_frames':a['reviewed_frames']+b['reviewed_frames'],
                'unknown_visibility_frames':a['unknown_visibility_frames']+b['unknown_visibility_frames'],
                'reference_role':'development_only_mixed_original_and_model_assisted'}
        report['completed_review_evaluation']=reviewed
        report['combined_development_ball']={key:{'aggregate':combined(originals[key]['aggregate']['ball'],value['ball']),
            'per_recording':{name:combined(originals[key]['per_recording'][name]['ball'],value['per_recording'][name]['ball']) for name in NAMES}}
            for key,value in reviewed['systems'].items()}
    write(root/'report.json',report)
    # Validate all optional artifacts before making them selectable. Registry is
    # written last, so an incomplete experiment cannot become application-ready.
    from src.serving.cv import CVReview
    from src.serving.cv_events import CVEventReview
    registry={}
    for name in NAMES:
        review=CVReview(Path('outputs/world_model')/(name+'_m9_ready'));review.attach_ball(root/name)
        if review.ball_review is None or review.ball_review.football is None:raise ValueError('M22 sidecar validation failed')
        CVEventReview(root/name/'passes',review)
        registry[name]={'root':str(root),'report_sha256':digest(root/'report.json')}
    if register:write('configs/serving/cv_m22.json',registry)
    print('M22 VALIDATED all three clips; metrics',aggregate['ball'],'passes',aggregate['passes'],flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--perception',required=True);p.add_argument('--training',required=True);p.add_argument('--output',required=True);p.add_argument('--no-register',action='store_true');p.add_argument('--completed-review-snapshot');a=p.parse_args();run(a.perception,a.training,a.output,not a.no_register,a.completed_review_snapshot)
