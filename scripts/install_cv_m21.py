"""Validate and register the isolated M21 experiments without promoting application defaults."""
import argparse
import json
from pathlib import Path
from src.serving.cv import CVReview, digest
from src.serving.cv_events import CVEventReview
from src.vision.events.passes import PassConfig, signature
from src.vision.advanced21.experiment import NAMES, write_json


def install(root):
    root=Path(root);report=json.loads((root/'report.json').read_text())
    selected=report['selection']['development_selected_variant']
    if any(set(result['per_recording'])!=set(NAMES) for result in report['variants'].values()):
        raise ValueError('All three source clips are required')
    registry={};validated={}
    for name in NAMES:
        review=CVReview(Path('outputs/world_model')/(name+'_m9_ready'))
        for variant in report['variants']:
            directory=root/name/variant
            bm=json.loads((directory/'manifest.json').read_text())
            cfg=PassConfig()
            event_dir=directory/'passes'
            manifest={k:review.manifest[k] for k in ['recording_id','frames','fps','width','height','video_sha256']}
            manifest.update(schema_version='cv-passes-v1',algorithm_version='cv-pass-m21-v1',
                independent_reference_status='development_evaluated_final_test_pending',world_frames_sha256=review.manifest['frames_sha256'],
                ball_version='m21',ball_frames_sha256=bm['ball_frames_sha256'],ball_detector_sha256=bm['detector_sha256'],
                events_sha256=digest(event_dir/'events.json'),evidence_sha256=digest(event_dir/'evidence_frames.jsonl'),
                config=cfg.model_dump(),config_sha256=signature(cfg.model_dump()),statistics=json.loads((event_dir/'summary.json').read_text()),
                second_stage_enabled=bm['config']['event_second_stage'],
                unavailable=['goals','shots','tackles','interceptions','pass_attempts','pass_accuracy','verified_possession'])
            write_json(event_dir/'manifest.json',manifest)
            review.attach_ball(directory)
            if review.ball_review is None or review.ball_review.advanced is None:
                raise ValueError('M21 ball/diagnostic validation failed: '+name+'/'+variant)
            layer=CVEventReview(event_dir,review)
            validated[name+':'+variant]={'events':len(layer.events),'qualified_estimates':sum(e['status']=='estimated' for e in layer.events)}
            print('M21 VALIDATED',name,variant,flush=True)
        registry[name]={'root':str(root),'default_experiment':selected,'report_sha256':digest(root/'report.json')}
    write_json('configs/serving/cv_m21.json',registry)
    lock={'report_sha256':digest(root/'report.json'),'development_selected_variant':selected,
          'development_annotations_sha256':report['annotations_sha256'],
          'selection_uses_only_development_labels':True,'final_test_labels_inspected':False,
          'model_promotion':'not_promoted; final test pending','validated_variants':validated}
    write_json(root/'selection_lock.json',lock)
    print(json.dumps({'registered':len(registry),'validated_variants':len(validated),'default_experiment':selected,'application_default_changed':False}),flush=True)
    return registry


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    install(parser.parse_args().root)
