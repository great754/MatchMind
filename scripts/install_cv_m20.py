"""Validate M20 with the existing M19 event gates; opt in without promoting defaults."""
import argparse
import json
from pathlib import Path
from src.serving.cv import CVReview, digest
from src.serving.cv_events import CVEventReview
from src.vision.events.passes import PassConfig, signature


def install(root):
    root = Path(root); report = json.loads((root/'report.json').read_text())
    balls = {}; events = {}; results = {}
    if set(report['clips']) != {'test_match', 'team_goal', 'southampton'}:
        raise ValueError('Process all three clips before registering M20')
    for name, result in report['clips'].items():
        base = Path('outputs/world_model')/(name+'_m9_ready'); review = CVReview(base)
        ball = root/name/'ball_m20'
        bm = json.loads((ball/'manifest.json').read_text()); comparison = {}
        for version, entry in [('m19', result['m19_same_global_inputs']), ('m20', result['m20'])]:
            comparison[version] = {k: entry['ball'][k] for k in ['direct_observed_frames','frames','observed_track_changes','longest_direct_run_frames']}
            comparison[version].update(qualified_pass_estimates=entry['passes']['completed_pass_estimates'], transfer_candidates=entry['passes']['transfer_candidates'])
        bm['comparison'] = comparison; (ball/'manifest.json').write_text(json.dumps(bm,indent=2)+'\n')
        review.attach_ball(ball)
        if review.ball_review is None or review.ball_review.trajectory is None: raise ValueError('M20 ball/trajectory validation failed')
        source = review.manifest; bm = review.ball_review.manifest; target = root/name/'passes_m20'; cfg = PassConfig()
        manifest = {k: source[k] for k in ['recording_id', 'frames', 'fps', 'width', 'height', 'video_sha256']}
        manifest.update(schema_version='cv-passes-v1', algorithm_version='cv-pass-m19-v1', independent_reference_status='pending_human_review',
                        world_frames_sha256=source['frames_sha256'], ball_version='m20', ball_frames_sha256=bm['ball_frames_sha256'],
                        ball_detector_sha256=bm['detector_sha256'], events_sha256=digest(target/'events.json'),
                        evidence_sha256=digest(target/'evidence_frames.jsonl'), config=cfg.model_dump(), config_sha256=signature(cfg.model_dump()),
                        statistics=result['m20']['passes'], unavailable=['goals', 'shots', 'tackles', 'pass_attempts', 'pass_accuracy', 'verified_possession'])
        (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        layer = CVEventReview(target, review)
        balls[name] = str(ball); events[name] = str(target)
        results[name] = {'validated_events': len(layer.events), 'qualified_estimates': manifest['statistics']['completed_pass_estimates'],
                         'review_candidates': manifest['statistics']['transfer_candidates'], 'default_changed': False}
    Path('configs/serving/cv_ball_m20.json').write_text(json.dumps(balls, indent=2)+'\n')
    Path('configs/serving/cv_events_m20.json').write_text(json.dumps(events, indent=2)+'\n')
    (root/'installation.json').write_text(json.dumps(results, indent=2)+'\n'); print(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); args = parser.parse_args(); install(args.root)
