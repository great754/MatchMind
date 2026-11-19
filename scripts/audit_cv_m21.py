"""Freeze completed reviews as development data and reproduce pre-M21 baselines."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil

from src.serving.cv import digest
from src.serving.cv_review_labels import HumanAnnotation, HumanReviewStore


def audit(output, replay_baseline=None):
    root = Path(output)
    if root.exists():
        raise ValueError('Use a fresh audit directory; original annotations are immutable')
    store = HumanReviewStore()
    path = store.root / 'labels.json'
    original = path.read_bytes()
    rows = store.labels()
    seen = set()
    counts = {}
    for row in rows:
        annotation = HumanAnnotation.model_validate({k: v for k, v in row.items() if k != 'source_video_sha256'})
        task = store.tasks[annotation.task_id]
        checked = store.validate(task['recording_id'], annotation)
        if json.loads(json.dumps(checked)) != row:
            raise ValueError('Annotation/source mismatch')
        key = (row['reviewer_id'], row['task_id'])
        if key in seen:
            raise ValueError('Duplicate reviewer/task')
        seen.add(key)
        count = counts.setdefault(task['recording_id'], dict(saved=0, ball_known=0, ball_visible=0,
                                ball_uncertain=0, pass_windows_eligible=0, pass_windows_excluded=0,
                                completed_passes=0, identity_pairs=0, identity_same=0, identity_different=0))
        count['saved'] += 1
        if task['kind'] == 'ball':
            count['ball_uncertain' if row['state'] == 'uncertain' else 'ball_known'] += 1
            count['ball_visible'] += row['state'] == 'visible'
        elif task['kind'] == 'passes':
            eligible = row['state'] == 'reviewed' and row['exhaustive'] and all(p['outcome'] != 'uncertain' for p in row['passes'])
            count['pass_windows_eligible' if eligible else 'pass_windows_excluded'] += 1
            count['completed_passes'] += sum(p['outcome'] == 'completed' for p in row['passes']) if eligible else 0
        else:
            count['identity_pairs'] += 1
            count['identity_same'] += row['state'] == 'same'
            count['identity_different'] += row['state'] == 'different'
    if len({r['reviewer_id'] for r in rows}) != 1:
        raise ValueError('Select/adjudicate reviewers explicitly before this campaign')
    root.mkdir(parents=True)
    snapshot = root / 'development_labels.json'
    snapshot.write_bytes(original)
    snapshot.chmod(0o600)
    shutil.copyfile('configs/evaluation/cv_m19_tasks.json', root / 'development_tasks.json')
    report = {'schema_version': 'cv-m21-audit-v1', 'reference_role': 'development_only',
              'reason': 'All 60 tasks were inspected in the public 2026-11-18 evaluation; none are untouched final test data.',
              'dataset_fingerprint': store.dataset['fingerprint'], 'annotations_sha256': hashlib.sha256(original).hexdigest(),
              'sealed_tasks': len(store.tasks), 'saved_annotations': len(rows), 'independent_reviewers': 1,
              'per_recording': counts, 'final_test_status': 'new_source_disjoint_human_references_required',
              'thresholds': store.dataset['protocol'], 'sources': store.dataset['sources']}
    (root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    if replay_baseline is None:
        from scripts.evaluate_cv_m19 import run as baseline
        from scripts.evaluate_cv_m20 import run as m20
        reviewer = rows[0]['reviewer_id']
        with (root / 'evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
            baseline('outputs/m19/run_v3', snapshot, reviewer, root / 'baseline_m14_m16_m19.json')
            m20('outputs/m20/run_v1', snapshot, reviewer, root / 'baseline_m19_m20.json')
    else:
        # Explicitly selected, previously verified byte-identical replay; never silently substitute a model.
        published = json.loads(Path('docs/evidence/cv_human_evaluation_2026_11_18.json').read_text())
        if published['reference_sha256'] != report['annotations_sha256']:
            raise ValueError('Published baseline references differ')
        from src.evaluation.cv_m19 import evaluate
        from scripts.evaluate_cv_m20 import localization
        from scripts.run_cv_m19 import compact_worlds
        from src.vision.events.passes import visible_players
        from src.vision.events.passes_m19 import detect_m19
        balls, events, players = {}, {}, {}
        for name, source in store.dataset['sources'].items():
            base = Path('outputs/world_model')/(name+'_m9_ready')
            metadata = json.loads((base/'manifest.json').read_text())
            if digest(base/'frames.jsonl') != metadata['frames_sha256'] or digest(metadata['source_video']) != source['video_sha256']:
                raise ValueError('Frozen world/video changed')
            directory = Path(replay_baseline)/name
            expected = published['provenance'][name]['variants']['m20']
            if digest(directory/'ball_m20/ball_frames.jsonl') != expected['ball_frames_sha256'] or digest(directory/'passes_m20/events.json') != expected['events_sha256']:
                raise ValueError('Replay is not the published byte-identical M20 baseline')
            balls[name] = [json.loads(line) for line in (directory/'ball_m20/ball_frames.jsonl').open()]
            worlds = compact_worlds(base/'frames.jsonl')
            events[name], _, _ = detect_m19(name, worlds, balls[name], source['fps'])
            if json.loads(json.dumps(events[name])) != json.loads((directory/'passes_m20/events.json').read_text()):
                raise ValueError('Frozen M20 pass replay mismatch')
            players[name] = {t[k]: list(visible_players(worlds[t[k]]).values())
                             for t in store.dataset['tasks'] if t['recording_id'] == name and t['kind'] == 'identity'
                             for k in ['frame', 'other_frame']}
        protocol = {**store.dataset['protocol'], 'sources': store.dataset['sources']}
        measured = evaluate(store.dataset['tasks'], rows, balls, events, protocol, players)
        measured['localization'] = localization(store.dataset['tasks'], rows, balls, protocol)
        for metric in ['ball', 'passes', 'identity', 'localization']:
            if measured[metric] != published['variants']['m20'][metric]:
                raise ValueError('M20 baseline metric changed: '+metric)
        result = {'reference_role': 'development_only', 'm20_baseline_source': str(replay_baseline),
                  'm20_byte_identical_replay_verified': True, 'm20_re_evaluated': measured,
                  'historical_baselines_from_hash_bound_2026_11_18_report': {
                      key: published['variants'][key] for key in ['original_m14', 'original_m16', 'm19']},
                  'historical_baselines_recomputed_this_run': False}
        (root/'baseline_reproduction.json').write_text(json.dumps(result, indent=2)+'\n')
    if path.read_bytes() != original or digest(snapshot) != report['annotations_sha256']:
        raise ValueError('Original annotations changed during audit')
    print(json.dumps({'output': str(root), 'saved': len(rows), 'per_recording': counts,
                      'final_test_status': report['final_test_status']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay-baseline', help='Explicit byte-identical M20 replay if original files are offline')
    args = parser.parse_args()
    audit(args.output, args.replay_baseline)
