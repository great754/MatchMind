"""Immutable experiment inputs, source provenance and compact existing M9 evidence."""
import json
from pathlib import Path
from src.serving.cv import digest
from src.vision.advanced21.players import observations

NAMES = ['test_match', 'team_goal', 'southampton']


def fresh_directory(path):
    root = Path(path)
    if root.exists():
        raise ValueError('Use a fresh experiment directory; never overwrite frozen results')
    root.mkdir(parents=True)
    return root


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_worlds(name):
    base = Path('outputs/world_model')/(name+'_m9_ready')
    metadata = json.loads((base/'manifest.json').read_text())
    if digest(base/'frames.jsonl') != metadata['frames_sha256'] or digest(metadata['source_video']) != metadata['video_sha256']:
        raise ValueError('Frozen source/world hash mismatch')
    worlds, people = [], []
    for line in (base/'frames.jsonl').open():
        row = json.loads(line)
        people.append(observations(row))
        camera = row['camera']
        worlds.append({'frame': row['frame'], 'timestamp': row['timestamp'],
            'camera': {k: camera[k] for k in ['epoch', 'continuity', 'calibration_valid', 'calibration']} | {
                'motion_evidence': {k: camera.get('motion_evidence', {}).get(k, {}) for k in ['field_motion', 'metric_camera', 'temporal_camera']}},
            'entities': [{k: e[k] for k in ['track_id', 'image_bbox', 'visibility']} for e in row['entities'] if e['visibility'] == 'visible'],
            'unassigned': [{'observation': {k: u['observation'][k] for k in ['track_id', 'bbox']}} for u in row['unassigned']],
            'player_world': {'players': [{k: p.get(k) for k in ['entity_id', 'current_segment_id', 'current_track_id', 'visibility_state',
                'identity_status', 'team', 'current_position', 'position_provenance', 'position_uncertainty_m']}
                for p in row['player_world']['players'] if p['visibility_state'] == 'VISIBLE']}})
    if len(worlds) != metadata['frames'] or any(row['frame'] != i or abs(row['timestamp']-i/metadata['fps']) > 1e-6 for i, row in enumerate(worlds)):
        raise ValueError('Frozen world clock mismatch')
    return metadata, worlds, people


def frozen_m20(name, root):
    expected = json.loads(Path('docs/evidence/cv_human_evaluation_2026_11_18.json').read_text())['provenance'][name]['variants']['m20']
    directory = Path(root)/name
    path = directory/'ball_m20/ball_frames.jsonl'
    events = directory/'passes_m20/events.json'
    if digest(path) != expected['ball_frames_sha256'] or digest(events) != expected['events_sha256']:
        raise ValueError('Explicit M20 baseline differs from the frozen evaluated output')
    return [json.loads(line) for line in path.open()], json.loads(events.read_text())
