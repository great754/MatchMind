"""Explicit, content-pinned copies of approved sources; no unrelated fallback."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from src.serving.cv import digest
from src.vision.advanced21.players import observations

NAMES = ('test_match', 'team_goal', 'southampton')
ORIGINAL_LABEL_SHA = 'c4ac71548269fd1fe4e601684dffa09a706e8a215ed978ff9e0c8feb2966991e'
M22_LABEL_SHA = '0b4c470a5f116f2a8c4811f7a5b7d66f11b017534d0f4d5146aae7a1312a51b4'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def prepare(output, base_inputs=None, resume=False):
    root = Path(output)
    if root.exists() and not resume:
        raise ValueError('Use a fresh immutable M23 input root')
    if (root/'manifest.json').exists():
        raise ValueError('Completed input snapshots are immutable; use Inputs to verify them')
    root.mkdir(parents=True, exist_ok=resume, mode=0o700)
    if base_inputs is not None:
        parent = Inputs(base_inputs)
        files = {}
        for key, item in parent.manifest['files'].items():
            source = parent.path(key)
            destination = root/'files'/key
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if digest(destination) != item['sha256']:raise ValueError('Partial parent input changed')
            else:os.link(source, destination)
            files[key] = {**item, 'path': str(destination)}
        trained = parent.json('outputs/m22/analysis_review_v1/report.json')
        m22_root = Path('outputs/m22/perception_review_v1')
        for source in [m22_root/'report.json', *[m22_root/n/'detections.jsonl' for n in NAMES]]:
            destination = root/'files'/source
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = (trained['perception_report_sha256'] if source.name == 'report.json' else
                        trained['perception']['clips'][source.parent.name]['detections_sha256'])
            if destination.exists() and digest(destination) == expected:
                files[str(source)] = {'path': str(destination), 'sha256': expected, 'bytes': destination.stat().st_size}
                continue
            destination.unlink(missing_ok=True)
            if source.stat().st_size and not source.stat().st_blocks and os.uname().sysname == 'Darwin':
                subprocess.run(['brctl', 'download', str(source.resolve())], check=True, capture_output=True)
            shutil.copyfile(source, destination)
            sha = digest(destination)
            if digest(source) != sha or sha != expected:raise ValueError('Complementary visual cache changed during copy')
            destination.chmod(0o400)
            files[str(source)] = {'path': str(destination), 'sha256': sha, 'bytes': destination.stat().st_size}
        if files[str(m22_root/'report.json')]['sha256'] != trained['perception_report_sha256']:
            raise ValueError('Wrong frozen M22 complementary perception')
        for name in NAMES:
            if files[str(m22_root/name/'detections.jsonl')]['sha256'] != trained['perception']['clips'][name]['detections_sha256']:
                raise ValueError('M22 raw current-pixel candidate cache mismatch')
        result = {**parent.manifest, 'files': files, 'parent_input_manifest_sha256': parent.sha256,
            'm22_global_root': str(m22_root),
            'complement_models': {n: {k: trained['training'][n][k] for k in ['checkpoint', 'checkpoint_sha256', 'split']} for n in NAMES}}
        write(root/'manifest.json', result)
        return result
    m21 = json.loads(Path('outputs/m21/analysis_final_v1/report.json').read_text())
    m22 = json.loads(Path('outputs/m22/analysis_review_v1/report.json').read_text())
    primary_root = Path(m21['perception_source'])
    wanted = {
        Path('configs/evaluation/cv_m19_tasks.json'), Path('outputs/m19/human_review/labels.json'),
        Path('outputs/m22/completed_review_snapshot_v1/tasks.json'),
        Path('outputs/m22/completed_review_snapshot_v1/labels.json'),
        Path('outputs/m21/analysis_final_v1/report.json'), Path('outputs/m22/analysis_review_v1/report.json'),
        primary_root/'report.json', primary_root/'config.json',
        Path('outputs/ml/ball_m16/yolo11s.pt'),
        Path('outputs/ml/ball_m16/training/soccer11s/weights/best.pt'),
        Path('configs/vision/cv_m20.json'), Path('configs/vision/cv_m21.json'), Path('configs/vision/cv_m22.json'),
    }
    for name in NAMES:
        base = Path('outputs/world_model')/(name+'_m9_ready')
        meta = json.loads((base/'manifest.json').read_text())
        wanted.update([base/'manifest.json', base/'frames.jsonl', Path(meta['source_video']),
                       primary_root/name/'inputs.jsonl', Path(m22['training'][name]['checkpoint'])])
        for directory in [Path('outputs/m20/run_v1')/name/'ball_m20',
                          Path('outputs/m20/run_v1')/name/'passes_m20',
                          Path('outputs/m21/analysis_final_v1')/name/'temporal_offline',
                          Path('outputs/m21/analysis_final_v1')/name/'offline_dag',
                          Path('outputs/m22/analysis_review_v1')/name]:
            wanted.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl'])

    def copied(source):
        source = source.resolve()
        relative = source.relative_to(Path.cwd().resolve())
        destination = root/'files'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                if source.stat().st_size and not source.stat().st_blocks and os.uname().sysname == 'Darwin':
                    subprocess.run(['brctl', 'download', str(source)], check=True, capture_output=True)
                shutil.copyfile(source, destination)
                sha = digest(destination)
                if digest(source) != sha:
                    raise ValueError('Input changed while copying: '+str(relative))
                destination.chmod(0o400)
                return str(relative), {'path': str(destination), 'sha256': sha, 'bytes': destination.stat().st_size}
            except TimeoutError:
                if attempt == 2:
                    raise
        raise RuntimeError('No materialized input')

    files = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(copied, p) for p in sorted(wanted)]):
            key, value = future.result()
            files[key] = value
            print('M23 SNAPSHOT', key, flush=True)
    result = {'schema_version': 'cv-m23-inputs-v1', 'approved_sources': list(NAMES),
              'reference_role': 'development_only', 'independent_test_available': False,
              'm21_perception_root': str(primary_root), 'files': files,
              'withdrawn_m21_review_used': False, 'full_match_used': False,
              'original_labels_sha256': files['outputs/m19/human_review/labels.json']['sha256'],
              'm22_labels_sha256': files['outputs/m22/completed_review_snapshot_v1/labels.json']['sha256']}
    if result['original_labels_sha256'] != ORIGINAL_LABEL_SHA or result['m22_labels_sha256'] != M22_LABEL_SHA:
        raise ValueError('The fixed development references changed; explicitly version a new protocol')
    write(root/'manifest.json', result)
    return result


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['schema_version'] != 'cv-m23-inputs-v1' or tuple(self.manifest['approved_sources']) != NAMES:
            raise ValueError('Only the three approved development clips are allowed')
        self.sha256 = digest(self.root/'manifest.json')

    def path(self, original, verify=True):
        path = Path(original)
        key = str(path.resolve().relative_to(Path.cwd().resolve())) if path.is_absolute() else str(path)
        item = self.manifest['files'][key]
        result = Path(item['path'])
        if not result.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Input escaped its pinned snapshot')
        if verify and digest(result) != item['sha256']:
            raise ValueError('Pinned M23 input changed: '+key)
        return result

    def json(self, path):
        return json.loads(self.path(path).read_text())

    def worlds(self, name):
        if name not in NAMES:
            raise ValueError('Unapproved M23 source')
        base = Path('outputs/world_model')/(name+'_m9_ready')
        metadata = self.json(base/'manifest.json')
        video = self.path(metadata['source_video'])
        data = self.path(base/'frames.jsonl')
        if digest(video) != metadata['video_sha256'] or digest(data) != metadata['frames_sha256']:
            raise ValueError('Frozen M9 source hash mismatch')
        # Only the explicitly hash-equivalent snapshot path is redirected.
        metadata = {**metadata, 'source_video': str(video)}
        worlds, people = [], []
        for line in data.open():
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
        if len(worlds) != metadata['frames'] or any(w['frame'] != i or abs(w['timestamp']-i/metadata['fps']) > 1e-6 for i, w in enumerate(worlds)):
            raise ValueError('Frozen source clock mismatch')
        return metadata, worlds, people
