"""Validate M23 before registering it; cache only byte-identical source content."""
import argparse
import json
import os
from pathlib import Path
from src.serving.cv import CVReview, digest
from src.serving.intelligent23 import IntelligentRegistry
from src.vision.intelligent23.inputs import Inputs


def install(root, registry=Path('configs/serving/cv_m23.json')):
    root = Path(root)
    report = json.loads((root/'report.json').read_text())
    inputs = Inputs(report['inputs_root'])
    if inputs.sha256 != report['input_manifest_sha256']:
        raise ValueError('Input snapshot changed')
    runtime = root.parent/'serving_worlds'
    runtime.mkdir(parents=True, exist_ok=True)
    worlds, evidence, bases = {}, {}, {}
    for name in inputs.manifest['approved_sources']:
        original = inputs.path(Path('outputs/world_model')/(name+'_m9_ready')/'manifest.json')
        manifest = json.loads(original.read_text())
        frames = inputs.path(Path('outputs/world_model')/(name+'_m9_ready')/'frames.jsonl')
        video = inputs.path(Path('data/raw')/(name+'.mp4'))
        if digest(video) != manifest['video_sha256']:
            raise ValueError('Source-video cache differs')
        directory = runtime/name
        directory.mkdir(exist_ok=True)
        target = directory/'frames.jsonl'
        if not target.exists():os.link(frames, target)
        if digest(target) != digest(frames):raise ValueError('M9 source frames differ')
        cached = manifest | {'source_video': str(video)}
        (directory/'manifest.json').write_text(json.dumps(cached, indent=2)+'\n')
        worlds[name] = str(directory)
        evidence[name] = {'original_manifest_sha256': digest(original),
            'runtime_manifest_sha256': digest(directory/'manifest.json'),
            'source_video_sha256': digest(video), 'm9_frames_sha256': digest(frames),
            'change': 'source_video points explicitly to a hash-equivalent cached video'}
        bases[name] = CVReview(directory)
    pending = root.parent/'pending_registry.json'
    entries = {name: {'root': str(root), 'report_sha256': digest(root/'report.json')} for name in worlds}
    pending.write_text(json.dumps(entries, indent=2)+'\n')
    selected = IntelligentRegistry(bases, pending)
    for name in worlds:
        review = selected.load(name)
        review.frame(0)
        selected.reference(name, 0)
        selected.metadata(name)
    runtime_registry = root.parent/'runtime_cv_registry.json'
    runtime_registry.write_text(json.dumps(worlds, indent=2)+'\n')
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(entries, indent=2)+'\n')
    (root.parent/'installation.json').write_text(json.dumps({'report_sha256': digest(root/'report.json'),
        'runtime_registry': str(runtime_registry), 'recordings': evidence}, indent=2)+'\n')
    print(json.dumps({'validated': list(worlds), 'registry': str(registry), 'runtime_registry': str(runtime_registry)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--registry', default=Path('configs/serving/cv_m23.json'), type=Path)
    args = parser.parse_args()
    install(args.analysis, args.registry)
