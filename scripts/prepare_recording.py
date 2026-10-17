"""Preflight and process an independent SoccerTrack clip; register only on success."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from src.vision.data_loading import DatasetPaths
from src.artifacts.store import recording_key,validate_bundle


def dependencies(paths):
    # Both halves are needed to discover the clip's half/offset without guessing.
    return [paths.mot,*[paths.root/'gsr'/paths.match_id/f'{paths.match_id}_{half}.json' for half in ('1st','2nd')],
            paths.ball(1),paths.ball(2),paths.bas,paths.keypoints,
            paths.raw/f'{paths.match_id}_tracker_box_metadata.xml']


def inspect(config):
    recording_key(config['recording_id']);paths=DatasetPaths(config['match_id'],Path(config['data_root']))
    missing=[p for p in dependencies(paths) if not p.is_file()]
    # Downloads always use canonical repository names, not browser-added suffixes.
    names=[str(p.relative_to(paths.root)) for p in missing]
    if not paths.video.is_file():names.append(f'mot/clips/{paths.match_id}.mp4')
    command=shlex.join(['hf','download','atomscott/soccertrack-v2','--repo-type','dataset',*names,'--local-dir',str(paths.root)]) if names else None
    return dict(recording_id=config['recording_id'],match_id=paths.match_id,video=str(paths.video),missing=names,download_command=command,ready=not names)


def protect_output(config):
    output=Path(config['output']).resolve();registry=Path(config['registry'])
    values=json.loads(registry.read_text()) if registry.exists() else {}
    if any(key!=config['recording_id'] and Path(value).resolve()==output for key,value in values.items()):
        raise ValueError('Output directory belongs to another registered recording')
    summary=output/'summary.json'
    if summary.exists() and str(json.loads(summary.read_text()).get('match_id'))!=config['match_id']:
        raise ValueError('Output directory belongs to another match')


def register(config):
    directory=validate_bundle(config['output']);summary=json.loads((directory/'summary.json').read_text())
    if str(summary.get('match_id'))!=config['match_id'] or summary.get('scope')!='clip':raise ValueError('Artifact recording identity mismatch')
    registry=Path(config['registry']);values=json.loads(registry.read_text()) if registry.exists() else {}
    key=recording_key(config['recording_id'])
    if key in values and Path(values[key]).resolve()!=directory:raise ValueError('Recording ID already belongs to another artifact directory')
    values[key]=str(Path(config['output']));registry.parent.mkdir(parents=True,exist_ok=True)
    temp=registry.with_suffix('.tmp');temp.write_text(json.dumps(values,indent=2)+'\n');temp.replace(registry)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());report=inspect(config);print(json.dumps(report,indent=2),flush=True)
    if not args.run:return
    if not report['ready']:raise SystemExit('Dependencies missing; run the printed download command first. No artifacts registered.')
    protect_output(config)
    match=config['match_id'];root=config['data_root']
    subprocess.run([sys.executable,'-m','src.vision.align_clip','--match-id',match,'--data-root',root],check=True)
    subprocess.run([sys.executable,'-m','src.analytics.pipeline','--match-id',match,'--data-root',root,'--scope','clip','--player-source','gsr','--summary-only','--commentary-provider','template','--output',config['output']],check=True)
    register(config);print('Processed and registered '+config['recording_id'])

if __name__=='__main__':main()
