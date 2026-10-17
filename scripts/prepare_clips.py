"""Download and process the ten short SoccerTrack clips, preserving ready bundles."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from scripts.prepare_recording import inspect
from src.artifacts.store import validate_bundle

MATCHES=('117092','117093','118575','118576','118577','118578','128057','128058','132831','132877')


def configuration(match,registry='configs/serving/local.json'):
    return dict(recording_id=match+'_clip',match_id=match,data_root='data/soccertrack',
                output=f'outputs/analytics_{match}_tactics',registry=registry)


def already_processed(config):
    registry=Path(config['registry']);values=json.loads(registry.read_text()) if registry.exists() else {}
    key=config['recording_id']
    if key not in values:return False
    try:
        directory=validate_bundle(values[key]);summary=json.loads((directory/'summary.json').read_text())
        from src.vision.data_loading import DatasetPaths
        return str(summary.get('match_id'))==config['match_id'] and summary.get('scope')=='clip' and DatasetPaths(config['match_id'],Path(config['data_root'])).video.is_file()
    except (ValueError,OSError):return False


def prepare_one(config,config_path):
    if already_processed(config):return 'already_processed'
    from huggingface_hub import hf_hub_download
    for filename in inspect(config)['missing']:
        print('Downloading '+filename,flush=True)
        hf_hub_download('atomscott/soccertrack-v2',filename,repo_type='dataset',local_dir=config['data_root'])
    if not inspect(config)['ready']:raise ValueError('Dependencies remain missing')
    subprocess.run([sys.executable,'-m','scripts.prepare_recording','--config',str(config_path),'--run'],check=True)
    if not already_processed(config):raise ValueError('Processed bundle did not validate')
    return 'processed'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',help='Download missing dependencies and process clips sequentially')
    args=parser.parse_args();results=[]
    for match in MATCHES:
        config=configuration(match);path=Path('configs/recordings')/(match+'_clip.json')
        if path.exists():
            config=json.loads(path.read_text())
            if config.get('match_id')!=match or config.get('recording_id')!=match+'_clip':raise ValueError('Configuration identity mismatch: '+str(path))
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(config,indent=2)+'\n')
        if not args.run:
            print(json.dumps(dict(**inspect(config),already_processed=already_processed(config)),indent=2));continue
        print('\nPreparing '+config['recording_id'],flush=True)
        try:status=prepare_one(config,path);result=dict(recording_id=config['recording_id'],status=status)
        except Exception as error:result=dict(recording_id=config['recording_id'],status='failed',error_type=type(error).__name__)
        print(json.dumps(result),flush=True);results.append(result)
        destination=Path('outputs/clip_batch_status.json');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status']=='failed' for r in results):raise SystemExit('Some clips failed; see outputs/clip_batch_status.json. Rerun to retry; valid processed clips are skipped.')

if __name__=='__main__':main()
