"""TactiqAI canonical artifact build, safe Azure upload and verified read-back."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from .store import LocalArtifactStore,BUNDLE_FILES,ArtifactError,recording_key
from .manifest import validate_snapshot


def export(registry,output):
    """Preserve the legacy export layout and behavior; never copy arbitrary files."""
    store=LocalArtifactStore(registry);output=Path(output)
    for key in store.ids():
        source=store.resolve(key);dest=output/'matches'/key;dest.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink() or not dest.resolve().is_relative_to(output.resolve()):raise ArtifactError('Unsafe export destination')
        files={}
        for name in BUNDLE_FILES:
            p=source/name;target=dest/name
            if p.is_symlink() or target.is_symlink():raise ArtifactError('Artifact export symlinks are not accepted')
            if p.is_file():
                temporary=dest/(name+'.tmp')
                if temporary.is_symlink():raise ArtifactError('Unsafe export temporary path')
                shutil.copyfile(p,temporary);temporary.replace(target)
                files[name]=hashlib.sha256(target.read_bytes()).hexdigest()
            elif target.exists():target.unlink()
        raw=json.dumps(dict(schema_version='1.0',files=files),sort_keys=True,indent=2)+'\n'
        temporary=dest/'bundle.json.tmp'
        if temporary.is_symlink() or (dest/'bundle.json').is_symlink():raise ArtifactError('Unsafe export manifest path')
        temporary.write_text(raw);temporary.replace(dest/'bundle.json')
        validate_snapshot(dest)
    output.mkdir(parents=True,exist_ok=True)
    (output/'registry.json').write_text(json.dumps({k:str((output/'matches'/k).resolve()) for k in store.ids()},indent=2)+'\n')


def parser():
    root=argparse.ArgumentParser(description=__doc__)
    # Legacy options remain usable without a subcommand.
    root.add_argument('--registry',type=Path,default=Path('configs/serving/local.json'))
    root.add_argument('--output',type=Path,default=Path('outputs/serving_bundle'))
    commands=root.add_subparsers(dest='command')
    for name in ('build','upload','verify'):
        command=commands.add_parser(name)
        command.add_argument('--recording',required=True)
        command.add_argument('--registry',type=Path,default=Path('configs/serving/local.json'))
        command.add_argument('--output',type=Path,default=Path('outputs/serving_bundle'),help='Canonical export root (matches/<recording>/bundle.json)')
        if name=='verify':command.add_argument('--cache',type=Path,default=Path('outputs/azure_artifact_cache'))
    return root


def main(argv=None):
    args=parser().parse_args(argv);client=credential=None
    try:
        if args.command in (None,'build'):
            registry=json.loads(args.registry.read_text())
            if args.command:
                key=recording_key(args.recording)
                if key not in registry:raise ArtifactError('Unknown recording')
                registry={key:registry[key]}
            export(registry,args.output)
            print(json.dumps(dict(operation=args.command or 'export',recordings=sorted(registry),output=str(args.output),validated=True)))
            return 0
        key=recording_key(args.recording)
        from .azure_workflow import client_from_environment,upload,verify
        # Existing env names only. Never read/bundle provider or storage secrets.
        from dotenv import load_dotenv
        load_dotenv(override=False)
        logging.getLogger('azure').setLevel(logging.ERROR)
        # Reject unsafe local artifacts before constructing credentials or contacting Azure.
        if args.command=='upload':validate_snapshot(args.output/'matches'/key)
        client,credential=client_from_environment()
        result=upload(client,key,args.output/'matches'/key) if args.command=='upload' else verify(client,key,args.cache)
        print(json.dumps(dict(operation=args.command,**result),sort_keys=True));return 0
    except Exception as error:
        from .azure_workflow import diagnostic
        code=diagnostic(error)
        print(json.dumps(dict(operation=args.command or 'export',verified=False,error=code)),file=sys.stderr);return 1
    finally:
        if client is not None:client.close()
        if credential is not None:credential.close()

if __name__=='__main__':raise SystemExit(main())
