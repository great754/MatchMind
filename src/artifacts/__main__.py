"""Package allowlisted serving files; no raw dataset, caches or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .store import LocalArtifactStore,BUNDLE_FILES

def export(registry,output):
    store=LocalArtifactStore(registry);output=Path(output)
    for key in store.ids():
        source=store.resolve(key);dest=output/'matches'/key;dest.mkdir(parents=True,exist_ok=True)
        files={}
        for name in BUNDLE_FILES:
            p=source/name
            if p.is_file():
                shutil.copyfile(p,dest/name);files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        (dest/'bundle.json').write_text(json.dumps(dict(schema_version='1.0',files=files),sort_keys=True,indent=2)+'\n')
    (output/'registry.json').write_text(json.dumps({k:str((output/'matches'/k).resolve()) for k in store.ids()},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,default=Path('configs/serving/local.json'));p.add_argument('--output',type=Path,default=Path('outputs/serving_bundle'));a=p.parse_args();export(json.loads(a.registry.read_text()),a.output)
if __name__=='__main__':main()
