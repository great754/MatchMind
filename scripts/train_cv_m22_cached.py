"""Run/resume the unchanged statistical trainer with verified local pixel I/O."""
import argparse,json,time
from pathlib import Path
from src.serving.cv import digest
from src.vision.football22.pixel_cache import PixelCache
from scripts.train_cv_m22 import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--resume',action='store_true');p.add_argument('--validation-layout',choices=['source','temporal_groups'],default='temporal_groups')
    p.add_argument('--pixel-cache');a=p.parse_args();data=Path(a.data)/'dataset.json'
    if not 1<=a.epochs<=40:raise ValueError('Use 1–40 bounded epochs')
    started=time.perf_counter();cache=PixelCache(data,a.pixel_cache or Path('/private/tmp')/('tactiq-m22-pixels-'+digest(data)[:16]))
    directory=Path(a.output);directory.mkdir(parents=True,exist_ok=True)
    execution={'dataset_sha256':cache.dataset_sha256,'cache_root':str(cache.root),'verified_unique_images':len(cache.paths),
        'statistical_training_code_unchanged':True,'launcher_sha256':digest(__file__),
        'pixel_cache_code_sha256':digest('src/vision/football22/pixel_cache.py'),'cache_prepare_seconds':time.perf_counter()-started}
    path=directory/'pixel_cache_executions.json';history=json.loads(path.read_text()) if path.exists() else [];history.append(execution)
    path.write_text(json.dumps(history,indent=2)+'\n');print('M22 local pixel cache ready:',len(cache.paths),'verified images',flush=True)
    with cache.training_io():run(a.data,a.output,a.epochs,a.resume,validation_layout=a.validation_layout)
