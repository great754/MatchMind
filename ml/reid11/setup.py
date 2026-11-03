"""Explicit offline M11 setup; small calibration files and official ViT weights only."""
import argparse,json,urllib.request,shutil
from pathlib import Path
from .common import ROOT,M10,save,sha,freeze
DINO_URL='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
DINO_SHA='1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7'
REVISION='325bd4c57ee8e135ab652f3921d274526ada496b'

def download():
 from huggingface_hub import HfApi,hf_hub_download
 ROOT.mkdir(parents=True,exist_ok=True)
 protocol=ROOT/'evaluation_protocol.json'
 if not protocol.exists():shutil.copy2('configs/ml/temporal-soccer-reid-v1-evaluation.json',protocol)
 p=ROOT/'dino_deitsmall16_pretrain.pth'
 if not p.exists():
  temp=p.with_suffix('.partial');urllib.request.urlretrieve(DINO_URL,temp)
  if sha(temp)!=DINO_SHA:temp.unlink();raise ValueError('Official ViT checkpoint content differs from recorded v1 initializer')
  temp.rename(p)
 if sha(p)!=DINO_SHA:raise ValueError('Existing ViT initializer changed')
 save('initializer.json',{'url':DINO_URL,'sha256':DINO_SHA,'bytes':p.stat().st_size,'pretraining':'DINO self-supervised ImageNet; ViT-S/16,384D','not_person_reid_pretrained':True})
 files=HfApi().list_repo_files('atomscott/soccertrack-v2',repo_type='dataset',revision=REVISION);save('official_dataset_inventory.json',{'repo':'atomscott/soccertrack-v2','revision':REVISION,'files':files});rows=json.loads((M10/'curated_manifest.json').read_text())['samples'];provenance={}
 for match in sorted({r['match_id'] for r in rows}):
  p=hf_hub_download('atomscott/soccertrack-v2',f'raw/{match}/{match}_keypoints.json',repo_type='dataset',revision=REVISION,local_dir=ROOT/'annotations');provenance[match]={'file':str(p),'sha256':sha(p),'source':'organizer paired nonlinear image/pitch keypoints; not independent camera uncertainty'}
 save('geometry_provenance.json',provenance)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
 if not (ROOT/'frozen_m10.json').exists():freeze()
 if args.download:download()
 else:print('Baseline frozen. Use --download explicitly for model/calibration setup.')
