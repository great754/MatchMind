"""Explicit bounded public-source setup. No full archives, cloud writes or startup hooks."""
import argparse,json,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .common import ROOT,load,save,sha,freeze
from .remote import open_remote,member_bytes
from .data import PLAN
REPO='SN-Tracking-2023';REVISION='3777f456d908570e10b19e7354744640e77c351e'
MODEL_URL='https://drive.google.com/uc?export=download&id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x'
MODEL_SHA='48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727'

def setup():
 ROOT.mkdir(parents=True,exist_ok=True)
 audit=ROOT/'source_audit.json'
 if not audit.exists():audit.write_bytes(Path('docs/evidence/cv_m12_source_audit.json').read_bytes())
 if not (ROOT/'frozen_baselines.json').exists():freeze()
 tree_response=requests.get(f'https://huggingface.co/api/datasets/SoccerNet/{REPO}/tree/{REVISION}?recursive=true&expand=false',timeout=30);tree_response.raise_for_status();tree=tree_response.json();save(REPO+'_tree.json',tree);save('soccernet_inventory.json',{REPO:{'sha':REVISION,'gated':False}});item=next(x for x in tree if x['path']=='train.zip');url=f'https://huggingface.co/datasets/SoccerNet/{REPO}/resolve/{REVISION}/train.zip';z,f=open_remote(url,item['size']);entries=[{'path':i.filename,'size':i.file_size,'compressed':i.compress_size,'crc':i.CRC,'offset':i.header_offset} for i in z.infolist()];save(REPO+'_zip_index.json',entries);sequences={s for p in PLAN.values() for s in p['sequences']};selected=[e for e in entries if len(e['path'].split('/'))>2 and e['path'].split('/')[1] in sequences and (e['path'].endswith(('/gt/gt.txt','/seqinfo.ini','/gameinfo.ini')))]
 def fetch(e):
  b,n=member_bytes(url,item['size'],e);p=ROOT/'source_metadata'/e['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);return n
 with ThreadPoolExecutor(max_workers=8) as pool:count=sum(pool.map(fetch,selected))
 protocol=ROOT/'evaluation_protocol.json'
 if not protocol.exists():protocol.write_bytes(Path('configs/ml/soccer-reid-v2-protocol.json').read_bytes())
 p=ROOT/'osnet_x1_0_msmt17.pth'
 if not p.exists():
  r=requests.get(MODEL_URL,timeout=45,stream=True);r.raise_for_status();temp=p.with_suffix('.partial');total=0
  try:
   with temp.open('xb') as out:
    for chunk in r.iter_content(1048576):
     total+=len(chunk)
     if total>40000000:raise ValueError('Unexpected initializer size')
     out.write(chunk)
   if sha(temp)!=MODEL_SHA:raise ValueError('Initializer differs from recorded author checkpoint')
   temp.rename(p)
  finally:temp.unlink(missing_ok=True)
 if sha(p)!=MODEL_SHA:raise ValueError('Existing alternative initializer changed')
 save('alternative_initializer.json',{'url':MODEL_URL,'sha256':MODEL_SHA,'bytes':p.stat().st_size,'architecture':'OSNet x1.0,512D','pretraining':'authors MSMT17 combineall','author_source':'https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO'})
 save('setup_transfer.json',{'directory_bytes':f.bytes,'annotation_bytes':count,'archive_bytes':item['size'],'archive_downloaded':False});print('Bounded metadata and initializer setup complete')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
 if a.download:setup()
 else:print('Use --download explicitly for public metadata and small initializer; see M12 dataset-use scope first.')
