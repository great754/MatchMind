"""Bounded public organizer metadata audit and official small detector initializer."""
import json,configparser,concurrent.futures,requests
from pathlib import Path
from ml.reid13.remote import member_bytes
from .common import ROOT,save,sha
SOURCE=Path('outputs/ml/soccer_reid_m13')
def remote():
 inv=json.loads((SOURCE/'soccernet_inventory.json').read_text());rev=inv['SN-Tracking-2023']['sha'];tree=json.loads((SOURCE/'SN-Tracking-2023_tree.json').read_text());size=next(x['size'] for x in tree if x['path']=='train.zip');entries={e['path']:e for e in json.loads((SOURCE/'SN-Tracking-2023_zip_index.json').read_text())};return f'https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/{rev}/train.zip',size,entries,rev

def run():
 url,size,entries,rev=remote();members=[e for k,e in entries.items() if k.endswith('gameinfo.ini')];dest=ROOT/'source_metadata';dest.mkdir(exist_ok=True);total=0
 def fetch(e):
  seq=e['path'].split('/')[1];p=dest/seq/'gameinfo.ini';p.parent.mkdir(exist_ok=True)
  if p.exists():b=p.read_bytes();n=0
  else:b,n=member_bytes(url,size,e,budget=100000);p.write_bytes(b)
  c=configparser.ConfigParser();c.read_string(b.decode());return seq,dict(c['Sequence']),n
 rows={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for seq,meta,n in pool.map(fetch,members):rows[seq]=meta;total+=n
 save('metadata_audit.json',{'sequences':rows,'network_bytes':total,'revision':rev,'source_terms':'SoccerNet research-use, same original source audited for M14; no commercial/deployment permission asserted'})
 model=ROOT/'yolo11n.pt'
 if not model.exists():
  r=requests.get('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',timeout=60);r.raise_for_status()
  if len(r.content)>15000000:raise ValueError('Unexpected checkpoint size')
  model.write_bytes(r.content)
 save('initializer.json',{'url':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt','sha256':sha(model),'bytes':model.stat().st_size,'architecture':'YOLO11n COCO, 2.6M parameters','license':'Ultralytics AGPL-3.0; offline research only, no commercial deployment approval','package':'ultralytics==8.3.228'})
 print('META',len(rows),'bytes',total,'model',model.stat().st_size,flush=True)
if __name__=='__main__':run()
