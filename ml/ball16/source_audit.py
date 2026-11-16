"""Inspect archive directory and annotation contracts before any large training payload."""
import json,configparser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ml.reid13.remote import open_remote,member_bytes
from .common import ROOT,save
SOURCE=Path('outputs/ml/soccer_reid_m13')
def run():
 inv=json.loads((SOURCE/'soccernet_inventory.json').read_text());rev=inv['SN-Tracking-2023']['sha'];tree=json.loads((SOURCE/'SN-Tracking-2023_tree.json').read_text());size=next(e['size'] for e in tree if e['path']=='test.zip');url=f'https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/{rev}/test.zip';p=ROOT/'test_zip_index.json'
 if not p.exists():
  z,remote=open_remote(url,size);entries=[{'path':e.filename,'size':e.file_size,'compressed':e.compress_size,'crc':e.CRC,'offset':e.header_offset} for e in z.infolist()];save('test_zip_index.json',entries);directory_bytes=remote.bytes
 else:entries=json.loads(p.read_text());directory_bytes=0
 selected=[e for e in entries if e['path'].endswith(('gameinfo.ini','seqinfo.ini','gt/gt.txt'))];network=0
 def fetch(e):
  p=ROOT/'source_metadata'/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():return 0
  b,n=member_bytes(url,size,e);p.write_bytes(b);return n
 with ThreadPoolExecutor(max_workers=8) as pool:network=sum(pool.map(fetch,selected))
 games={}
 for p in (ROOT/'source_metadata').rglob('gameinfo.ini'):
  cfg=configparser.ConfigParser();cfg.read(p);m=dict(cfg['Sequence']);games.setdefault(m['gameid'],[]).append({'sequence':p.parent.name,'clipstart_ms':m['clipstart'],'ball_ids':[k for k,v in m.items() if k.startswith('trackletid_') and v.startswith('ball;')]})
 save('source_audit.json',{'revision':rev,'url':url,'archive_size':size,'directory_bytes':directory_bytes,'metadata_bytes':network,'games':games,'gt_members':sum(e['path'].endswith('gt/gt.txt') for e in entries),'visibility_contract':'Organizer tracked boxes/identities; no reliable explicit ball visibility/occlusion/height labels; absent row not certified negative','terms':'Same public SoccerNet Tracking research scope as M14/M15; no commercial rights asserted','full_archive_downloaded':False});print('AUDIT',games,'metadata',network,flush=True)
if __name__=='__main__':run()
