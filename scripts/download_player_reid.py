"""Explicit offline checkpoint download; never called by serving/startup."""
import argparse,urllib.request
from pathlib import Path
from src.vision.world.player_appearance import MODEL_URL,MODEL_SHA256,digest

def download(path):
 p=Path(path)
 if p.exists():
  if digest(p)!=MODEL_SHA256:raise ValueError('Existing checkpoint hash mismatch; refusing overwrite')
  return p
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.download')
 try:
  with urllib.request.urlopen(MODEL_URL,timeout=60) as source,temp.open('xb') as target:
   total=0
   while chunk:=source.read(1048576):
    total+=len(chunk)
    if total>12000000:raise ValueError('Unexpected model download size')
    target.write(chunk)
  if digest(temp)!=MODEL_SHA256:raise ValueError('Checkpoint hash mismatch')
  temp.rename(p)
 finally:temp.unlink(missing_ok=True)
 return p
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/models/m9/osnet_x0_25_msmt17.pth');a=p.parse_args();print(download(a.output))
