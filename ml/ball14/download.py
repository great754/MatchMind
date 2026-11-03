"""Fetch the bounded, SHA-pinned official initializer for a fresh research run."""
import requests
from .common import ROOT,save,sha
URL='https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth'
SHA='907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5'
def run():
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'generic_fasterrcnn_mobilenet320.pth'
 if not p.exists():
  temporary=p.with_suffix('.partial')
  try:
   with requests.get(URL,timeout=60,stream=True) as r,temporary.open('xb') as f:
    r.raise_for_status();total=0
    for b in r.iter_content(1048576):
     total+=len(b)
     if total>100000000:raise ValueError('Unexpected checkpoint size')
     f.write(b)
   if sha(temporary)!=SHA:raise ValueError('Checkpoint SHA mismatch')
   temporary.replace(p)
  finally:temporary.unlink(missing_ok=True)
 if sha(p)!=SHA:raise ValueError('Checkpoint SHA mismatch')
 save('initializer.json',{'url':URL,'sha256':SHA,'bytes':p.stat().st_size,'architecture':'torchvision FasterRCNN MobileNetV3-Large FPN320,COCO pretrained','ball_class':37,'code_license':'torchvision BSD-3-Clause; pretrained weight use follows source data conditions','source':'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html'})
if __name__=='__main__':run()
