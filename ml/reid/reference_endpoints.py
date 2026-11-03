"""Exact endpoint crop diagnostics, separate from tracklet association scores."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from .protocol import ROOT
from .model import choose_device,seed_all
from .evaluate import load_model
from .shadow import tensors
from src.vision.world.player_appearance import BodyEncoder

def run():
 lock=json.loads((ROOT/'selection_lock.json').read_text());shadow=json.loads((ROOT/'shadow_comparison.json').read_text());device=choose_device();seed_all(20261102);soccer,strategy=load_model(lock['selected_checkpoint'],device=device);generic=BodyEncoder('outputs/models/m9/osnet_x0_25_msmt17.pth');results=[]
 for name,summary in shadow.items():
  source=Path('outputs/world_model')/(name+'_m9_ready');m=json.loads((source/'manifest.json').read_text());features=json.loads((Path('outputs/world_model/m9_cache')/name/'features.json').read_text());cap=cv2.VideoCapture(m['source_video']);raw=[];old=[];keys=[]
  for i,c in enumerate(summary['reference_cases']):
   for frame,track in zip(c['frames'],c['tracks']):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;p=next(p for p in features[frame] if p['track_id']==track);x,y,w,h=p['bbox'];full=image[max(0,int(y)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))];body=image[max(0,int(y+.18*h)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))];raw.append(full.copy());old.append(body.copy());keys.append((i,frame,track))
  ov=generic.encode(old)
  with torch.inference_mode():nv=soccer(tensors(raw,strategy).to(device))[1].cpu().numpy()
  for i,c in enumerate(summary['reference_cases']):
   c['generic_endpoint_crop_similarity']=float(ov[2*i]@ov[2*i+1]);c['soccer_endpoint_crop_similarity']=float(nv[2*i]@nv[2*i+1]);c['endpoint_vs_graph_note']='Endpoint crop similarity is diagnostic; assignment uses the complete eligible-tracklet medians and frozen M9 evidence.';results.append(c)
  cap.release();(ROOT/'m9_shadow'/name/'reference_comparison.json').write_text(json.dumps(summary['reference_cases'],indent=2))
 (ROOT/'reference_endpoint_comparison.json').write_text(json.dumps(results,indent=2));(ROOT/'shadow_comparison.json').write_text(json.dumps(shadow,indent=2))
if __name__=='__main__':run()
