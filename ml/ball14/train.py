"""Bounded head/RPN fine-tuning and validation-only detector/view/threshold selection."""
import time,random,json
from pathlib import Path
import cv2,numpy as np,torch
from ml.reid.model import seed_all,choose_device
from ml.reid.train import cpu_tree
from .common import ROOT,SEED,load,save,sha
from .model import make_model,training_item,infer
from .evaluate import detection_metrics
CONFIG={'seed':SEED,'epochs':4,'steps_per_epoch':60,'batch':2,'lr':.0001,'optimizer':'AdamW','weight_decay':.0005,'trainable':'pretrained RPN + ROI heads; frozen MobileNet/FPN backbone and BN','head':'background/ball, transferred COCO37 and background0 weights','fine_anchor_sizes':[[4,8,16,32,64],[8,16,32,64,128],[16,32,64,128,256]],'input':384,'training_tile':320,'inference_tile':480,'overlap':.2,'validation_sample':'every8th validation frame (20frames), identical for all configurations; no TEST inference','selection':'maximum validation F1 at IoU0.25, confidence grid0.05,0.1,0.2,0.35,0.5; ties favor precision and full-frame lower cost','comparison':['generic_full','generic_tiled','fine_full','fine_tiled'],'test':'locked model and operating point evaluated once; original matched frame list for all ablations','ground_plane':'automatic ground contact is a conservative heuristic, not true3D; otherwise image-only'}

def validate(m,rows,fine,device):
 results=[]
 for tiled in [False,True]:
  predictions=[];seconds=0
  for row in rows:
   p,t=infer(m,cv2.imread(row['path']),fine,tiled,device,tile_size=(load('trial_settings.json').get('inference_tile',480) if (ROOT/'trial_settings.json').exists() else 480));predictions.append(p);seconds+=t['seconds']
  for threshold in [.05,.1,.2,.35,.5]:
   metrics=detection_metrics(rows,predictions,threshold);results.append({'kind':'finetuned' if fine else 'generic','tiled':tiled,'threshold':threshold,'metrics':metrics,'inference_seconds':seconds})
 return results

def run():
 if (ROOT/'selection_lock.json').exists() or (ROOT/'training_history.json').exists():raise ValueError('Experiment already started; use a new root/version')
 seed_all(SEED);CONFIG['trial_settings']=load('trial_settings.json') if (ROOT/'trial_settings.json').exists() else {};save('config.json',CONFIG);
 if not Path('configs/ml/ball-m14.json').exists():Path('configs/ml/ball-m14.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 device=choose_device();data=load('dataset.json')['samples'];train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='validation'][::8];save('validation_selection_samples.json',[r['sample_id'] for r in val]);tick=time.perf_counter();generic=make_model(device=device).eval();candidates=validate(generic,val,False,device);del generic
 if device=='mps':torch.mps.empty_cache()
 model=make_model(True,device=device)
 for p in model.backbone.parameters():p.requires_grad=False
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay']);history=[]
 for epoch in range(CONFIG['epochs']):
  model.train();model.backbone.eval();rng=random.Random(SEED+epoch);losses=[];start=time.perf_counter()
  for step in range(CONFIG['steps_per_epoch']):
   batch=[training_item(rng.choice(train),rng) for _ in range(2)];images=[x.to(device) for x,y in batch];targets=[{k:v.to(device) for k,v in y.items()} for x,y in batch];parts=model(images,targets);loss=sum(parts.values())
   if not torch.isfinite(loss):raise ValueError('Nonfinite detector training loss')
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
  train_seconds=time.perf_counter()-start;model.eval();metrics=validate(model,val,True,device);p=ROOT/f'fine_epoch_{epoch+1:02d}.pth';torch.save({'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),'epoch':epoch+1,'torch_rng_state':torch.get_rng_state(),'initializer_sha256':load('initializer.json')['sha256'],'config_sha256':sha(ROOT/'config.json'),'dataset_sha256':sha(ROOT/'dataset.json'),'trial_settings':CONFIG['trial_settings']},p);entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'training_seconds':train_seconds,'checkpoint':str(p),'sha256':sha(p),'validation':metrics,'mps_allocated_bytes_snapshot':torch.mps.current_allocated_memory() if device=='mps' else None};history.append(entry);save('training_history.json',history)
  for c in metrics:candidates.append({**c,'checkpoint':str(p),'checkpoint_sha256':sha(p),'epoch':epoch+1})
  print('EPOCH',epoch+1,'loss',entry['loss'],'bestF1',max(c['metrics']['f1'] for c in metrics),'train_s',train_seconds,flush=True)
 selected=max(candidates,key=lambda c:(c['metrics']['f1'],c['metrics']['precision'],not c['tiled']));selection={**selected,'checkpoint':selected.get('checkpoint',str(ROOT/'generic_fasterrcnn_mobilenet320.pth')),'checkpoint_sha256':selected.get('checkpoint_sha256',load('initializer.json')['sha256']),'config_sha256':sha(ROOT/'config.json'),'dataset_sha256':sha(ROOT/'dataset.json'),'test_seen':False,'model_selection_wall_seconds':time.perf_counter()-tick,'device':device,'trial_settings':CONFIG['trial_settings']};save('validation_candidates.json',candidates);save('selection_lock.json',selection);print('SELECTION',selection,flush=True)
if __name__=='__main__':run()
