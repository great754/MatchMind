"""Hash-validated frozen-prefix caches; TEST requires a sealed selection."""
import time,json
import numpy as np,torch
from ml.reid.model import choose_device,seed_all
from .model import Visual,image_tensor
from .common import ROOT,SEED,save,load,sha

def cache(split):
 if split=='test' and not (ROOT/'selection_lock.json').exists():raise ValueError('TEST sealed until all models/scorer/thresholds are locked')
 rows=[r for r in load('dataset.json')['samples'] if r['split']==split];p=ROOT/f'{split}_prefix.npy';meta=ROOT/f'{split}_prefix.json'
 if p.exists():
  m=json.loads(meta.read_text())
  if m['samples']!=[r['sample_id'] for r in rows] or sha(p)!=m['sha256']:raise ValueError('Prefix cache mismatch')
  return np.load(p,mmap_mode='r'),rows
 seed_all(SEED);device=choose_device();model=Visual().to(device).eval();start=time.perf_counter();out=np.lib.format.open_memmap(p,mode='w+',dtype=np.float16,shape=(len(rows),197,384));generic=[]
 with torch.inference_mode():
  for i in range(0,len(rows),32):
   x=torch.from_numpy(np.stack([image_tensor(r) for r in rows[i:i+32]])).to(device);tokens=model.prefix(x);out[i:i+len(x)]=tokens.cpu().numpy().astype(np.float16);generic.extend(model.finish(tokens).cpu().numpy())
   if i%512==0:print('PREFIX',split,i,len(rows),'elapsed',round(time.perf_counter()-start,1),flush=True)
 out.flush();np.save(ROOT/f'{split}_pretrained.npy',np.asarray(generic));save(f'{split}_prefix.json',{'samples':[r['sample_id'] for r in rows],'sha256':sha(p),'initializer_sha256':load('initializer.json')['sha256'],'dataset_sha256':sha(ROOT/'dataset.json'),'device':device,'seconds':time.perf_counter()-start,'bytes':p.stat().st_size});return out,rows

def fine_vectors(model,tokens,device):
 v=[]
 with torch.inference_mode():
  for i in range(0,len(tokens),48):v.extend(model.finish(torch.from_numpy(np.array(tokens[i:i+48],dtype=np.float32)).to(device)).cpu().numpy())
 return np.asarray(v)
if __name__=='__main__':
 import sys
 for s in sys.argv[1:] or ['train','validation']:cache(s)
