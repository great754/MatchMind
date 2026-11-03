"""Fresh broadcast TEST evaluation and calibration fixed exclusively on VAL."""
import json,time
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from ml.reid.model import choose_device,seed_all
from ml.reid.evaluate import load_model as old_load,embed as old_embed
from ml.reid11.model import Visual,image_tensor as vit_image
from ml.reid11.common import ROOT as M11ROOT
from ml.reid11.scorer import choose_threshold
from .common import ROOT,M10,M11,SEED,save,load,sha
from .train import load_model,embed
from .model import Pooler
from .protocol import evaluate,retrieval

def vectors(rows,name,checkpoint=None):
 device=choose_device();key=sha(checkpoint) if checkpoint else name;p=ROOT/'embedding_cache'/f'{rows[0]["split"]}-{name}-{key[:16]}.npz';p.parent.mkdir(exist_ok=True)
 if rows[0]['split']=='test' and not (ROOT/'selection_lock.json').exists():raise ValueError('TEST sealed before selection lock')
 if p.exists():
  meta=json.loads(p.with_suffix('.json').read_text())
  if meta['samples']!=[r['sample_id'] for r in rows] or meta['sha256']!=sha(p):raise ValueError('Cache identity/hash mismatch')
  return np.load(p)['vectors']
 start=time.perf_counter()
 if name in ['m9','m10']:
  cp=json.loads((M10/'selection_lock.json').read_text())['selected_checkpoint'] if name=='m10' else None;m,strategy=old_load(cp,device=device);v,_=old_embed(rows,m,strategy,device)
 elif name=='m11':
  s=json.loads((M11/'selection_lock.json').read_text());m=Visual().to(device);payload=torch.load(s['visual_checkpoint'],map_location=device,weights_only=True);m.net.blocks[-1].load_state_dict(payload['last_block']);m.net.norm.load_state_dict(payload['norm']);m.eval();out=[]
  with torch.inference_mode():
   for i in range(0,len(rows),32):out.extend(m(torch.tensor(np.stack([vit_image(r) for r in rows[i:i+32]]),device=device)).cpu().numpy())
  v=np.array(out)
 else:m=load_model(checkpoint,device);v=embed(rows,m,device)
 np.savez_compressed(p,vectors=v);p.with_suffix('.json').write_text(json.dumps({'samples':[r['sample_id'] for r in rows],'sha256':sha(p),'model':name,'checkpoint_sha256':sha(checkpoint) if checkpoint else None,'device':device,'seconds':time.perf_counter()-start}));del m
 if device=='mps':torch.mps.empty_cache()
 return v

def pair_values(rows,v):
 q,g,_,_=retrieval(rows);by={r['sample_id']:i for i,r in enumerate(rows)};y=[];scores=[]
 for a in q:
  for b in g:
   if a['match_id']==b['match_id'] and a['team_identity']==b['team_identity']:y.append(a['player_identity']==b['player_identity']);scores.append(float(v[by[a['sample_id']]]@v[by[b['sample_id']]]))
 return np.array(y,bool),np.array(scores)

def calibrate():
 if (ROOT/'calibration_lock.json').exists():raise ValueError('Calibration already locked')
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='validation'];models=[('m9',None),('m10',None),('m11',None)]+[(v['name'],v['selected']['checkpoint']) for v in load('variant_selections.json')];out={}
 for name,cp in models:
  v=vectors(rows,name,cp);y,s=pair_values(rows,v);t=choose_threshold(y,s,.01);out[name]={'threshold':t,'validation_false_match_rate':float(np.mean(s[~y]>=t)),'validation_false_nonmatch_rate':float(np.mean(s[y]<t)),'pair_count':len(y),'positives':int(y.sum()),'validation_sample_sha256':sha(ROOT/'dataset.json'),'test_seen':False}
 save('calibration_lock.json',out);save('evaluation_lock.json',{'selection_sha256':sha(ROOT/'selection_lock.json'),'calibration_sha256':sha(ROOT/'calibration_lock.json'),'protocol_sha256':sha(ROOT/'evaluation_protocol.json'),'test_seen':False});return out

def final():
 if (ROOT/'heldout_results.json').exists():raise ValueError('Final TEST completed; no retuning')
 lock=load('selection_lock.json');cal=load('calibration_lock.json');seal=load('evaluation_lock.json')
 if sha(ROOT/'selection_lock.json')!=seal['selection_sha256'] or sha(ROOT/'calibration_lock.json')!=seal['calibration_sha256'] or sha(ROOT/'evaluation_protocol.json')!=seal['protocol_sha256']:raise ValueError('Evaluation seal changed')
 if sha(ROOT/'dataset.json')!=lock['dataset_sha256']:raise ValueError('Dataset changed')
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='test'];device=choose_device();pooler=Pooler().to(device);pooler.load_state_dict(torch.load(lock['pool_checkpoint'],map_location=device,weights_only=True)['model']);pooler.eval();models=[('m9',None),('m10',None),('m11',None)]+[(v['name'],v['selected']['checkpoint']) for v in lock['variants']];out={};start=time.perf_counter()
 for name,cp in models:
  v=vectors(rows,name,cp);report=evaluate(rows,v,pooler if name==lock['visual_variant'] else None,device);y,s=pair_values(rows,v);threshold=cal[name]['threshold'];pred=s>=threshold;report['validation_calibrated_operating_point']={'threshold':threshold,'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'false_match_rate':float(np.mean(pred[~y])),'false_nonmatch_rate':float(np.mean(~pred[y])),'roc_auc':float(roc_auc_score(y,s))};out[name]=report;print('TEST',name,'crop',report['same_team_crop']['rank1'],report['same_team_crop']['map'],'track',report['tracklets']['quality_weighted']['map'],flush=True)
 selected=out[lock['visual_variant']];baseline=out['m10'];method=lock['selected_aggregation'];beat=selected['same_team_crop']['map']>=baseline['same_team_crop']['map']+.03 and selected['tracklets'][method]['map']>=baseline['tracklets']['quality_weighted']['map']+.03 and selected['same_team_crop']['rank1']>=baseline['same_team_crop']['rank1'];result={'models':out,'m13_beats_m10_preregistered_gate':beat,'selected_model':lock['visual_variant'],'selected_aggregation':method,'test_used_for_selection':False,'fresh_test_domain':'SoccerNet single moving broadcast camera,game9; same-match GT,20eligible query identities','global_physical_identity_registry_verified':False,'wall_seconds':time.perf_counter()-start};save('heldout_results.json',result);return result
if __name__=='__main__':
 import sys
 calibrate() if '--calibrate' in sys.argv else final()
