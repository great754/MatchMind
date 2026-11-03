"""Measured failure evidence and portable M11 results; no model fitting."""
import json,shutil,math,pickle
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,M10,load,save,sha,verify
from .evaluate import osnet_vectors,pooled,paired

def run():
 result=load('heldout_results.json');shadow=load('shadow_results.json');lock=load('selection_lock.json');data=load('dataset.json');rows=[r for r in data['samples'] if r['split']=='test'];ts=data['tracklets']['test'];by={r['sample_id']:r for r in rows};mv,ids=osnet_vectors('test','m10');m10=pooled(ts,rows,mv);temporal=np.load(ROOT/'test_temporal.npy');qi,gi=paired(ts);scores={'m10':m10[qi]@m10[gi].T,'m11':temporal[qi]@temporal[gi].T};failures=[]
 for i,a in enumerate(qi):
  eligible=[j for j,b in enumerate(gi) if ts[a]['match_id']==ts[b]['match_id'] and ts[a]['team_identity']==ts[b]['team_identity']];rank={k:sorted(eligible,key=lambda j:-v[i,j]) for k,v in scores.items()};best={k:ts[gi[v[0]]]['player_identity']==ts[a]['player_identity'] for k,v in rank.items()}
  if not best['m11']:failures.append({'query_tracklet':ts[a],'m10_correct_rank1':best['m10'],'m11_correct_rank1':False,'candidates':{k:[{'tracklet':ts[gi[j]],'similarity':float(scores[k][i,j]),'same_annotated_identity':ts[gi[j]]['player_identity']==ts[a]['player_identity']} for j in order[:3]] for k,order in rank.items()}})
 failures.sort(key=lambda x:not x['m10_correct_rank1']);save('failure_analysis.json',{'m11_same_team_tracklet_rank1_failures':len(failures),'m10_success_m11_failure':sum(x['m10_correct_rank1'] for x in failures),'cases':failures,'interpretation':'annotation-defined identity, not independently human-verified physical identity; similar kits, small crops and viewpoint/lighting changes remain; no definite label error inferred from embedding scores'})
 def tile(tracklet,title):
  r=max([by[k] for k in tracklet['samples']],key=lambda r:r['quality']['score']);im=cv2.imread(r['crop_path']);canvas=np.full((220,165,3),245,np.uint8);h,w=im.shape[:2];scale=min(140/w,175/h);im=cv2.resize(im,(round(w*scale),round(h*scale)));y=33;x=(165-im.shape[1])//2;canvas[y:y+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,title[:24],(4,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1);cv2.putText(canvas,tracklet['player_identity'][-20:],(4,29),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1);return canvas
 for page in range(min(2,math.ceil(len(failures)/4))):
  panels=[]
  for failure in failures[page*4:page*4+4]:
   a=tile(failure['query_tracklet'],'QUERY');cells=[a]
   for kind in ['m10','m11']:
    for c in failure['candidates'][kind]:cells.append(tile(c['tracklet'],f"{kind} {'same' if c['same_annotated_identity'] else 'DIFF'} {c['similarity']:.3f}"))
   panels.append(np.hstack(cells))
  if panels:cv2.imwrite(str(ROOT/f'failure_examples_{page}.jpg'),np.vstack(panels))
 history=load('training_history.json');fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,stage in zip(axes,['visual','temporal']):
  h=[r for r in history if r['stage']==stage];ax.plot([r['epoch'] for r in h],[r['loss'] for r in h],label='train CE + triplet');ax.set_xlabel('Epoch');ax.set_ylabel('Training loss');ax.set_title(stage);ay=ax.twinx();ay.plot([r['epoch'] for r in h],[r['validation_score'] for r in h],color='orange',label='validation selection metric');ay.set_ylabel('Validation metric');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'training_curves.png',dpi=140);plt.close(fig)
 names=list(result['models']);values=[result['models'][k]['matched_8_12frame_tracklets']['same_team']['map'] for k in names];fig,ax=plt.subplots(figsize=(11,4));ax.barh(names,values);ax.set_xlim(0,1);ax.set_xlabel('Same-team mAP · identical 8–12-frame tracklets');ax.set_title('M11 does not beat M10');fig.tight_layout();fig.savefig(ROOT/'heldout_ablation.png',dpi=140);plt.close(fig)
 attention=load('test_attention.json');q=[];w=[]
 for t,weights in zip(attention['tracklets'],attention['weights']):
  q.extend([by[k]['quality']['score'] for k in t['samples']]);w.extend(weights)
 save('attention_diagnostic.json',{'frames':len(q),'quality_weight_correlation':float(np.corrcoef(q,w)[0,1]),'weight_min':min(w),'weight_max':max(w),'learned_quality_prior':True,'pose_view_labels_available':False})
 # Reference evidence remains explicitly qualitative.
 references={}
 for source,values in shadow['sources'].items():
  references[source]={}
  for kind in ['m9','m10','vit_pretrained','vit_finetuned','temporal','temporal_scorer']:
   references[source][kind]={assessment:sum(c.get('assessment')==assessment and c['decisions'][kind]=='LINKED' for c in values['references']) for assessment in ['APPEARS_SAME','APPEARS_DIFFERENT','INDETERMINATE']}
 save('reference_summary.json',references)
 conclusion='M11 DOES NOT BEAT M10' if not result['representation_beats_m10'] else 'M11 IMPROVES RESEARCH RESULTS BUT INTEGRATION IS NOT YET JUSTIFIED';decision={'conclusion':conclusion,'production_activated':False,'representation_beats_m10':result['representation_beats_m10'],'reason':'Fine-tuned ViT improves its own pretrained baseline, but temporal attention and learned scorer fail to beat the frozen M10 representation. Shadow learned-scorer reconnects also regress. No independent identity accuracy or valid broadcast camera-uncertainty supervision established. Keep M9 serving, preserve M10 research baseline.','no_test_tuning':True};save('integration_decision.json',decision)
 registry={**lock,'model_name':'tactiqai-temporal-soccer-reid-m11-v1','architecture':'DINO ViT-S/16 last-block fine-tuning + 1-layer quality-biased temporal Transformer + L2 logistic reassociation','initializer':load('initializer.json'),'test_evaluated':True,'production_activated':False,'conclusion':conclusion,'heldout_sha256':sha(ROOT/'heldout_results.json'),'shadow_sha256':sha(ROOT/'shadow_results.json'),'current_code_sha256':{str(p):sha(p) for p in Path('ml/reid11').glob('*.py')}};save('model_registry.json',registry);verify()
 portable={'decision':decision,'dataset':load('dataset_audit.json'),'selection':lock,'metrics':result,'shadow':shadow,'references':references,'runtime':{'training_seconds':sum(h['training_seconds'] for h in history),'selection_wall_seconds':lock['model_selection_seconds'],'prefix_seconds':sum(load(f'{s}_prefix.json')['seconds'] for s in ['train','validation','test']),'test_seconds':result['wall_seconds'],'shadow_seconds':shadow['wall_seconds'],'device':lock['device']},'attention':load('attention_diagnostic.json'),'baseline_preservation':load('baseline_preservation.json')};Path('docs/evidence/cv_m11_results.json').write_text(json.dumps(portable,indent=2)+'\n');save('summary.json',portable);return portable
if __name__=='__main__':run()
