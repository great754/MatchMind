"""Seal measured research outcomes and charts. Does not select or train models."""
import json,shutil,platform,importlib.metadata
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,M10,load,save,sha,verify

def run():
 result=load('heldout_results.json');shadow=load('shadow_results.json');lock=load('selection_lock.json');history=load('training_history.json');review=load('changed_connection_review.json')
 if not all(r.get('visually_reviewed') for r in review):raise ValueError('Changed connections still awaiting visual review')
 conclusion='M13 DOES NOT BEAT M10 UNDER THE LOCKED ACCEPTANCE GATE' if not result['m13_beats_m10_preregistered_gate'] else 'M13 BEATS M10 ON THE LOCKED BENCHMARK; INTEGRATION IS NOT YET JUSTIFIED'
 decision={'conclusion':conclusion,'selected_variant':lock['visual_variant'],'selected_aggregation':lock['selected_aggregation'],'m13_beats_m10_locked_gate':result['m13_beats_m10_preregistered_gate'],'unseen_broadcast_generalization':'Partial crop-level improvement on one unseen broadcast game; selected tracklet retrieval and actual reconnect counts do not improve consistently. No broad independent broadcast identity accuracy established.','production_integration_justified':False,'production_activated':False,'global_player_split_certification':'NOT_VERIFIED: original within-clip GT and metadata guards, not an independent worldwide physical-player registry','reason':'Keep frozen M9 serving. Selected model fails combined crop/tracklet gate; fresh TEST is only one game with20 query identities; validation only8 query identities. Better TEST alternative is not substituted post hoc. Shadow/context evidence is qualitative; physical invariants are not identity accuracy.','m8_m9_and_m10_m11_preserved':True};save('integration_decision.json',decision)
 # Quantitative ablations only; all models already locked/evaluated.
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for kind in dict.fromkeys(h['variant'] for h in history):
  h=[x for x in history if x['variant']==kind];axes[0].plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label=kind);axes[1].plot([x['epoch'] for x in h],[x['validation_score'] for x in h],label=kind)
 axes[0].set_ylabel('Train CE + metric loss');axes[1].set_ylabel('Fresh VAL selection score')
 for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'training_curves.png',dpi=140);plt.close(fig)
 names=list(result['models']);crop=[result['models'][k]['same_team_crop']['map'] for k in names];track=[result['models'][k]['tracklets']['quality_weighted']['map'] for k in names];fig,ax=plt.subplots(figsize=(11,5));y=np.arange(len(names));ax.barh(y-.18,crop,.35,label='crop mAP');ax.barh(y+.18,track,.35,label='quality-weighted tracklet mAP');ax.set_yticks(y,names);ax.set_xlim(0,1);ax.set_xlabel('Same-team mAP · identical fresh broadcast TEST samples');ax.legend();fig.tight_layout();fig.savefig(ROOT/'fresh_test_ablations.png',dpi=140);plt.close(fig)
 for filename in ['training_curves.png','fresh_test_ablations.png']:shutil.copy2(ROOT/filename,Path('docs/evidence')/('cv_m13_'+filename))
 verify();ref={}
 for source,v in shadow['sources'].items():
  ref[source]={kind:{label:sum(c.get('assessment')==label and c['decisions'][kind]=='LINKED' for c in v['references']) for label in ['APPEARS_SAME','APPEARS_DIFFERENT','INDETERMINATE']} for kind in ['m9','m10','m13_median','m13_selected_pool']}
 save('reference_summary.json',ref)
 registry={'model_name':'tactiqai-soccer-reid-v2','architecture':'OSNet x0.25 orx1.0,512D; late-block fine-tuning; validation-selected aggregation','selected':lock,'test_evaluated':True,'decision':decision,'dataset_sha256':sha(ROOT/'dataset.json'),'evaluation_sha256':sha(ROOT/'heldout_results.json'),'shadow_sha256':sha(ROOT/'shadow_results.json'),'historical_sha256':sha(ROOT/'historical_cross_dataset.json'),'current_source_sha256':{str(p):sha(p) for p in Path('ml/reid13').glob('*.py')},'research_only_source_terms':True};save('model_registry.json',registry)
 packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','timm','scikit-learn','numpy','matplotlib','huggingface-hub']};save('environment.json',{'python':platform.python_version(),'packages':packages,'device':load('visual_selection.json')['device'],'hardware':'Apple M4,16GB unified memory; existing CV runtime','peak_training_memory_measured':False})
 runtime={'training_seconds':sum(h['training_seconds'] for h in history),'validation_seconds':sum(h['validation_seconds'] for h in history),'visual_selection_wall_seconds':load('visual_selection.json')['wall_seconds'],'fresh_test_seconds':result['wall_seconds'],'shadow_seconds':shadow['wall_seconds'],'epochs':dict(Counter(h['variant'] for h in history)),'checkpoint_bytes':sum(p.stat().st_size for p in ROOT.rglob('*.pth')),'artifact_logical_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())};save('runtime.json',runtime)
 summary={'decision':decision,'dataset':load('dataset_summary.json'),'selection':lock,'source_audit':load('source_audit.json'),'fresh_test':result,'historical_cross_dataset':load('historical_cross_dataset.json'),'shadow':shadow,'reference_summary':ref,'changed_connection_review':review,'runtime':runtime,'baseline_preservation':load('baseline_preservation.json'),'environment':load('environment.json'),'verification':load('verification_results.json') if (ROOT/'verification_results.json').exists() else {'status':'NOT_RUN_IN_THIS_REPLAY'}};save('summary.json',summary);Path('docs/evidence/cv_m13_results.json').write_text(json.dumps(summary,indent=2)+'\n')
 directory=ROOT/'verification';directory.mkdir(exist_ok=True)
 for p in Path('/tmp').glob('tactiq-m1[23]-*.log'):shutil.copy2(p,directory/p.name)
 executed=ROOT/'source_snapshot';executed.mkdir(exist_ok=True)
 for p in Path('ml/reid13').glob('*.py'):shutil.copy2(p,executed/p.name)
 return summary
if __name__=='__main__':run()
