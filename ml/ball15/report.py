"""Portable measured results and figures; no post-TEST model selection."""
import os,json,shutil
from pathlib import Path
import numpy as np
from .common import ROOT,load,save,sha,verify

def run():
 os.environ.setdefault('MPLCONFIGDIR','/tmp/tactiq-m15-mpl')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 verify();dest=Path('docs/evidence');dest.mkdir(exist_ok=True);test=load('test_results.json');real=load('real_video_summary.json');demo=load('demo_sidecar_evaluation.json');det=load('detector_lock.json');pipeline=load('pipeline_lock.json');motion=load('motion_lock.json');search=load('search_lock.json');report={'milestone':'m15','status':'EXPERIMENTAL; targets not achieved; M14 remains default','dataset':{k:v for k,v in load('dataset.json').items() if k!='samples'},'manual_annotation_seal':load('demo_seal.json'),'initializer':load('initializer.json'),'detector_training':load('detector_training.json'),'selected_detector':det['selected'],'frozen_m14_validation_comparison':det['frozen_m14_validation_comparison'],'selected_pipeline':pipeline['selected'],'verification_enabled':pipeline['verification_enabled'],'best_verifier_validation':pipeline['best_candidate_verifier']['metrics'],'local_search':{k:v for k,v in search.items()},'motion_selection':{'selected':motion['selected'],'train_samples':motion['train_samples'],'validation_samples':motion['validation_samples'],'training_seconds':motion['training_seconds'],'checkpoint_sha256':motion['checkpoint_sha256']},'test':test,'real_videos':real,'sparse_demo_sidecars':demo,'ground_audit':load('ground_audit.json'),'preservation':load('review_preservation.json'),'camera_prediction_diagnostic':load('camera_motion_prediction_diagnostic.json'),'motion_region_sizes':load('motion_region_sizes.json'),'live':load('live_regression.json'),'limitations':['Original organizer ball GT lacks independent visibility/height/metric labels; manual demo audit is source-only Codex annotation, not external adjudication.','Detection evaluator inherits an unavailable-sharpness default bucket: blur strata are NOT measured; do not interpret that bucket as blur truth.','Tracking center consistency uses12px, distinct from detector IoU.25. Adjacent samples are correlated.','No reliable learned ground classifier, surveyed metric references or predicted pitch coordinates; supported metric frames0.','No learned discrete motion-state classifier; image velocity/direction-change diagnostics only.','No demonstration of improved overall detection; lower predicted-error rate comes with less observed/predicted coverage.']}
 # Keep legacy raw metrics in local artifacts, but never publish unmeasured blur strata as observations.
 for metric in list(report['test']['detectors'].values())+[c[v] for c in report['test']['manual_demo'].values() for v in ['m14','m15']]+[report['selected_detector']['metrics'],report['frozen_m14_validation_comparison']['metrics']]:
  metric['strata']={k:v for k,v in metric.get('strata',{}).items() if 'sharpness' not in k};metric['sharpness_strata_available']=False
 (dest/'cv_m15_results.json').write_text(json.dumps(report,indent=2)+'\n')
 for name,frame in [('test_match','0049'),('team_goal','0021'),('southampton','0165')]:shutil.copyfile(f'outputs/world_model/ball_m15/{name}/frame_{frame}.jpg',dest/f'm15_{name}.jpg')
 for n in ['miss_0','false_proposal_0','matched_0']:shutil.copyfile(ROOT/'failure_examples'/f'{n}.jpg',dest/f'm15_{n}.jpg')
 fig,axes=plt.subplots(1,2,figsize=(11,4));labels=['Precision','Recall','F1'];x=np.arange(3)
 for i,v in enumerate(['m14','m15']):
  values=[test['detectors'][v][k] for k in ['precision','recall','f1']];axes[0].bar(x+(i-.5)*.35,values,.35,label=v.upper());
 axes[0].set_xticks(x,labels);axes[0].set_ylim(0,1);axes[0].set_title('Same fresh200-frame TEST · IoU ≥ .25');axes[0].legend()
 names=list(real);x=np.arange(3)
 for i,v in enumerate(['m14','m15']):
  values=[real[n]['m14_observed_frames']/real[n]['frames'] if v=='m14' else real[n]['observed_coverage'] for n in names];axes[1].bar(x+(i-.5)*.35,values,.35,label=v.upper())
 axes[1].set_xticks(x,names);axes[1].set_ylim(0,1);axes[1].set_title('Full-video observed coverage · not accuracy');axes[1].legend();fig.suptitle('TactiqAI M15 experiment: detection regresses; ground reconstruction remains unvalidated');fig.tight_layout();fig.savefig(dest/'cv_m15_comparison.svg');plt.close(fig)
 print('Portable M15 results and evidence saved')
if __name__=='__main__':run()
