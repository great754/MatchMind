"""Portable measured tables/figures; explicit acceptance blockers and cluster uncertainty."""
import os,json,shutil
from pathlib import Path
import numpy as np
from .common import ROOT,load,sha,verify

def cluster_intervals(result,seed=20261116):
 matches=list(result['per_match']);rng=np.random.default_rng(seed);scores=[]
 for _ in range(5000):
  tp=fp=fn=0
  for match in rng.choice(matches,len(matches),replace=True):
   m=result['per_match'][match]['detection'];tp+=m['tp'];fp+=m['fp'];fn+=m['fn']
  scores.append([tp/max(1,tp+fp),tp/max(1,tp+fn),2*tp/max(1,2*tp+fp+fn)])
 a=np.asarray(scores);return {'precision':np.quantile(a[:,0],[.025,.975]).tolist(),'recall':np.quantile(a[:,1],[.025,.975]).tolist(),'f1':np.quantile(a[:,2],[.025,.975]).tolist(),'clusters':len(matches),'method':'5000whole-match bootstrap resamples, adjacent frames retained together; only2matches, exploratory intervals do not certify broad generalization'}

def run():
 verify();dest=Path('docs/evidence');dest.mkdir(exist_ok=True);test=load('test_results.json');real=load('real_video_summary.json');sel=load('selection.json');selected=test['results']['complete_m16'];intervals={v:cluster_intervals(test['results'][v]) for v in ['m14','m15','new_detector_motion','new_multi']};report={'milestone':'m16','status':'EXPERIMENTAL; defaultM14retained','date':'2026-11-03','dataset_summary':load('dataset.json')['summary'],'data_integrity':load('data_integrity.json'),'annotation_contracts':load('annotation_contract_verification.json'),'failure_taxonomy':load('failure_taxonomy.json'),'initializer':load('initializer.json'),'training_config':load('training_config.json'),'training':load('training_result.json'),'selection':sel,'temporal_visual':load('temporal_result.json'),'metric_geometry_reference':load('metric_geometry_results.json'),'metric_reference_source':load('metric_reference_download.json'),'local_recovery':load('local_recovery_audit.json'),'test':test,'exploratory_match_cluster_intervals':intervals,'real_videos':real,'source_only_demo':load('demo_reference_results.json'),'preservation':load('review_preservation.json'),'environment':load('environment.json'),'acceptance':{'detector_reference_f1_at_least85':selected['detection']['f1']>=.85,'independent_correct_visible_coverage_established':False,'independent_ground_coverage_established':False,'verified_negative_test_frames':0,'precision_at_least90':selected['detection']['precision']>=.9,'recall_at_least80':selected['detection']['recall']>=.8,'accepted_center_consistency_at_least95':selected['tracking']['accepted_center_consistency']>=.95,'prediction_inconsistency_at_most5':selected['tracking']['prediction_inconsistency_rate']<=.05,'production_promotion':False,'blockers':['Organizer visibility/occlusion/height unavailable; no verified no-visible-ball final TEST cases.','Only2fresh TEST matches; source-only40box demo audit not externally adjudicated.','No independent ground/metric references; user confirmed none available.','Learned visual verifier has no independent reviewed negative sequence calibration; diagnostic only.']}}
 for filename,key in [('live_regression.json','live'),('frame_api_timing.json','frame_api_timing'),('render_correction.json','presentation_correction'),('verification_results.json','verification')]:
  if (ROOT/filename).exists():report[key]=load(filename)
 (dest/'cv_m16_results.json').write_text(json.dumps(report,indent=2)+'\n')
 for name in ['test_match','team_goal','southampton']:
  manifest=json.loads(Path(f'outputs/world_model/ball_m16/{name}/manifest.json').read_text());examples=manifest['examples'];chosen=next((e for e in examples if e['state'] in ['DETECTED','TRACKED']),examples[0]);shutil.copyfile(chosen['path'],dest/f'm16_{name}.jpg')
 for name in ['miss','matched','unmatched']:
  path=ROOT/'failure_examples'/f'{name}_0.jpg'
  if path.exists():shutil.copyfile(path,dest/f'm16_{name}.jpg')
 os.environ.setdefault('MPLCONFIGDIR','/tmp/tactiq-m16-mpl');import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4));versions=['m14','m15','complete_m16'];x=np.arange(3)
 for i,key in enumerate(['precision','recall','f1']):axes[0].bar(x+(i-1)*.25,[test['results'][v]['detection'][key] for v in versions],.25,label=key)
 axes[0].set_xticks(x,['M14','M15','M16']);axes[0].set_ylim(0,1);axes[0].set_title(f'Identical{test["test_frames"]}organizer TEST frames');axes[0].legend()
 for i,v in enumerate(versions):
  t=test['results'][v]['tracking'];axes[1].bar(i,t['counts'].get('observed_center_matched',0)/test['test_frames'],label=v)
 axes[1].set_xticks(x,['M14','M15','M16']);axes[1].set_ylim(0,1);axes[1].set_title('Correct-reference observed coverage; visibility unverified');fig.suptitle('TactiqAI M16 · independently validated ground coordinates unavailable');fig.tight_layout();fig.savefig(dest/'cv_m16_comparison.svg');plt.close(fig);print('M16 portable results/figures saved')
if __name__=='__main__':run()
