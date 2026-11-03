"""One-time held-out evaluation after immutable validation selection."""
import json,time
from pathlib import Path
from .protocol import ROOT,integration_gate
from .evaluate import evaluate_split
from src.vision.world.player_appearance import digest

def run():
 output=ROOT/'heldout_comparison.json'
 if output.exists():raise ValueError('Held-out evaluation already completed; do not use TEST for another selection')
 lock=json.loads((ROOT/'selection_lock.json').read_text());checkpoint=lock['selected_checkpoint']
 for path,expected in [(checkpoint,lock['checkpoint_sha256']),(ROOT/'curated_manifest.json',lock['training_manifest_sha256']),(ROOT/'training_config.json',lock['training_configuration_sha256'])]:
  if digest(path)!=expected:raise ValueError('Selection/data/config changed after lock')
 start=time.perf_counter();generic=evaluate_split('test',strategy='m9_body',output=ROOT/'generic_test_m9_body.json');matched=generic if lock['strategy']=='m9_body' else evaluate_split('test',strategy=lock['strategy'],output=ROOT/'generic_test_matching_crop.json');soccer=evaluate_split('test',checkpoint=checkpoint,strategy=lock['strategy'],output=ROOT/'soccer_test.json');gate_m9=integration_gate(generic,soccer);gate_matched=integration_gate(matched,soccer);passed=gate_m9['passed'] and gate_matched['passed'];report={'generic_m9_body':generic,'generic_matching_crop':matched,'soccer_reid_v1':soccer,'gate_against_m9_body':gate_m9,'gate_against_matching_crop':gate_matched,'representation_gate_passed':passed,'selection_lock_sha256':digest(ROOT/'selection_lock.json'),'integration_gate_sha256':digest('configs/ml/soccer-reid-v1-gate.json'),'wall_seconds':time.perf_counter()-start,'test_evaluation_runs':1,'threshold_retuned':False,'test_used_for_selection':False,'integration_status':'NOT_INTEGRATED; pending world-model/reference safety ablation' if passed else 'NOT_INTEGRATED; held-out integration gate failed'};output.write_text(json.dumps(report,indent=2));registry={**lock,'model_checkpoint_sha256':digest(checkpoint),'evaluation_file':str(output),'evaluation_sha256':digest(output),'status':report['integration_status'],'generic_initializer_untouched':True};(ROOT/'model_registry.json').write_text(json.dumps(registry,indent=2));print('HELDOUT_COMPARISON',{k:{m:v[m] for m in ['rank1','map','same_team_rank1','same_team_map','tracklet_map']} for k,v in [('generic_m9',generic),('generic_same_crop',matched),('soccer',soccer)]},flush=True);print('REPRESENTATION_GATE',passed,gate_m9,gate_matched,flush=True)
 return report
if __name__=='__main__':run()
