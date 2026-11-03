"""Measured diagnostics, separated from independent metric accuracy."""
import json
from pathlib import Path
import numpy as np

def stats(v):
 a=np.asarray(v,float);return {'count':len(a),'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'max':float(np.max(a)) if len(a) else None}
def run():
 for name in ['test_match','team_goal','southampton']:
  root=Path('outputs/world_model/m8_verification')/name/'release';rows=[json.loads(l) for l in (root/'timeline.jsonl').open()];accepted=[r for r in rows if r['accepted']];durations=[];start=None
  for r in rows+[{'frame':len(rows),'accepted':False,'state':'UNKNOWN'}]:
   if r['accepted'] and r['state']=='MOTION_PROPAGATED':
    if start is None:start=r['frame']
   elif start is not None:durations.append(r['frame']-start);start=None
  report={'units':{'uncertainty':'uncalibrated metres proxy','motion':'metres of viewing-ray ground intersection, not physical camera position','conic':'source pixels','propagation':'frames'},'motion_propagation_lengths':stats(durations),'uncertainty_m_proxy':stats([r['metric']['uncertainty_pitch_m_proxy'] for r in accepted]),'uncertainty_source_pixel_budget':stats([r['metric']['uncertainty_pixel_budget'] for r in accepted]),'incremental_view_shift_m':stats([np.linalg.norm(r['metric']['incremental_view_shift_m']) for r in accepted]),'conic_heldout_p90_px':stats([r['metric']['landmark_certificate']['conic']['heldout_p90_px'] for r in accepted if (r['metric'].get('landmark_certificate') or {}).get('accepted') and 'heldout_p90_px' in r['metric']['landmark_certificate'].get('conic',{})]),'state_transitions':sum(a['state']!=b['state'] for a,b in zip(rows,rows[1:])),'accepted_to_unknown_transitions':sum(a['accepted'] and not b['accepted'] for a,b in zip(rows,rows[1:])),'reference_evaluated':False,'interpretation':'Dependent diagnostics, not calibrated accuracy or measured physical camera position.'}
  (root/'metric_statistics.json').write_text(json.dumps(report,indent=2));print(name,report)
if __name__=='__main__':run()
