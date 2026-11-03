"""Diagnostic unique-candidate center baseline from sealed TEST cache; no inference."""
from .common import ROOT,load,save
from .benchmark import gt_match

def run():
 rows=[r for r in load('dataset.json')['samples'] if r['split']=='test']
 predictions=load('test_cache/generic_tile256.json')
 observed=matched=ambiguous=0
 for row,pred in zip(rows,predictions,strict=True):
  if row['sample_id']!=pred['sample_id']:raise ValueError('TEST cache sample mismatch')
  candidates=[c for c in pred['candidates'] if c['confidence']>=load('campaign_selection.json')['selected']['threshold']]
  ambiguous+=len(candidates)>1
  if len(candidates)==1:
   observed+=1;x,y,w,h=candidates[0]['bbox'];matched+=gt_match([x+w/2,y+h/2],row['balls'])[0] is not None
 report={'frames':len(rows),'observed':observed,'matched':matched,'unmatched':observed-matched,'ambiguous':ambiguous,'coverage':observed/len(rows),'accepted_gt_consistency_precision':matched/max(1,observed),'protocol':'Exactly one candidate >=0.5, center <=12px organizer GT. No temporal state. Diagnostic computed from sealed TEST cache; no retuning.'}
 save('detector_only_center_baseline.json',report);return report
if __name__=='__main__':print(run())
