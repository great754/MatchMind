"""Locked, historical cross-dataset diagnostic; never used for selection."""
import json
from ml.reid.evaluate import evaluate_vectors
from .common import ROOT,M10,load,save
from .evaluate import vectors

def run():
 if not (ROOT/'heldout_results.json').exists():raise ValueError('Fresh TEST comparison must finish first')
 if (ROOT/'historical_cross_dataset.json').exists():raise ValueError('Historical diagnostic already evaluated')
 rows=[{**r,'split':'historical_test'} for r in json.loads((M10/'curated_manifest.json').read_text())['samples'] if r['split']=='test'];out={}
 for v in load('variant_selections.json'):
  z=vectors(rows,v['name'],v['selected']['checkpoint']);out[v['name']]=evaluate_vectors(rows,z);print('HISTORICAL',v['name'],out[v['name']]['same_team_map'],out[v['name']]['tracklet_map'],flush=True)
 save('historical_cross_dataset.json',{'models':out,'protocol':'unchanged original M10 crop and10–30-frame tracklets; historical only, not fresh TEST','model_selection_used_this_result':False,'original_m10_results_unchanged':True})
if __name__=='__main__':run()
