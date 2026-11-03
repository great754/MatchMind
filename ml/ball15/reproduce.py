"""Recompute historical M14 metrics without altering its model or artifacts."""
import json
from pathlib import Path
from ml.ball14.evaluate import detection_metrics
from .common import save,verify

def run():
 verify();root=Path('outputs/ml/ball_m14');read=lambda n:json.loads((root/n).read_text());rows=[r for r in read('dataset.json')['samples'] if r['split']=='test'];ps=read('test_cache/generic_tile256.json');m=detection_metrics(rows,[p['candidates'] for p in ps],.5,.25);old=read('test_results.json')['detectors']['generic_tile256']['primary'];assert m==old;save('m14_reproduced.json',{'identical_to_original':True,'metrics':m,'original_artifacts_unchanged':verify()});print(m['precision'],m['recall'],m['f1'])
if __name__=='__main__':run()
