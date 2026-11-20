"""Freeze approved short-clip human supervision and model-free source contact sheets."""
import argparse
from src.vision.football22.data import build
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--temporal-from');p.add_argument('--weak-local',action='store_true');a=p.parse_args()
    if a.temporal_from:
        from src.vision.football22.training import prepare_temporal
        result=prepare_temporal(a.temporal_from,a.output,a.weak_local)
    else:result=build(a.output)
    print('M22 verified source frames:',sum(not r.get('weak_supervision') for r in result['rows']),
          '| explicitly weak training frames:',sum(bool(r.get('weak_supervision')) for r in result['rows']))
