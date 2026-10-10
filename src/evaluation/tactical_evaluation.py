"""One-at-a-time sensitivity, with baseline analytics held fixed; no optimization."""
from dataclasses import asdict,replace
import hashlib
from pathlib import Path
from collections import Counter
from src.tactics.engine import analyze_tactics

VARIANTS={
 'pressure_local_radius_m':(4.,6.),'press_persistence_seconds':(1.,3.),
 'block_low_max_m':(30.,40.),'block_high_min_m':(65.,75.),
 'formation_max_rmse_m':(3.5,5.5),'formation_persistence_seconds':(3.,7.)}


def implementation_signature():
    root=Path(__file__).resolve().parents[1]
    files=list((root/'tactics').glob('*.py'))
    files += [root/'analytics'/f'{name}.py' for name in ('config','data','motion','passes','possession','quality','shots','validation')]
    files += [root/'intelligence'/f'{name}.py' for name in ('events','labels','schema')]
    digest=hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def describe(tactics):
    states=tactics['states']
    return dict(event_counts=dict(Counter(e.type.value for e in tactics['events'])),samples=len(states),
        pressure_candidate_samples=sum(s.pressure.press_candidate for s in states),
        sustained_press_samples=sum(s.pressure.sustained_press for s in states),
        blocks={team:dict(Counter(getattr(s,team).block for s in states)) for team in ('left','right')},
        formations={team:dict(Counter(getattr(s,team).formation.estimate for s in states)) for team in ('left','right')})


def sensitivity(recording,result,analytics_config,config,enabled=True):
    baseline=analyze_tactics(recording,result,analytics_config,config)
    variants=[]
    if enabled:
        for name,values in VARIANTS.items():
            for value in values:
                variant=replace(config,**{name:value})
                variants.append(dict(parameter=name,value=value,config=asdict(variant),
                    measurements=describe(analyze_tactics(recording,result,analytics_config,variant))))
    return dict(implementation_sha256=implementation_signature(),baseline_config=asdict(config),baseline=describe(baseline),variants=variants,
        interpretation='Descriptive threshold sensitivity, not tuning, accuracy, or independent tactical validation. Synthetic known-answer tests cover geometric correctness.')
