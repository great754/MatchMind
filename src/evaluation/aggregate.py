"""Micro aggregation excludes recordings without references from precision claims."""
import numpy as np

def aggregate_events(results):
    eligible=[r for r in results if r['reference_events']>0]
    tp=sum(r['matched'] for r in eligible);fp=sum(r['false_positives'] for r in eligible);fn=sum(r['false_negatives'] for r in eligible)
    p=tp/(tp+fp) if tp+fp else (0. if eligible else None)
    r=tp/(tp+fn) if tp+fn else None
    errors=[abs(m['timing_error_seconds']) for r in eligible for m in r.get('matching',[]) if m['match_status']=='matched']
    return dict(timing_error_seconds={k:float(fn(errors)) if errors else None for k,fn in [('mean',np.mean),('median',np.median),('p95',lambda a:np.percentile(a,95))]},recordings=len(results),evaluated_recordings=len(eligible),no_reference_recordings=len(results)-len(eligible),
        all_detections=sum(x['detections'] for x in results),reference_events=sum(x['reference_events'] for x in results),
        matched=tp,false_positives=fp if eligible else None,false_negatives=fn if eligible else None,
        precision=p,recall=r,f1=2*p*r/(p+r) if p is not None and r is not None and p+r else (0. if eligible else None),
        independent_validation=False,scope='Micro counts over reference-containing recordings; zero-reference recordings retained separately.')


def aggregate_transfer_classes(results):
    from .event_evaluation import PASS_CLASSES
    classes={}
    for label in PASS_CLASSES:
        refs=sum(r.get('by_bas_class',{}).get(label,{}).get('reference_events',0) for r in results)
        matched=[m for r in results for m in r.get('matching',[]) if m['match_status']=='matched' and m['annotation_label']==label]
        errors=[abs(m['timing_error_seconds']) for m in matched]
        classes[label]=dict(reference_events=refs,matched=len(matched),false_negatives=refs-len(matched),
            recall=len(matched)/refs if refs else None,precision=None,false_positives=None,f1=None,
            timing_error_seconds={k:float(fn(errors)) if errors else None for k,fn in [('mean',np.mean),('median',np.median),('p95',lambda a:np.percentile(a,95))]},
            reason='Observable transfer predictions do not carry BAS class labels; class-specific precision is undefined.')
    return classes
