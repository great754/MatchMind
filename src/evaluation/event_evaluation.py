"""One-to-one matching shared with analytics; absolute timing errors in seconds."""
import numpy as np
from src.analytics.validation import validate_events

PASS_CLASSES=('PASS','HIGH PASS','CROSS','THROW IN')


def evaluate_events(predictions,references,fps,tolerance_seconds=1.,actor_field=None):
    summary,matches=validate_events(predictions,references,fps,tolerance_seconds,actor_field)
    errors=matches.loc[matches.match_status=='matched','timing_error_seconds'].abs().dropna()
    summary['timing_error_seconds']={name:float(fn(errors)) if len(errors) else None for name,fn in
        [('mean',np.mean),('median',np.median),('p95',lambda a:np.percentile(a,95))]}
    summary['matching']=matches.where(matches.notna(),None).astype(object).where(matches.notna(),None).to_dict('records')
    return summary


def evaluate_transfers(predictions,references,fps,tolerance_seconds=1.):
    refs=[r for r in references if r['label'] in PASS_CLASSES]
    result=evaluate_events(predictions,refs,fps,tolerance_seconds,'sender_id')
    # Class attribution uses a single global assignment, never reuses a prediction.
    result['by_bas_class']={}
    for label in PASS_CLASSES:
        count=sum(r['label']==label for r in refs)
        matched=[m for m in result['matching'] if m['match_status']=='matched' and m['annotation_label']==label]
        result['by_bas_class'][label]=dict(reference_events=count,matched=len(matched),false_negatives=count-len(matched),
            recall=len(matched)/count if count else None,precision=None,false_positives=None,f1=None,
            reason='Predictions are observable transfers without BAS class labels; class-specific precision/FP/F1 are not identifiable.',
            timing_error_seconds={key:float(fn([abs(m['timing_error_seconds']) for m in matched])) if matched else None for key,fn in [('mean',np.mean),('median',np.median),('p95',lambda a:np.percentile(a,95))]})
    result['observation_kind']='observable_transfer versus annotated pass-like reference'
    return result
