"""Measured camera outcomes. Failure labels describe gates, never guessed causes."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .pose import residual_gate,PoseFitter
from .semantic_pitch import GEOMETRY

def absolute_outcome(fit, diagnostic, classes=()):
    if fit is not None:return 'DIRECT_ACCEPTED'
    if diagnostic.get('outcome') in {'POSE_OPTIMIZATION_FAILED','POSE_RESIDUAL_TOO_HIGH','POSE_HYPOTHESES_DISAGREE','INVALID_VISIBLE_REGION','INSUFFICIENT_SEMANTIC_CLASSES'}:return diagnostic['outcome']
    reason=diagnostic.get('reason','')
    if 'Insufficient distinct semantic' in reason or reason=='Insufficient semantic evidence':
        return 'INSUFFICIENT_SEMANTIC_CLASSES'
    if 'heldout_mean_px' in diagnostic:
        means=diagnostic['heldout_mean_px'];p90=diagnostic['heldout_p90_px']
        good=sum(residual_gate({k:v[i] for k,v in means.items()},{k:v[i] for k,v in p90.items()}) for i in range(len(means['lines'])))
        if good<2:return 'POSE_RESIDUAL_TOO_HIGH'
    if 'varies between' in reason:return 'POSE_HYPOTHESES_DISAGREE'
    if 'Invalid projected' in reason:return 'INVALID_VISIBLE_REGION'
    if not classes and not diagnostic:return 'NO_PITCH_EVIDENCE'
    return 'UNKNOWN_FAILURE'

def runs(rows):
    result=[]
    for row in rows:
        valid=row['available']
        if not result or result[-1]['available']!=valid:result.append({'available':valid,'start_frame':row['frame'],'end_frame':row['frame']})
        else:result[-1]['end_frame']=row['frame']
    return result

def report(directory, poses, output):
    root=Path(directory);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    fits=json.loads((Path(poses)/'pose_fits.json').read_text())['frames'];manifest=json.loads((root/'manifest.json').read_text());fps=manifest['fps'];rows=[]
    cache=np.load(Path(poses)/'semantic_masks.npz',allow_pickle=False)
    masks=cache['mask'];confidence=cache['confidence'];cache.close()
    with (root/'frames.jsonl').open() as source:
        for index,line in enumerate(source):
            c=json.loads(line)['camera'];pose=fits[index];motion=c['motion_evidence'];evidence=c['pitch_evidence']
            classes=sorted(set([l.get('kind','unidentified') for l in evidence.get('lines',[])]+[l['kind'] for l in evidence.get('curves',[])]))
            solver_classes=pose['diagnostic'].get('source_classes')
            if solver_classes is None:solver_classes=list(PoseFitter.groups(masks[index],confidence[index]))
            absolute=absolute_outcome(pose['fit'],pose['diagnostic'],solver_classes)
            available=c['calibration_valid'];checks=[]
            if not pose['fit']:
                age=motion.get('propagation_age_seconds');matches=motion.get('feature_matches',0);inliers=motion.get('affine_inliers',0);error=motion.get('affine_median_error_small_px')
                if age is None:checks.append('NO_ACTIVE_TRACKING_ANCHOR')
                else:
                    if not 0<age<=.25:checks.append('PROPAGATION_EXPIRED')
                    if c['continuity']!='continuous' or motion.get('image_affine') is None:checks.append('TEMPORAL_TRACKING_FAILED')
                    if matches<50:checks.append('INSUFFICIENT_FEATURE_MATCHES')
                    if inliers/max(matches,1)<.8:checks.append('INSUFFICIENT_MOTION_INLIERS')
                    if error is None or error>1:checks.append('MOTION_RESIDUAL_TOO_HIGH')
                    agreement=motion.get('propagated_line_agreement_px')
                    if 'propagated_line_agreement_px' in motion:
                        if agreement is None:checks.append('INSUFFICIENT_GEOMETRIC_CONSTRAINTS')
                        elif agreement['mean']>12 or agreement['p90']>24:checks.append('TRACKED_PITCH_RESIDUAL_TOO_HIGH')
            if c['continuity']=='cut':checks.append('CAMERA_CUT')
            loc=motion.get('localization',{})
            rows.append({'frame':index,'timestamp':index/fps,'available':available,'source':c['calibration_source_kind'] if available else 'unknown',
                'absolute_outcome':absolute,'solver_attempted':pose['diagnostic'].get('solver_attempted','heldout_mean_px' in pose['diagnostic']),'semantic_classes':classes,
                'solver_classes':solver_classes,'constraint_axes':sorted({GEOMETRY[n][0] for n in solver_classes if n in GEOMETRY}),
                'absolute_diagnostic':pose['diagnostic'],'tracking_blockers':loc.get('tracking_blockers',checks),
                'outcome':loc.get('outcome','DIRECT_ACCEPTED' if pose['fit'] else 'PROPAGATED_ACCEPTED' if available else absolute),
                'flow':{k:v for k,v in motion.items() if k not in ('pose','localization')},'transition':loc.get('transition')})
    sequences=runs(rows)
    for r in sequences:r['frames']=r['end_frame']-r['start_frame']+1;r['duration_seconds']=r['frames']/fps
    counts=Counter(r['absolute_outcome'] for r in rows);outcomes=Counter(r['outcome'] for r in rows)
    summary={'frames':len(rows),'fps':fps,'available_frames':sum(r['available'] for r in rows),'unavailable_frames':sum(not r['available'] for r in rows),'coverage_percent':100*sum(r['available'] for r in rows)/len(rows),
        'absolute_outcomes':{k:{'count':v,'percent_all_frames':100*v/len(rows)} for k,v in counts.items()},'final_outcomes':dict(outcomes),'final_outcome_percent_all_frames':{k:100*v/len(rows) for k,v in outcomes.items()},
        'tracking_blocker_percent_all_frames_overlapping':{k:100*v/len(rows) for k,v in Counter(b for r in rows if not r['available'] for b in r['tracking_blockers']).items()},
        'tracking_blockers_overlapping':dict(Counter(b for r in rows if not r['available'] for b in r['tracking_blockers'])),
        'solver_attempts':sum(r['solver_attempted'] for r in rows),'acquisitions':sum(r['available'] for r in sequences),'losses':sum(not r['available'] for r in sequences[1:]),
        'direct_fit_runs':sum(r['source']=='direct_calibrated' and (i==0 or rows[i-1]['source']!='direct_calibrated') for i,r in enumerate(rows)),
        'direct_frames':sum(r['source']=='direct_calibrated' for r in rows),'tracked_frames':sum(r['source']=='camera_propagated' for r in rows),
        'camera_cuts':sum('CAMERA_CUT' in r['tracking_blockers'] for r in rows),
        'semantic_class_count_distribution':dict(Counter(len(r['solver_classes']) for r in rows)),
        'transitions':dict(Counter(r['transition'] for r in rows if r['transition'])),
        'longest_localized':max((r for r in sequences if r['available']),key=lambda r:r['frames'],default=None),
        'longest_unlocalized':max((r for r in sequences if not r['available']),key=lambda r:r['frames'],default=None),
        'sequences':sequences,'independent_accuracy':None,'notes':['Cached original optimizer gates; source video/reference not used to fabricate rejection causes.','Tracking blockers overlap; absolute outcomes are mutually exclusive.','Unrecorded internal optimizer failures cannot be reconstructed. No out-of-distribution classification is inferred.']}
    (out/'frame_outcomes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    colors={'DIRECT_ACCEPTED':'#76d4a6','PROPAGATED_ACCEPTED':'#75b8ed','TEMPORALLY_TRACKED':'#75b8ed'}
    bars=''.join(f'<rect x="{r["frame"]}" y="20" width="1" height="50" fill="{colors.get(r["outcome"],"#755568")}"><title>Frame {r["frame"]}: {r["outcome"]}; {r["absolute_outcome"]}; {", ".join(r["tracking_blockers"])}</title></rect>' for r in rows)
    (out/'timeline.html').write_text('<!doctype html><meta charset="utf-8"><title>TactiqAI Camera Diagnosis</title><body style="background:#102423;color:#eee;font:16px system-ui"><h1>TactiqAI · Camera diagnosis</h1><p>Green direct; blue tracked; purple unavailable. Hover a frame for measured gates. Coverage is not accuracy.</p><svg width="100%" viewBox="0 0 '+str(len(rows))+' 90">'+bars+'</svg><pre>'+json.dumps(summary,indent=2)+'</pre>')
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(report(a.directory,a.poses,a.output),indent=2))
