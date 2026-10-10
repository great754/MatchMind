"""Conservative longitudinal line-count fits from a trailing, identity-stable window."""
import numpy as np
from .schema import Formation
from .shape import usable

TEMPLATES={'4-3-3':(4,3,3),'4-4-2':(4,4,2),'4-2-3-1':(4,2,3,1),'3-5-2':(3,5,2)}


def estimate_formation(history,frames,direction,config,goalkeeper_known=True,fps=25) -> Formation:
    """Fit line counts to a trailing roster history or explain why evidence is unknown."""
    a=np.asarray(history,float)
    metadata=dict(window_start_frame=int(frames[0]) if len(frames) else None,window_end_frame=int(frames[-1]) if len(frames) else None,samples=len(a))
    def unknown(reason,**extra):return Formation(reason=reason,**metadata,**extra)
    if not goalkeeper_known:return unknown('goalkeeper_unknown')
    if a.ndim!=3 or a.shape[1]!=10:return unknown('incomplete_roster')
    if len(a)<2 or (frames[-1]-frames[0])/fps<config.formation_window_seconds-1/config.sample_hz-1e-8:return unknown('insufficient_history')
    valid=np.array([usable(row).all() for row in a]);a=a[valid];metadata['samples']=len(a)
    expected=max(1,round(config.formation_window_seconds*config.sample_hz))
    if len(a)<expected*config.formation_min_fraction:return unknown('insufficient_coverage')
    longitudinal=a[:,:,0]*direction
    mean=longitudinal.mean(axis=0);order=np.argsort(mean,kind='stable');sorted_mean=mean[order]
    candidates=[]
    for name,counts in TEMPLATES.items():
        boundaries=np.cumsum((0,)+counts);centers=[];squared=0.;assignment=np.empty(10,int);fits=True
        for line,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            values=sorted_mean[lo:hi];centers.append(values.mean());squared+=float(((values-values.mean())**2).sum());assignment[order[lo:hi]]=line
            if np.ptp(values)>config.formation_max_line_depth_m:fits=False
        if np.any(np.diff(centers)<config.formation_min_line_gap_m):fits=False
        if not fits:continue
        membership=[]
        for row in longitudinal:
            ranked=np.argsort(row,kind='stable');labels=np.empty(10,int)
            for line,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])):labels[ranked[lo:hi]]=line
            membership.append(np.mean(labels==assignment))
        candidates.append((np.sqrt(squared/10),name,float(np.mean(membership))))
    candidates.sort()
    if not candidates:return unknown('poor_fit')
    error,name,stability=candidates[0];margin=float(candidates[1][0]-error) if len(candidates)>1 else None
    details=dict(fit_rmse_m=float(error),fit_margin_m=margin,membership_stability=stability)
    if error>config.formation_max_rmse_m:return unknown('poor_fit',**details)
    if margin is not None and margin<config.formation_min_fit_margin_m:return unknown('ambiguous_fit',**details)
    if stability<config.formation_min_membership_stability:return unknown('unstable_membership',**details)
    return Formation(estimate=name,fit_quality='clear',**metadata,**details)
