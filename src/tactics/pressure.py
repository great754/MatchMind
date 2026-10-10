"""Nearby opponents are local pressure; stronger labels need closing and persistence."""
import numpy as np
from .schema import Pressure
from .shape import usable


def causal_velocities(positions,frame,fps,lookback_seconds,max_speed):
    """Backward displacement only; do not bridge missing samples or use centered filters."""
    steps=max(1,round(lookback_seconds*fps));start=frame-steps
    result=np.full_like(positions[frame],np.nan,dtype=float)
    if start<0: return result
    history=positions[start:frame+1]
    valid=np.isfinite(history).all(axis=(0,2))
    values=(positions[frame]-positions[start])/(steps/fps)
    valid &= np.linalg.norm(values,axis=1)<=max_speed
    result[valid]=values[valid]
    return result


def local_pressure(carrier,carrier_velocity,defenders,defender_velocities,teammates,config,**identity) -> Pressure:
    """Separate observed proximity from an instantaneous spatial/closing candidate."""
    d=np.asarray(defenders,float).reshape(-1,2);dv=np.asarray(defender_velocities,float).reshape(-1,2)
    keep=usable(d);d=d[keep];dv=dv[keep]
    friendly=np.asarray(teammates,float).reshape(-1,2);friendly=friendly[usable(friendly)]
    if not usable(np.asarray(carrier).reshape(1,2))[0]: return Pressure(**identity,reason='missing_carrier_or_ball')
    vectors=d-carrier;distances=np.linalg.norm(vectors,axis=1)
    near=int((distances<=config.pressure_near_radius_m).sum());local=int((distances<=config.pressure_local_radius_m).sum());outer=int((distances<=config.pressure_outer_radius_m).sum())
    friendly_count=int((np.linalg.norm(friendly-carrier,axis=1)<=config.pressure_local_radius_m).sum())
    velocity_valid=np.isfinite(dv).all(axis=1)&np.isfinite(carrier_velocity).all()&(distances>1e-6)
    closing=np.full(len(d),np.nan)
    closing[velocity_valid]=-np.sum((dv[velocity_valid]-carrier_velocity)*vectors[velocity_valid]/distances[velocity_valid,None],axis=1)
    count=int(((closing>=config.closing_min_mps)&(distances<=config.pressure_outer_radius_m)).sum()) if velocity_valid.any() else None
    enough=len(d)>=config.pressure_min_observed_defenders
    candidate=enough and local>=config.pressure_min_defenders and (count or 0)>=1
    return Pressure(**identity,near_radius_m=config.pressure_near_radius_m,local_radius_m=config.pressure_local_radius_m,outer_radius_m=config.pressure_outer_radius_m,available_defenders=len(d),nearest_defender_distance_m=float(distances.min()) if len(d) else None,
        defenders_within_near=near,defenders_within_local=local,defenders_within_outer=outer,closing_defenders=count,
        peak_closing_speed_mps=float(np.nanmax(closing)) if velocity_valid.any() else None,
        teammates_within_local_including_carrier=friendly_count,local_numerical_balance=friendly_count-local,
        local_pressure='multiple_nearby' if local>=config.pressure_min_defenders else 'nearby' if outer else 'low',press_candidate=candidate,
        reason='incomplete_defenders' if not enough else 'closing_unavailable' if count is None else None)
