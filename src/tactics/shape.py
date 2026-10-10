"""Pitch geometry with explicit goalkeeper and observation policies."""
import numpy as np
from .schema import Shape,BallRelative


def usable(points):
    """Mask finite positions on the closed field rectangle."""
    a=np.asarray(points,float).reshape(-1,2)
    return np.isfinite(a).all(axis=1)&(a[:,0]>=0)&(a[:,0]<=105)&(a[:,1]>=0)&(a[:,1]<=68)


def team_shape(points,direction,goalkeeper_mask=None,goalkeeper_source='unavailable',min_players=8,include_goalkeeper=False) -> Shape:
    """Summarize observed outfield geometry, withholding unknown goalkeeper cases."""
    a=np.asarray(points,float).reshape(-1,2)
    keep=usable(a)
    known=goalkeeper_mask is not None and bool(np.asarray(goalkeeper_mask).any())
    if known and not include_goalkeeper: keep &= ~np.asarray(goalkeeper_mask,bool)
    selected=a[keep];n=len(selected)
    base=dict(available_players=n,goalkeeper_excluded=known and not include_goalkeeper,goalkeeper_source=goalkeeper_source)
    if not known and not include_goalkeeper: return Shape(**base,reason='Goalkeeper identity unavailable; outfield shape withheld.')
    if n<min_players or n>(11 if include_goalkeeper else 10): return Shape(**base,reason='Insufficient or ambiguous active outfield roster.')
    center=selected.mean(axis=0);radius=np.linalg.norm(selected-center,axis=1)
    distances=np.linalg.norm(selected[:,None]-selected[None,:],axis=2)
    spread=float(distances[np.triu_indices(n,1)].mean()) if n>1 else 0.
    return Shape(**base,centroid_x_m=float(center[0]),centroid_y_m=float(center[1]),width_m=float(np.ptp(selected[:,1])),
        depth_m=float(np.ptp(selected[:,0])),compactness_m=float(np.sqrt(np.mean(radius**2))),spread_m=spread,
        longitudinal_m=float(center[0] if direction==1 else 105-center[0]),lateral_m=float(center[1]))


def block_candidate(shape: Shape,phase: str,low: float=35.,high: float=70.) -> str:
    """Return a geometric defending band; persistence is applied by the engine."""
    if phase!='defending' or shape.longitudinal_m is None: return 'unknown'
    return 'low' if shape.longitudinal_m<=low else 'high' if shape.longitudinal_m>=high else 'mid'


def players_relative_to_ball(attackers,defenders,ball,attack_direction,tolerance=.5) -> BallRelative:
    """Count observed outfield players using the attacking direction as the sign."""
    a=np.asarray(attackers,float).reshape(-1,2);d=np.asarray(defenders,float).reshape(-1,2)
    a=a[usable(a)];d=d[usable(d)]
    if not usable(np.asarray(ball).reshape(1,2))[0]: return BallRelative(attacking_available=len(a),defending_available=len(d))
    signed=(a[:,0]-ball[0])*attack_direction
    goal_side=(d[:,0]-ball[0])*attack_direction>tolerance
    count=int(goal_side.sum())
    return BallRelative(attacking_ahead=int((signed>tolerance).sum()),attacking_behind=int((signed< -tolerance).sum()),
        attacking_level=int((np.abs(signed)<=tolerance).sum()),defenders_goal_side=count,defenders_behind_ball_to_own_goal=count,
        attacking_available=len(a),defending_available=len(d))
