"""Goal-directed ball-flight candidates and separately annotated shot windows."""
import numpy as np
import pandas as pd

from src.analytics.motion import segments

SHOT_COLUMNS = ['shot_id','start_frame','end_frame','start_seconds','end_seconds',
                'shooter_id','team','goal_x_m','start_x_m','start_y_m',
                'projected_goal_y_m','speed_peak_mps','speed_median_mps','speed_peak_kmh',
                'valid_speed_samples','window_coverage_fraction','window_requested_frames','window_truncated','source']


def speed_window(ball_motion, start, end, config):
    # end is inclusive; truncate at the first unavailable position after release.
    requested_end = min(end, len(ball_motion.positions)-1)
    requested_frames = max(1, requested_end-start+1)
    end = requested_end
    for n in range(start, end+1):
        if not ball_motion.valid[n]:
            end = n-1
            break
    samples = ball_motion.smoothed_speed_mps[start:end+1]
    samples = samples[np.isfinite(samples)]
    return dict(end_frame=max(start,end), speed_peak_mps=float(samples.max()) if len(samples) else None,
                speed_median_mps=float(np.median(samples)) if len(samples) else None,
                speed_peak_kmh=float(samples.max()*3.6) if len(samples) else None,
                valid_speed_samples=len(samples), window_coverage_fraction=len(samples)/requested_frames,
                window_requested_frames=requested_frames, window_truncated=end < requested_end)


def detect_shots(possession, ball_motion, config):
    n = len(ball_motion.positions)
    flags = np.zeros(n, dtype=bool)
    targets = np.full(n, np.nan)
    projected_y = np.full(n, np.nan)
    contexts = []
    last_actor = None
    last_team = None
    last_frame = -100000
    for r in possession.itertuples(index=False):
        i = int(r.frame)
        if r.state in ('unknown','out_of_play'):
            last_actor = last_team = None
        if not pd.isna(r.possessing_player_id):
            last_actor, last_team, last_frame = int(r.possessing_player_id), r.possessing_team, i
        actor = last_actor if i-last_frame <= config.fps else None
        team = last_team if actor is not None else None
        contexts.append((actor,team))
        p = ball_motion.positions[i]
        v = ball_motion.velocity[i]
        speed = ball_motion.smoothed_speed_mps[i]
        if r.state not in ('free','candidate','contested') or not np.isfinite([*p,*v,speed]).all() or speed < config.shot_min_speed_mps:
            continue
        direction = 1 if v[0]>0 else -1
        if team is not None and config.attack_direction(team) != direction:
            continue
        goal_x = 105.0 if direction == 1 else 0.0
        dx = (goal_x-p[0])*direction
        if not 0 < dx <= config.shot_max_goal_distance_m or abs(v[0])/speed < 0.6:
            continue
        t = (goal_x-p[0])/v[0]
        goal_y = p[1]+t*v[1]
        if 0 < t <= config.shot_max_time_to_goal_seconds and abs(goal_y-34) <= 3.66+config.shot_goal_margin_m:
            flags[i] = True
            targets[i] = goal_x
            projected_y[i] = goal_y
    rows = []
    last_end = -1
    for start,end in segments(flags):
        if end-start < config.shot_persistence_frames or start <= last_end:
            continue
        window = speed_window(ball_motion, start, start+round(config.shot_window_seconds*config.fps), config)
        # Stop at a controlled reception, rather than including a subsequent kick.
        controlled = np.flatnonzero(possession.state.iloc[start:window['end_frame']+1].to_numpy() == 'controlled')
        if len(controlled):
            window = speed_window(ball_motion, start, start+int(controlled[0])-1, config)
        last_end = window['end_frame']
        actor,team = contexts[start]
        rows.append(dict(shot_id=len(rows)+1, start_frame=int(start), end_frame=window['end_frame'],
                         start_seconds=start/config.fps, end_seconds=window['end_frame']/config.fps,
                         shooter_id=actor, team=team, goal_x_m=float(targets[start]),
                         start_x_m=float(ball_motion.positions[start,0]), start_y_m=float(ball_motion.positions[start,1]),
                         projected_goal_y_m=float(projected_y[start]), source='trajectory_candidate',
                         **{k:v for k,v in window.items() if k != 'end_frame'}))
    result = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    result['shooter_id'] = result['shooter_id'].astype('Int64')
    return result


def annotated_shot_windows(events, ball_motion, config):
    """Measure known BAS shots regardless of whether the detector found them."""
    rows = []
    for e in events:
        if e['label'].upper() != 'SHOT':
            continue
        start = e['frame']
        # The release sample itself may still contain pre-shot motion. Include a
        # short 1.2s window, never reaching across unavailable position samples.
        window = speed_window(ball_motion,start,start+round(config.shot_window_seconds*config.fps),config)
        p = ball_motion.positions[start]
        rows.append(dict(annotation_id=e['annotation_id'], frame=start, seconds=start/config.fps,
                         player_id=e['player_id'], team=e['team'], x_m=float(p[0]) if np.isfinite(p[0]) else None,
                         y_m=float(p[1]) if np.isfinite(p[1]) else None, source='BAS_reference_window', **window))
    return pd.DataFrame(rows, columns=['annotation_id','frame','seconds','player_id','team','x_m','y_m',
                                      'source','end_frame','speed_peak_mps','speed_median_mps',
                                      'speed_peak_kmh','valid_speed_samples','window_coverage_fraction','window_requested_frames','window_truncated'])
