"""Goal-directed ball-flight candidates and separately annotated shot windows."""
import numpy as np
import pandas as pd

from src.analytics.motion import segments
from src.analytics.shot_evidence import trace_shot_frames

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


def detect_shots(possession, ball_motion, config, options=None):
    evidence=trace_shot_frames(possession,ball_motion,config,options)
    flags=np.array([e['eligible'] for e in evidence],dtype=bool)
    targets=np.array([e['goal_x'] if e['eligible'] else np.nan for e in evidence])
    projected_y=np.array([e['projected_goal_crossing_y'] if e['eligible'] else np.nan for e in evidence])
    contexts=[(e['recent_possessing_player'],e['recent_possessing_team']) for e in evidence]
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
