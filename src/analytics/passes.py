"""Infer observable transfers; do not use BAS labels to create detections."""
import numpy as np
import pandas as pd

PASS_COLUMNS = ['pass_id','start_frame','end_frame','start_seconds','end_seconds',
                'sender_id','receiver_id','sender_team','receiver_team','outcome',
                'start_x_m','start_y_m','end_x_m','end_y_m','distance_m','path_distance_m',
                'duration_seconds','independent_frames','ball_speed_mean_mps',
                'ball_speed_median_mps','ball_speed_peak_mps','ball_speed_peak_kmh',
                'observed_step_fraction','termination_reason']


def detect_passes(possession, ball_motion, config):
    rows = []
    last_owner = None
    last_team = None
    last_control = None
    pending = None

    def finish(end_frame, receiver=None, receiver_team=None, reason='received'):
        if pending is None:
            return
        start = pending['start']
        end = max(start, end_frame)
        independent = max(0, end-start-1)
        if independent < config.pass_min_flight_frames:
            return
        points = ball_motion.positions[start:end+1]
        if len(points) < 2 or not np.isfinite(points[[0,-1]]).all():
            return
        straight = float(np.linalg.norm(points[-1]-points[0]))
        speed = ball_motion.smoothed_speed_mps[start+1:end+1]
        speed = speed[np.isfinite(speed)]
        steps = ball_motion.step_distance_m[start+1:end+1]
        steps = steps[np.isfinite(steps)]
        if straight < config.pass_min_distance_m or not len(speed) or speed.max() < config.pass_min_speed_mps:
            return
        if receiver == pending['sender']:
            return  # A player recovering their own loose ball is not a pass.
        outcome = ('completed' if receiver_team == pending['team'] else 'intercepted') if receiver is not None else ('out_of_play' if reason == 'out_of_play' else 'unresolved')
        rows.append(dict(pass_id=len(rows)+1, start_frame=start, end_frame=end,
                         start_seconds=start/config.fps, end_seconds=end/config.fps,
                         sender_id=pending['sender'], receiver_id=receiver,
                         sender_team=pending['team'], receiver_team=receiver_team, outcome=outcome,
                         start_x_m=float(points[0,0]), start_y_m=float(points[0,1]),
                         end_x_m=float(points[-1,0]), end_y_m=float(points[-1,1]),
                         distance_m=straight, path_distance_m=float(steps.sum()),
                         duration_seconds=(end-start)/config.fps, independent_frames=independent,
                         ball_speed_mean_mps=float(steps.mean()*config.fps) if len(steps) else None,
                         ball_speed_median_mps=float(np.median(speed)), ball_speed_peak_mps=float(speed.max()),
                         ball_speed_peak_kmh=float(speed.max()*3.6),
                         observed_step_fraction=len(steps)/max(1,end-start), termination_reason=reason))

    for r in possession.itertuples(index=False):
        frame = int(r.frame)
        if r.state in ('unknown','out_of_play'):
            if pending is not None:
                # Missing coordinates terminate at the last observed sample.
                end = frame if r.state == 'out_of_play' else frame-1
                finish(end, reason=r.state)
            pending = None
            last_owner = last_team = last_control = None
            continue
        controlled = not pd.isna(r.possessing_player_id)
        if controlled:
            pid = int(r.possessing_player_id)
            if pending is not None:
                contact = int(r.control_start_frame) if not pd.isna(r.control_start_frame) else frame
                finish(max(pending['start']+1, contact), pid, r.possessing_team)
                pending = None
            last_owner, last_team, last_control = pid, r.possessing_team, frame
        elif last_owner is not None and pending is None:
            pending = {'sender':last_owner,'team':last_team,'start':last_control}
            last_owner = last_team = last_control = None
        if pending is not None and frame-pending['start'] > config.pass_timeout_seconds*config.fps:
            finish(frame, reason='timeout')
            pending = None
    if pending is not None:
        finish(len(possession)-1, reason='end_of_recording')
    result = pd.DataFrame(rows, columns=PASS_COLUMNS)
    for field in ('sender_id','receiver_id','start_frame','end_frame','independent_frames'):
        result[field] = result[field].astype('Int64')
    return result
