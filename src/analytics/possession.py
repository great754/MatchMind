"""Conservative ownership from proximity, relative motion and persistence."""
import numpy as np
import pandas as pd


def detect_possession(player_positions, player_velocities, player_ids, teams, ball_motion, config):
    positions = np.asarray(player_positions)
    velocities = np.asarray(player_velocities)
    count = len(ball_motion.positions)
    if positions.shape != velocities.shape or positions.shape != (count, len(player_ids), 2):
        raise ValueError('Expected player arrays shaped (frames, players, 2)')
    if len(teams) != len(player_ids):
        raise ValueError('Each player must have a team')
    distances = np.linalg.norm(positions-ball_motion.positions[:, None, :], axis=2)
    relative = np.linalg.norm(velocities-ball_motion.velocity[:, None, :], axis=2)
    rows = []
    owner = None
    candidate = None
    streak = 0
    acquired_at = None
    for frame in range(count):
        bp = ball_motion.positions[frame]
        row = {'frame':frame, 'possessing_player_id':None, 'possessing_team':None,
               'state':'unknown', 'candidate_player_id':None, 'nearest_distance_m':None,
               'relative_speed_mps':None, 'control_confidence':0.0,
               'control_start_frame':None}
        finite = np.flatnonzero(np.isfinite(distances[frame]))
        if not np.isfinite(bp).all() or not len(finite):
            owner = candidate = None
            streak = 0
            rows.append(row)
            continue
        if not (0 <= bp[0] <= 105 and 0 <= bp[1] <= 68):
            row['state'] = 'out_of_play'
            owner = candidate = None
            streak = 0
            rows.append(row)
            continue
        if not np.isfinite(ball_motion.smoothed_speed_mps[frame]):
            owner = candidate = None
            streak = 0
            rows.append(row)
            continue
        order = finite[np.argsort(distances[frame,finite])]
        nearest = int(order[0])
        gap = distances[frame,order[1]]-distances[frame,nearest] if len(order)>1 else np.inf
        row['nearest_distance_m'] = float(distances[frame,nearest])
        speed = ball_motion.smoothed_speed_mps[frame]
        control = np.isfinite(speed) and (
            speed <= config.control_ball_speed_mps or
            (np.isfinite(relative[frame,nearest]) and relative[frame,nearest] <= config.control_relative_speed_mps))
        contested = distances[frame,nearest] <= config.acquisition_radius_m and gap < config.ambiguity_margin_m
        proposed = nearest if distances[frame,nearest] <= config.acquisition_radius_m and control and not contested else None
        if proposed is not None:
            row['candidate_player_id'] = int(player_ids[proposed])
        if proposed is not None and proposed == candidate:
            streak += 1
        else:
            candidate = proposed
            streak = 1 if proposed is not None else 0
        if contested:
            owner = None
            acquired_at = None
            row['state'] = 'contested'
        else:
            retain = owner is not None and distances[frame,owner] <= config.retention_radius_m and (
                np.isfinite(speed) and (speed <= config.control_ball_speed_mps or
                (np.isfinite(relative[frame,owner]) and relative[frame,owner] <= config.control_relative_speed_mps)))
            if not retain:
                owner = None
                acquired_at = None
            if candidate is not None and streak >= config.possession_frames and candidate != owner:
                owner = candidate
                acquired_at = frame-config.possession_frames+1
            row['state'] = 'controlled' if owner is not None else ('candidate' if proposed is not None else 'free')
        if owner is not None:
            proximity = max(0.0, 1-distances[frame,owner]/config.retention_radius_m)
            row.update(possessing_player_id=int(player_ids[owner]), possessing_team=teams[owner],
                       control_start_frame=acquired_at,
                       relative_speed_mps=float(relative[frame,owner]) if np.isfinite(relative[frame,owner]) else None,
                       control_confidence=float(0.5+0.5*proximity))
        rows.append(row)
    result = pd.DataFrame(rows)
    for field in ('possessing_player_id','candidate_player_id','control_start_frame'):
        result[field] = result[field].astype('Int64')
    return result
