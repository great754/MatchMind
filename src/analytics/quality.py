"""An inspectable retrospective difficulty score, not a learned probability."""
import numpy as np

QUALITY_COLUMNS = ['forward_progression_m','sender_nearby_defenders',
                   'passing_lane_defenders','target_nearest_defender_at_launch_m',
                   'target_nearby_defenders_at_launch','receiver_pressure_at_arrival',
                   'difficulty_distance_component','difficulty_progression_component',
                   'difficulty_lane_component','difficulty_target_pressure_component',
                   'difficulty_sender_pressure_component','difficulty_score_0_100',
                   'heuristic_completion_score_0_1','quality_available']


def pass_features(start, end, defenders_at_launch, defenders_at_arrival, direction, config):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    defenders = np.asarray(defenders_at_launch, dtype=float).reshape(-1,2)
    defenders = defenders[np.isfinite(defenders).all(axis=1)]
    arrivals = np.asarray(defenders_at_arrival, dtype=float).reshape(-1,2)
    arrivals = arrivals[np.isfinite(arrivals).all(axis=1)]
    distance = float(np.linalg.norm(end-start))
    progression = float((end[0]-start[0])*direction)
    sender_dist = np.linalg.norm(defenders-start, axis=1)
    target_dist = np.linalg.norm(defenders-end, axis=1)
    arrival_dist = np.linalg.norm(arrivals-end, axis=1)
    if distance > 0:
        along = ((defenders-start)@(end-start))/(distance**2)
        projected = start+along[:,None]*(end-start)
        lateral = np.linalg.norm(defenders-projected, axis=1)
        lane = int(np.sum((along>0.05)&(along<0.95)&(lateral<=config.lane_radius_m)))
    else:
        lane = 0
    sender_pressure = int(np.sum(sender_dist <= config.defender_radius_m))
    target_pressure = int(np.sum(target_dist <= config.defender_radius_m))
    nearest = float(target_dist.min()) if len(target_dist) else None
    arrival_pressure = int(np.sum(arrival_dist <= config.defender_radius_m)) if len(arrival_dist) else None
    available = bool(len(defenders))
    components = [35*min(distance/50,1), 10*min(max(progression,0)/30,1),
                  25*min(lane/3,1), 20*(max(0,1-nearest/config.defender_radius_m) if nearest is not None else 0),
                  10*min(sender_pressure/3,1)]
    score = float(sum(components)) if available else None
    return dict(forward_progression_m=progression,
                sender_nearby_defenders=sender_pressure if available else None,
                passing_lane_defenders=lane if available else None,
                target_nearest_defender_at_launch_m=nearest,
                target_nearby_defenders_at_launch=target_pressure if available else None,
                receiver_pressure_at_arrival=arrival_pressure,
                difficulty_distance_component=components[0], difficulty_progression_component=components[1],
                difficulty_lane_component=components[2] if available else None,
                difficulty_target_pressure_component=components[3] if available else None,
                difficulty_sender_pressure_component=components[4] if available else None,
                difficulty_score_0_100=score,
                heuristic_completion_score_0_1=1-score/100 if score is not None else None,
                quality_available=available)


def score_passes(passes, player_positions, player_teams, config):
    scored = passes.copy()
    values = []
    for r in passes.itertuples(index=False):
        opponent = np.asarray(player_teams) != r.sender_team
        values.append(pass_features([r.start_x_m,r.start_y_m], [r.end_x_m,r.end_y_m],
                                    player_positions[int(r.start_frame),opponent],
                                    player_positions[int(r.end_frame),opponent],
                                    config.attack_direction(r.sender_team), config))
    for column in QUALITY_COLUMNS:
        scored[column] = [v[column] for v in values]
    return scored
