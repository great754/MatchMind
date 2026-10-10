"""Only these deterministic, fact-backed sentences may reach the user."""
from ..schema import EventType as T

def sentences(events,context: dict,mode: str) -> dict[str,str]:
    result={}
    for event in events:
        p=f'Player {event.player_id}' if event.player_id is not None else 'An unidentified player'
        receiver=f'Player {event.receiver_id}' if event.receiver_id is not None else 'an unidentified receiver'
        m=event.metrics; kind=event.type; tactical=event.tactical
        if kind==T.TACTICAL_BLOCK_CHANGED and tactical:
            text=f'MatchMind estimates a {tactical.block} defensive block for {event.team}, after {tactical.duration_s:.1f} seconds of geometric evidence under the configured centroid thresholds.'
        elif kind==T.TEAM_WIDTH_INCREASED and tactical:
            text=f'The {event.team} team’s recent mean outfield width increases from {tactical.previous_width_m:.1f} to {tactical.width_m:.1f} meters; goalkeeper excluded.'
        elif kind==T.TEAM_COMPACTNESS_CHANGED and tactical:
            text=f'The {event.team} team’s recent mean RMS spread around its outfield centroid changes from {tactical.previous_compactness_m:.1f} to {tactical.compactness_m:.1f} meters; this is geometric compactness.'
        elif kind==T.HIGH_PRESSURE_SEQUENCE and tactical:
            text=f'Tracking shows {tactical.defenders_within_local} observed opponents within {tactical.local_radius_m:g} meters of {p}, with {tactical.closing_defenders} closing opponents after {tactical.duration_s:.1f} seconds of sustained spatial and movement evidence. This is a geometric pressure estimate.'
        elif kind==T.TRANSFER_NETWORK_PATTERN and tactical:
            text=f'{p} connects with {receiver} in {tactical.transfer_count} estimated completed observable transfers during the trailing {tactical.network_window_seconds:g} seconds.'
        elif kind==T.FORMATION_ESTIMATE_CHANGED and tactical:
            text=f'MatchMind estimates a {tactical.formation} longitudinal line pattern for {event.team}, using {tactical.formation_samples} trailing observations; fit RMSE {tactical.formation_fit_rmse_m:.1f} meters. This is not a verified lineup or assignment of player roles.'
        elif kind in (T.PASS_COMPLETED,T.LONG_PASS,T.DIFFICULT_PASS,T.PROGRESSIVE_PASS,T.BALL_PROGRESSION):
            text=f'MatchMind estimates a completed transfer from {p} to {receiver}.'
            if mode=='beginner': text+= ' A completed transfer means the ball reaches a teammate under the possession rules.'
            if mode=='fan': text=f'{p} finds {receiver}, according to the estimated possession sequence.'
            if m.distance_m is not None: text+=f' The transfer covers {m.distance_m:.1f} meters.'
            if mode=='analyst':
                if m.forward_progression_m is not None: text+=f' Forward progression: {m.forward_progression_m:.1f} meters, using the configured attack direction.'
                if m.difficulty_score_0_100 is not None: text+=f' Retrospective difficulty: {m.difficulty_score_0_100:.1f}/100 (heuristic, not completion probability).'
        elif kind==T.TRANSFER_INTERCEPTED: text=f'MatchMind estimates an opposite-team transfer from {p} to {receiver}; this may be an interception, tackle or clearance.'
        elif kind==T.POSSESSION_CHANGED: text=f'MatchMind estimates possession changing to {p}.'
        elif kind==T.SHOT_CANDIDATE:
            text=f'MatchMind detects a possible shot by {p}.'
            if m.ball_speed_kmh is not None: text+=f' Peak interpolated 2D ball speed in the completed window: {m.ball_speed_kmh:.1f} km/h; strike speed and ball height are unavailable.'
        elif kind==T.SHOT_VALIDATED: text=f'The BAS reference annotates a shot by {p}; this is not independent validation of the reconstructed ball trajectory.'
        elif kind==T.GOAL: text='The BAS reference annotates a goal.'
        elif kind in (T.PLAYER_SPEED_THRESHOLD,T.BALL_SPEED_THRESHOLD):
            who=p if kind==T.PLAYER_SPEED_THRESHOLD else 'The interpolated 2D ball path'
            text=f'{who} reaches an estimated smoothed speed of {m.speed_kmh:.1f} km/h, above the configured {m.threshold_kmh:g} km/h threshold.'
        elif kind==T.PLAYER_DISTANCE_MILESTONE: text=f'{p} reaches {m.milestone_m:g} meters of measured travel; missing intervals contribute no distance.'
        else: continue
        result[event.event_id]=text
    if mode=='analyst' and result:
        result['context_possession']=f"In the available {context['sampled_seconds']:.2f}-second window, estimated possession is left {context['possession_percent']['left']:.1f}% and right {context['possession_percent']['right']:.1f}% of all frames; unknown ownership is {context['unknown_percent']:.1f}%."
    if mode=='analyst' and result and context['fastest_players']:
        fastest=context['fastest_players'][0]
        result['context_fastest']=f"Player {fastest['player_id']} has the highest accepted peak smoothed player speed in the available window: {fastest['peak_speed_kmh']:.1f} km/h (estimated)."
    if result and len(context['recent_pass_sequence'])>=3:
        sequence=' → '.join(f'Player {pid}' for pid in context['recent_pass_sequence'])
        result['context_sequence']=f'The recent estimated completed-transfer sequence is {sequence}.'
    state=context.get('tactical_state')
    if mode=='analyst' and result and state:
        for team in ('left','right'):
            shape=state[team]['shape']
            if shape['width_m'] is not None:
                result['context_shape_'+team]=f"At the latest tactical sample, {team} outfield width is {shape['width_m']:.1f} meters, depth {shape['depth_m']:.1f} meters and RMS compactness {shape['compactness_m']:.1f} meters; goalkeeper excluded."
    return result
