"""Deterministic English from supported facts; no model-authored claims."""

def sentences(facts, mode='fan'):
    data={f.field:f.value for f in facts if f.value is not None}
    team=data.get('team') or data.get('attacking_team')
    subject={'left':'Home side','right':'Away side'}.get(team,'The team')
    actor=data.get('actor_id') or data.get('player_id') or data.get('carrier_actor_id') or data.get('sender_id')
    player=f'Actor {actor}' if actor is not None else 'The player'
    out=[]
    if isinstance(data.get('x_m'),(float,int)):
        region='left third' if data['x_m']<35 else 'middle third' if data['x_m']<70 else 'right third'
        out.append(f'{player} was in the {region} of the pitch.')
    def number(field,template):
        if isinstance(data.get(field),(int,float)) and not isinstance(data[field],bool):out.append(template.format(value=f'{data[field]:.1f}'.rstrip('0').rstrip('.') if data[field]%1 else str(int(data[field]))))
    number('max_speed_kmh',player+(f' on {subject}' if team else '')+' reached an estimated top speed of {value} km/h.')
    number('measured_distance_m',player+' covered {value} metres in the tracked intervals.')
    number('confirmed_passes',subject+' completed {value} observable transfers.')
    number('total_confirmed_passes','Both teams completed {value} observable transfers altogether.')
    if 'team_with_more_confirmed_passes' in data:
        winner=data['team_with_more_confirmed_passes']
        out.append('Both teams completed the same number of observable transfers.' if winner=='equal' else f"{ {'left':'Home side','right':'Away side'}.get(winner,'Neither side')} completed more observable transfers.")
    number('pass_count_difference','The difference was {value} completed observable transfers.')
    if isinstance(data.get('confirmed_pass_share'),(int,float)):out.append(f"{subject} had {data['confirmed_pass_share']:.0%} overall possession by confirmed completed-pass share.")
    if isinstance(data.get('attributed_share'),(int,float)):out.append(f"{subject} was attributed possession for {data['attributed_share']:.0%} of the recording.")
    if 'attribution' in data:
        out.append(f'{player} on {subject} '+('controls the ball.' if data['attribution']=='controlled' else 'retains possession while control is being re-established.' if data['attribution']=='retained' else 'has no established possession.'))
    formation=data.get('estimate',data.get('formation'))
    if formation is not None:out.append(f'{subject} has an estimated {formation} shape.' if formation!='unknown' else f'There is not enough consistent evidence to identify {subject}\'s formation.')
    if 'pressure_level' in data:out.append('The available tracking cannot establish the pressure level.' if data['pressure_level']=='unknown' else f"Pressure on {subject} is {data['pressure_level']}.")
    if 'sustained_press' in data:out.append('The stricter sustained-pressure test is met.' if data['sustained_press'] else 'A sustained press is not established at this moment.')
    number('defenders_within_local',f'For {subject}, '+'{value} opponents are close to the ball carrier.')
    number('distance_m','The ball travelled {value} metres.')
    number('difficulty_score_0_100','The transfer has a retrospective difficulty estimate of {value} out of 100.')
    number('ball_speed_peak_kmh','The estimated peak two-dimensional ball speed was {value} km/h.')
    number('distinct_teammates',player+' exchanged completed transfers with {value} teammates.')
    for key,title in [('width_m','width'),('depth_m','depth'),('compactness_m','grouping radius')]:number(key,subject+f"'s {title} was {{value}} metres.")
    if data.get('block') in ('low','mid','high'):out.append(f"{subject} occupied a {data['block']} defensive block.")
    if data.get('outcome')=='completed':
        receiver=data.get('receiver_actor_id') or (data.get('receiver_id') if facts[0].source.startswith('transfers.csv') else None)
        out.append(f'{player} transferred the ball to Actor {receiver}, who established control.' if receiver else 'The receiving teammate established control.')
    elif data.get('outcome')=='intercepted':out.append('An opponent established control; this is an interception candidate.')
    if data.get('validation_state')=='bas_time_actor_match' or data.get('validated_by_bas') is True:out.append('The annotation supports the release timing and sender, rather than proving the full trajectory.')
    elif data.get('validation_state')=='estimated':out.append('This event is estimated from tracking; annotation support is not established.')
    number('bas_shots','The clip contains {value} annotated shots.')
    number('trajectory_shot_candidates','Tracking produced {value} heuristic shot candidates.')
    if not out and data.get('type'):out.append('A '+str(data['type']).replace('PASS','TRANSFER').replace('_',' ').lower()+' was recorded.')
    if not out:out.append('The available evidence does not support a more specific description.')
    return ' '.join(out)
