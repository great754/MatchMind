"""Descriptions of existing visual evidence, never new possession or touch facts."""


def interaction_frames(evidence,events):
    endpoints={}
    for event in events:
        if event['status']!='estimated':continue
        endpoints[event['start_frame']]=('RELEASE_SUPPORTED',event['event_id'])
        endpoints[event['end_frame']]=('RECEIPT_SUPPORTED',event['event_id'])
    result=[]
    for row in evidence:
        owner=row['people'].get(row['owner']) if row['owner'] else None
        state='UNKNOWN';reason=row['reason'];event_id=None
        if row['observed']:
            state='POSSIBLE_CONTACT' if owner else 'BALL_OBSERVED_WITHOUT_CLEAR_CONTACT'
            if row['frame'] in endpoints:state,event_id=endpoints[row['frame']]
        result.append({'frame':row['frame'],'state':state,'direct_ball_observation':row['observed'],
            'possible_player_ref':owner['ref'] if owner else None,'possible_player_entity':owner['entity'] if owner else None,
            'team':owner['team'] if owner else None,'team_support':owner['team_support'] if owner else None,
            'reason':reason,'event_id':event_id,'verified_touch':False,'verified_possession':False,
            'mode':'offline_existing_pass_rules','definition':'Image-space lower-body proximity and existing evidence-qualified pass endpoints; not independent contact or possession truth.'})
    return result


def pass_trace(reference,task,balls,evidence,events,fps):
    start,end=reference['release_frame'],reference['receipt_frame'];radius=max(1,round(.25*fps))
    release=[i for i in range(max(0,start-radius),min(len(balls),start+radius+1)) if balls[i]['directly_observed']]
    receipt=[i for i in range(max(0,end-radius),min(len(balls),end+radius+1)) if balls[i]['directly_observed']]
    flight=[i for i in range(start,end+1) if balls[i]['directly_observed']]
    related=[e for e in events if abs(e['start']-start/fps)<=.25 and abs(e['end']-end/fps)<=.25]
    reasons=[]
    if not release:reasons.append('NO_DIRECT_BALL_NEAR_HUMAN_RELEASE')
    if not receipt:reasons.append('NO_DIRECT_BALL_NEAR_HUMAN_RECEIPT')
    if not any(evidence[i]['owner'] for i in release):reasons.append('NO_UNAMBIGUOUS_M9_LOWER_BODY_CONTACT_NEAR_RELEASE')
    if not any(evidence[i]['owner'] for i in receipt):reasons.append('NO_UNAMBIGUOUS_M9_LOWER_BODY_CONTACT_NEAR_RECEIPT')
    for event in related:reasons.extend(event['reasons'])
    accepted=any(e['status']=='estimated' for e in related)
    if not accepted and not reasons:reasons.append('NO_EVENT_WITH_BOTH_ENDPOINTS_INSIDE_FIXED_MATCH_TOLERANCE; CHECK_FLIGHT_CONTINUITY_TEAM_AND_CONTACT_GATES')
    def players(indices):
        return [{'frame':i,'owner':evidence[i]['owner'],'candidates':[{k:p[k] for k in ['ref','entity','bbox','team','team_support']} for p in evidence[i]['people'].values()]} for i in indices]
    return {'review_task_id':task['task_id'],'release_frame':start,'receipt_frame':end,'release_seconds':start/fps,'receipt_seconds':end/fps,
        'direct_observations_near_release':release,'direct_observations_near_receipt':receipt,'direct_flight_observations':flight,
        'sender_evidence':players(release),'receiver_evidence':players(receipt),'related_events':[e['event_id'] for e in related],
        'matched_completed_pass':accepted,'diagnostic_reasons':sorted(set(reasons)),
        'limits':'Sparse human pass endpoints do not independently validate each model ball/player observation; reasons describe model evidence, not new human labels.'}
