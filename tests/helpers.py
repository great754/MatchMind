"""Shared synthetic fixtures, independent of unittest discovery names."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.intelligence.schema import EventType as T,EventMetrics,make_event

def formation_points(counts,direction=1):
    points=[]
    for line,count in enumerate(counts):
        points.extend([[20+line*20,float(y)] for y in np.linspace(10,58,count)] if count>1 else [[20+line*20,34.]])
    points=np.array(points)
    if direction==-1:points[:,0]=105-points[:,0]
    return points


def tactical_fixture(seconds=30):
    n=round(seconds*25)
    left=np.vstack(([5,34],formation_points((4,3,3))))
    right=np.vstack(([100,34],formation_points((4,4,2),-1)))
    positions=np.tile(np.vstack((left,right)),(n,1,1))
    ids=np.arange(100,122)
    recording=SimpleNamespace(positions=positions,ball=np.tile(left[1],(n,1)),fps=25.,player_ids=ids,
        player_teams=['left']*11+['right']*11,mot_ids={int(actor):int(actor-99) for actor in ids},goalkeeper_ids={100,111},
        source_files=[],source_offset=0,half=1,scope='clip',player_source='gsr',sync={'match_id':'synthetic','method':'synthetic'})
    ownership=pd.DataFrame(dict(frame=np.arange(n),possessing_player_id=pd.array([101]*n,dtype='Int64'),possessing_team=['left']*n,
        state=['controlled']*n,control_confidence=[.9]*n))
    motion=[SimpleNamespace(smoothed_speed_mps=np.zeros(n)) for _ in ids]
    ball_motion=SimpleNamespace(segment_id=np.zeros(n),positions=recording.ball)
    result={'possession':ownership,'players':motion,'ball':ball_motion}
    return recording,result


def transfer(frame=100,distance=10,progression=5,difficulty=20,validated=False):
    return make_event('synthetic',frame=frame,fps=25,type=T.PASS_COMPLETED,group_id=f'transfer:{frame}',source='transfer',significance=60,
        team='left',player_id=2,actor_id=101,receiver_id=3,receiver_actor_id=102,
        validation_state='bas_time_actor_match' if validated else 'estimated',validated_by_bas=validated,
        metrics=EventMetrics(start_frame=frame-25,end_frame=frame,distance_m=distance,forward_progression_m=progression,
            difficulty_score_0_100=difficulty,outcome='completed',start_x_m=20,start_y_m=20,end_x_m=30,end_y_m=30))


def write_query_artifacts(path):
    import json
    event=transfer()
    summary=dict(match_id='synthetic',scope='clip',half=1,frames=250,duration_seconds=10,trajectory_shot_candidates=0,
        limitations=['Synthetic fixture; not accuracy evidence.'],team_possession={'left':dict(controlled_seconds=6),'right':dict(controlled_seconds=2)})
    for name,value in [('summary.json',summary),('events.json',dict(events=[event.model_dump(mode='json')],player_labels={}))]:
        (path/name).write_text(json.dumps(value))
    pd.DataFrame([dict(player_id=101,mot_id=2,team='left',max_speed_kmh=20)]).to_csv(path/'player_summary.csv',index=False)
    pd.DataFrame(columns=['sender_team','outcome']).to_csv(path/'transfers.csv',index=False)
    return event


def shot_fixture(speed=12.,missing=False):
    from src.analytics.config import AnalyticsConfig
    from src.analytics.shots import detect_shots
    from src.analytics.validation import validate_events
    n=80;config=AnalyticsConfig();rec,result=tactical_fixture(seconds=n/25)
    rec.ball_status=np.ones(n,dtype=int)
    pos=np.column_stack((80+np.arange(n)*speed/25,np.full(n,34.)))
    velocity=np.tile([speed,0.],(n,1))
    if missing:pos[:]=np.nan;velocity[:]=np.nan
    ball=SimpleNamespace(positions=pos,velocity=velocity,smoothed_speed_mps=np.full(n,np.nan if missing else speed),
        valid=np.full(n,not missing),observed=np.ones(n,dtype=bool),raw_speed_flag=np.zeros(n,dtype=bool),rejected_speed_flag=np.zeros(n,dtype=bool),segment_id=np.zeros(n))
    result['ball']=ball
    result['possession'].loc[:,'possessing_player_id']=None
    result['possession'].loc[:,'possessing_team']=None
    result['possession'].loc[:,'state']='free'
    rec.events=[dict(annotation_id=1,frame=0,seconds=0.,label='SHOT',player_id=101,team='left')]
    result['shots']=detect_shots(result['possession'],ball,config)
    _,result['shot_matches']=validate_events(result['shots'],rec.events,25)
    return rec,result,config
