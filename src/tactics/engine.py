"""5 Hz deterministic tactical states from raw metric coordinates and confirmed ownership."""
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from src.intelligence.labels import recording_labels
from src.intelligence.schema import EventType as T
from .config import TacticsConfig
from .schema import TacticalState,TeamState,Pressure,Formation,BallRelative
from .shape import team_shape,block_candidate,players_relative_to_ball,usable
from .pressure import causal_velocities,local_pressure
from .formations import estimate_formation
from .events import PersistentState,PersistentTrigger,tactical_event
from .passing_network import transfer_network


def goalkeeper_roles(recording):
    """Use explicit actor roles or provider metadata; do not infer a formation keeper by color."""
    if hasattr(recording,'goalkeeper_ids'):
        return set(map(int,recording.goalkeeper_ids)),'explicit',None
    candidates=[Path(p).parent/f"{recording.sync['match_id']}_tracker_box_metadata.xml" for p in recording.source_files if str(p).endswith('_keypoints.json')]
    # Dataset source paths identify the dataset root, independently of current working directory.
    for p in recording.source_files:
        path=Path(p)
        if path.parent.name=='mot':candidates.append(path.parent.parent/'raw'/str(recording.sync['match_id'])/f"{recording.sync['match_id']}_tracker_box_metadata.xml")
    for path in candidates:
        if path.exists():
            roles={int(p.attrib['id']) for p in ET.parse(path).getroot().findall('.//players/player') if p.attrib.get('position')=='GK'}
            if roles:return roles,'provider_metadata',path
    return set(),'unavailable',None


def phase_comparison(states,config: TacticsConfig) -> dict:
    """Compare observed phase means only when both meet sample-count gates."""
    result={}
    for team in ('left','right'):
        phases={}
        for phase in ('possessing','defending'):
            shapes=[getattr(s,team).shape for s in states if getattr(s,team).phase==phase and getattr(s,team).shape.width_m is not None]
            phases[phase]=dict(samples=len(shapes),sampled_seconds=len(shapes)/config.sample_hz,
                **{f'mean_{name}':float(np.mean([getattr(s,name) for s in shapes])) if shapes else None for name in ('width_m','depth_m','compactness_m','spread_m','centroid_x_m','centroid_y_m')})
        enough=all(phases[p]['samples']>=config.comparison_min_samples for p in phases)
        delta=phases['possessing']['mean_width_m']-phases['defending']['mean_width_m'] if enough else None
        result[team]=dict(**phases,comparison_available=enough,possession_minus_defending_width_m=delta,
            wider_in_possession=bool(delta>=config.comparison_width_margin_m) if enough else None)
    return result


def analyze_tactics(recording,result,analytics_config,config=None,events=None):
    """Tactical calculations use no coordinate sample after their published frame."""
    config=config or TacticsConfig();stride=config.stride(recording.fps);started=time.perf_counter()
    if events is None:
        from src.intelligence.events import build_events
        events=build_events(recording,result)
    coords=np.asarray(recording.positions,float);labels=recording_labels(recording);actors=list(map(int,recording.player_ids));slots={actor:i for i,actor in enumerate(actors)}
    teams=np.asarray(recording.player_teams);keepers,role_source,role_file=goalkeeper_roles(recording)
    keeper_mask=np.isin(recording.player_ids,list(keepers))
    team_masks={team:teams==team for team in ('left','right')}
    outfield={team:team_masks[team]&~keeper_mask for team in team_masks}
    known={team:bool((team_masks[team]&keeper_mask).sum()==1) for team in team_masks}
    blocks={team:PersistentState(config.block_persistence_seconds) for team in team_masks}
    formation_states={team:PersistentState(config.formation_persistence_seconds,invalidate_pending=True) for team in team_masks}
    shape_triggers={team:{name:PersistentTrigger(config.shape_persistence_seconds,config.event_cooldown_seconds) for name in ('width','compactness')} for team in team_masks}
    formation_cache={team:Formation(reason='insufficient_history') for team in team_masks};last_formation_update={team:-math.inf for team in team_masks}
    press=PersistentTrigger(config.press_persistence_seconds,config.event_cooldown_seconds);press_carrier=None
    network_triggers={};sample_frames=[];states=[];tactical_events=[];direction_previous=None;direction_since=0
    completed=[e for e in events if e.type==T.PASS_COMPLETED];completed_frames=[e.frame for e in completed]
    formation_time=0.;network_time=0.
    for frame in range(0,len(coords),stride):
        timestamp=frame/recording.fps;sample_frames.append(frame)
        direction_left=config.direction(frame,'left',analytics_config)
        if direction_previous is not None and direction_left!=direction_previous:
            direction_since=frame
            blocks={team:PersistentState(config.block_persistence_seconds) for team in team_masks}
            formation_states={team:PersistentState(config.formation_persistence_seconds,invalidate_pending=True) for team in team_masks}
            shape_triggers={team:{name:PersistentTrigger(config.shape_persistence_seconds,config.event_cooldown_seconds) for name in ('width','compactness')} for team in team_masks}
            formation_cache={team:Formation(reason='direction_changed') for team in team_masks};last_formation_update={team:-math.inf for team in team_masks}
            press=PersistentTrigger(config.press_persistence_seconds,config.event_cooldown_seconds);press_carrier=None
        direction_previous=direction_left
        ownership=result['possession'].iloc[frame]
        owner=None if pd.isna(ownership.possessing_player_id) else int(ownership.possessing_player_id)
        owner_team=None if pd.isna(ownership.possessing_team) or owner is None else str(ownership.possessing_team)
        current=coords[frame];state_teams={}
        for team in ('left','right'):
            direction=config.direction(frame,team,analytics_config);mask=team_masks[team]
            shape=team_shape(current[mask],direction,keeper_mask[mask] if known[team] else None,role_source,config.min_outfield_players)
            phase='possessing' if owner_team==team else 'defending' if owner_team is not None else 'unknown'
            candidate=block_candidate(shape,phase,config.block_low_max_m,config.block_high_min_m)
            previous=blocks[team].value;block,changed,duration=blocks[team].update(candidate,frame,recording.fps)
            if changed:tactical_events.append(tactical_event(recording,frame,T.TACTICAL_BLOCK_CHANGED,team,dict(previous_block=previous,block=block,evidence_start_frame=blocks[team].since,duration_s=duration)))
            formation=formation_cache[team]
            if timestamp-last_formation_update[team]>=config.formation_update_seconds-1e-8:
                tick=time.perf_counter();lower=max(direction_since,frame-round(config.formation_window_seconds*recording.fps)+stride)
                start=bisect_right(sample_frames,lower-1);history_frames=sample_frames[start:]
                formation=estimate_formation(coords[history_frames][:,outfield[team]],history_frames,direction,config,known[team],recording.fps)
                formation_cache[team]=formation;last_formation_update[team]=timestamp;formation_time+=time.perf_counter()-tick
            complete=known[team] and int(outfield[team].sum())==10 and usable(current[outfield[team]]).all()
            estimate=formation.estimate if shape.width_m is not None and complete else 'unknown'
            if not complete:formation=Formation(reason='incomplete_roster' if known[team] else 'goalkeeper_unknown')
            stable,formation_changed,_=formation_states[team].update(estimate,frame,recording.fps)
            if stable=='unknown' and estimate!='unknown':formation=formation.model_copy(update={'estimate':'unknown','reason':'pending_persistence','fit_quality':'unavailable'})
            elif stable=='unknown' and estimate=='unknown' and shape.width_m is None:formation=Formation(reason='incomplete_roster')
            if formation_changed:tactical_events.append(tactical_event(recording,frame,T.FORMATION_ESTIMATE_CHANGED,team,dict(formation=stable,formation_fit_rmse_m=formation.fit_rmse_m,formation_window_start_frame=formation.window_start_frame,formation_window_end_frame=formation.window_end_frame,formation_samples=formation.samples,duration_s=config.formation_persistence_seconds)))
            state_teams[team]=TeamState(team=team,attack_direction=direction,phase=phase,shape=shape,block_candidate=candidate,block=block,block_evidence_s=duration,formation=formation)
            # Compare two trailing windows, requiring coverage in each and persistence.
            window_frames=round(config.shape_window_seconds*recording.fps)
            past_start=bisect_right(sample_frames,max(direction_since,frame-2*window_frames))
            past_end=bisect_right(sample_frames,frame-window_frames)
            recent_start=bisect_right(sample_frames,max(direction_since,frame-window_frames))
            past=[getattr(s,team).shape for s in states[past_start:past_end]]
            recent=[getattr(s,team).shape for s in states[recent_start:]]
            recent.append(shape)
            expected=config.shape_window_seconds*config.sample_hz*config.shape_window_min_fraction
            for metric,threshold,kind in [('width',config.width_change_m,T.TEAM_WIDTH_INCREASED),('compactness',config.compactness_change_m,T.TEAM_COMPACTNESS_CHANGED)]:
                before=[getattr(s,metric+'_m') for s in past if getattr(s,metric+'_m') is not None]
                after=[getattr(s,metric+'_m') for s in recent if getattr(s,metric+'_m') is not None]
                available=len(before)>=expected and len(after)>=expected and shape.width_m is not None
                difference=float(np.mean(after)-np.mean(before)) if available else 0
                condition=available and (difference>=threshold if metric=='width' else abs(difference)>=threshold)
                fire,evidence=shape_triggers[team][metric].update(condition,frame,recording.fps)
                if fire:tactical_events.append(tactical_event(recording,frame,kind,team,{metric+'_m':float(np.mean(after)),'previous_'+metric+'_m':float(np.mean(before)), 'duration_s':evidence,'evidence_start_frame':shape_triggers[team][metric].since}))
        pressure=Pressure(reason='no_confirmed_carrier');relative=BallRelative()
        ball=recording.ball[frame]
        if owner is not None and owner in slots and owner_team in team_masks and usable(np.asarray(ball).reshape(1,2))[0]:
            attacking=team_masks[owner_team];defending=~attacking;defending_team='right' if owner_team=='left' else 'left'
            velocities=causal_velocities(coords,frame,recording.fps,config.closing_lookback_seconds,config.closing_max_mps)
            pressure=local_pressure(current[slots[owner]],velocities[slots[owner]],current[defending],velocities[defending],current[attacking],config,
                carrier_id=labels[owner].tracker_id,carrier_actor_id=owner,attacking_team=owner_team,defending_team=defending_team,expected_defenders=int(defending.sum()))
            if known[owner_team] and known[defending_team]:relative=players_relative_to_ball(current[outfield[owner_team]],current[outfield[defending_team]],ball,config.direction(frame,owner_team,analytics_config),config.ball_line_tolerance_m)
        elif owner is not None:pressure=Pressure(reason='missing_carrier_or_ball')
        if owner!=press_carrier:
            press=PersistentTrigger(config.press_persistence_seconds,config.event_cooldown_seconds);press_carrier=owner
        fire,duration=press.update(pressure.press_candidate,frame,recording.fps)
        sustained=pressure.press_candidate and duration+1e-8>=config.press_persistence_seconds
        pressure=pressure.model_copy(update={'sustained_press':sustained,'evidence_duration_s':duration})
        if fire:tactical_events.append(tactical_event(recording,frame,T.HIGH_PRESSURE_SEQUENCE,pressure.defending_team,dict(nearest_defender_distance_m=pressure.nearest_defender_distance_m,defenders_within_local=pressure.defenders_within_local,local_radius_m=config.pressure_local_radius_m,closing_defenders=pressure.closing_defenders,duration_s=duration,evidence_start_frame=press.since),pressure.carrier_id,pressure.carrier_actor_id))
        a,b=state_teams['left'].shape,state_teams['right'].shape
        separation=float(np.hypot(a.centroid_x_m-b.centroid_x_m,a.centroid_y_m-b.centroid_y_m)) if a.centroid_x_m is not None and b.centroid_x_m is not None else None
        states.append(TacticalState(frame=frame,timestamp=timestamp,left=state_teams['left'],right=state_teams['right'],pressure=pressure,ball_relative=relative,
            centroid_distance_m=separation,position_source=('gsr' if recording.player_source=='gsr' else 'pitch_transform') if np.isfinite(current).any() else 'unavailable',
            identity_source='mot_to_gsr_mapping' if recording.mot_ids else 'gsr_actor',possession_state=ownership.state,
            possession_confidence=float(ownership.control_confidence) if not pd.isna(ownership.control_confidence) else None))
        tick=time.perf_counter()
        upper=bisect_right(completed_frames,frame);lower=bisect_right(completed_frames,frame-config.network_window_seconds*recording.fps)
        counts=Counter((e.team,e.player_id,e.receiver_id) for e in completed[lower:upper])
        for edge in set(counts)|set(network_triggers):
            trigger=network_triggers.setdefault(edge,PersistentTrigger(config.network_pattern_persistence_seconds,config.event_cooldown_seconds))
            fire,duration=trigger.update(counts[edge]>=config.network_pattern_min_transfers,frame,recording.fps)
            if fire:
                team,sender,receiver=edge
                tactical_events.append(tactical_event(recording,frame,T.TRANSFER_NETWORK_PATTERN,team,dict(transfer_count=counts[edge],network_window_seconds=config.network_window_seconds,duration_s=duration,evidence_start_frame=trigger.since),sender,receiver_id=receiver))
        network_time+=time.perf_counter()-tick
    tactical_events.sort(key=lambda e:(e.frame,e.event_id))
    duration=(len(coords)-1)/recording.fps
    network=transfer_network(events,end_seconds=duration,long_transfer_m=config.network_long_transfer_m)
    windows=list(np.arange(config.network_window_seconds,duration,config.network_window_seconds))+[duration]
    rolling=[transfer_network(events,max(-1/recording.fps,end-config.network_window_seconds),float(end),team,config.network_long_transfer_m) for end in windows for team in ('left','right')]
    return dict(states=states,events=tactical_events,network=network,
        network_by_team={team:transfer_network(events,end_seconds=duration,team=team,long_transfer_m=config.network_long_transfer_m) for team in ('left','right')},network_windows=rolling,
        summary=dict(schema_version='1.0',sample_hz=config.sample_hz,source_fps=recording.fps,samples=len(states),config=asdict(config),goalkeeper_actor_ids=sorted(keepers),goalkeeper_source=role_source,
            phase_comparison=phase_comparison(states,config),block_samples={team:dict(Counter(getattr(s,team).block for s in states)) for team in team_masks},
            formation_samples={team:dict(Counter(getattr(s,team).formation.estimate for s in states)) for team in team_masks},
            sustained_press_samples=sum(s.pressure.sustained_press for s in states),event_counts=dict(Counter(e.type.value for e in tactical_events)),
            metadata_source=str(role_file) if role_file else None),
        profile=dict(total_seconds=time.perf_counter()-started,formation_seconds=formation_time,network_pattern_seconds=network_time))
