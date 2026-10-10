"""Run the complete offline analytics pipeline without decoding a video."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.config import AnalyticsConfig
from src.analytics.data import load_recording
from src.analytics.motion import calculate_motion
from src.analytics.passes import detect_passes
from src.analytics.possession import detect_possession
from src.analytics.quality import score_passes
from src.analytics.shots import annotated_shot_windows, detect_shots
from src.analytics.validation import validate_events
from src.vision.data_loading import DatasetPaths


def analyze_recording(recording, config):
    if recording.fps != config.fps:
        raise ValueError('Configuration FPS must match recording metadata')
    players = [calculate_motion(recording.positions[:,i],config.fps,config.player_smoothing_frames,
                                config.median_frames,config.player_speed_limit_mps)
               for i in range(len(recording.player_ids))]
    ball = calculate_motion(recording.ball,config.fps,config.ball_smoothing_frames,
                            config.median_frames,config.ball_speed_limit_mps)
    positions = np.stack([p.positions for p in players],axis=1)
    velocities = np.stack([p.velocity for p in players],axis=1)
    possession = detect_possession(positions,velocities,recording.player_ids,recording.player_teams,ball,config)
    passes = score_passes(detect_passes(possession,ball,config),positions,recording.player_teams,config)
    shots = detect_shots(possession,ball,config)
    bas_shots = annotated_shot_windows(recording.events,ball,config)
    pass_refs = [e for e in recording.events if e['label'] in ('PASS','HIGH PASS','CROSS','THROW IN')]
    shot_refs = [e for e in recording.events if e['label'] == 'SHOT']
    pass_validation,pass_matches = validate_events(passes,pass_refs,config.fps,
                                                   config.validation_tolerance_seconds,'sender_id')
    shot_validation,shot_matches = validate_events(shots,shot_refs,config.fps,config.validation_tolerance_seconds)
    return dict(players=players,ball=ball,positions=positions,possession=possession,
                passes=passes,shots=shots,bas_shots=bas_shots,
                pass_validation=pass_validation,shot_validation=shot_validation,
                pass_matches=pass_matches,shot_matches=shot_matches)


def motion_table(motion, fps):
    n = len(motion.positions)
    return pd.DataFrame(dict(frame=np.arange(n),seconds=np.arange(n)/fps,
                             raw_x_m=motion.raw_positions[:,0],raw_y_m=motion.raw_positions[:,1],
                             x_m=motion.positions[:,0],y_m=motion.positions[:,1],
                             vx_mps=motion.velocity[:,0],vy_mps=motion.velocity[:,1],
                             instantaneous_speed_mps=motion.instantaneous_speed_mps,
                             instantaneous_speed_kmh=motion.instantaneous_speed_mps*3.6,
                             smoothed_speed_mps=motion.smoothed_speed_mps,
                             smoothed_speed_kmh=motion.smoothed_speed_mps*3.6,
                             distance_m=motion.distance_m,step_distance_m=motion.step_distance_m,
                             observed=motion.observed,position_usable=motion.valid,
                             raw_speed_flag=motion.raw_speed_flag,rejected_speed_flag=motion.rejected_speed_flag,
                             segment_id=motion.segment_id))


def player_summary(recording, result, config):
    rows = []
    for i,(pid,team,m) in enumerate(zip(recording.player_ids,recording.player_teams,result['players'])):
        speeds = m.smoothed_speed_mps[np.isfinite(m.smoothed_speed_mps)]
        rows.append(dict(player_id=int(pid),mot_id=recording.mot_ids.get(int(pid)),team=team,
                         measured_distance_m=float(m.distance_m[-1]),
                         mean_speed_mps=float(speeds.mean()) if len(speeds) else None,
                         max_speed_mps=float(speeds.max()) if len(speeds) else None,
                         max_speed_kmh=float(speeds.max()*3.6) if len(speeds) else None,
                         peak_speed_seconds=float(np.nanargmax(m.smoothed_speed_mps)/config.fps) if len(speeds) else None,
                         observed_position_fraction=float(m.observed.mean()),
                         valid_motion_fraction=float(np.isfinite(m.smoothed_speed_mps).mean()),
                         raw_speed_flag_frames=int(m.raw_speed_flag.sum()),
                         rejected_speed_frames=int(m.rejected_speed_flag.sum()),
                         controlled_seconds=float((result['possession'].possessing_player_id == int(pid)).sum()/config.fps)))
    return pd.DataFrame(rows)


def write_outputs(recording, result, config, output, summary_only=False, intelligence_config=None, commentary_provider=None,tactics_config=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summaries = player_summary(recording,result,config)
    summaries.to_csv(output/'player_summary.csv',index=False)
    if not summary_only:
        frames = []
        for pid,team,m in zip(recording.player_ids,recording.player_teams,result['players']):
            table = motion_table(m,config.fps)
            table.insert(2,'player_id',int(pid))
            table.insert(3,'mot_id',recording.mot_ids.get(int(pid)))
            table.insert(4,'team',team)
            table['position_source']=np.where(m.observed,'gsr' if recording.player_source=='gsr' else 'pitch_transform','unavailable')
            table['identity_source']='gsr_actor' if int(pid) not in recording.mot_ids else 'mot_to_gsr_mapping'
            frames.append(table)
        pd.concat(frames,ignore_index=True).to_csv(output/'player_frames.csv',index=False,float_format='%.6f')
    ball_frames = motion_table(result['ball'],config.fps)
    ball_frames['source_status'] = recording.ball_status
    ball_frames['position_source'] = 'event_interpolated'
    ball_frames['half_video_seconds'] = (ball_frames.frame+recording.source_offset)/config.fps
    ball_frames.to_csv(output/'ball_frames.csv',index=False,float_format='%.6f')
    possession = result['possession'].copy()
    possession.insert(1,'seconds',possession.frame/config.fps)
    possession.insert(2,'half_video_seconds',(possession.frame+recording.source_offset)/config.fps)
    possession.to_csv(output/'possession.csv',index=False,float_format='%.6f')
    for name,key in (('passes','passes'),('shots','shots'),('bas_shot_windows','bas_shots'),
                     ('pass_validation','pass_matches'),('shot_validation','shot_matches')):
        result[key].to_csv(output/f'{name}.csv',index=False,float_format='%.6f')
    transfers=result['passes'].copy()
    transfers['transfer_id']=transfers['pass_id']
    transfers['inverse_geometric_difficulty_0_1']=transfers['heuristic_completion_score_0_1']
    transfers['observation_kind']='observable_transfer'
    matched={int(r.prediction_index) for r in result['pass_matches'].itertuples() if r.match_status=='matched'}
    transfers['bas_supported_pass']=[i in matched for i in range(len(transfers))]
    transfers['event_source']=['inferred+validated' if i in matched else 'inferred' for i in range(len(transfers))]
    transfers['validation_scope']=['release_time_and_sender_only' if i in matched else 'unvalidated' for i in range(len(transfers))]
    transfers.to_csv(output/'transfers.csv',index=False,float_format='%.6f')
    semantics=dict(schema_version='1.0',legacy_files={'passes.csv':'Observable transfer candidates; filename retained for compatibility.',
        'shots.csv':'Trajectory shot candidates; speeds are estimated 2D ball motion.'},
        legacy_fields={'pass_id':'transfer candidate ID','heuristic_completion_score_0_1':'Inverse geometric difficulty; NOT calibrated probability.',
            'speed_peak_kmh':'Peak estimated 2D ball speed in the shot window, NOT strike velocity.',
            'ball_speed_peak_kmh':'Peak estimated 2D ball speed during the observable transfer.'},
        legacy_event_types={k:k.replace('PASS','TRANSFER') for k in ('PASS_COMPLETED','LONG_PASS','DIFFICULT_PASS','PROGRESSIVE_PASS')},
        preferred_transfer_export='transfers.csv',ball_position_source='event_interpolated',ball_height_available=False,
        player_position_source=recording.player_source,synchronization=recording.sync['method'],source_offset_frames=recording.source_offset,
        control_confidence='Rule-derived proximity score, not calibrated probability.',bas_matching='Release timing/sender agreement does not validate outcome or intent; not independent of BAS ball anchors.')
    (output/'measurement_definitions.json').write_text(json.dumps(semantics,indent=2)+'\n')
    pd.DataFrame(recording.events,columns=['annotation_id','frame','seconds','half_video_seconds','label','player_id','team']).to_csv(output/'reference_events.csv',index=False)
    owned = int((possession.state == 'controlled').sum())
    by_team = {team:dict(controlled_seconds=float((possession.possessing_team == team).sum()/config.fps),
                         fraction_of_all_frames=float((possession.possessing_team == team).mean()),
                         fraction_of_controlled_frames=float((possession.possessing_team == team).sum()/owned) if owned else None)
               for team in ('left','right')}
    sources = []
    for filename in recording.source_files:
        p = Path(filename)
        sources.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    report = dict(schema_version=1,match_id=recording.sync['match_id'],scope=recording.scope,
                  half=recording.half,player_coordinate_source=recording.player_source,
                  fps=config.fps,frames=len(recording.ball),duration_seconds=len(recording.ball)/config.fps,
                  source_offset_frames=recording.source_offset,players=len(recording.player_ids),
                  ball_position_fraction=float(result['ball'].observed.mean()),
                  ball_motion_fraction=float(np.isfinite(result['ball'].smoothed_speed_mps).mean()),
                  ball_raw_speed_flag_frames=int(result['ball'].raw_speed_flag.sum()),
                  ball_rejected_speed_frames=int(result['ball'].rejected_speed_flag.sum()),
                  possession_state_frames=possession.state.value_counts().to_dict(),team_possession=by_team,
                  measurement_definitions='measurement_definitions.json',
                  tactics_enabled=tactics_config is not False,tactical_summary='tactical_summary.json' if tactics_config is not False else None,
                  transfer_completion_rate=float((transfers.outcome=='completed').mean()) if len(transfers) else None,
                  transfer_completion_rate_definition='Completed observable transfers / all detected transfer candidates; not official pass accuracy.',
                  transfer_outcomes=result['passes'].outcome.value_counts().to_dict(),
                  trajectory_shot_candidates=len(result['shots']),bas_shots=len(result['bas_shots']),
                  bas_shot_windows_with_speed=int((result['bas_shots'].valid_speed_samples>0).sum()),
                  bas_shot_windows_without_speed=int((result['bas_shots'].valid_speed_samples==0).sum()),
                  pass_validation=result['pass_validation'],shot_validation=result['shot_validation'],
                  config=asdict(config),source_files=sources,sync_method=recording.sync['method'],
                  limitations=[
                      'Ball positions are event-anchored and interpolated; speeds are 2D estimates, not measured strike speeds.',
                      'Centered smoothing uses future frames: this pipeline is offline and adds no claim of live latency.',
                      'Missing/rejected intervals contribute no distance; measured distance is partial coverage, not full ground truth.',
                      'Ownership and transfer detections are heuristic; opposite-team transfers can be interceptions, tackles or clearances.',
                      'Difficulty is a retrospective heuristic with a realized endpoint, not calibrated completion probability.',
                      'BAS-based checks are not independent because the supplied ball reconstruction uses those event anchors.',
                      'Clip alignment is inferred at approximately one-frame precision; goal direction is configurable.',
                      'Ball height is unavailable. Tracker IDs are not verified jersey numbers or player identities.',
                      'Goalkeeper/team assignment and panoramic calibration can introduce errors; GSR labels are preferred when available.'
                  ])
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    from src.intelligence.pipeline import write_intelligence
    intelligence = write_intelligence(recording,result,output,intelligence_config,commentary_provider,tactics_config,config)
    from src.analytics.report import create_report
    create_report(recording,result,summaries,report,output,intelligence)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--half',type=int,choices=(1,2))
    parser.add_argument('--scope',choices=('clip','half'),default='clip')
    parser.add_argument('--player-source',choices=('gsr','mot'),default='gsr')
    parser.add_argument('--match-id',default='118575')
    parser.add_argument('--data-root',type=Path,default=Path('data/soccertrack'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,help='JSON file overriding AnalyticsConfig fields')
    parser.add_argument('--summary-only',action='store_true',help='Skip the large player_frames.csv; other outputs remain')
    parser.add_argument('--commentary-provider',choices=('none','template','gemini'),default='none')
    parser.add_argument('--tactics-config',type=Path,help='JSON overrides for deterministic TacticsConfig')
    parser.add_argument('--no-tactics',action='store_true',help='Skip tactical calculations and report section')
    parser.add_argument('--gemini-model',default='gemini-3.5-flash-lite')
    parser.add_argument('--intelligence-config',type=Path,help='JSON overrides for IntelligenceConfig')
    parser.add_argument('--max-commentary-events',type=int)
    parser.add_argument('--max-provider-calls',type=int)
    parser.add_argument('--significance-threshold',type=float)
    parser.add_argument('--commentary-start',type=float)
    parser.add_argument('--commentary-end',type=float)
    args = parser.parse_args()
    from src.intelligence.config import IntelligenceConfig
    from src.intelligence.commentary.base import TemplateProvider,CommentaryError
    intelligence_values=json.loads(args.intelligence_config.read_text()) if args.intelligence_config else {}
    for argument,field in [('max_commentary_events','max_commentary_moments'),('max_provider_calls','max_provider_calls'),('significance_threshold','significance_threshold'),('commentary_start','start_seconds'),('commentary_end','end_seconds')]:
        if getattr(args,argument) is not None: intelligence_values[field]=getattr(args,argument)
    try:
        intelligence_config=IntelligenceConfig(**intelligence_values)
        provider=None
        if args.commentary_provider=='template': provider=TemplateProvider()
        elif args.commentary_provider=='gemini':
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[2]/'.env',override=False)
            from src.intelligence.commentary.gemini_provider import GeminiCommentaryProvider
            provider=GeminiCommentaryProvider(args.gemini_model)
    except (ValueError,CommentaryError) as error:
        parser.error(str(error))
    from src.tactics.config import TacticsConfig
    tactics_config=False if args.no_tactics else TacticsConfig(**(json.loads(args.tactics_config.read_text()) if args.tactics_config else {}))
    values = json.loads(args.config.read_text()) if args.config else {}
    output = args.output or Path('outputs')/f'analytics_{args.match_id}_{args.scope}'+(f'_h{args.half}' if args.half else '')
    print(f'Loading {args.scope} positions from {args.player_source}...',flush=True)
    recording = load_recording(DatasetPaths(args.match_id,args.data_root),args.scope,args.player_source,args.half)
    values.setdefault('left_attack_direction',1 if recording.half == 1 else -1)
    values.setdefault('fps',recording.fps)
    config = AnalyticsConfig(**values)
    print(f'Analyzing {len(recording.ball)} frames, {len(recording.player_ids)} players...',flush=True)
    result = analyze_recording(recording,config)
    summary = write_outputs(recording,result,config,output,args.summary_only,intelligence_config,provider,tactics_config)
    print(json.dumps({k:summary[k] for k in ('possession_state_frames','transfer_outcomes','trajectory_shot_candidates','bas_shots','shot_validation')},indent=2),flush=True)
    print(f'Report: {output.resolve()}/report.html',flush=True)


if __name__ == '__main__':
    main()
