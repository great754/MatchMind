"""Tactical machine-readable outputs, compact report data and two focused plot panels."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .evidence import pre_event_evidence

DEFINITIONS={
    'coordinates':'105×68 meters, corner origin; x to right, y toward near touchline.',
    'position_input':'Raw metric recording positions, not centered-smoothed positions. Causal trailing histories only.',
    'shape':'Goalkeepers excluded using explicit/provider role identities; unavailable if role unknown or fewer than min_outfield_players usable observations.',
    'width_depth':'Width = max(y)-min(y); depth = max(x)-min(x), usable outfield only.',
    'compactness':'RMS distance to outfield centroid, meters; lower is more compact.',
    'spread':'Mean pairwise Euclidean outfield distance, meters.',
    'longitudinal':'Outfield centroid distance from own goal: centroid x if attacking +x, 105-x if attacking -x.',
    'block':'Only out of possession with confirmed opposing ownership. Centroid longitudinal position in configurable pitch bands; persistence required.',
    'pressure':'Observed opponent counts near current confirmed carrier, including goalkeepers; availability retained. Nearby opponents alone are not pressing.',
    'closing':'Backward mean relative velocity projected toward carrier; requires a full finite history and caps implausible individual displacement velocities.',
    'press':'At least pressure_min_defenders within local radius and at least one closing defender within outer radius; same carrier and sustained spatial/closing evidence required.',
    'ball_relative':'Outfield only; ahead = positive (player_x-ball_x)*attacking_direction beyond tolerance. Goal-side defenders lie toward their own defended goal; behind_ball_to_own_goal is an explicit synonym.',
    'formation':'Trailing identity-stable longitudinal line-count fit, full 10-player outfield roster and known keeper, deterministic RMSE/gap/stability gates plus persistence; not verified lineup or player roles.',
    'transfer_network':'Directed completed observable transfers at confirmed receipt, including non-pass movements; not official passing statistics.',
    'comparison':'Sample means in confirmed possession/defending phases, with minimum counts in both phases; repeated frames are not independent observations.',
    'evidence':'Strictly earlier than original release/start; later-confirmed outcomes and retrospective realized-target geometry are separated.',
    'event_timing':'Persistent event timestamp is confirmation time; evidence_start_frame describes the earlier start. No backward-dated tactical event.',
}


def write_tactics(recording,result,tactics,events,output) -> dict:
    """Export full audit facts and return compact synchronized report data."""
    output=Path(output);summary=tactics['summary'];summary['definitions']=DEFINITIONS
    summary['units']={'length':'m','velocity':'m/s','time':'s','counts':'observed players/samples/transfers'}
    summary['source_offset_frames']=recording.source_offset;summary['synchronization_method']=recording.sync['method']
    summary['limitations']=['Tactical labels are configurable geometric heuristics, not verified tactics or causal conclusions.',
        'GSR is quantized; pitch fallback can be inaccurate, especially in interior regions. Team/GK classification and timing remain uncertain.',
        'Only backward coordinate histories enter tactics; upstream ball interpolation and centered-smoothed ownership are offline inputs.',
        'Missing/off-field players reduce coverage; counts are observed counts, not guarantees of all active players.',
        'Formation averages can obscure movement. Unknown is intentional; fit quality is not a calibrated probability.']
    files=list(recording.source_files)
    if summary.get('metadata_source'):files.append(summary['metadata_source'])
    summary['source_files']=[dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in files]
    states=tactics['states'];serialized=[s.model_dump(mode='json') for s in states]
    pd.json_normalize(serialized,sep='_').to_csv(output/'tactical_states.csv',index=False,float_format='%.6f')
    export=dict(schema_version='1.0',full=tactics['network'],by_team=tactics['network_by_team'],windows=tactics['network_windows'])
    pd.DataFrame(tactics['network']['edges'],columns=['team','sender_id','receiver_id','transfer_count','mean_distance_m','total_progression_m','mean_progression_m','mean_heuristic_difficulty','distance_samples','progression_samples','difficulty_samples','long_transfer_count','bas_supported_pass_count']).to_csv(output/'transfer_network.csv',index=False)
    evidence=pre_event_evidence(events,states,result['passes'],recording.fps)
    for name,payload in [('tactical_states.json',dict(schema_version='1.0',states=serialized)),('tactical_events.json',dict(schema_version='1.0',events=[e.model_dump(mode='json',exclude_none=True) for e in tactics['events']])),('tactical_summary.json',summary),('transfer_network.json',export),('pre_event_evidence.json',dict(schema_version='1.0',items=evidence)),('tactical_profile.json',tactics['profile'])]:
        (output/name).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    make_plots(states,output,recording.sync.get('side_to_color'))
    compact=[]
    for s in states:
        compact.append(dict(frame=s.frame,timestamp=s.timestamp,centroid_distance_m=s.centroid_distance_m,
            teams={team:dict(width_m=getattr(s,team).shape.width_m,depth_m=getattr(s,team).shape.depth_m,compactness_m=getattr(s,team).shape.compactness_m,
                centroid_x_m=getattr(s,team).shape.centroid_x_m,centroid_y_m=getattr(s,team).shape.centroid_y_m,phase=getattr(s,team).phase,
                block=getattr(s,team).block,formation=getattr(s,team).formation.estimate,formation_reason=getattr(s,team).formation.reason,formation_fit_rmse_m=getattr(s,team).formation.fit_rmse_m,
                players=getattr(s,team).shape.available_players,goalkeeper_excluded=getattr(s,team).shape.goalkeeper_excluded) for team in ('left','right')},
            pressure=s.pressure.model_dump(mode='json'),ball_relative=s.ball_relative.model_dump(mode='json')))
    return dict(states=compact,events=[e.model_dump(mode='json',exclude_none=True) for e in tactics['events']],network=export,
        summary=summary,config=summary['config'])


def make_plots(states,output,side_to_color=None):
    """Plot actual shape/pressure observations, leaving unavailable intervals blank."""
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matchmind-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=[s.timestamp for s in states]
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,constrained_layout=True)
    side_to_color=side_to_color or {'0':1,'1':0}
    colors={team:('#3677c9' if side_to_color[str(i)]==0 else '#d64b4b') for i,team in enumerate(('left','right'))}
    for team,color in colors.items():
        for ax,metric,label in zip(axes,('width_m','depth_m','compactness_m'),('Outfield width (m)','Outfield depth (m)','RMS compactness (m)')):
            ax.plot(t,[getattr(getattr(s,team).shape,metric) for s in states],color=color,lw=.8,label=team);ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Recording time (s)');fig.suptitle('How does team shape change? Goalkeepers excluded; gaps mean unavailable.')
    fig.savefig(output/'tactical_shape.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True,constrained_layout=True)
    axes[0].plot(t,[s.pressure.nearest_defender_distance_m for s in states],lw=.8);axes[0].set_ylabel('Nearest defender (m)')
    axes[1].step(t,[s.pressure.defenders_within_local if s.pressure.defenders_within_local is not None else np.nan for s in states],where='post',label='Opponents within local radius')
    axes[1].fill_between(t,0,[s.pressure.defenders_within_local or 0 if s.pressure.sustained_press else 0 for s in states],alpha=.2,label='Sustained closing-pressure evidence')
    axes[1].legend();axes[1].set(xlabel='Recording time (s)',ylabel='Observed opponents');fig.suptitle('When did local pressure increase? Proximity alone does not establish pressing.')
    fig.savefig(output/'tactical_pressure.png',dpi=130);plt.close(fig)
