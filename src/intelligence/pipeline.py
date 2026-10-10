"""Write intelligence alongside existing exports; default requires no network."""
from dataclasses import asdict
import json
import numpy as np
from .events import build_events
from .context import ContextBuilder
from .config import IntelligenceConfig
from .labels import recording_labels
from .commentary.service import generate_commentary,select_moments

def write_intelligence(recording,result,output,config: IntelligenceConfig | None = None,provider=None,tactics_config=None,analytics_config=None) -> dict:
    config=config or IntelligenceConfig()
    events=build_events(recording,result,config)
    tactical_payload=None;tactical_states=[]
    if tactics_config is not False:
        from src.analytics.config import AnalyticsConfig
        from src.tactics.engine import analyze_tactics
        from src.tactics.output import write_tactics
        tactics=analyze_tactics(recording,result,analytics_config or AnalyticsConfig(fps=recording.fps),tactics_config,events)
        tactical_states=tactics['states']
        events=sorted(events+tactics['events'],key=lambda e:(e.frame,e.event_id))
        tactical_payload=write_tactics(recording,result,tactics,events,output)
    builder=ContextBuilder(recording,result,events,tactical_states)
    duration=(len(recording.ball)-1)/recording.fps
    times=set(float(t) for t in np.arange(0,duration,config.context_snapshot_seconds));times.add(duration)
    times.update(max(e.timestamp for e in group) for group in select_moments(events,config))
    contexts=[builder.at(t,w) for t in sorted(times) for w in (15,30,60)]
    commentary=generate_commentary(events,builder,config,provider,output/'commentary_cache')
    event_export=dict(schema_version='1.0',fps=recording.fps,source_offset_frames=recording.source_offset,
        timestamp_origin='recording-local; half-video timestamp = timestamp + source_offset_frames/fps',
        config=asdict(config),player_labels={str(actor):dict(display=label.display,tracker_id=label.tracker_id,id_namespace='mot' if actor in recording.mot_ids else 'gsr_actor') for actor,label in recording_labels(recording).items()},
        events=[e.model_dump(mode='json',exclude_none=True) for e in events])
    for filename,payload in [('events.json',event_export),('rolling_context.json',dict(schema_version='1.0',snapshots=contexts)),('commentary.json',commentary)]:
        (output/filename).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return dict(events=event_export,commentary=commentary,tactics=tactical_payload)
