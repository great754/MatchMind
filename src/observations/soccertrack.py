"""Incremental adapter from the existing analytics Recording, without reprocessing.

Future providers can emit RecordingObservation directly. Production loading remains
unchanged; this boundary does not claim to solve CV calibration or identity.
"""
import numpy as np
from .schema import RecordingObservation,PlayerObservation,BallObservation,Provenance

def _point(value):
    return tuple(float(v) for v in value) if np.isfinite(value).all() else None

def observations(recording, recording_id):
    if recording.positions.shape!=(len(recording.ball),len(recording.player_ids),2) or len(recording.player_teams)!=len(recording.player_ids):
        raise ValueError('Unaligned SoccerTrack observations')
    source=','.join(map(str,recording.source_files)) or 'SoccerTrack Recording'
    base=Provenance(provider='soccertrack',source=source,quality='provided')
    ball_source=Provenance(provider='soccertrack',source=source,quality='reconstructed')
    for frame in range(len(recording.ball)):
        yield RecordingObservation(recording_id=recording_id,frame=frame,timestamp=frame/recording.fps,
            fps=recording.fps,source_offset_frames=recording.source_offset,half=getattr(recording,"half",None),provenance=base,
            players=[PlayerObservation(actor_id=str(actor) if actor is not None else None,team=team if team in ('left','right') else None,
                pitch=_point(recording.positions[frame,i]),position=base,
                identity=Provenance(provider='soccertrack',source=source,quality='unknown'))
                for i,(actor,team) in enumerate(zip(recording.player_ids,recording.player_teams))],
            ball=BallObservation(pitch=_point(recording.ball[frame]),provenance=ball_source))

def analytics_arrays(frames):
    """Compatibility proof: fixed actor ordering, NaN for unavailable coordinates."""
    frames=list(frames)
    if not frames: raise ValueError('Empty observations')
    ids=[p.actor_id for p in frames[0].players]
    if any([p.actor_id for p in f.players]!=ids for f in frames):raise ValueError('Unaligned actor ordering')
    if any(f.frame!=i or f.recording_id!=frames[0].recording_id or f.fps!=frames[0].fps or f.half!=frames[0].half or f.source_offset_frames!=frames[0].source_offset_frames for i,f in enumerate(frames)):
        raise ValueError('Unaligned recording clock')
    missing=(float('nan'),float('nan'))
    return np.asarray([[p.pitch or missing for p in f.players] for f in frames]),np.asarray([f.ball.pitch or missing for f in frames])
