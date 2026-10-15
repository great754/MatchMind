"""Load frozen per-recording exports for diagnostics, without changing detections."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analytics.config import AnalyticsConfig
from src.analytics.data import load_recording
from src.analytics.motion import Motion
from src.vision.data_loading import DatasetPaths


def load_evaluation_recording(directory,data_root=Path('data/soccertrack')):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    recording=load_recording(DatasetPaths(str(summary['match_id']),Path(data_root)),summary['scope'],'gsr',summary['half'])
    for source in summary['source_files']:
        p=Path(source['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=source['sha256']:raise ValueError(f'Frozen source changed: {p.name}')
    b=pd.read_csv(directory/'ball_frames.csv')
    if len(b)!=len(recording.ball):raise ValueError('Artifact frame count does not match recording')
    pair=lambda x,y:b[[x,y]].to_numpy(float)
    ball=Motion(pair('raw_x_m','raw_y_m'),pair('x_m','y_m'),pair('vx_mps','vy_mps'),
        b.instantaneous_speed_mps.to_numpy(),b.smoothed_speed_mps.to_numpy(),b.distance_m.to_numpy(),b.step_distance_m.to_numpy(),
        b.observed.to_numpy(bool),b.position_usable.to_numpy(bool),b.raw_speed_flag.to_numpy(bool),b.rejected_speed_flag.to_numpy(bool),b.segment_id.to_numpy(int))
    result=dict(ball=ball,possession=pd.read_csv(directory/'possession.csv'),shots=pd.read_csv(directory/'shots.csv'),
        shot_matches=pd.read_csv(directory/'shot_validation.csv'),passes=pd.read_csv(directory/'transfers.csv'),pass_matches=pd.read_csv(directory/'pass_validation.csv'))
    return recording,result,AnalyticsConfig(**summary['config'])
