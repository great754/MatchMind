"""Normalize recording time, IDs and coordinates before any event logic."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from src.vision.coordinate_transform import PitchCoordinateTransformer
from src.vision.data_loading import DatasetPaths, load_mot


@dataclass
class Recording:
    positions: np.ndarray
    ball: np.ndarray
    ball_status: np.ndarray
    player_ids: np.ndarray
    player_teams: list
    mot_ids: dict
    events: list
    fps: float
    source_offset: int
    half: int
    scope: str
    player_source: str
    source_files: list
    sync: dict


def load_recording(paths=None, scope='clip', player_source='gsr', half=None):
    paths = paths or DatasetPaths()
    if scope not in ('clip','half') or player_source not in ('gsr','mot'):
        raise ValueError('scope must be clip/half and player_source must be gsr/mot')
    sync_path = paths.sync
    sync = json.loads(sync_path.read_text())
    bas_path = paths.bas
    bas = json.loads(bas_path.read_text())
    fps = float(sync['fps'])
    if bas['fps'] != fps or 'first frame' not in bas.get('clock',''):
        raise ValueError('BAS must use the synchronized released-half clock')
    if str(sync['match_id']) != str(paths.match_id):
        raise ValueError('Synchronization match ID does not match selected recording')
    half = int(sync['half']) if half is None else half
    if half not in (1, 2):
        raise ValueError('half must be 1 or 2')
    if scope == 'clip' and half != int(sync['half']):
        raise ValueError('Requested half does not contain the synchronized clip')
    suffix = '1st' if half == 1 else '2nd'
    ball_path = paths.ball(half)
    with np.load(ball_path) as raw:
        frames = raw['frame']
        if not np.array_equal(frames,np.arange(1,len(frames)+1)):
            raise ValueError('Expected consecutive one-based ball frames')
        ball = np.column_stack((raw['x']+52.5,raw['y']+34)).astype(float)
        status = raw['status'].copy()
    files = [str(sync_path),str(bas_path),str(ball_path)]
    offset = int(sync['clip_start_frame_zero_based']) if scope == 'clip' else 0
    count = len(ball)
    if scope == 'clip':
        cap = cv2.VideoCapture(str(paths.video))
        try:
            if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) != fps:
                raise ValueError('Clip must be readable and match metadata FPS')
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if offset < 0 or offset+count > len(ball):
            raise ValueError('Ball timeline does not cover clip')
    ball = ball[offset:offset+count].copy()
    status = status[offset:offset+count]
    ball[status == 0] = np.nan
    mot_ids = {int(p['player_id']):int(t) for t,p in sync['players'].items()}
    if player_source == 'gsr':
        compact = paths.gsr_compact(half)
        if not compact.exists():
            raise FileNotFoundError(f'{compact} is required for GSR analytics. Run python -m src.vision.align_clip after downloading GSR, or use --player-source mot for the clip.')
        with np.load(compact) as gsr:
            coords = gsr['pitch'][offset:offset+count].astype(float)
            ids = gsr['ids'][offset:offset+count]
            sides = gsr['sides'][offset:offset+count]
        if len(coords) != count:
            raise ValueError('GSR timeline does not cover recording')
        player_ids = np.unique(ids[ids > 0])
        positions = np.full((count,len(player_ids),2),np.nan)
        teams = []
        for slot,pid in enumerate(player_ids):
            frame,track = np.where(ids == pid)
            if len(np.unique(frame)) != len(frame):
                raise ValueError(f'Duplicate GSR actor {pid} within a frame')
            positions[frame,slot] = coords[frame,track]
            side = np.unique(sides[frame,track])
            if len(side) != 1 or side[0] not in (0,1):
                raise ValueError(f'Inconsistent team for actor {pid}')
            teams.append('left' if side[0] == 0 else 'right')
        files.append(str(compact))
    else:
        if scope != 'clip':
            raise ValueError('MOT coordinates cover only the four-minute clip; half scope requires GSR')
        mot = load_mot(paths.mot)
        if mot.frame.min() != 0 or mot.frame.max() != count-1 or mot.duplicated(['frame','player_id']).any():
            raise ValueError('MOT frame coverage/IDs do not match clip')
        transform = PitchCoordinateTransformer(paths.keypoints)
        projected = transform.transform_points(mot[['foot_x','foot_y']].to_numpy())
        player_ids = np.array(sorted(mot_ids))
        slots = {pid:i for i,pid in enumerate(player_ids)}
        positions = np.full((count,len(player_ids),2),np.nan)
        for i,r in enumerate(mot.itertuples()):
            actor = int(sync['players'][str(int(r.player_id))]['player_id'])
            positions[int(r.frame),slots[actor]] = projected[i]
        teams = ['left' if sync['players'][str(mot_ids[int(pid)])]['team'] == 0 else 'right' for pid in player_ids]
        files += [str(paths.mot),str(paths.keypoints)]
    events = []
    for eid,e in enumerate(bas['actions']):
        event_half = int(e['gameTime'].split('-')[0].strip())
        frame = int(e['frame'])-1-offset
        if event_half == half and 0 <= frame < count:
            events.append({'annotation_id':eid,'frame':frame,'seconds':frame/fps,
                           'half_video_seconds':float(e['position'])/1000,
                           'label':e['label'],'player_id':int(e['player_id']) if e.get('player_id') is not None else None, 'team':e.get('team')})
    events.sort(key=lambda e:e['frame'])
    recording = Recording(positions,ball,status,player_ids,teams,mot_ids,events,fps,offset,
                     half,scope,player_source,files,sync)
    recording.video_path = paths.video
    return recording
