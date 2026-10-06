"""GSR-first tactical positions keyed by the synchronized MOT-to-actor mapping."""
from dataclasses import dataclass
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.vision.coordinate_transform import PitchCoordinateTransformer
from src.vision.team_classifier import classify_teams


@dataclass
class TacticalTracking:
    rows: pd.DataFrame
    teams: dict
    goalkeepers: set
    comparison: dict
    sync: dict | None


def mapped_gsr_samples(mot, ids, pitch, sides, mapping, offset):
    """Join by actor ID in each frame, never by the compact array's slot number."""
    positions = np.full((len(mot),2),np.nan)
    labels = np.full(len(mot),-1,dtype=int)
    for track,entry in mapping.items():
        selected = np.flatnonzero(mot.player_id.to_numpy() == int(track))
        frames = mot.frame.to_numpy(dtype=int)[selected]+offset
        covered = (frames >= 0) & (frames < len(ids))
        selected,frames = selected[covered],frames[covered]
        actor = int(entry['player_id'])
        matches = ids[frames] == actor
        counts = matches.sum(axis=1)
        if np.any(counts > 1):
            raise ValueError(f'Duplicate GSR actor {actor} within a frame')
        present = counts == 1
        slots = matches[present].argmax(axis=1)
        positions[selected[present]] = pitch[frames[present],slots]
        labels[selected[present]] = sides[frames[present],slots]
    if np.any(~np.isin(labels,[-1,0,1])):
        raise ValueError('Invalid GSR team side')
    return positions,labels


def compare_positions(mot, projected, gsr):
    valid = np.isfinite(projected).all(axis=1) & np.isfinite(gsr).all(axis=1)
    errors = np.full(len(mot),np.nan)
    errors[valid] = np.linalg.norm(projected[valid]-gsr[valid],axis=1)
    def stats(mask):
        values = errors[mask & valid]
        return dict(samples=int(len(values)),mean_m=float(values.mean()) if len(values) else None,
                    median_m=float(np.median(values)) if len(values) else None,
                    p95_m=float(np.percentile(values,95)) if len(values) else None,
                    max_m=float(values.max()) if len(values) else None)
    result = stats(np.ones(len(mot),dtype=bool))
    result.update(frames_compared=int(mot.loc[valid,'frame'].nunique()),
                  detections_total=len(mot),comparison_coverage_fraction=float(valid.mean()),
                  reference='Synchronized GSR pitch positions',
                  interpretation='Image-to-pitch disagreement relative to GSR, not an independently measured absolute error. GSR itself is quantized and clip synchronization is inferred.',
                  per_player={str(pid):stats(mot.player_id.to_numpy()==pid) for pid in sorted(mot.player_id.unique())})
    frame_errors = pd.DataFrame({'frame':mot.frame.to_numpy(),'error_m':errors}).groupby('frame').error_m.agg(['count','mean','median','max']).reset_index()
    return result,frame_errors


def prepare_tactical_tracking(paths, mot, fps):
    if mot.duplicated(['frame','player_id']).any():
        raise ValueError('Duplicate MOT actor in a frame')
    transformer = PitchCoordinateTransformer(paths.keypoints)
    projected = transformer.transform_points(mot[['foot_x','foot_y']].to_numpy())
    sync_path = paths.root/'mot'/f'{paths.match_id}_sync.json'
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else None
    gsr = np.full_like(projected,np.nan)
    sides = np.full(len(mot),-1,dtype=int)
    gsr_source = None
    if sync:
        if sync['fps'] != fps or str(sync['match_id']) != paths.match_id:
            raise ValueError('Synchronization metadata does not match video')
        if set(map(int,sync['players'])) != set(mot.player_id.astype(int)):
            raise ValueError('Sync mapping does not cover MOT players')
        if set(sync['side_to_color']) != {'0','1'} or set(sync['side_to_color'].values()) != {0,1}:
            raise ValueError('Team sides must map to distinct blue/red colors')
        if sync['half'] not in (1,2) or int(sync['clip_start_frame_zero_based']) < 0:
            raise ValueError('Invalid half or source frame offset')
        actors = [int(p['player_id']) for p in sync['players'].values()]
        if len(set(actors)) != len(actors):
            raise ValueError('Sync mapping must be one-to-one')
        suffix = '1st' if sync['half'] == 1 else '2nd'
        compact = paths.root/'gsr'/paths.match_id/f'{suffix}_compact.npz'
        raw = paths.root/'gsr'/paths.match_id/f'{paths.match_id}_{suffix}.json'
        if compact.exists():
            with np.load(compact) as data:
                gsr,sides = mapped_gsr_samples(mot,data['ids'],data['pitch'],data['sides'],
                                             sync['players'],int(sync['clip_start_frame_zero_based']))
            gsr_source = str(compact)
        elif raw.exists():
            # Stream raw GSR if the compact cache has not been built. Only keep
            # detections from this clip; do not load a multi-GB JSON into memory.
            from src.vision.align_clip import read_annotations
            lookup = {(int(r.frame),int(sync['players'][str(int(r.player_id))]['player_id'])):i
                      for i,r in enumerate(mot.itertuples())}
            offset = int(sync['clip_start_frame_zero_based'])
            for a in read_annotations(raw):
                if a.get('category_id') not in (1,2):
                    continue
                frame = int(a['image_id'])%1000000-1-offset
                attrs = a['attributes']
                index = lookup.get((frame,int(attrs['player_id'])))
                if index is None:
                    continue
                if sides[index] != -1:
                    raise ValueError('Duplicate actor in raw GSR')
                b = a['bbox_pitch']
                gsr[index] = [b['x_bottom_middle']+52.5,b['y_bottom_middle']+34]
                sides[index] = {'left':0,'right':1}[attrs['team']]
            gsr_source = str(raw)
    elif any((paths.root/'gsr'/paths.match_id).glob('*_compact.npz')) or any((paths.root/'gsr'/paths.match_id).glob(f'{paths.match_id}_*st.json')) or any((paths.root/'gsr'/paths.match_id).glob(f'{paths.match_id}_2nd.json')):
        raise ValueError('GSR requires the sync mapping; cannot guess actor IDs or clip offset')
    use_gsr = np.isfinite(gsr).all(axis=1)
    positions = projected.copy()
    positions[use_gsr] = gsr[use_gsr]
    source = np.where(use_gsr,'gsr',np.where(np.isfinite(projected).all(axis=1),'pitch_transform','unavailable'))
    rows = mot.copy()
    rows['pitch_x'],rows['pitch_y'] = positions[:,0],positions[:,1]
    rows['position_source'] = source
    teams = {}
    team_sources = {}
    for track in sorted(mot.player_id.unique()):
        observed_sides = np.unique(sides[mot.player_id.to_numpy()==track])
        observed_sides = observed_sides[observed_sides >= 0]
        if len(observed_sides) > 1:
            raise ValueError(f'GSR team label changed for MOT track {track}')
        if len(observed_sides):
            side = int(observed_sides[0])
            if side != int(sync['players'][str(track)]['team']):
                raise ValueError(f'GSR and sync team labels disagree for track {track}')
            teams[int(track)] = int(sync['side_to_color'][str(side)])
            team_sources[str(track)] = 'gsr_team_label'
        elif sync:
            side = int(sync['players'][str(track)]['team'])
            teams[int(track)] = int(sync['side_to_color'][str(side)])
            team_sources[str(track)] = 'sync_gsr_team_label'
    if len(teams) != mot.player_id.nunique():
        teams.update(classify_teams(paths.video,rows))
        team_sources.update({str(t):'jersey_and_goalkeeper_geometry' for t in teams})
    goalkeepers = set()
    role_source = 'metadata'
    metadata = paths.raw/f'{paths.match_id}_tracker_box_metadata.xml'
    if metadata.exists() and sync:
        keeper_actors = {int(p.attrib['id']) for p in ET.parse(metadata).getroot().findall('.//players/player')
                         if p.attrib.get('position') == 'GK'}
        goalkeepers = {int(t) for t,p in sync['players'].items() if int(p['player_id']) in keeper_actors}
    if not goalkeepers:
        from src.vision.team_classifier import infer_goalkeepers
        goalkeepers = set(infer_goalkeepers(rows))
        role_source = 'persistent_depth_heuristic'
    rows['is_goalkeeper'] = rows.player_id.isin(goalkeepers)
    report,frame_errors = compare_positions(mot,projected,gsr)
    report.update(selected_primary='gsr' if use_gsr.any() else 'pitch_transform',gsr_source=gsr_source,
                  position_source_counts=pd.Series(source).value_counts().to_dict(),
                  team_sources=team_sources,teams={str(t):v for t,v in teams.items()},
                  goalkeepers=[dict(mot_id=t,player_id=sync['players'][str(t)]['player_id'] if sync else None,
                                    color='blue' if teams[t]==0 else 'red',role_source=role_source)
                               for t in sorted(goalkeepers)])
    # Keep exact per-frame disagreements available for visual audit.
    rows['transform_gsr_error_m'] = np.linalg.norm(projected-gsr,axis=1)
    return TacticalTracking(rows,teams,goalkeepers,report,sync),frame_errors
