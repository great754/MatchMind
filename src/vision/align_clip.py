import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.vision.data_loading import DatasetPaths,load_mot

def read_annotations(path):
    with open(path) as f:
        for line in f:
            if '"annotations": [' in line: break
        obj=[]
        for line in f:
            if line.startswith('    ],'): break
            obj.append(line)
            if line.startswith('        }'):
                yield json.loads(''.join(obj).strip().rstrip(','))
                obj=[]


def main():
    p=DatasetPaths();mot=load_mot(p.mot)
    target=mot[mot.frame==0][['foot_x','foot_y']].to_numpy()
    results=[]
    for half,name,count in ((1,'1st',71850),(2,'2nd',73425)):
        cache=Path(f'data/soccertrack/gsr/118575/{name}_compact.npz')
        if cache.exists():
            z=np.load(cache);coords=z['coords'];pitch=z['pitch'];ids=z['ids'];sides=z['sides']
        else:
            coords=np.full((count,40,2),np.nan,dtype=np.float32);pitch=coords.copy()
            ids=np.zeros((count,40),dtype=np.int32);sides=np.full((count,40),-1,dtype=np.int8)
            for a in read_annotations(p.root/'gsr'/'118575'/f'118575_{name}.json'):
                if a.get('category_id') not in (1,2):continue
                n=int(a['image_id'])%1000000-1;t=int(a['track_id'])-1
                if t>=40:raise ValueError(t)
                b=a['bbox_image'];q=a['bbox_pitch'];attrs=a['attributes']
                coords[n,t]=[b['x']+b['w']/2,b['y']+b['h']]
                pitch[n,t]=[q['x_bottom_middle']+52.5,q['y_bottom_middle']+34]
                ids[n,t]=int(attrs['player_id']);sides[n,t]=0 if attrs['team']=='left' else 1
            np.savez_compressed(cache,coords=coords,pitch=pitch,ids=ids,sides=sides)
        score=np.empty(count)
        for start in range(0,count,500):
            d=np.linalg.norm(coords[start:start+500,:,None,:]-target[None,None,:,:],axis=-1)
            d=np.where(np.isfinite(d),d,1e6)
            score[start:start+500]=np.min(d,axis=1).mean(axis=1)
        candidates=np.argsort(score)[:20]
        print('half',half,'candidates',[(int(n),float(score[n])) for n in candidates[:10]],flush=True)
        for n in candidates:
            errors=[]
            for off in (0,250,1000,3000,5999):
                if n+off>=count:break
                dst=mot[mot.frame==off][['foot_x','foot_y']].to_numpy()
                valid=np.isfinite(coords[n+off]).all(axis=1)
                c=coords[n+off,valid];d=np.linalg.norm(c[:,None]-dst[None],axis=-1)
                r,s=linear_sum_assignment(d);errors.append(float(d[r,s].mean()))
            if len(errors)==5:results.append((float(np.mean(errors)),half,int(n),errors))
    best_error, best_half, coarse_offset, _ = min(results)
    name = '1st' if best_half == 1 else '2nd'
    z = np.load(p.root/'gsr'/p.match_id/f'{name}_compact.npz')
    coords = z['coords']
    checks = list(range(0, 6000, 300)) + [5999]
    refined = []
    for offset in range(max(0, coarse_offset-50), coarse_offset+51):
        if offset+5999 >= len(coords):
            continue
        errors = []
        for off in checks:
            q = coords[offset+off]
            q = q[np.isfinite(q).all(axis=1)]
            dst = mot[mot.frame == off][['foot_x','foot_y']].to_numpy()
            d = np.linalg.norm(q[:,None]-dst[None], axis=-1)
            r,s = linear_sum_assignment(d)
            errors.append(float(d[r,s].mean()))
        refined.append((float(np.mean(errors)), offset))
    error, offset = min(refined)
    votes = defaultdict(Counter)
    sidevotes = defaultdict(Counter)
    for off in checks:
        rows = mot[mot.frame == off]
        q = coords[offset+off]
        valid = np.flatnonzero(np.isfinite(q).all(axis=1))
        d = np.linalg.norm(q[valid,None]-rows[['foot_x','foot_y']].to_numpy()[None], axis=-1)
        r,s = linear_sum_assignment(d)
        for i,j in zip(r,s):
            track = int(rows.iloc[j].player_id)
            slot = valid[i]
            votes[track][int(z['ids'][offset+off,slot])] += 1
            sidevotes[track][int(z['sides'][offset+off,slot])] += 1
    # Jersey-derived colors are needed only to name the authoritative team sides.
    from src.vision.team_classifier import classify_teams
    colors = classify_teams(p.video, mot)
    side_colors = defaultdict(Counter)
    for track,v in sidevotes.items():
        side_colors[v.most_common(1)[0][0]][colors[track]] += 1
    players = {str(t): {'player_id':v.most_common(1)[0][0],
                       'team':sidevotes[t].most_common(1)[0][0],
                       'votes':v.most_common(1)[0][1], 'samples':sum(v.values())}
               for t,v in votes.items()}
    report = {'match_id':p.match_id, 'half':best_half, 'clip_start_frame_zero_based':offset,
              'fps':25, 'method':'Player bbox foot-position matching over 21 checkpoints; inferred offset, approximately one-frame precision.',
              'mean_assignment_error_px':error, 'players':players,
              'side_to_color':{str(s):v.most_common(1)[0][0] for s,v in side_colors.items()}}
    output = p.root/'mot'/f'{p.match_id}_sync.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved inferred clip alignment to {output}: half {best_half}, offset {offset}, mean error {error:.2f}px', flush=True)


if __name__ == '__main__':
    main()
