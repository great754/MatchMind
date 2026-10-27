"""Compare immutable camera reviews and render measured success/failure examples.

No inference, Azure writes, labels, or reference accuracy are fabricated.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from src.serving.cv import CVReview
from src.vision.world.diagnosis import report
from scripts.verify_cv_review import verify

def evaluate(before,after,poses,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summaries={name:report(directory,poses,output/name) for name,directory in [('before',before),('after',after)]}
    baseline=CVReview(before);current=CVReview(after)
    if baseline.manifest['video_sha256']!=current.manifest['video_sha256']:raise ValueError('Different source recordings')
    outcomes={name:[json.loads(l) for l in (output/name/'frame_outcomes.jsonl').read_text().splitlines()] for name in summaries}
    removed=[{'frame':a['frame'],'blockers':b['tracking_blockers'],'motion':b['flow'].get('projective_motion')} for a,b in zip(outcomes['before'],outcomes['after'],strict=True) if a['available'] and not b['available']]
    added=[b['frame'] for a,b in zip(outcomes['before'],outcomes['after'],strict=True) if not a['available'] and b['available']]
    comparison={'removed_available_frames':removed,'new_available_frames':added,'independent_accuracy':None,'interpretation':'Stricter measured motion gates; lower coverage does not establish higher absolute accuracy.'}
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    tracked=[r['flow']['projective_motion'] for r in outcomes['after'] if r['source']=='camera_propagated']
    corrections=[{'frame':r['frame'],'p90_m':r['flow']['relocalization_correction_p90_m']} for r in outcomes['after'] if 'relocalization_correction_p90_m' in r['flow']]
    stats={'tracking_heldout_p90_small_px':{'min':min(r['heldout_p90_small_px'] for r in tracked),'max':max(r['heldout_p90_small_px'] for r in tracked)} if tracked else None,
           'corrections':corrections,'interpretation':'Dependent image/pose agreement, not independent metric accuracy.'}
    (output/'tracking_quality.json').write_text(json.dumps(stats,indent=2))
    verify(after,output/'semantic_integrity.json')
    # Select genuine direct, tracked and unavailable examples from the source.
    selected=[]
    for kind in ['direct_calibrated','camera_propagated','unknown']:
        matches=[r for r in outcomes['after'] if r['source']==kind]
        if matches:selected.extend([matches[0]['frame'],matches[len(matches)//2]['frame']])
    cap=cv2.VideoCapture(str(current.video));tiles=[]
    for index in sorted(set(selected)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
        if not ok:raise ValueError('Cannot decode example')
        state=current.frame(index/current.manifest['fps']);camera=state.camera
        image=cv2.resize(image,(480,480));pitch=np.full((480,360,3),(35,50,30),np.uint8)
        def p(point):return (int(12+point[0]*3.2),int(35+point[1]*5.8))
        cv2.rectangle(pitch,p((0,0)),p((105,68)),(170,190,170),1);cv2.line(pitch,p((52.5,0)),p((52.5,68)),(170,190,170),1)
        if camera.visible_pitch_region:
            cv2.polylines(pitch,[np.array([p(v) for v in camera.visible_pitch_region])],True,(220,210,100),2)
        for e in state.entities:
            if e.current_pitch:cv2.circle(pitch,p(e.current_pitch),4,(130,160,250),1 if e.position_source=='camera_propagated' else -1)
        info=camera.motion_evidence['localization'];cv2.putText(image,f'Frame {index} / {state.timestamp:.3f}s / {info["status"]}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        tile=np.hstack([image,pitch]);cv2.putText(tile,info['outcome'],(8,466),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);tiles.append(tile)
    cap.release();cv2.imwrite(str(output/'camera-examples.jpg'),np.vstack(tiles))
    result={'before':summaries['before'],'after':summaries['after'],'comparison':comparison,'quality':stats}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--poses',required=True);p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.before,a.after,a.poses,a.output)
