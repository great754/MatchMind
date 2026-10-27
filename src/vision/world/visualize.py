"""Local debug visualization; no production frontend or football inference."""
import html,json,os
from pathlib import Path
import cv2
import numpy as np
from src.vision.pitch import PITCH_LENGTH,PITCH_WIDTH

def compact(state):
    return dict(frame=state.frame,timestamp=state.timestamp,camera=state.camera.model_dump(),
        players=[dict(id=e.entity_id,track=e.track_id,team=e.team,visibility=e.visibility,bbox=e.image_bbox,pitch=e.current_pitch,
            confidence=e.identity_confidence,last_pitch=e.last_reliable_pitch,last_seen=e.last_seen,
            hypotheses=[h.model_dump() for h in e.hypotheses]) for e in state.entities],
        unknown=[dict(track=u.observation.track_id,bbox=u.observation.bbox,pitch=u.observation.pitch,team=u.observation.team,status=u.status,
            hypotheses=[dict(id=h.entity_id,score=h.support_score) for h in u.hypotheses]) for u in state.unassigned])

def annotated_frame(frame,state):
    image=frame.copy()
    linked={e.track_id:e for e in state.entities if e.track_id}
    for t in [u.observation for u in state.unassigned]:
        x,y,w,h=map(int,t.bbox);cv2.rectangle(image,(x,y),(x+w,y+h),(180,180,180),2)
        cv2.putText(image,f'{t.track_id} {next(u.status for u in state.unassigned if u.observation.track_id==t.track_id)} identity',(x,max(12,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(220,220,220),1)
    for e in linked.values():
        x,y,w,h=map(int,e.image_bbox);color=(90,140,255) if e.team=='team_a' else (90,230,230) if e.team=='team_b' else (200,200,200)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        score=f'{e.identity_confidence:.2f}' if e.identity_confidence is not None else '?'
        text=f'{e.last_track_id} {e.entity_id} {e.team or "team?"} support:{score} loc:{"calibrated" if e.current_pitch else "UNKNOWN"}'
        cv2.putText(image,text,(max(0,x),max(14,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        cv2.circle(image,tuple(map(int,e.image_foot)),3,color,-1)
    pitch=np.full((image.shape[0],600,3),(35,55,35),np.uint8);sx=540/PITCH_LENGTH;sy=350/PITCH_WIDTH
    cv2.rectangle(pitch,(30,80),(570,430),(160,190,160),1);cv2.line(pitch,(300,80),(300,430),(160,190,160),1);cv2.circle(pitch,(300,255),int(9.15*sx),(160,190,160),1)
    cv2.putText(pitch,f'{state.timestamp:.2f}s  frame {state.frame}',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
    cv2.putText(pitch,'Metric positions: '+('calibration supplied' if state.camera.calibration_valid else 'UNKNOWN - no calibration'),(15,50),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,220,130),1)
    if state.camera.visible_pitch_region:
        polygon=np.array([[30+p[0]*sx,80+p[1]*sy] for p in state.camera.visible_pitch_region],np.int32);cv2.polylines(pitch,[polygon],True,(80,200,230),2)
    unlocalized=0
    for e in state.entities:
        current=e.current_pitch if e.visibility=='visible' else None
        p=current or e.last_reliable_pitch
        if p:
            q=(int(30+p[0]*sx),int(80+p[1]*sy));color=(240,240,240) if current else (100,100,100)
            cv2.circle(pitch,q,5,color,-1 if current else 1)
            cv2.putText(pitch,e.entity_id if current else f'last {state.timestamp-e.last_seen:.1f}s',q,cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
        if e.visibility=='visible' and current is None:unlocalized+=1
    rows=[f'Visible unlocalized: {unlocalized}',f'Off camera/missing: {sum(e.visibility!="visible" for e in state.entities)}',f'Unknown/ambiguous tracks: {len(state.unassigned)}','No invented current pitch markers.','Gray rings mean LAST KNOWN, not current.']
    for i,text in enumerate(rows):cv2.putText(pitch,text,(15,470+i*22),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
    return np.concatenate([image,pitch],axis=1)

def write_debug(output,video,frames,summary):
    output=Path(output);relative=os.path.relpath(Path(video).resolve(),output.resolve())
    payload=json.dumps(frames,allow_nan=False,separators=(',',':')).replace('<',r'\u003c')
    template=Path(__file__).with_name('debug.html').read_text()
    page=template.replace('__VIDEO__',html.escape(relative,quote=True)).replace('__DATA__',payload).replace('__SUMMARY__',html.escape(json.dumps(summary,indent=2)))
    (output/'debug.html').write_text(page)


def replay_debug(video,output):
    """Refresh debug presentation from saved typed states; no detector re-run."""
    import hashlib
    output=Path(output);summary=json.loads((output/'diagnostics.json').read_text())
    if hashlib.sha256(Path(video).read_bytes()).hexdigest()!=summary['input']['sha256']:raise ValueError('Video does not match this world-state experiment')
    from .schema import WorldState
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary=output/'annotated.refresh.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width+600,height));frames=[]
    try:
        with (output/'world.jsonl').open() as source:
            for i,line in enumerate(source):
                state=WorldState.model_validate_json(line);ok,image=cap.read()
                if not ok or state.frame!=i or abs(state.timestamp-i/fps)>1e-6:raise ValueError('World/video clock mismatch')
                frames.append(compact(state));writer.write(annotated_frame(image,state))
    finally:writer.release();cap.release()
    if len(frames)!=summary['input']['frames']:temporary.unlink(missing_ok=True);raise ValueError('Incomplete visualization')
    temporary.replace(output/'annotated.mp4');write_debug(output,video,frames,summary)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Replay stored world-state diagnostics without neural inference.')
    parser.add_argument('--video',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();replay_debug(args.video,args.output)
