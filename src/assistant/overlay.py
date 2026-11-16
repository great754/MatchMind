"""Per-frame player marks from explicit source boxes; no off-camera ghosts."""
import os,time
from pathlib import Path
import cv2
from src.highlights.render import run,file_hash,probe

def annotated_video(directory,segments,catalog,ref,label,cancel,review=False):
    started=time.monotonic();directory=Path(directory);movie=directory/'highlight.mp4';cap=cv2.VideoCapture(str(movie));temp=directory/'overlay_silent.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720));drawn=0;hidden=0;before=0;positions=[];trail=[];last_segment=None
    info=probe(movie);source_boxes=catalog.boxes.get(ref,{})
    # Original dimensions come from the authorized source's verified MOT boxes/video.
    source=catalog.video_dimensions
    scale=min(1280/source[0],720/source[1]);padx=(1280-source[0]*scale)/2;pady=(720-source[1]*scale)/2
    try:
        if not cap.isOpened() or not writer.isOpened():raise ValueError('Overlay renderer unavailable')
        i=0
        while True:
            ok,image=cap.read()
            if not ok:break
            if cancel.is_set():raise ValueError('cancelled')
            t=i/30;segment=next((s for s in segments if s['output_start']-1e-8<=t<s['output_end']-1e-8),None);box=None;frame=None
            if segment:
                frame=min(catalog.frames-1,int((segment['start']+t-segment['output_start'])*catalog.fps+1e-7));box=source_boxes.get(frame)
            if segment is not last_segment:trail=[];last_segment=segment
            point_source=catalog.image_point(ref,frame) if segment and hasattr(catalog,'image_point') and not box else None
            occluded=False
            if box:
                x,y,w,h=box
                for other_ref,boxes in catalog.boxes.items():
                    other=boxes.get(frame)
                    if other_ref==ref or other is None:continue
                    overlap=max(0,min(x+w,other[0]+other[2])-max(x,other[0]))*max(0,min(y+h,other[1]+other[3])-max(y,other[1]))
                    if overlap/(w*h)>.35:occluded=True;break
                if x<0 or y<0 or x+w>source[0] or y+h>source[1]:occluded=True
            if point_source is not None:
                px,py=map(float,point_source)
                if not (0<=px<source[0] and 0<=py<source[1]):point_source=None
                elif getattr(catalog,'point_coords',None) is not None:
                    others=catalog.point_coords[frame];distances=((others-point_source)**2).sum(axis=1)
                    # Ambiguous overlapping feet do not grant a guessed marker.
                    if ((distances>0)&(distances<18**2)).any() or (distances==0).sum()>1:point_source=None
            if box and not occluded:
                x,y,w,h=box;x1,y1=round(x*scale+padx),round(y*scale+pady);x2,y2=round((x+w)*scale+padx),round((y+h)*scale+pady)
                cv2.rectangle(image,(x1,y1),(x2,y2),(90,255,220),2);cv2.putText(image,label[:50],(max(0,x1),max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,(90,255,220),2)
                point=((x1+x2)//2,y2);trail.append(point);trail=trail[-12:]
                for a,b in zip(trail,trail[1:]):cv2.line(image,a,b,(90,255,220),1)
                drawn+=1;before+=int(segment is not None and segment['start']+t-segment['output_start']<segment['event_start']);positions.append({'output_frame':i,'source_frame':frame,'box':[x1,y1,x2-x1,y2-y1]})
            elif point_source is not None:
                px,py=point_source;x=round(px*scale+padx);y=round(py*scale+pady)
                lead=segment['start']+t-segment['output_start']<segment['event_start']
                colour=(90,255,220)
                cv2.ellipse(image,(x,y),(12,5),0,0,360,colour,2)
                cv2.arrowedLine(image,(x,max(18,y-46)),(x,y-9),colour,2,tipLength=.25)
                cv2.putText(image,label[:50],(max(4,min(1120,x-35)),max(18,y-53)),cv2.FONT_HERSHEY_SIMPLEX,.55,colour,2)
                drawn+=1;before+=int(lead);positions.append({'output_frame':i,'source_frame':frame,'image_foot':[x,y]})
            else:hidden+=1;trail=[]
            if review:
                cv2.rectangle(image,(0,0),(1280,42),(22,35,50),-1)
                cv2.putText(image,'POSSIBLE TRANSFER - NEEDS REVIEW',(18,29),cv2.FONT_HERSHEY_SIMPLEX,.72,(220,235,255),2)
            writer.write(image);i+=1
    finally:cap.release();writer.release()
    if ref is not None and not drawn:
        from src.highlights.render import HighlightError
        temp.unlink(missing_ok=True)
        raise HighlightError('player_marker_unavailable')
    final=directory/'overlay_final.mp4';run(['ffmpeg','-nostdin','-v','error','-y','-i',str(temp),'-i',str(movie),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-threads','2','-c:a','copy','-movflags','+faststart',str(final)],cancel,120);os.replace(final,movie);temp.unlink(missing_ok=True)
    run(['ffmpeg','-nostdin','-v','error','-y','-i',str(movie),'-frames:v','1',str(directory/'thumbnail.jpg')],cancel,20)
    return {'version':'player-focus-overlay-v3','player_ref':ref,'label':label,'review_banner':review,'drawn_frames':drawn,'hidden_frames':hidden,'pre_event_drawn_frames':before,'samples':positions,'provenance':'Frame-specific source boxes or hash-verified image foot points; overlap/truncation/loss suppresses marker; no propagation across identities or cuts.','seconds':time.monotonic()-started,'sha256':file_hash(movie)}
