"""Real YOLOX proposal/NMS audit; no changes to detector confidence or suppression."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from src.vision.world.detection import YoloXDetector
from src.vision.world.review import digest

def audit(video,model,output):
    detector=YoloXDetector(model);cap=cv2.VideoCapture(str(video));rows=[];counts={str(t):0 for t in (.3,.5,.65)};started=time.perf_counter();expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in range(expected):
        ok,image=cap.read()
        if not ok:raise ValueError('Incomplete video')
        found=detector.detect(image);proposals=detector.last_proposals;boxes=[b for b,s in proposals];scores=[s for b,s in proposals]
        totals={str(t):len(np.asarray(cv2.dnn.NMSBoxes(boxes,scores,.4,t)).reshape(-1)) for t in (.3,.5,.65)}
        for key,value in totals.items():counts[key]+=value
        rows.append({'frame':index,'detections':[d.model_dump() for d in found],'proposals':proposals,'raw_proposals':len(proposals),'nms_counts':totals,'suppressed_at_0_5':len(proposals)-totals['0.5']})
    cap.release();result={'source_sha256':digest(video),'frames':rows,'wall_seconds':time.perf_counter()-started,'nms_sensitivity_totals':counts,'detector_threshold':.4,'selected_nms_threshold':.5,'labels':'COCO person candidates, not football role ground truth'}
    Path(output).write_text(json.dumps(result,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.video,a.model,a.output)
