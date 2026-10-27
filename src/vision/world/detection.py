"""Optional YOLOX COCO person inference on the existing OpenCV CPU backend.

Model: OpenCV Zoo YOLOX-S, Apache-2.0. No neural weights auto-download here.
COCO person is a candidate person, not proof of football role or team.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from .schema import Detection,Config

MODEL_SHA256='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
MODEL_URL='https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx'

class YoloXDetector:
    def __init__(self,path,confidence=.40,nms_threshold=.5):
        if not 0<confidence<=1 or not 0<nms_threshold<=1:raise ValueError("Detector thresholds must be in (0,1]")
        self.nms_threshold=nms_threshold;self.last_proposals=[]
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('Expected the documented OpenCV Zoo YOLOX model hash')
        self.net=cv2.dnn.readNet(str(path));self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.confidence=confidence
        grids=[];strides=[]
        for stride in (8,16,32):
            y,x=np.mgrid[:640//stride,:640//stride];grids.append(np.stack([x,y],axis=-1).reshape(-1,2));strides.append(np.full((x.size,1),stride))
        self.grid=np.concatenate(grids);self.stride=np.concatenate(strides)
    def detect(self,frame):
        height,width=frame.shape[:2];scale=min(640/width,640/height)
        canvas=np.full((640,640,3),114,dtype=np.uint8)
        resized=cv2.resize(frame,(int(width*scale),int(height*scale)))
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        # This model expects unnormalized RGB CHW values, as its Zoo demo specifies.
        self.net.setInput(cv2.dnn.blobFromImage(canvas,scalefactor=1.,swapRB=True))
        out=self.net.forward()[0]
        scores=out[:,4]*out[:,5] # COCO person class zero only
        selected=np.flatnonzero(scores>=self.confidence)
        xy=(out[selected,:2]+self.grid[selected])*self.stride[selected]
        wh=np.exp(np.clip(out[selected,2:4],-20,20))*self.stride[selected]
        boxes=np.column_stack([xy-wh/2,wh])/scale
        self.last_proposals=[(list(map(float,b)),float(s)) for b,s in zip(boxes,scores[selected])]
        keep=cv2.dnn.NMSBoxes(boxes.tolist(),scores[selected].tolist(),self.confidence,self.nms_threshold)
        detections=[]
        for k in np.asarray(keep).reshape(-1):
            x,y,w,h=boxes[k];x1=max(0,float(x));y1=max(0,float(y));x2=min(width,float(x+w));y2=min(height,float(y+h))
            if x2<=x1 or y2<=y1:continue
            bbox=(x1,y1,x2-x1,y2-y1)
            # Torso HSV histogram: cheap weak appearance, not learned person ReID.
            crop=frame[int(y1+.2*(y2-y1)):int(y1+.6*(y2-y1)),int(x1+.2*(x2-x1)):int(x1+.8*(x2-x1))]
            appearance=None
            if crop.size:
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,np.array([0,40,35]),np.array([179,255,255]))
                grass=cv2.inRange(hsv,np.array([30,40,0]),np.array([95,255,255]));mask=cv2.bitwise_and(mask,cv2.bitwise_not(grass))
                hist=cv2.calcHist([hsv],[0,1],mask,[12,4],[0,180,0,256]).ravel()
                if hist.sum()>0:appearance=tuple(map(float,hist/np.linalg.norm(hist)))
            detections.append(Detection(bbox=bbox,detection_confidence=float(scores[selected[k]]),appearance=appearance))
        return sorted(detections,key=lambda d:d.bbox)

class TeamClassifier:
    """Two fixed torso-color prototypes; unknown when separation/margin is weak."""
    def __init__(self,centers=None,config=None):self.centers=centers;self.config=config or Config()
    @staticmethod
    def color(frame,d):
        from src.vision.team_classifier import extract_jersey_color
        x,y,w,h=d.bbox
        return extract_jersey_color(frame,SimpleNamespace(x=x,y=y,width=w,height=h))
    @classmethod
    def fit(cls,samples,config=None):
        config=config or Config()
        if len(samples)<config.team_min_samples:return cls(config=config)
        from sklearn.cluster import KMeans
        model=KMeans(n_clusters=2,random_state=42,n_init=10).fit(np.asarray(samples))
        centers=sorted(model.cluster_centers_.tolist(),key=lambda p:tuple(p))
        return cls(np.asarray(centers),config) if np.linalg.norm(centers[0]-np.array(centers[1]))>=config.team_center_separation else cls(config=config)
    def classify(self,frame,d):
        c=self.color(frame,d)
        if c is None or self.centers is None:return d
        distance=np.linalg.norm(self.centers-c,axis=1);order=np.argsort(distance)
        margin=(distance[order[1]]-distance[order[0]])/max(distance[order[1]],1)
        if margin<self.config.team_margin or distance[order[0]]>self.config.team_max_color_distance:return d
        return d.model_copy(update={'team':'team_a' if order[0]==0 else 'team_b','team_confidence':float(margin)})
