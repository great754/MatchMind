"""Inspectable classical pitch evidence and guarded temporal camera localization.

Anonymous lines never acquire semantic IDs by proximity to a guessed pitch.
Metric solutions require independently identified correspondences. No weights.
"""
import cv2
import numpy as np
from pydantic import Field
from src.query.schema import Closed
from src.observations.schema import PitchCalibrationObservation,Provenance
from .schema import CameraState

class LineEvidence(Closed):
    start: tuple[float,float]
    end: tuple[float,float]
    kind: str = 'unidentified_white_line'
    support: float = Field(ge=0,le=1)

class CurveEvidence(Closed):
    kind: str
    points: list[tuple[float,float]]
    support: float = Field(ge=0,le=1)

class PitchEvidence(Closed):
    status: str
    lines: list[LineEvidence] = Field(default_factory=list)
    curves: list[CurveEvidence] = Field(default_factory=list)
    grass_fraction: float | None
    reason: str

class PitchPerception:
    def detect(self,image):
        height,width=image.shape[:2];scale=min(1.,640/width)
        small=cv2.resize(image,None,fx=scale,fy=scale)
        hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        grass=cv2.inRange(hsv,(25,35,25),(95,255,255))
        neighborhood=cv2.dilate(grass,np.ones((13,13),np.uint8))
        white=cv2.inRange(hsv,(0,0,145),(179,75,255))
        mask=cv2.bitwise_and(white,neighborhood)
        lines=cv2.HoughLinesP(mask,1,np.pi/180,35,minLineLength=45,maxLineGap=12)
        evidence=[]
        for x1,y1,x2,y2 in ([] if lines is None else lines.reshape(-1,4)):
            length=float(np.hypot(x2-x1,y2-y1))
            evidence.append(LineEvidence(start=(x1/scale,y1/scale),end=(x2/scale,y2/scale),support=min(1,length/200)))
        evidence=sorted(evidence,key=lambda x:-x.support)[:40]
        fraction=float(np.count_nonzero(grass)/grass.size)
        return PitchEvidence(status='partial' if evidence else 'unavailable',lines=evidence,grass_fraction=fraction,
            reason='Anonymous line segments; semantic correspondences and pitch orientation unresolved' if evidence else 'Insufficient visible pitch-line evidence')

class TemporalCamera:
    """Conservative cuts + optional semantic fits. Last reliable H is history only.

    Correspondences must be supplied with provenance by a semantic provider, never
    synthesized from green-mask corners. Temporal flow diagnoses motion; it does
    not automatically authorize a stale metric transform.
    """
    def __init__(self):
        self.previous=None;self.hist=None;self.epoch=0;self.last_transform=None;self.last_timestamp=None;self.was_valid=False
    def update(self,image,frame,timestamp,evidence,correspondences=None,source='automatic anonymous line evidence'):
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(320,180))
        hist=cv2.calcHist([cv2.cvtColor(cv2.resize(image,(320,180)),cv2.COLOR_BGR2HSV)],[0,1],None,[24,16],[0,180,0,256]);cv2.normalize(hist,hist)
        distance=float(cv2.compareHist(self.hist,hist,cv2.HISTCMP_BHATTACHARYYA)) if self.hist is not None else None
        matches=0;flow=None;inlier_count=0;motion_error=None
        if self.previous is not None:
            points=cv2.goodFeaturesToTrack(self.previous,150,.02,8)
            if points is not None:
                nxt,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None)
                back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None) if nxt is not None else (None,None,None)
                if back is not None:
                    good=(status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.5)
                    matches=int(good.sum())
                    if matches>=12:
                        motion,inliers=cv2.estimateAffinePartial2D(points[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2)
                        if motion is not None and inliers.sum()>=12:
                            flow=motion.tolist();inlier_count=int(inliers.sum())
                            predicted=cv2.transform(points[good],motion)
                            motion_error=float(np.median(np.linalg.norm(predicted[:,0]-nxt[good,0],axis=1)))
        cut=distance is not None and distance>.65 and matches<12
        if cut:self.epoch+=1
        continuity='cut' if cut else 'continuous' if flow is not None else 'unknown'
        calibration=None;region=None;reason=evidence.reason;h=None
        # Four independently constraining correspondences are the minimum.
        # A nonzero determinant alone does not reject nearly degenerate fits.
        if correspondences and len(correspondences)>=4:
            image_points=np.array([p[0] for p in correspondences],np.float64)
            pitch_points=np.array([p[1] for p in correspondences],np.float64)
            normalized=[]
            for points in (image_points,pitch_points):
                centered=points-points.mean(axis=0)
                normalized.append(centered/max(float(np.sqrt(np.mean(centered**2))),1e-12))
            design=[]
            for (x,y),(u,v) in zip(*normalized):
                design.extend([[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]])
            singular=np.linalg.svd(design,compute_uv=False)
            constrained=sum(singular>singular[0]*1e-4)>=8
            if constrained:
                candidate,inliers=cv2.findHomography(image_points,pitch_points,cv2.RANSAC,1.)
                if candidate is not None and inliers is not None and int(inliers.sum())>=4 and abs(np.linalg.det(candidate))>1e-12:
                    errors=np.linalg.norm(cv2.perspectiveTransform(image_points[None],candidate)[0]-pitch_points,axis=1)
                    if float(np.quantile(errors,.9))<=1.:
                        hull=cv2.convexHull(image_points.astype(np.float32)).reshape(-1,2)
                        mapped=cv2.perspectiveTransform(hull[None],candidate)[0]
                        if np.isfinite(mapped).all() and ((mapped>=[-1e-6,-1e-6])&(mapped<=[105+1e-6,68+1e-6])).all():
                            h=candidate/candidate[2,2];region=np.clip(mapped,[0,0],[105,68]).tolist()
                            calibration=PitchCalibrationObservation(start_frame=frame,end_frame=frame,landmarks=image_points.tolist(),
                                image_to_pitch=tuple(h.flat),validity_region=hull.tolist(),
                                provenance=Provenance(provider='cv-camera-m2',source=source,quality='inferred'))
                            reason='Identified correspondences passed geometric checks; residual is dependent validation'
        valid=calibration is not None
        if valid:self.last_transform=tuple(h.flat);self.last_timestamp=timestamp
        self.previous=gray;self.hist=hist;self.was_valid=valid
        return CameraState(frame=frame,timestamp=timestamp,epoch=self.epoch,continuity=continuity,
            calibration=calibration,calibration_valid=valid,visible_pitch_region=region,source=source,
            calibration_status='valid' if valid else 'partial' if evidence.lines else 'lost' if self.last_transform else 'unknown',
            pitch_evidence=evidence.model_dump(),motion_evidence={'histogram_distance':distance,'feature_matches':matches,'affine_inliers':inlier_count,'affine_median_error_small_px':motion_error,'image_affine':flow,'reason':reason},
            last_reliable_transform=self.last_transform,last_reliable_timestamp=self.last_timestamp)

    @staticmethod
    def project(point,camera):
        if not camera.calibration_valid or camera.calibration is None:return None
        c=camera.calibration
        if cv2.pointPolygonTest(np.array(c.validity_region,np.float32),tuple(map(float,point)),False)<0:return None
        p=cv2.perspectiveTransform(np.array([[point]],np.float64),np.array(c.image_to_pitch).reshape(3,3))[0,0]
        return tuple(map(float,p)) if np.isfinite(p).all() and ((p>=[0,0])&(p<=[105,68])).all() else None
