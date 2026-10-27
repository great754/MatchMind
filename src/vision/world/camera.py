"""Explicit calibration/camera provider. No automatic pitch-line recognition."""
import hashlib,json
from pathlib import Path
from typing import Protocol
import cv2
import numpy as np
from src.observations.schema import PitchCalibrationObservation,Provenance
from .schema import CameraState

class PitchProvider(Protocol):
    def state(self,frame:int,timestamp:float)->CameraState: ...
    def project(self,point,frame:int): ...

class CameraProvider:
    def __init__(self,video_path,calibration_path=None,events_path=None):
        self.video_hash=hashlib.sha256(Path(video_path).read_bytes()).hexdigest()
        self.calibrations=[];self.events=[];self.transformer=None;self.epoch=0
        for path,kind in ((calibration_path,'calibration'),(events_path,'camera')):
            if not path:continue
            raw=json.loads(Path(path).read_text())
            if raw.get('video_sha256')!=self.video_hash:raise ValueError('Camera metadata does not match this video')
            if kind=='calibration':
                self.calibrations=[PitchCalibrationObservation.model_validate(c) for c in raw['intervals']]
                for c in self.calibrations:
                    if c.image_to_pitch and abs(np.linalg.det(np.array(c.image_to_pitch).reshape(3,3)))<1e-12:raise ValueError('Singular calibration homography')
                ordered=sorted(self.calibrations,key=lambda c:c.start_frame)
                if any(a.end_frame>=b.start_frame for a,b in zip(ordered,ordered[1:])):raise ValueError('Overlapping calibration intervals')
            else:
                self.events=sorted(raw['events'],key=lambda e:e['frame'])
                if len({e['frame'] for e in self.events})!=len(self.events) or any(e['frame']<0 for e in self.events):raise ValueError('Invalid camera event frames')
    def state(self,frame,timestamp):
        cuts=[e for e in self.events if e.get('continuity')=='cut' and e['frame']<=frame]
        self.epoch=len(cuts)
        entry=next((e for e in self.events if e['frame']==frame),{})
        calibration=next((c for c in self.calibrations if c.start_frame<=frame<=c.end_frame),None)
        # An interval must have been explicitly re-established after the last cut.
        if calibration and cuts and calibration.start_frame<cuts[-1]['frame']:calibration=None
        valid=calibration is not None and calibration.image_to_pitch is not None and len(calibration.validity_region)>=3
        region=None
        if valid and len(calibration.validity_region)>=3:
            projected=[self._homography(p,calibration) for p in calibration.validity_region]
            if all(p is not None for p in projected):region=projected
        return CameraState(frame=frame,timestamp=timestamp,epoch=self.epoch,mode=entry.get('mode','unknown'),
            continuity=entry.get('continuity','unknown'),calibration=calibration,calibration_valid=valid,
            calibration_confidence=calibration.provenance.confidence if calibration else None,visible_pitch_region=region,
            source='hash-bound supplied metadata' if calibration or self.events else 'calibration unavailable; continuity unverified')
    @staticmethod
    def _homography(point,calibration):
        h=np.array(calibration.image_to_pitch).reshape(3,3)
        v=h@np.array([*point,1.])
        if abs(v[2])<1e-10:return None
        p=v[:2]/v[2]
        return tuple(map(float,p)) if np.isfinite(p).all() and 0<=p[0]<=105 and 0<=p[1]<=68 else None
    def project(self,point,frame):
        state=self.state(frame,0)
        c=state.calibration
        if not state.calibration_valid or not c or len(c.validity_region)<3:return None
        polygon=np.array(c.validity_region,dtype=np.float32)
        if cv2.pointPolygonTest(polygon,tuple(map(float,point)),False)<0:return None
        return self._homography(point,c)

class SoccerTrackCameraProvider(CameraProvider):
    """Reuse the existing piecewise transformer only with video-hash-bound metadata.

    Panoramic SoccerTrack calibration is nonlinear; never replace it with a fake
    single homography or apply its landmarks to an unrelated broadcast video.
    """
    def __init__(self,video_path,metadata_path):
        super().__init__(video_path)
        raw=json.loads(Path(metadata_path).read_text())
        if raw['video_sha256']!=self.video_hash:raise ValueError('SoccerTrack calibration belongs to another video')
        from src.vision.coordinate_transform import PitchCoordinateTransformer
        self.transformer=PitchCoordinateTransformer(Path(metadata_path).parent/raw['keypoints'])
        self.start=raw['start_frame'];self.end=raw['end_frame'];self.source=str(metadata_path)
    def state(self,frame,timestamp):
        valid=self.start<=frame<=self.end
        c=PitchCalibrationObservation(start_frame=self.start,end_frame=self.end,
            landmarks=[tuple(p) for p in self.transformer.image_points],validity_region=[tuple(p) for p in cv2.convexHull(self.transformer.image_points.astype(np.float32)).reshape(-1,2)],
            provenance=Provenance(provider='soccertrack-piecewise',source=self.source,quality='provided')) if valid else None
        return CameraState(frame=frame,timestamp=timestamp,calibration_valid=valid,calibration=c,source=self.source)
    def project(self,point,frame):
        if not self.start<=frame<=self.end:return None
        p=self.transformer.transform_point(*point)
        return p if p and 0<=p[0]<=105 and 0<=p[1]<=68 else None
