"""Validated, bounded, read-only precomputed CV review access."""
import hashlib,json,math
from pathlib import Path
from fastapi import HTTPException
from src.vision.world.schema import WorldState

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

class CVReview:
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.manifest=json.loads((self.directory/'manifest.json').read_text())
        m=self.manifest;self.video=Path(m['source_video']).resolve();self.data=self.directory/'frames.jsonl'
        if digest(self.video)!=m['video_sha256'] or digest(self.data)!=m['frames_sha256']:raise ValueError('CV review hash mismatch')
        if not math.isfinite(m['fps']) or m['fps']<=0 or m['frames']<1 or len(m['frame_offsets'])!=m['frames']:raise ValueError('Invalid CV index')
        # Validate all indexed frames once before exposure. No corruption enters UI.
        with self.data.open('rb') as f:
            for i,offset in enumerate(m['frame_offsets']):
                if offset!=f.tell():raise ValueError('Invalid CV byte index')
                row=WorldState.model_validate_json(f.readline())
                if row.frame!=i or row.recording_id!=m['recording_id'] or abs(row.timestamp-i/m['fps'])>1e-6:raise ValueError('CV clock/recording mismatch')
            if f.read(1):raise ValueError('Unindexed CV frames')
        self.data_stamp=self.data.stat().st_mtime_ns;self.video_stamp=self.video.stat().st_mtime_ns
    def frame(self,timestamp):
        try:
            changed=self.data.stat().st_mtime_ns!=self.data_stamp or self.video.stat().st_mtime_ns!=self.video_stamp
        except OSError:
            raise HTTPException(503,'CV source unavailable; restart and revalidate') from None
        if changed:raise HTTPException(503,'CV source changed; restart and revalidate')
        m=self.manifest;i=min(m['frames']-1,max(0,math.floor(timestamp*m['fps']+1e-8)))
        with self.data.open('rb') as f:
            f.seek(m['frame_offsets'][i]);return WorldState.model_validate_json(f.readline())
    def metadata(self):
        return {k:v for k,v in self.manifest.items() if k not in ('source_video','frame_offsets','frames_sha256') }|{'media_url':f"/matches/{self.manifest['recording_id']}/video.mp4"}
