"""Content-verified local copies for an isolated training process's image I/O.

The frozen dataset, statistical training code and checkpoint provenance stay
unchanged. Only equivalent, verified input bytes are read from outside iCloud.
"""
from contextlib import contextmanager
import json,os
from pathlib import Path
import cv2
from src.serving.cv import digest


class PixelCache:
    def __init__(self,dataset_path,root):
        self.dataset_path=Path(dataset_path);self.dataset_sha256=digest(self.dataset_path)
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_uid!=os.getuid():raise ValueError('Use an owned, real local pixel-cache directory')
        rows=json.loads(self.dataset_path.read_text())['rows'];expected={}
        for row in rows:
            for item in row['context']:
                path=str(Path(item['path']).resolve());sha=item['sha256']
                if path in expected and expected[path]!=sha:raise ValueError('Conflicting source-image identities')
                expected[path]=sha
        self.paths={};self.expected=expected
        for source,sha in expected.items():
            suffix=Path(source).suffix.lower()
            if suffix not in ['.png','.jpg','.jpeg'] or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Unsupported source-image cache entry')
            destination=self.root/(sha+suffix)
            if destination.is_symlink():raise ValueError('Cache entry cannot be a symlink')
            if not destination.exists():
                payload=Path(source).read_bytes()
                import hashlib
                if hashlib.sha256(payload).hexdigest()!=sha:raise ValueError('Source image changed before caching')
                temporary=destination.with_suffix(destination.suffix+'.tmp')
                with temporary.open('xb') as stream:stream.write(payload)
                temporary.chmod(0o600);os.replace(temporary,destination)
            if digest(destination)!=sha:raise ValueError('Cached image hash mismatch')
            self.paths[source]=destination

    def imread(self,path,*args,**kwargs):
        original=str(Path(path).resolve())
        if original not in self.paths:raise ValueError('Image is outside the frozen training dataset')
        pixels=cv2.imread(str(self.paths[original]),*args,**kwargs)
        if pixels is None:raise ValueError('Verified cached source image could not decode')
        return pixels

    def digest(self,path):
        original=str(Path(path).resolve())
        result=digest(self.paths[original]) if original in self.paths else digest(path)
        if original in self.expected and result!=self.expected[original]:raise ValueError('Cached source pixels changed')
        return result

    @contextmanager
    def training_io(self):
        # Dependency injection is confined to the training module in this
        # dedicated, workers=0 process; the global OpenCV library is untouched.
        from . import training
        old_cv2,old_digest=training.cv2,training.digest
        cache=self
        class Reader:
            def __getattr__(self,name):return cache.imread if name=='imread' else getattr(old_cv2,name)
        training.cv2=Reader();training.digest=self.digest
        try:yield
        finally:training.cv2=old_cv2;training.digest=old_digest
