"""Content-pinned Azure sources for the explicitly migrated short CV clips.

Videos live in Azure. Decoding/rendering leases an automatically removed private
temporary file; there is no persistent local video cache or unrelated fallback.
"""
from contextlib import contextmanager
import hashlib,json,re,tempfile,threading,time
from pathlib import Path
from .media import AzureClipMedia

NAMES=('giroud','southampton','team_goal','test_match')


class AzureCVSource:
    def __init__(self,client,name,entry):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',name):raise ValueError('Invalid CV recording')
        self.name=name;self.entry=dict(entry);self.sha256=entry['sha256'];self.size=entry['bytes'];self.etag=entry['etag']
        if not re.fullmatch('[0-9a-f]{64}',self.sha256) or not isinstance(self.size,int) or not 0<self.size<=20*1024**2:raise ValueError('Invalid bounded CV source identity')
        if entry['blob']!=f'cv/clips/{name}/{self.sha256}.mp4' or not isinstance(self.etag,str) or not self.etag:raise ValueError('Invalid pinned Azure CV location')
        self.blob=client.get_blob_client(entry['blob']);self.media=AzureClipMedia(client);self.media.sources[name]=entry['blob']
        self.lock=threading.RLock();self._data=None;self.checked_at=None

    def validate(self,expected_sha256=None):
        if expected_sha256 is not None and expected_sha256!=self.sha256:raise ValueError('Azure CV source does not match the frozen recording')
        with self.lock:
            if self.checked_at is not None and time.monotonic()-self.checked_at<30:return
            try:
                props=self.blob.get_blob_properties()
                if props.size!=self.size or props.etag!=self.etag or props.metadata.get('sha256')!=self.sha256 or props.metadata.get('recording_id')!=self.name:raise ValueError('Pinned Azure CV revision changed')
                if self._data is None:
                    from azure.core import MatchConditions
                    data=bytearray()
                    for chunk in self.blob.download_blob(etag=self.etag,match_condition=MatchConditions.IfNotModified).chunks():
                        data.extend(chunk)
                        if len(data)>self.size:raise ValueError('Azure CV source exceeds its pinned size')
                    if len(data)!=self.size or hashlib.sha256(data).hexdigest()!=self.sha256:raise ValueError('Azure CV full-content SHA mismatch')
                    self._data=bytes(data)
                self.media.metadata[self.name]=props;self.checked_at=time.monotonic()
            except ValueError:raise
            except Exception as exc:raise ValueError('Azure CV source unavailable: '+type(exc).__name__) from None

    @contextmanager
    def open(self):
        self.validate()
        with tempfile.TemporaryDirectory(prefix='tactiq-cv-source-') as directory:
            path=Path(directory)/(self.name+'.mp4');path.write_bytes(self._data)
            try:yield path
            finally:path.unlink(missing_ok=True)

    def response(self,request):
        from fastapi import HTTPException
        try:self.validate()
        except ValueError as exc:raise HTTPException(503,str(exc)) from None
        return self.media.response(self.name,request)


class UnavailableSource:
    def __init__(self,reason):self.reason=reason
    def validate(self,*args):raise ValueError(self.reason)


def load_sources(client,account_url,container,path='configs/serving/cv_media.json'):
    path=Path(path)
    if not path.exists():return {}
    try:
        data=json.loads(path.read_text())
        if data['schema_version']!='cv-azure-media-v1' or data['account_url']!=account_url or data['container']!=container:raise ValueError('Azure CV account/container mismatch')
        if set(data['recordings'])!=set(NAMES):raise ValueError('Expected the four migrated short clips')
    except (OSError,ValueError,KeyError,TypeError):return {name:UnavailableSource('Azure CV media registry unavailable or invalid') for name in NAMES}
    sources={}
    for name,entry in data['recordings'].items():
        try:sources[name]=AzureCVSource(client,name,entry)
        except (ValueError,KeyError,TypeError):sources[name]=UnavailableSource('Azure CV source registration invalid')
    return sources
