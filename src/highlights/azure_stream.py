"""Ephemeral loopback range gateway for one registered, ETag-pinned Azure video."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib,json,re,threading,uuid
from .render import HighlightError

@contextmanager
def azure_video_source(media,recording,summary,cancel,byte_budget=512*1024**2):
    match=str(summary['match_id']);half=int(summary['half'])
    media.register_half(recording,match,half)
    blob=media.client.get_blob_client(media.blob_name(recording));prop=blob.get_blob_properties()
    if prop.size<=0:raise HighlightError('source_unavailable')
    etag=str(prop.etag);name=media.blob_name(recording);size=int(prop.size)
    identity=hashlib.sha256(json.dumps([name,etag,size],separators=(',',':')).encode()).hexdigest()
    state={'bytes':0,'failure':None};lock=threading.Lock();token='/'+uuid.uuid4().hex+'/source.mp4'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):self.serve(False)
        def do_GET(self):self.serve(True)
        def serve(self,body):
            if self.path!=token:self.send_error(404);return
            if cancel.is_set():self.send_error(503);return
            start,end=0,size-1;status=200;value=self.headers.get('Range')
            if value:
                m=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
                if not m or not any(m.groups()):self.send_error(416);return
                if m[1]:start=int(m[1]);end=min(size-1,int(m[2])) if m[2] else size-1
                else:start=max(0,size-int(m[2]))
                if start>end or start>=size:self.send_error(416);return
                status=206
            download=None
            try:
                if body:
                    from azure.core import MatchConditions
                    download=blob.download_blob(offset=start,length=end-start+1,etag=etag,match_condition=MatchConditions.IfNotModified,max_concurrency=1)
                self.send_response(status);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('ETag',etag);self.send_header('Content-Length',str(end-start+1))
                if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                if body:
                    for chunk in download.chunks():
                        if cancel.is_set():return
                        with lock:
                            state['bytes']+=len(chunk)
                            if state['bytes']>byte_budget:state['failure']='source_read_budget_exceeded';return
                        self.wfile.write(chunk)
            except (BrokenPipeError,ConnectionResetError):pass # FFmpeg closes unneeded source ranges after seeking.
            except Exception as error:
                with lock:state['failure']='source_changed' if getattr(error,'status_code',None)==412 else 'azure_stream_failed'
                try:self.send_error(503)
                except OSError:pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;server.block_on_close=False
    thread=threading.Thread(target=server.serve_forever,daemon=True,name='azure-highlight-ranges');thread.start()
    origin={'origin':'registered_azure_stream','blob_name':name,'etag':etag,'size_bytes':size,'identity_sha256':identity,'integrity':'ETag-conditioned Azure ranges; no whole-source byte hash claimed','registered_media_access':True}
    try:
        yield f'http://127.0.0.1:{server.server_port}{token}',identity,origin,state
        if state['failure']:raise HighlightError(state['failure'])
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
