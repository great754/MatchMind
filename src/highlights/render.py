"""Bounded accurate FFmpeg rendering from server-resolved files only."""
import hashlib,json,os,signal,subprocess,time,threading
from pathlib import Path

MAX_SOURCE=1024**3
class HighlightError(RuntimeError):pass

def file_hash(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(command,cancel,timeout=120):
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    started=time.monotonic()
    while True:
        try:
            stdout,stderr=p.communicate(timeout=.2)
            if p.returncode:raise HighlightError('media_process_failed')
            return stdout
        except subprocess.TimeoutExpired:
            if cancel.is_set() or time.monotonic()-started>timeout:
                os.killpg(p.pid,signal.SIGKILL);p.communicate();raise HighlightError('cancelled' if cancel.is_set() else 'media_process_timeout')

def probe(path,cancel=None):
    raw=run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],cancel or threading.Event(),20)
    data=json.loads(raw);video=next((s for s in data['streams'] if s['codec_type']=='video'),None)
    if video is None:raise HighlightError('no_video_stream')
    duration=float(data['format']['duration'])
    if not 0<duration<14400:raise HighlightError('invalid_media_duration')
    return {'duration':duration,'audio':any(s['codec_type']=='audio' for s in data['streams']),'width':video['width'],'height':video['height'],'start_time':float(data['format'].get('start_time',0)),'video_start_time':float(video.get('start_time',data['format'].get('start_time',0))),'fps':video.get('avg_frame_rate'),'nb_frames':video.get('nb_frames'),'streams':data['streams']}

class SourceResolver:
    def __init__(self,root,loaded,cv_reviews,media_store=None,media_root=None):
        self._lock=threading.RLock();self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self.loaded=loaded;self.cv=cv_reviews;self.media=media_store;self.local=Path(media_root).resolve() if media_root else None
    def resolve(self,recording,cancel):
        with self._lock:return self._resolve(recording,cancel)
    def _resolve(self,recording,cancel):
        if recording not in self.loaded and recording not in self.cv:raise HighlightError('unknown_source')
        if recording in self.cv and recording not in self.loaded:
            review=self.cv[recording];review.frame(0);p=review.video.resolve()
            if not p.is_file() or file_hash(p)!=review.manifest['video_sha256']:raise HighlightError('source_changed')
            return p,file_hash(p),{'origin':'registered_cv','registered_media_access':True}
        tools=self.loaded[recording];match=str(tools.summary.get('match_id',''))
        if tools.summary.get('scope')=='half':raise HighlightError('full_half_highlight_not_supported')
        if tools.summary.get('scope')!='clip':raise HighlightError('unverified_source_timebase')
        if self.media:
            prop=self.media.properties(match)
            if prop is None or not 0<prop.size<=MAX_SOURCE:raise HighlightError('source_missing_or_too_large')
            identity=hashlib.sha256((recording+str(prop.etag)).encode()).hexdigest();p=self.root/(identity+'.mp4');meta=p.with_suffix('.json')
            if p.exists() and meta.exists():
                d=json.loads(meta.read_text())
                if p.stat().st_size==prop.size and file_hash(p)==d['sha256']:return p,d['sha256'],{'origin':'registered_azure','etag':str(prop.etag),'registered_media_access':True}
                p.unlink();meta.unlink()
            if sum(x.stat().st_size for x in self.root.glob('*.mp4'))+prop.size>2*1024**3:raise HighlightError('source_cache_capacity')
            temp=p.with_suffix('.part')
            try:
                from azure.core import MatchConditions
                blob=self.media.client.get_blob_client(f'mot/clips/{match}.mp4');download=blob.download_blob(etag=prop.etag,match_condition=MatchConditions.IfNotModified);total=0
                with temp.open('wb') as f:
                    for chunk in download.chunks():
                        if cancel.is_set():raise HighlightError('cancelled')
                        total+=len(chunk)
                        if total>prop.size or total>MAX_SOURCE:raise HighlightError('source_size_changed')
                        f.write(chunk)
                if total!=prop.size:raise HighlightError('source_incomplete')
                os.replace(temp,p);h=file_hash(p);meta.write_text(json.dumps({'sha256':h,'etag':str(prop.etag)}));return p,h,{'origin':'registered_azure','etag':str(prop.etag),'registered_media_access':True}
            finally:temp.unlink(missing_ok=True)
        if self.local:
            for suffix in ['.mp4','.txt.mp4']:
                p=(self.local/(match+suffix)).resolve()
                if p.is_relative_to(self.local) and p.is_file():return p,file_hash(p),{'origin':'registered_local','registered_media_access':True}
        raise HighlightError('source_unavailable')

class HighlightRenderer:
    settings={'version':'m17.2','fps':30,'size':[1280,720],'video':'libx264','audio':'aac','max_output_seconds':120,'threads':2}
    def render(self,path,segments,out,cancel,download_permitted=True):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);info=probe(path,cancel);started=time.monotonic()
        for i,s in enumerate(segments):
            if not 0<=s['start']<s['end']<=info['duration']+1e-6:raise HighlightError('interval_outside_media')
            length=s['end']-s['start'];seek=max(0,s['start']+info['video_start_time']-info['start_time']);command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-ss',str(seek),'-i',str(path)]
            if not info['audio']:command+=['-f','lavfi','-i','anullsrc=r=48000:cl=stereo']
            command+=['-map','0:v:0','-map','0:a:0' if info['audio'] else '1:a:0','-t',str(length),'-vf','setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30','-af',f'aresample=48000:async=1:first_pts=0,apad,atrim=duration={length},asetpts=PTS-STARTPTS','-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-crf','23','-threads','2','-filter_threads','1','-c:a','aac','-ac','2','-ar','48000','-b:a','128k','-fs','250000000','-movflags','+faststart',str(out/f'part_{i:02d}.mp4')]
            run(command,cancel,120)
            if time.monotonic()-started>300:raise HighlightError('render_budget_exceeded')
        (out/'concat.txt').write_text(''.join(f"file 'part_{i:02d}.mp4'\n" for i in range(len(segments))))
        run(['ffmpeg','-nostdin','-v','error','-y','-f','concat','-safe','1','-i',str(out/'concat.txt'),'-c','copy','-movflags','+faststart',str(out/'highlight.mp4')],cancel,30)
        final=probe(out/'highlight.mp4',cancel);expected=sum(s['end']-s['start'] for s in segments)
        if abs(final['duration']-expected)>.15 or (out/'highlight.mp4').stat().st_size>250000000:raise HighlightError('output_duration_mismatch')
        run(['ffmpeg','-nostdin','-v','error','-y','-i',str(out/'highlight.mp4'),'-frames:v','1','-threads','1',str(out/'thumbnail.jpg')],cancel,20)
        for p in out.glob('part_*.mp4'):p.unlink()
        (out/'concat.txt').unlink()
        return {'duration':final['duration'],'expected_duration':expected,'size_bytes':(out/'highlight.mp4').stat().st_size,'sha256':file_hash(out/'highlight.mp4'),'render_seconds':time.monotonic()-started,'audio':True,'original_audio':info['audio'],'download_permitted':download_permitted,'source_probe':{k:v for k,v in info.items() if k!='streams'},'output_streams':[{k:s.get(k) for k in ['codec_type','codec_name','start_time','duration','avg_frame_rate','sample_rate']} for s in final['streams']]}
