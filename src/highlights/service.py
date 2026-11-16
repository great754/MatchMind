"""Bounded asynchronous highlight jobs and session-scoped conversation."""
import json,os,shutil,threading,time,uuid,re
from contextlib import ExitStack
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import APIRouter,Depends,HTTPException
from fastapi.responses import FileResponse
from .schema import HighlightRequest,HighlightQueryPlan,HighlightManifest
from .agent import HighlightAgent
from .index import MatchEventIndex,digest
from .boundaries import ClipBoundaryEngine
from .planner import HighlightQueryPlanner
from .render import HighlightRenderer,SourceResolver,HighlightError,probe,file_hash

class HighlightService:
    def __init__(self,loaded,cv_reviews,media_store=None,media_root=None,root=None):
        self.root=Path(root or os.environ.get('MATCHMIND_HIGHLIGHT_CACHE','outputs/highlights_m17')).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.indexes={k:self.make_index(k,v) for k,v in loaded.items()};self.indexes.update({k:MatchEventIndex(k) for k in cv_reviews if k not in loaded})
        self.source=SourceResolver(self.root/'sources',loaded,cv_reviews,media_store,media_root);self.renderer=HighlightRenderer();self.agent=HighlightAgent();self.jobs={};self.lock=threading.Lock();self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tactiq-highlight');self.downloads=os.environ.get('MATCHMIND_HIGHLIGHT_DOWNLOADS','0')=='1'
    @staticmethod
    def make_index(recording,tools):
        try:return MatchEventIndex(recording,tools)
        except (ValueError,TypeError,KeyError):
            index=MatchEventIndex(recording);index.unavailable_reason='Highlight search is unavailable because the event index could not be validated.';return index
    def owned(self,job_id,session,recording):
        job=self.jobs.get(str(job_id))
        if job is None or job['session']!=str(session) or job['recording']!=recording:raise HTTPException(404,'Highlight unavailable in this session')
        return job
    def public(self,job):return {k:v for k,v in job.items() if not k.startswith('_') and k not in ['session','cancel','directory','created']}
    def cleanup(self):
        now=time.time()
        with self.lock:
            for key,j in list(self.jobs.items()):
                if now-j['created']>86400 and j['state'] not in ['QUEUED','PROCESSING']:del self.jobs[key]
            active={j.get('directory') for j in self.jobs.values() if j['state'] in ['QUEUED','PROCESSING']}
            for p in self.root.iterdir():
                if p.is_dir() and p.name!='sources' and str(p) not in active and now-p.stat().st_mtime>86400:shutil.rmtree(p)
            # Azure source cache is bounded separately; expiry only when worker idle.
            if not any(j['state'] in ['QUEUED','PROCESSING'] for j in self.jobs.values()):
                for p in (self.root/'sources').glob('*.mp4'):
                    if now-p.stat().st_mtime>86400:p.unlink();p.with_suffix('.json').unlink(missing_ok=True)
    def submit(self,recording,request,selection=None):
        if recording not in self.indexes:raise HTTPException(404,'Recording unavailable')
        self.cleanup();previous=self.owned(request.previous_job,request.session_id,recording) if request.previous_job else None
        if previous and previous['state']!='COMPLETED':raise HTTPException(409,'Wait for the previous result first')
        if selection is None:
            previous_plan=HighlightQueryPlan.model_validate(previous['plan']) if previous else None
            try:plan,provider,diagnostic=self.agent.interpret(request.question,previous_plan,request.provider)
            except ValueError:raise HTTPException(422,'Choose a valid request with a duration between 5 and 120 seconds.') from None
            rows,message=HighlightQueryPlanner().resolve(self.indexes[recording],plan)
            if previous and plan.selection=='ordinal' and re.search(r'\b(ones?)\b',request.question.lower()):
                pool=previous['events'];rows=pool[plan.ordinal-1:plan.ordinal]
                message=None if rows else f"I found {len(pool)} {'moment' if len(pool)==1 else 'moments'} in that result, but there isn’t a moment number {plan.ordinal}."
            if plan.intent=='unsupported':rows=[];message='I can help make highlights from recorded goals, shots and passes. I couldn’t safely interpret that request or its filters.'
            if plan.intent=='identity':
                rows=previous.get('events',[]) if previous else []
                if len(rows)!=1:message='Choose one moment first so I know which player you mean.'
                elif rows[0]['player'] is None:message='The player’s identity isn’t supported for this moment.'
                else:message=f"This event is linked to tracked player {rows[0]['player']}. That is a tracking ID, not a verified jersey number or name."
        else:
            plan,rows,selected_index,overlay=selection
            provider='unified';diagnostic=None;message=None
        job={'job_id':str(uuid.uuid4()),'recording':recording,'session':str(request.session_id),'state':'QUEUED','plan':plan.model_dump(),'events':rows,'message':message or '', 'provider':provider,'provider_diagnostic':diagnostic,'created':time.time(),'cancel':threading.Event(),'video_url':None,'thumbnail_url':None,'manifest':None,'cache_hit':False,'download_permitted':self.downloads}
        if selection is not None:
            job['_selection_index']=selected_index;job['_overlay']=overlay
            job['_variant']={'index_hash':selected_index.hash,'overlay':{k:v for k,v in (overlay or {}).items() if k!='catalog'}}
        if not rows or plan.intent=='identity':job['state']='COMPLETED';job['events']=rows;job['message']=message
        with self.lock:
            active=[j for j in self.jobs.values() if j['state'] in ['QUEUED','PROCESSING']]
            if len(active)>=4 or sum(j['session']==str(request.session_id) for j in active)>=2 or len(self.jobs)>=500:raise HTTPException(429,'Too many highlight requests; wait or cancel one')
            self.jobs[job['job_id']]=job
        if job['state']=='QUEUED':self.executor.submit(self.work,job)
        return self.public(job)
    def work(self,j):
        directory=None;stack=ExitStack()
        try:
            if j['cancel'].is_set():j['state']='CANCELLED';return
            j['state']='PROCESSING';started=time.monotonic();idx=j.get('_selection_index') or self.indexes[j['recording']]
            streamed=False;stream_state=None
            tools=self.source.loaded.get(j['recording'])
            if tools and tools.summary.get('scope')=='half' and self.source.media:
                from .azure_stream import azure_video_source
                path,source_hash,origin,stream_state=stack.enter_context(azure_video_source(self.source.media,j['recording'],tools.summary,j['cancel']));streamed=True
            else:path,source_hash,origin=self.source.resolve(j['recording'],j['cancel'])
            info=probe(path,j['cancel'])
            tb=idx.timebase
            if tb is None or abs(info['duration']-tb.frames/tb.fps)>1.0 or abs(float(Fraction(info['fps']))-tb.fps)>.01 or (info['nb_frames'] not in [None,'N/A'] and int(info['nb_frames'])!=tb.frames):raise HighlightError('source_clock_not_verified')
            duration=min(info['duration'],tb.frames/tb.fps)
            plan=HighlightQueryPlan.model_validate(j['plan']);segments=ClipBoundaryEngine().build(j['events'],plan,duration,tb.fps)
            if not segments:j.update(state='COMPLETED',events=[],message='Those moments won’t fit that duration without cutting out the play. Try a longer video.');return
            included={x for s in segments for x in s['event_ids']};omitted=[e['event_id'] for e in j['events'] if e['event_id'] not in included];j['events']=[e for e in j['events'] if e['event_id'] in included]
            key_parts=[source_hash,idx.version,idx.hash,j['plan'],self.renderer.settings,segments,self.downloads]
            if j.get('_variant'):key_parts.append(j['_variant'])
            key=digest(key_parts);directory=self.root/key;j['directory']=str(directory);manifest_path=directory/'manifest.json'
            cached=False
            if manifest_path.exists():
                manifest=HighlightManifest.model_validate_json(manifest_path.read_text()).model_dump()
                if (manifest.get('source_identity_sha256') if streamed else manifest['source_sha256'])==source_hash and manifest['source']==origin and manifest['index_hash']==idx.hash and manifest['plan']==j['plan'] and manifest['segments']==segments and file_hash(directory/'highlight.mp4')==manifest['render']['sha256'] and (directory/'thumbnail.jpg').is_file():cached=True
            if not cached:
                manifest_path.unlink(missing_ok=True)
                if sum(p.stat().st_size for p in self.root.rglob('highlight.mp4'))>1024**3:raise HighlightError('output_cache_capacity')
                directory.mkdir(exist_ok=True);render=self.renderer.render(path,segments,directory,j['cancel'],self.downloads)
                annotation=None
                if j.get('_overlay'):
                    from src.assistant.overlay import annotated_video
                    overlay=j['_overlay'];overlay['catalog'].video_dimensions=(info['width'],info['height']);annotation=annotated_video(directory,segments,overlay['catalog'],overlay['player_ref'],overlay['label'],j['cancel'],review=overlay.get('candidate_review',False));render['sha256']=annotation['sha256'];render['size_bytes']=(directory/'highlight.mp4').stat().st_size;render['render_seconds']+=annotation['seconds']
                if stream_state and stream_state['failure']:raise HighlightError(stream_state['failure'])
                manifest={'schema_version':'highlight-m17-v1','recording_id':j['recording'],'source_sha256':None if streamed else source_hash,'source_identity_sha256':source_hash if streamed else None,'source':origin,'index_version':idx.version,'index_hash':idx.hash,'plan':j['plan'],'render_settings':self.renderer.settings,'events':j['events'],'segments':segments,'omitted_event_ids':omitted,'render':render,'visual_annotation':annotation,'ranking':'goals, annotated shots, crosses, tackles, then tracking-estimated passes; chronological within priority; not objective excitement','buildup':'Fixed context windows; no verified associated possession sequence','timebase':tb.model_dump()}
                if streamed:manifest['render']['source_range_bytes_read']=stream_state['bytes']
                manifest=HighlightManifest.model_validate(manifest).model_dump()
                temp=manifest_path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2,allow_nan=False));os.replace(temp,manifest_path)
            if j['cancel'].is_set():raise HighlightError('cancelled')
            kinds=sorted({e['event_type'] for e in j['events']});label=' / '.join(kinds);n=len(j['events']);review=any(not e['safe_for_automatic'] for e in j['events'])
            singular=kinds[0].replace('_',' ') if len(kinds)==1 else 'recorded moment'
            plural={'pass':'passes','cross':'crosses'}.get(singular,singular+'s')
            message=f"I found {n} {singular if n==1 else plural}. Here’s the video."
            if plan.metric=='distance_m':message=f"The longest recorded pass was {j['events'][0]['metrics']['distance_m']:.1f} meters (a tracking estimate). Here’s the play."
            missing_types=set(plan.event_types)-set(kinds)
            if missing_types:message+=' I couldn’t confirm matching '+', '.join(t.replace('_',' ') for t in sorted(missing_types))+' events.'
            if omitted:message+=f' {len(omitted)} other matching moments did not fit the video limits.'
            if review:message+=' This is review footage, including unconfirmed events.'
            if 'pass' in kinds:message+=' Passes are tracking estimates; any annotation support is listed below.'
            if plan.duration:message+=f" It uses up to {plan.duration:g} seconds of available moments; it isn’t padded to fill the requested length."
            base=f"/matches/{j['recording']}/highlights/{j['job_id']}";session=f"?session_id={j['session']}"
            j.update(state='COMPLETED',message=message,manifest=manifest,cache_hit=cached,video_url=base+'/video.mp4'+session,thumbnail_url=base+'/thumbnail.jpg'+session,generation_seconds=time.monotonic()-started)
        except Exception as error:
            cancelled=j['cancel'].is_set();j.update(state='CANCELLED' if cancelled else 'FAILED',message='Highlight cancelled.' if cancelled else 'I couldn’t create this video. The source may be unavailable or invalid.',diagnostic=str(error) if isinstance(error,HighlightError) else type(error).__name__)
            if directory and directory.exists() and not (directory/'manifest.json').exists():shutil.rmtree(directory,ignore_errors=True)
        finally:
            try:stack.close()
            except HighlightError as error:j.update(state='CANCELLED' if j['cancel'].is_set() else 'FAILED',video_url=None,diagnostic=str(error))
    def router(self,authorize):
        router=APIRouter(dependencies=[Depends(authorize)])
        @router.get('/matches/{recording}/highlights/capabilities')
        def capabilities(recording:str):
            if recording not in self.indexes:raise HTTPException(404,'Recording unavailable')
            return self.indexes[recording].capabilities()
        @router.post('/matches/{recording}/highlights')
        def submit(recording:str,body:HighlightRequest):return self.submit(recording,body)
        @router.get('/matches/{recording}/highlights/{job_id}')
        def status(recording:str,job_id:uuid.UUID,session_id:uuid.UUID):return self.public(self.owned(job_id,session_id,recording))
        @router.delete('/matches/{recording}/highlights/{job_id}')
        def cancel(recording:str,job_id:uuid.UUID,session_id:uuid.UUID):
            j=self.owned(job_id,session_id,recording)
            if j['state'] in ['QUEUED','PROCESSING']:
                j['cancel'].set()
                if j['state']=='QUEUED':j.update(state='CANCELLED',message='Highlight cancelled.')
            return self.public(j)
        @router.get('/matches/{recording}/highlights/{job_id}/{asset}')
        def asset(recording:str,job_id:uuid.UUID,asset:str,session_id:uuid.UUID,download:bool=False):
            j=self.owned(job_id,session_id,recording)
            if j['state']!='COMPLETED' or not j.get('directory'):raise HTTPException(404,'Video not ready')
            if asset not in ['video.mp4','thumbnail.jpg','manifest.json']:raise HTTPException(404,'Unknown highlight asset')
            if download and not self.downloads:raise HTTPException(403,'Downloads are not enabled for these media rights')
            filename={'video.mp4':'highlight.mp4','thumbnail.jpg':'thumbnail.jpg','manifest.json':'manifest.json'}[asset];p=Path(j['directory'])/filename
            if not p.is_file():raise HTTPException(404,'Highlight expired; create it again')
            if asset=='video.mp4' and file_hash(p)!=j['manifest']['render']['sha256']:raise HTTPException(503,'Highlight changed; create it again')
            return FileResponse(p,media_type={'video.mp4':'video/mp4','thumbnail.jpg':'image/jpeg','manifest.json':'application/json'}[asset],filename='tactiqai-highlight.mp4' if download else None)
        return router
