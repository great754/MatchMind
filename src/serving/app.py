"""Read-only service. Paths come from a trusted startup registry, never requests."""
import hmac
import json
import os
import logging
import time
import uuid
import re
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,HTTPException,Depends,Header,Query,Request
from fastapi.responses import FileResponse,JSONResponse,HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import Field
from src.query import MatchMindTools,explain_event
from src.query.dispatch import dispatch
from src.query.schema import Closed
from src.providers.agent import ask,AgentResponse,evidence_for
from src.providers.conversation import remember
from src.query.schema import Answer
from src.providers.conversation import ConversationState
from src.artifacts import LocalArtifactStore,AzureBlobArtifactStore,ArtifactError
from src.providers.factory import build_provider
from .review import Review,load_replay,review

class AskRequest(Closed):
    question:str=Field(min_length=1,max_length=2000)
    mode:Literal['beginner','fan','analyst']='fan'
    timestamp:float=Field(default=0,ge=0)
    event_id:str|None=Field(default=None,min_length=1,max_length=100)
    provider:Literal['local','gemini','microsoft','microsoft-framework']|None=None
    conversation:ConversationState=Field(default_factory=ConversationState)


def create_app(registry,provider=None,token=None,default_provider='local',media_root=None,recording_catalog=None,media_store=None,cv_registry=None):
    # Resolve once before traffic. Azure recordings fail independently; local stays strict.
    store=registry if hasattr(registry,'resolve') and hasattr(registry,'ids') else LocalArtifactStore(registry)
    logger=logging.getLogger('matchmind.requests')
    azure=isinstance(store,AzureBlobArtifactStore)
    catalog={item['recording_id']:item for item in (recording_catalog or [])}
    statuses={};loaded={}
    for key in store.ids():
        item=catalog.get(key,{})
        status=dict(recording_id=key,match_id=str(item.get('match_id','')),known=True,configured=True,
                    processed=item.get('processed'),artifact_store='azure' if azure else 'local')
        try:
            loaded[key]=MatchMindTools(store.resolve(key))
        except ArtifactError as error:
            if not azure:raise
            diagnostic=getattr(error,'diagnostic','azure_artifact_invalid')
            status.update(status='unavailable' if diagnostic=='azure_bundle_missing' else 'error',
                          artifact_available=getattr(error,'bundle_available',None),artifact_ready=False,diagnostic=diagnostic)
            logger.warning('Recording %s unavailable: %s',key,diagnostic)
        else:
            status.update(match_id=str(loaded[key].summary.get('match_id','')),status='available',
                          processed=True,artifact_available=True,artifact_ready=True)
            status['media_available']=media_store.available(status['match_id']) if media_store and loaded[key].summary.get('scope')=='clip' else None
        statuses[key]=status
    for key,item in catalog.items():
        if key not in statuses:
            statuses[key]=dict(recording_id=key,match_id=str(item.get('match_id','')),status='unavailable' if item.get('processed') else 'not_processed',
                               known=True,configured=False,processed=item.get('processed'),artifact_store='azure' if azure else 'local',
                               artifact_available=None,artifact_ready=False,diagnostic='not_configured')
    # Video availability is independent of validated analytics availability.
    media_ids={}
    if media_store:
        for key,status in statuses.items():
            match=status['match_id']
            available=bool(key.endswith('_clip') and match and media_store.available(match))
            status.update(media_available=available,media_url=f'/matches/{key}/video.mp4' if available else None)
            if available:media_ids[key]=match
    cv_reviews={}
    for key,directory in (cv_registry or {}).items():
        from .cv import CVReview
        try:
            cv_reviews[key]=CVReview(directory)
            if cv_reviews[key].manifest['recording_id']!=key:raise ValueError('CV registry identity mismatch')
            if key in loaded:
                summary=loaded[key].summary;metadata=cv_reviews[key].manifest
                if summary.get('fps')!=metadata['fps'] or summary.get('frames')!=metadata['frames']:
                    raise ValueError('CV/analytics recording clocks differ')
        except (ValueError,OSError) as error:
            cv_reviews.pop(key,None)
            logger.warning('CV recording %s unavailable: %s',key,type(error).__name__)
        if key in statuses:
            statuses[key]['cv_diagnostic']=None if key in cv_reviews else 'cv_review_unavailable_or_clock_mismatch'
            if key in cv_reviews and not statuses[key].get('media_available'):
                statuses[key].update(media_available=True,media_url=f'/matches/{key}/video.mp4')
            continue
        statuses[key]=dict(recording_id=key,match_id=key,status='available' if key in cv_reviews else 'unavailable',
            source='cv_experimental',artifact_ready=False,media_available=key in cv_reviews,
            media_url=f'/matches/{key}/video.mp4' if key in cv_reviews else None,
            diagnostic=None if key in cv_reviews else 'cv_review_unavailable')
    for key,status in statuses.items():
        cv=cv_reviews.get(key)
        status.setdefault('source','soccertrack_structured')
        status['capabilities']=dict(video_available=bool(status.get('media_available')),analytics_available=key in loaded,
            tactical_view_available=key in loaded,cv_world_model_available=cv is not None,
            pitch_calibration_available=bool(cv and cv.manifest.get('pitch_calibration_available')),ask_available=key in loaded)
    replays={key:load_replay(tools.directory) for key,tools in loaded.items()}
    from src.tactics.review import review_pressure
    for key,tools in loaded.items():
        tools.replay=replays[key][0]
        tools.replay_source=replays[key][1] or 'review_frames.json'
        tools.review_pressure=review_pressure(tools.replay,tools.ownership)
    provider_cache={};logger=logging.getLogger('matchmind.requests')
    metrics=dict(requests=0,unknown_questions=0,fallbacks=0)
    app=FastAPI(title='MatchMind artifact API',version='1.0')
    @app.middleware('http')
    async def observe(request,call_next):
        correlation=uuid.uuid4().hex;start=time.perf_counter()
        if request.method=='POST':
            body=bytearray()
            async for chunk in request.stream():
                if len(body)+len(chunk)>16384:return JSONResponse(status_code=413,content={'detail':'Request too large'})
                body.extend(chunk)
            request._body=bytes(body)
        response=await call_next(request)
        response.headers['X-Request-ID']=correlation
        metrics['requests']+=1
        logger.info(json.dumps(dict(request_id=correlation,method=request.method,route=getattr(request.scope.get('route'),'path','unmatched'),status=response.status_code,latency_ms=round((time.perf_counter()-start)*1000,3),**getattr(request.state,'query_metrics',{}))))
        return response
    @app.get('/ready')
    def ready():return dict(status='ready',recordings=len(set(loaded)|set(cv_reviews))) if loaded or cv_reviews else JSONResponse(status_code=503,content={'status':'empty'})
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,error):
        return JSONResponse(status_code=422,content={'detail':[{'loc':list(e['loc']),'type':e['type'],'msg':e['msg']} for e in error.errors()]})
    def authorize(authorization:str|None=Header(default=None)):
        if token and not hmac.compare_digest(authorization or '', 'Bearer '+token):raise HTTPException(401,'Authentication required')
    def recording(match_id):
        if azure and match_id in statuses and match_id not in loaded:raise HTTPException(503,dict(message='Recording unavailable',diagnostic=statuses[match_id]['diagnostic']))
        if match_id not in loaded:raise HTTPException(404,'Unknown recording')
        return loaded[match_id]
    secured=[Depends(authorize)]
    @app.get('/health')
    def health():return dict(status='ok',recordings=len(loaded),processing='precomputed_artifacts')
    @app.get('/recordings',dependencies=secured)
    def recordings():
        return dict(recordings=list(statuses.values()))

    @app.get('/matches/{match_id}/cv',dependencies=secured)
    def cv_metadata(match_id:str):
        if match_id not in cv_reviews:raise HTTPException(404,'CV review unavailable')
        return cv_reviews[match_id].metadata()

    @app.get('/matches/{match_id}/cv/frame',dependencies=secured)
    def cv_frame(match_id:str,timestamp:float=Query(default=0,ge=0,le=86400,allow_inf_nan=False)):
        if match_id not in cv_reviews:raise HTTPException(404,'CV review unavailable')
        return cv_reviews[match_id].frame(timestamp)

    @app.get('/matches/{match_id}/view',response_model=Review,dependencies=secured)
    def match_view(match_id:str):
        tools=recording(match_id);replay,source=replays[match_id]
        result=review(tools,match_id,replay,source,default_provider,media_root if not media_store else None)
        if media_store and tools.summary.get('scope')=='clip' and media_store.available(str(tools.summary.get('match_id',''))):
            result=result.model_copy(update={'media_url':f'/matches/{match_id}/video.mp4'})
        return result
    @app.get('/matches/{match_id}/summary',dependencies=secured)
    def summary(match_id:str):return recording(match_id).get_match_summary()
    @app.get('/matches/{match_id}/events',dependencies=secured)
    def events(match_id:str,event_type:str|None=None,team:Literal['left','right']|None=None):
        return recording(match_id).get_events(event_type=event_type,team=team)
    @app.get('/matches/{match_id}/players/{player_id}',dependencies=secured)
    def player(match_id:str,player_id:int):
        if player_id<=0:raise HTTPException(422,'Actor ID must be positive')
        return dispatch(recording(match_id),'get_player_summary',dict(player_id=player_id))
    @app.post('/matches/{match_id}/ask',dependencies=secured)
    def question(match_id:str,body:AskRequest,request:Request):
        tools=recording(match_id)
        if body.event_id and not any(e.event_id==body.event_id for e in tools.events):raise HTTPException(404,'Unknown event')
        name=body.provider or default_provider
        chosen=provider
        if chosen is None and name!='local':
            key=(match_id,name)
            if key not in provider_cache:
                try:provider_cache[key]=build_provider(name,tools)
                except Exception:provider_cache[key]=None
            chosen=provider_cache[key]
        if name=='local':chosen=None
        result=ask(tools,body.question,body.mode,body.timestamp,chosen,body.event_id,body.conversation,match_id,name)
        metrics['unknown_questions']+=int(result.answer.status!='ok');metrics['fallbacks']+=int(result.fallback)
        request.state.query_metrics=dict(provider_requested=name,provider=result.provider,query_type=result.answer.question_type,tools=result.tools,provider_latency_ms=round(result.provider_latency_ms,3),fallback=result.fallback,unknown_question_count=metrics['unknown_questions'])
        return result
    @app.get('/matches/{match_id}/events/{event_id}/explain',response_model=Answer|AgentResponse,dependencies=secured)
    def explain(match_id:str,event_id:str,mode:Literal['beginner','fan','analyst']='fan',with_evidence:bool=False):
        tools=recording(match_id)
        if not any(e.event_id==event_id for e in tools.events):raise HTTPException(404,'Unknown event')
        answer=explain_event(tools,event_id,mode)
        if with_evidence:return AgentResponse(answer=answer,provider='local',provider_status='local_explanation',evidence=evidence_for(answer),conversation=remember(answer,match_id))
        return answer
    @app.get('/matches/{match_id}/report',dependencies=secured)
    def report(match_id:str):
        path=recording(match_id).directory/'report.html'
        if not path.is_file():raise HTTPException(404,'Report unavailable')
        from .playback import report_playback
        tools=recording(match_id)
        page=report_playback(path.read_text(),tools.events,tools.summary.get('fps',25))
        config='<script>window.MATCHMIND_API_BASE='+json.dumps('/matches/'+match_id)+';window.MATCHMIND_DEFAULT_PROVIDER='+json.dumps(default_provider)+';</script>'
        page=page.replace('<script>',config+'<script>',1)
        if (media_root or media_store) and recording(match_id).summary.get('scope')=='clip':
            page=re.sub(r'(<video[^>]*\bsrc=")[^"]+("[^>]*>)',r'\1video.mp4\2',page)
        return HTMLResponse(page)
    @app.head('/matches/{match_id}/{asset}',dependencies=secured,include_in_schema=False)
    @app.get('/matches/{match_id}/{asset}',dependencies=secured)
    def image(match_id:str,asset:str,request:Request):
        if asset=='video.mp4' and match_id in cv_reviews and match_id not in media_ids:
            cv_reviews[match_id].frame(0)
            return FileResponse(cv_reviews[match_id].video,media_type='video/mp4')
        if asset=='video.mp4' and media_store and match_id in media_ids:
            return media_store.response(media_ids[match_id],request)
        tools=recording(match_id)
        if asset=='video.mp4' and media_store and tools.summary.get('scope')=='clip':
            return media_store.response(str(tools.summary.get('match_id','')),request)
        if asset=='video.mp4' and media_root and tools.summary.get('scope')=='clip':
            match=str(tools.summary.get('match_id',''))
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',match):raise HTTPException(404,'Media unavailable')
            root=Path(media_root).resolve();path=(root/(match+'.mp4')).resolve()
            if not path.is_file():path=(root/(match+'.txt.mp4')).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise HTTPException(404,'Media unavailable')
            return FileResponse(path,media_type='video/mp4')
        if asset not in {'motion.png','passes.png','tactical_pressure.png','tactical_shape.png'}:raise HTTPException(404,'Unknown asset')
        path=tools.directory/asset
        if not path.is_file():raise HTTPException(404,'Asset unavailable')
        return FileResponse(path,media_type='image/png')
    return app


def from_environment():
    from dotenv import load_dotenv
    load_dotenv(override=False)
    media_store=None
    if os.environ.get('MATCHMIND_ARTIFACT_STORE','local')=='azure':
        from azure.storage.blob import ContainerClient
        from azure.identity import DefaultAzureCredential
        from src.artifacts import AzureBlobArtifactStore
        credential=DefaultAzureCredential()
        client=ContainerClient(account_url=os.environ['MATCHMIND_BLOB_ACCOUNT_URL'],container_name=os.environ['MATCHMIND_BLOB_CONTAINER'],credential=credential)
        registry=AzureBlobArtifactStore(client,json.loads(os.environ['MATCHMIND_RECORDINGS']),os.environ.get('MATCHMIND_CACHE','/tmp/matchmind-artifacts'))
        if os.environ.get('MATCHMIND_MEDIA_BLOB_CONTAINER'):
            from .media import AzureClipMedia
            # An open-ended browser range must not buffer the SDK's default
            # 32 MiB initial GET before response headers can reach the player.
            media_store=AzureClipMedia(ContainerClient(account_url=os.environ['MATCHMIND_BLOB_ACCOUNT_URL'],container_name=os.environ['MATCHMIND_MEDIA_BLOB_CONTAINER'],credential=credential,max_single_get_size=1024*1024,max_chunk_get_size=1024*1024))
    else:registry=json.loads(Path(os.environ.get('MATCHMIND_REGISTRY','configs/serving/local.json')).read_text())
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    catalog_root=Path(os.environ.get('MATCHMIND_REGISTRY','configs/serving/local.json')).parent.parent/'recordings'
    registry_path=Path(os.environ.get('MATCHMIND_REGISTRY','configs/serving/local.json'))
    processed=json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    catalog=[dict(json.loads(p.read_text()),processed=p.stem in processed) for p in catalog_root.glob('*.json')]
    # Registry membership records local processing, never Azure availability.
    known={item['recording_id'] for item in catalog}
    catalog.extend(dict(recording_id=key,match_id='',processed=True) for key in processed if key not in known)
    cv_path=Path(os.environ.get('MATCHMIND_CV_REGISTRY','configs/serving/cv.json'))
    cv_registry=json.loads(cv_path.read_text()) if cv_path.is_file() else {}
    return create_app(registry,recording_catalog=catalog,token=os.environ.get('MATCHMIND_API_TOKEN'),default_provider=os.environ.get('MATCHMIND_PROVIDER','local'),media_root=os.environ.get('MATCHMIND_MEDIA_ROOT'),media_store=media_store,cv_registry=cv_registry)
