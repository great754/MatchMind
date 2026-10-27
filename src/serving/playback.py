"""Adapt report navigation in memory; immutable Azure HTML remains unchanged."""
import json
from pathlib import Path

def report_playback(page, events, fps):
    if 'id="tactiq-playback-policy"' in page:return page
    policy=(Path(__file__).parent/'static'/'playback.js').read_text().replace('export ','')
    payload=json.dumps([e.model_dump(mode='json') if hasattr(e,'model_dump') else e for e in events],allow_nan=False).replace('<',r'\u003c')
    replacements={
        'seek(item.timestamp)':'seekMoment(item.timestamp,item.event_ids?.[0])',
        'seek(event.timestamp)':'seekMoment(event.timestamp,event.event_id)',
        'seek(statement.timestamp)':'seekMoment(statement.timestamp,statement.event_id)',
        'seek(askContext.last_timestamp)':'seekMoment(askContext.last_timestamp,askContext.last_event_id)',
        'seek(Number(b.dataset.seek))':'seekMoment(Number(b.dataset.seek))',
        'seek(Number.isFinite(requestedTime)?Math.max(0,Math.min((data.frames-1)/data.fps,requestedTime)):0);':"seekMoment(Number.isFinite(requestedTime)?Math.max(0,Math.min((data.frames-1)/data.fps,requestedTime)):0,typeof window==='undefined'?null:new URLSearchParams(window.location.search).get('event_id'));",
        'showAsk(askData.explanations[event.event_id]);highlightAsk(event.event_id);':'seekMoment(event.timestamp,event.event_id);showAsk(askData.explanations[event.event_id]);highlightAsk(event.event_id);',
    }
    for before,after in replacements.items():page=page.replace(before,after)
    script='<script id="tactiq-playback-policy">'+policy+'\nconst playbackEvents='+payload+';\n'
    script+='''function seekMoment(timestamp,eventId){
 const matches=playbackEvents.filter(e=>eventId?e.event_id===eventId:e.timestamp===timestamp);
 const event=matches.length===1?matches[0]:null;
 const end=video&&Number.isFinite(video.duration)?Math.min(video.duration,(data.frames-1)/data.fps):(data.frames-1)/data.fps;
 seek(getPlaybackSeekTime(timestamp,event,'''+json.dumps(fps)+''',end));
 if(event&&typeof highlightAsk==='function')highlightAsk(event.event_id);
}\n</script>'''
    return page.replace('<script>',script+'<script>',1)
