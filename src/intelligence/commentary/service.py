"""Bounded generation, deterministic cache and strict output grounding."""
import hashlib
import json
from pathlib import Path
from ..schema import ContextFacts,EventType as T
from .catalogue import sentences

PROMPT_VERSION='1.2'
MODES=('beginner','fan','analyst')
INSTRUCTION='Select 1–3 sentence_ids from the supplied catalogue. Return only JSON with sentence_ids. Unsupported identities, jersey numbers, events, statistics, scores, formations, tactics, speeds and outcomes are forbidden. Tracker IDs are not jerseys. Use only supplied deterministic tactical facts; formations are estimates, not verified lineups or player roles. Tactical intentions, manager instructions, pressing traps and causal explanations are forbidden. Legacy PASS event names denote observable transfers, not proven intentional passes. Difficulty is retrospective geometry, never calibrated probability. Ball speed is estimated 2D motion, never measured strike velocity. Preserve uncertainty: ball data is interpolated 2D with no height; possession, transfers and shot candidates are estimates; BAS is not independent validation; team/goalkeeper classification, panoramic calibration and timing may be wrong. Do not write free-form prose. Select useful, non-redundant statements in the requested mode.'

def request_for(events,context: dict,mode: str) -> dict:
    if mode not in MODES: raise ValueError('Unknown commentary mode')
    context=ContextFacts.model_validate(context).model_dump(mode='json',exclude_none=False)
    # Closed event models and deterministic contexts only; never arbitrary raw logs.
    return dict(prompt_version=PROMPT_VERSION,schema_version='1.0',mode=mode,instruction=INSTRUCTION,
        events=[e.model_dump(mode='json',exclude_none=True) for e in events],context=context,
        sentences=sentences(events,context,mode))

def cache_key(request: dict,configuration: dict) -> str:
    return hashlib.sha256(json.dumps({'request':request,'model':configuration},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_plan(raw,request: dict) -> list[str]:
    plan=json.loads(raw) if isinstance(raw,str) else raw
    if not isinstance(plan,dict) or set(plan)!= {'sentence_ids'}: raise ValueError('Invalid commentary selection')
    ids=plan['sentence_ids']
    if not isinstance(ids,list) or not 1<=len(ids)<=3 or any(not isinstance(i,str) or i not in request['sentences'] for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Unsupported commentary selection')
    return ids

def select_moments(events,config):
    eligible=[e for e in events if e.significance>=config.significance_threshold and e.timestamp>=config.start_seconds and (config.end_seconds is None or e.timestamp<=config.end_seconds)]
    grouped=[]
    for event in eligible:
        if grouped and event.timestamp-grouped[-1][0].timestamp<=config.grouping_seconds: grouped[-1].append(event)
        else: grouped.append([event])
    selected=[]; last=-float('inf')
    for group in grouped:
        # One representative per detector group (e.g. difficult + long + completed transfer).
        representatives={}
        transfers=[e for e in group if e.source=='transfer']
        for event in group:
            if event.type==T.POSSESSION_CHANGED and any(e.receiver_id==event.player_id and e.frame==event.frame for e in transfers): continue
            if event.group_id not in representatives or event.significance>representatives[event.group_id].significance: representatives[event.group_id]=event
        group=sorted(representatives.values(),key=lambda e:(-e.significance,e.event_id))[:3]
        timestamp=max(e.timestamp for e in group)
        if timestamp-last<config.commentary_cooldown_seconds and max(e.significance for e in group)<80: continue
        selected.append(group);last=timestamp
        if len(selected)>=config.max_commentary_moments: break
    return selected if config.max_commentary_moments else []

def generate_commentary(events,builder,config,provider=None,cache_dir=None):
    output=dict(schema_version='1.0',provider=provider.configuration if provider else {'provider':'none'},prompt_version=PROMPT_VERSION,items=[],errors=[],provider_calls=0,cache_hits=0)
    if provider is None: return output
    cache=Path(cache_dir) if cache_dir else None
    if cache: cache.mkdir(parents=True,exist_ok=True)
    for group in select_moments(events,config):
        timestamp=max(e.timestamp for e in group);context=builder.at(timestamp,config.context_window_seconds)
        for mode in MODES:
            request=request_for(group,context,mode)
            if not request['sentences']: continue
            key=cache_key(request,provider.configuration);path=cache/f'{key}.json' if cache else None
            ids=None;hit=False
            if path and path.exists():
                try: ids=validate_plan(json.loads(path.read_text()),request);hit=True
                except (ValueError,OSError): pass
            if ids is None:
                if output['provider_calls']>=config.max_provider_calls: continue
                output['provider_calls']+=1
                try:
                    ids=validate_plan(provider.generate(request),request)
                except Exception:
                    output['errors'].append({'timestamp':timestamp,'mode':mode,'message':'Commentary unavailable: provider failure or unsupported response.'});continue
                if path:
                    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps({'sentence_ids':ids}));temporary.replace(path)
            output['cache_hits']+=int(hit)
            output['items'].append(dict(timestamp=timestamp,frame=int(round(timestamp*builder.recording.fps)),mode=mode,
                event_ids=[e.event_id for e in group],event_types=[e.type.value for e in group],context_window=config.context_window_seconds,
                commentary=' '.join(request['sentences'][i] for i in ids),sentence_ids=ids,cache_key=key,cached=hit))
    return output
