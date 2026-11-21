"""UnifiedMatchAssistant: one session, deterministic evidence, reusable M17 rendering."""
import json,os,re,time,uuid,threading,copy
from typing import Literal
from pathlib import Path
from fastapi import APIRouter,Depends,HTTPException
from fastapi.responses import FileResponse
from .schema import AssistantRequest,IdentityConfirmation,Metric,TeamLabelsRequest
from .catalog import PlayerCatalog
from .identity import ConfirmedIdentityStore
from .portraits import PlayerImageService
from .sprints import SprintService
from .evidence import AssistantEventIndex,PlayerIntelligenceService,TacticalExplanationService
from .planner import IntentAndQueryPlanner
from .composer import ResponseComposer
from .profiles import MatchPresentation
from src.highlights.schema import HighlightRequest,HighlightQueryPlan
from src.highlights.index import digest

class UnifiedMatchAssistant:
    def __init__(self,loaded,cv,highlights,root=None,cv_variants=None,cv_resolver=None,cv_m22_resolver=None,cv_m23_resolver=None):
        self.cv_m23_resolver=cv_m23_resolver
        self.cv_m22_resolver=cv_m22_resolver
        self.cv_resolver=cv_resolver
        self.cv_variants=cv_variants or {}
        self.loaded=loaded;self.cv=cv;self.highlights=highlights;self.root=Path(root or os.environ.get('MATCHMIND_ASSISTANT_CACHE','outputs/assistant_m18'));self.root.mkdir(parents=True,exist_ok=True);self.identities=ConfirmedIdentityStore(self.root/'identities');self.catalogs={};self.indexes={};self.sprints={};self.turns={};self.lock=threading.RLock();self.recording_locks={};self.planner=IntentAndQueryPlanner();self.presentation=MatchPresentation(self);self.portraits=PlayerImageService(self.root/'pictures',highlights,self.catalog)
    def recording_lock(self,recording):
        with self.lock:return self.recording_locks.setdefault(recording,threading.RLock())
    def catalog(self,recording):
        if recording not in self.loaded and recording not in self.cv:raise HTTPException(404,'Recording unavailable')
        with self.recording_lock(recording):
            if recording not in self.catalogs:self.catalogs[recording]=PlayerCatalog(recording,self.loaded.get(recording),self.cv.get(recording))
            return self.catalogs[recording]
    def index(self,recording,pipeline='m14',experiment=None):
        if experiment is not None and pipeline!='m21':raise HTTPException(422,'Experiment selection requires M21')
        key=recording if pipeline=='m14' else (recording,pipeline,experiment) if pipeline=='m21' else (recording,pipeline)
        review=self.cv_resolver(recording,experiment) if pipeline=='m21' and self.cv_resolver else self.cv.get(recording) if pipeline=='m14' else self.cv_variants.get(pipeline,{}).get(recording)
        if pipeline=='m22' and getattr(self,'cv_m22_resolver',None):review=self.cv_m22_resolver(recording)
        if pipeline=='m23' and getattr(self,'cv_m23_resolver',None):review=self.cv_m23_resolver(recording)
        revision=None
        if pipeline in ['m22','m23'] and review is not None:
            revision=':'.join([review.ball_review.manifest['detector_sha256'],review.ball_review.manifest['ball_frames_sha256'],review.event_review.manifest['events_sha256']])
            key=(recording,pipeline,revision)
        if pipeline!='m14' and (recording in self.loaded or review is None):raise HTTPException(503,'Selected experimental event pipeline unavailable')
        with self.recording_lock(recording):
            if key not in self.indexes:
                idx=AssistantEventIndex(recording,self.loaded.get(recording),self.catalog(recording))
                layer=getattr(review,'event_review',None)
                if recording not in self.loaded and layer is not None:idx.attach_cv(layer)
                if revision is not None:idx.cv_model_revision=revision
                self.indexes[key]=idx
            return self.indexes[key]
    def sprint_service(self,recording):
        with self.recording_lock(recording):
            if recording not in self.sprints:self.sprints[recording]=SprintService(self.catalog(recording))
            return self.sprints[recording]
    def previous(self,recording,request):
        if not request.previous_turn:return None
        t=self.turns.get(str(request.previous_turn))
        if not t or t['_recording']!=recording or t['_session']!=str(request.session_id):raise HTTPException(404,'This conversation is not available for this match. Start a new question.')
        return t
    def ask(self,recording,request):
        cat=self.catalog(recording);idx=self.index(recording,request.cv_pipeline,request.cv_experiment) if request.cv_experiment else self.index(recording,request.cv_pipeline);previous=self.previous(recording,request);identity=self.identities.load(recording,request.session_id,cat)
        if request.selected_player and request.selected_player not in cat.by_ref:raise HTTPException(422,'Select a player from this match')
        # Changing focus starts fresh event context; never reuse another player's play.
        if request.selected_player and previous and previous.get('selected_player')!=request.selected_player:previous=None
        if previous and previous.get('cv_pipeline','m14')!=request.cv_pipeline:previous=None
        if previous and previous.get('cv_experiment')!=request.cv_experiment:previous=None
        if previous and previous.get('cv_model_revision')!=getattr(idx,'cv_model_revision',None):previous=None
        selected=request.selected_player or (previous.get('selected_player') if previous else None)
        refs,identity_error=self.identities.resolve(request.question,identity,cat,selected)
        if request.selected_player and not identity_error:
            if refs and request.selected_player not in refs:identity_error='You’re focusing on '+self.identities.label(request.selected_player,identity,cat)+'. Choose the other player on the pitch to ask about them.'
            refs=[request.selected_player]
        planned_question=request.question
        for side,team_label in identity.get('teams',{}).items():planned_question=re.sub(r'(?<!\w)'+re.escape(team_label['name'])+r'(?!\w)','Team A' if side=='left' else 'Team B',planned_question,flags=re.I)
        unresolved=re.search(r'\b(?:by|for)\s+(?:the\s+)?([A-Za-z]+)',planned_question,re.I)
        if (request.provider=='local' or not os.environ.get('GEMINI_API_KEY')) and not refs and unresolved and unresolved[1].lower() not in ['team','player','this','that','both','all','left','right','best','available']:
            identity_error='Please confirm which player or team you mean in Match setup, or select their card.'
        try:plan,reason,provider,diagnostic=self.planner.plan(planned_question,refs,previous,'local' if identity_error else request.provider,roster=[{'player_ref':p['player_ref'],'label':p['label'],'name':identity['players'].get(p['player_ref'],{}).get('name'),'jersey':identity['players'].get(p['player_ref'],{}).get('jersey'),'team':identity['players'].get(p['player_ref'],{}).get('team',p['team'])} for p in cat.players],match_context={'duration_seconds':cat.frames/cat.fps,'period':self.loaded[recording].summary.get('half') if recording in self.loaded else None,'selected_player':selected,'recording_id':recording})
        except ValueError:raise HTTPException(422,'Please use a valid time range and a video length between 5 and 120 seconds.') from None
        if request.selected_player and plan.operation not in ['score','unsupported']:
            plan=plan.model_copy(update={'players':[request.selected_player]})
        evidence_recording=recording
        # A full-match goal request must not miss first-half goals simply because
        # the user is currently viewing the second half. Redirect only within the
        # same verified match, with the original source retained on every event.
        match_id=str(self.loaded[recording].summary.get('match_id')) if recording in self.loaded else None
        if previous and plan.reference in ['event','player']:
            candidate=previous.get('source_recording',recording)
            if candidate in self.loaded and str(self.loaded[candidate].summary.get('match_id'))==match_id:evidence_recording=candidate
        if plan.operation=='actions' and plan.actions==['goal'] and plan.video and plan.period is None and not plan.players and plan.team is None and not re.search(r'\b(this half|selected half|this clip|selected footage)\b',request.question,re.I):
            half_sources=[key for key,t in self.loaded.items() if str(t.summary.get('match_id'))==match_id and t.summary.get('scope')=='half']
            goal_sources=[key for key in half_sources if self.index(key).select(['goal'])]
            if len(goal_sources)==1:evidence_recording=goal_sources[0]
        if evidence_recording!=recording:
            cat=self.catalog(evidence_recording);idx=self.index(evidence_recording);identity=self.identities.load(evidence_recording,request.session_id,cat)
        def retrieve(actions=None,refs=None,team=None,start=None,end=None):
            rows=idx.select(actions,refs,None,start,end)
            if team is None:return rows
            return [e for e in rows if identity['players'].get('actor:'+str(e['actor_id']),{}).get('team',e['team'])==team]
        cv_layer=getattr(idx,'cv_layer',None)
        if cv_layer and (plan.actions and any(k not in ['pass','reception'] for k in plan.actions) or plan.operation=='score'):
            plan=plan.model_copy(update={'operation':'unsupported'});reason='This video analysis estimates passes and receptions only. Goals, shots, tackles and a match score are not available yet.'
        if cv_layer and plan.review and plan.operation=='player_profile':plan=plan.model_copy(update={'operation':'actions','actions':['pass'],'video':True})
        rows=[];metrics=[];illustration=None;text='';highlight=None;sprint=None;old_answer=None;profile=None
        if identity_error:text=identity_error
        elif plan.period is not None and (idx.timebase is None or idx.timebase.period!=plan.period):text='I can’t verify which half this footage belongs to.' if idx.timebase is None or idx.timebase.period is None else 'That half is not included in this recording.'
        elif plan.operation=='unsupported':text=reason
        elif plan.clarification:text=plan.clarification
        elif plan.operation=='score':
            score=self.presentation.score(recording);text=f"The recorded score is Team A {score['left']}–{score['right']} Team B."+(' This covers the available full match.' if score['scope']=='full_match' else ' This covers the available footage, not a verified full-match final score.')
        elif plan.operation=='player_profile':
            if not plan.players:text='Select a player on the pitch, or confirm their number in Match setup, and I’ll show their statistics.'
            else:
                selected=plan.players[0];profile=self.presentation.profile(evidence_recording,selected,request.session_id,request.cv_pipeline,request.cv_experiment) if request.cv_experiment else self.presentation.profile(evidence_recording,selected,request.session_id,request.cv_pipeline);metrics=profile['metrics'];text=ResponseComposer.player_summary(profile['label'],metrics)
                if plan.video:rows=retrieve(plan.actions or None,[selected],plan.team,plan.start,plan.end)
        elif plan.operation=='sprint':
            service=self.sprint_service(evidence_recording);sprint=next((r for r in service.intervals if (not plan.players or r['player_ref'] in plan.players) and (plan.team is None or identity['players'].get(r['player_ref'],{}).get('team',r['team'])==plan.team) and (plan.start is None or r['start']>=plan.start) and (plan.end is None or r['end']<=plan.end)),None)
            if not sprint:
                sprint=next((r for r in getattr(service,'runs',[]) if (not plan.players or r['player_ref'] in plan.players) and (plan.team is None or identity['players'].get(r['player_ref'],{}).get('team',r['team'])==plan.team) and (plan.start is None or r['start']>=plan.start) and (plan.end is None or r['end']<=plan.end)),None)
            if not sprint:text=service.diagnostic or 'There isn’t enough continuous, reliable position tracking to measure this player’s fastest run in this footage.'
            else:
                selected=sprint['player_ref'];row=idx.add_sprint(sprint);rows=[row];name=self.identities.label(selected,identity,cat);text=f"The fastest qualifying recorded sprint was by {name}, at an estimated sustained speed of {sprint['speed_mps']:.2f} meters per second. This is an estimate over a continuous run, not a single-frame peak."
                if not sprint.get('qualifies_as_sprint',True):text=f"{name}’s fastest measured run averaged {sprint['speed_mps']:.2f} meters per second over one second. It doesn’t meet the sustained sprint criteria, so this is labelled as a run."
                metrics=[Metric(name='Estimated sustained sprint speed' if sprint.get('qualifies_as_sprint',True) else 'Fastest measured run speed',value=sprint['speed_mps'],unit='m/s',definition=sprint['definition'],source='source_pitch_trajectories',support='estimated',samples=sprint['samples'],window=(sprint['start'],sprint['end']),player_ref=selected,limitations=['No independent speed ground truth; camera/pitch annotation uncertainty is inherited.']).model_dump()]
        elif plan.operation in ['identity','followup'] or plan.operation=='explain' and plan.reference=='event':
            event_ids=previous.get('event_ids',[]) if previous else []
            if previous and previous.get('highlight') and plan.ordinal:
                job=self.highlights.jobs.get(previous['highlight']['job_id'])
                if job and job['state']!='COMPLETED':text='Let the current video finish before choosing a moment.'
                elif job:event_ids=[e['event_id'] for e in job['events']]
            if plan.ordinal:event_ids=event_ids[plan.ordinal-1:plan.ordinal]
            rows=[e for e in idx.events if e['event_id'] in event_ids]
            if not text and not rows:text='Choose a recorded moment first so I know which play you mean.'
            if rows and not text:
                selected='actor:'+str(rows[0]['actor_id']) if rows[0]['actor_id'] else selected
                if plan.operation=='identity':text=(f"That was {self.identities.label(selected,identity,cat)}."+(' You haven’t confirmed their number yet; you can do that in Match setup.' if selected not in identity['players'] else ' You confirmed this identity in Match setup.')) if selected else 'I can’t reliably identify the player in that moment.'
                elif plan.explain and rows[0]['event_type']=='pass':text=TacticalExplanationService.pass_explanation(rows[0])
                else:text='Here is that recorded play again.'
        elif plan.operation=='pass_rank':
            rows=retrieve(['pass'],plan.players,plan.team,plan.start,plan.end);key='difficulty_score_0_100' if plan.metric=='difficulty' else 'distance_m';rows=[e for e in rows if key in e['metrics']];rows=sorted(rows,key=lambda e:(-e['metrics'][key],e['start']))[:1]
            if not rows:text='I don’t have enough reliable pass measurements to rank that request.'
            else:
                e=rows[0];selected='actor:'+str(e['actor_id']);name=self.identities.label(selected,identity,cat)
                if plan.metric=='difficulty':text=f"The most difficult recorded completed pass was made by {name}, according to the estimated difficulty score. "+TacticalExplanationService.pass_explanation(e)
                else:text=f"The longest recorded completed pass was made by {name} and covered about {e['metrics']['distance_m']:.1f} meters. This is an estimated start-to-finish distance."
                if plan.explain and plan.metric=='distance':text+=' '+TacticalExplanationService.pass_explanation(e)
        elif plan.operation=='compare':
            if not self.loaded.get(recording):text='I don’t have reliable football-action statistics to compare players or teams in this footage.'
            elif len(plan.players)==2:
                stats=PlayerIntelligenceService(idx,self.loaded.get(recording),cat);parts=[]
                for ref in plan.players:
                    ms=stats.statistics(ref);metrics+=ms;counts={m['name']:m['value'] for m in ms};distance=counts.get('Tracked distance');distance_text=f"{distance:.1f} meters of tracked movement" if distance is not None else 'tracked distance unavailable';parts.append(f"{self.identities.label(ref,identity,cat)}: {int(counts.get('Recorded shots',0))} recorded shots, {int(counts.get('Completed pass estimates',0))} completed pass estimates, and {distance_text}")
                text='; '.join(parts)+'. These figures cover only the available recording.'
            else:
                kind='pass' if plan.metric=='passes' else 'tackle' if plan.metric=='tackles' else 'shot';left=retrieve([kind],team='left');right=retrieve([kind],team='right');text=f"Team A has {len(left)} recorded {kind} entries and Team B has {len(right)} in this footage. These are {'completed pass estimates' if kind=='pass' else 'recorded entries'}; this is not a claim that every action was recorded. Goals are listed separately."
        elif plan.operation=='statistic' and plan.metric=='distance' and not plan.players:
            ts=self.loaded.get(recording);eligible=[p for p in ts.players if p.get('measured_distance_m') is not None and (p.get('valid_motion_fraction') or 0)>=.9] if ts else []
            if not eligible:text='I don’t have enough reliable pitch measurements to compare distances here.'
            else:
                winner=max(eligible,key=lambda p:p['measured_distance_m']);selected='actor:'+str(int(winner['player_id']));text=f"{self.identities.label(selected,identity,cat)} covered the farthest tracked distance in the available footage: about {winner['measured_distance_m']:.1f} meters. Missing observations are not filled in."
                metrics=PlayerIntelligenceService(idx,ts,cat).statistics(selected)
        elif plan.operation=='statistic' and not plan.players and re.search(r'\b(how many|count)\b',request.question,re.I):
            if not self.loaded.get(recording) and not cv_layer:text='I don’t have verified football-action records for this footage.'
            else:
                rows=retrieve(plan.actions or None,None,plan.team,plan.start,plan.end);kind=plan.actions[0] if plan.actions else 'action'
                text=f"I found {len(rows)} recorded {kind} entries in this footage. "
                if kind=='pass':text+='These are completed pass estimates, not independently verified match pass totals. '
                text+='This does not guarantee a complete match count.'
                metrics=[Metric(name='Recorded '+kind+' entries',value=len(rows),unit='events',definition='Eligible existing event entries in the requested recording and time range.',source='canonical_event_exports',support='estimated' if kind in ['pass','reception','interception'] else 'recorded',samples=len(rows),window=(plan.start or 0,plan.end if plan.end is not None else cat.frames/cat.fps)).model_dump()]
        elif plan.operation=='statistic' and not plan.players and plan.metric in ['shots','passes','tackles','possession','speed']:
            ts=self.loaded.get(recording);stats=PlayerIntelligenceService(idx,ts,cat)
            title={'shots':'Recorded shots','passes':'Completed pass estimates','tackles':'Recorded tackles','possession':'Estimated direct ball control','speed':'Peak estimated speed'}[plan.metric]
            candidates=[m for p in cat.players for m in stats.statistics(p['player_ref']) if m['name']==title and m['value'] is not None]
            if not candidates:text='I don’t have reliable measurements to rank that request in this footage.'
            else:
                winner=max(candidates,key=lambda m:m['value']);metrics=[winner];selected=winner['player_ref'];text=f"{self.identities.label(selected,identity,cat)} has the highest {title.lower()} in the available records: {winner['value']:.2f} {winner['unit']}. This covers the available footage only."
                if plan.metric=='possession':text+=' This counts estimated direct ball control, not a complete possession sequence.'
                if plan.metric=='speed':text+=' This is a smoothed peak estimate, not a sustained sprint ranking.'
        elif plan.operation=='statistic' and plan.players:
            service=PlayerIntelligenceService(idx,self.loaded.get(recording),cat);selected=plan.players[0];metrics=service.statistics(selected)
            if 'sprint' in plan.actions:
                for sprint_item in self.sprint_service(evidence_recording).intervals:
                    if sprint_item['player_ref']==selected:idx.add_sprint(sprint_item)
            rows=retrieve(plan.actions or None,plan.players,plan.team,plan.start,plan.end)
            if not metrics:text='I don’t have verified football-action records for this player in this footage.'
            else:
                name=self.identities.label(selected,identity,cat);kind=plan.actions[0] if plan.actions else 'actions'
                if re.search(r'completion rate|completion percentage|percentage.*pass',request.question,re.I):
                    completion=next(m for m in metrics if m['name']=='Estimated pass completion');metrics=[completion];text=f"{name} completed an estimated {completion['value']:.1f}% of the coverage-qualified pass attempts in this recording." if completion['value'] is not None else 'I don’t have enough qualifying pass attempts to calculate a completion rate.'
                else:
                    phrase={'shot':f"took {len(rows)} shot{'s' if len(rows)!=1 else ''}",'goal':f"scored {len(rows)} goal{'s' if len(rows)!=1 else ''}",'pass':f"completed {len(rows)} recorded passes",'tackle':f"made {len(rows)} tackle{'s' if len(rows)!=1 else ''}",'interception':f"made {len(rows)} estimated interceptions"}.get(kind,f"has {len(rows)} recorded {kind}")
                    text=f"{name} {phrase} in the requested footage."+(' Pass counts are tracking estimates.' if kind=='pass' else '')
                    metrics=[Metric(name='Recorded '+kind+' entries',value=len(rows),unit='events',definition='Eligible existing event entries explicitly assigned to this player in the requested window; no nearest-player attribution.',source='existing_event_records',support='estimated' if kind in ['pass','reception','interception','sprint','actions'] else 'recorded',samples=len(rows),window=(plan.start or cat.by_ref[selected]['start'],plan.end if plan.end is not None else cat.by_ref[selected]['end']),player_ref=selected).model_dump()]
        elif plan.operation=='recap':
            rows,_=idx.retrieve(HighlightQueryPlan(selection='recap',duration=plan.duration))
            text=f'I found {len(rows)} recorded moments for a short recap. It prioritizes recorded goals, shots and other available plays; it is not an objective excitement rating.' if rows else 'I couldn’t find reliable recorded moments for a recap in this footage.'
        elif plan.operation in ['actions','explain']:
            refs=plan.players or ([selected] if selected and re.search(r'\b(his|her|their|this player|other actions)\b',request.question,re.I) else [])
            if 'sprint' in plan.actions or refs and not plan.actions:
                for s in self.sprint_service(evidence_recording).intervals:
                    if not refs or s['player_ref'] in refs:idx.add_sprint(s)
            actions=plan.actions or (['goal','shot','pass','reception','cross','carry','header'] if 'attacking' in request.question.lower() else None)
            rows=retrieve(actions,refs,plan.team,plan.start,plan.end)
            if cv_layer and plan.review and (not actions or 'pass' in actions):
                rows=rows+[e for e in idx.candidates if (not refs or e['player_ref'] in refs) and (plan.team is None or e['team']==plan.team) and (plan.start is None or e['start']>=plan.start) and (plan.end is None or e['end']<=plan.end)]
                rows.sort(key=lambda e:e['start'])
            if 'other' in request.question.lower() and previous:rows=[e for e in rows if e['event_id'] not in previous.get('event_ids',[])]
            if plan.ordinal:rows=rows[plan.ordinal-1:plan.ordinal]
            if not rows:
                if cv_layer:text='No completed pass estimates meet the visual evidence gates for this selection. You can ask to review possible passes; goals and shots remain unavailable.'
                elif not self.loaded.get(recording):text='I can show this footage, but I don’t have confirmed football actions for it yet.'
                else:text='I couldn’t confirm any matching recorded actions in this footage.'
            else:
                if refs:selected=refs[0]
                elif len(rows)==1 and rows[0]['actor_id']:selected='actor:'+str(rows[0]['actor_id'])
                if refs and selected:
                    summary_metrics=PlayerIntelligenceService(idx,self.loaded.get(evidence_recording),cat).statistics(selected)
                    text=ResponseComposer.player_summary(self.identities.label(selected,identity,cat),summary_metrics)
                    if plan.video:text+=' Here’s the video of '+('their '+', '.join(plan.actions) if plan.actions else 'their available actions')+'.'
                else:
                    counts={kind:sum(e['event_type']==kind for e in rows) for kind in dict.fromkeys(e['event_type'] for e in rows)}
                    text='Here '+('is ' if len(rows)==1 else 'are ')+', '.join(f"{n} {kind}{'s' if n!=1 else ''}" for kind,n in counts.items())+' from this footage.'
                if plan.explain and plan.actions==['goal']:text='I can show the recorded goal and the surrounding play, but I can’t prove why it happened from these records.'
        else:
            tools=self.loaded.get(recording)
            if tools:
                from src.providers.agent import ask
                from src.providers.factory import build_provider
                q=request.question
                for ref in plan.players:
                    annotation=identity['players'].get(ref,{})
                    if annotation.get('jersey') is not None:q=re.sub(r'\bplayer\s*#?'+str(annotation['jersey'])+r'\b','player '+str(cat.by_ref[ref]['actor_id']),q,flags=re.I)
                    if annotation.get('name'):q=re.sub(re.escape(annotation['name']),'player '+str(cat.by_ref[ref]['actor_id']),q,flags=re.I)
                chosen=None;name='local' if request.provider=='auto' else request.provider
                if name!='local':
                    try:chosen=build_provider(name,tools)
                    except Exception:chosen=None
                answer=ask(tools,q,'fan',0,chosen,None,None,recording,name);old_answer=answer.model_dump(mode='json');text=' '.join(s.text for s in answer.answer.statements[:3]) if answer.answer.status=='ok' else 'I don’t have enough reliable match evidence to answer that question.'
            else:text='I can help inspect players and footage, but verified football statistics are not available for this recording.'
        if cv_layer:
            if any(not e['safe_for_automatic'] for e in rows):text=f"Here are {len(rows)} possible transfers to review. They are not confirmed passes and are excluded from completed-pass counts."
            text+=f" This is experimental video analysis using {cv_layer.manifest['ball_version'].upper()} ball observations; sparse visibility can miss actions."
        for side,team_label in identity.get('teams',{}).items():text=text.replace('Team A' if side=='left' else 'Team B',team_label['name'])
        text=ResponseComposer.readable(text or 'I couldn’t verify that request from the available match evidence.',cat,identity,self.identities)
        if rows and rows[0]['event_type']=='pass':
            m=rows[0]['metrics']
            if all(k in m for k in ['start_x_m','start_y_m','end_x_m','end_y_m']):illustration={'origin':[m['start_x_m'],m['start_y_m']],'destination':[m['end_x_m'],m['end_y_m']],'label':'Start-to-finish illustration, not the actual ball path'}
        if rows and plan.video and not identity_error and plan.operation!='identity':
            hp=HighlightQueryPlan(event_types=list(dict.fromkeys(e['event_type'] for e in rows)),selection='all',duration=plan.duration or 120,buildup=plan.buildup,review=plan.review)
            overlay=None
            mark_player=selected if (plan.players or plan.overlay or plan.operation=='sprint') else None
            can_mark=bool(mark_player and (cat.has_marker(mark_player) if hasattr(cat,'has_marker') else cat.boxes.get(mark_player)))
            if can_mark:overlay={'catalog':cat,'player_ref':mark_player,'label':self.identities.label(mark_player,identity,cat),'source_signature':cat.source_signature,'point_source_hash':getattr(cat,'point_source_hash',None),'identity_revision':identity['revision'],'version':'player-focus-overlay-v3','candidate_review':any(not e['safe_for_automatic'] for e in rows)}
            elif mark_player:text+=' A reliable on-video player marker isn’t available for this source.'
            if any(not e['safe_for_automatic'] for e in rows) and overlay is None:
                overlay={'catalog':cat,'player_ref':None,'label':'Possible transfer','source_signature':cat.source_signature,'version':'player-focus-overlay-v3','candidate_review':True}
            highlight=self.highlights.submit(evidence_recording,HighlightRequest(question=request.question,session_id=request.session_id,provider='local'),selection=(hp,rows,copy.copy(idx),overlay))
        if evidence_recording!=recording and rows:
            text+=' The source footage is the '+('first' if self.loaded[evidence_recording].summary['half']==1 else 'second')+' half of this match.'
        turn_id=str(uuid.uuid4());turn={'turn_id':turn_id,'cv_pipeline':request.cv_pipeline,'cv_experiment':request.cv_experiment,'cv_model_revision':getattr(idx,'cv_model_revision',None),'source_recording':evidence_recording,'text':text,'plan':plan.model_dump(),'metrics':metrics,'events':rows,'illustration':illustration,'selected_player':selected,'players':plan.players or ([selected] if selected else []),'event_ids':[e['event_id'] for e in rows],'highlight':highlight,'player_profile':profile,'provider':provider,'provider_note':diagnostic,'identity_revision':identity['revision'],'evidence':old_answer,'_session':str(request.session_id),'_recording':recording,'_created':time.time()}
        with self.lock:
            expired=[k for k,v in self.turns.items() if time.time()-v['_created']>86400]
            for k in expired:del self.turns[k]
            own=[k for k,v in self.turns.items() if v['_session']==str(request.session_id) and v['_recording']==recording]
            for key in own[:-99]:del self.turns[key]
            while len(self.turns)>=2000:del self.turns[next(iter(self.turns))]
            self.turns[turn_id]=turn
        return {k:v for k,v in turn.items() if not k.startswith('_')}
    def router(self,authorize):
        r=APIRouter(dependencies=[Depends(authorize)])
        @r.post('/matches/{recording}/assistant')
        def ask(recording:str,body:AssistantRequest):return self.ask(recording,body)
        @r.get('/matches/{recording}/score')
        def score(recording:str):return self.presentation.score(recording)
        @r.get('/matches/{recording}/players/profile')
        def profile(recording:str,player_ref:str,session_id:uuid.UUID,cv_pipeline:Literal['m14','m19','m20','m21','m22','m23']='m14',cv_experiment:str|None=None):return self.presentation.profile(recording,player_ref,session_id,cv_pipeline,cv_experiment) if cv_experiment else self.presentation.profile(recording,player_ref,session_id,cv_pipeline)
        @r.get('/matches/{recording}/players/roster')
        def roster(recording:str,session_id:uuid.UUID):
            cat=self.catalog(recording);return {**cat.public(),'identities':self.identities.load(recording,session_id,cat),'pictures':self.portraits.public(recording)}
        @r.post('/matches/{recording}/players/pictures')
        def pictures(recording:str):
            self.catalog(recording)
            try:return self.portraits.start(recording)
            except ValueError as e:raise HTTPException(429,str(e)) from None
        @r.get('/matches/{recording}/players/pictures/{image_id}.jpg')
        def picture(recording:str,image_id:str):
            p=self.portraits.image(recording,image_id)
            if not p:raise HTTPException(404,'Player picture unavailable')
            return FileResponse(p,media_type='image/jpeg')
        @r.post('/matches/{recording}/players/team-labels')
        def teams(recording:str,body:TeamLabelsRequest):return self.identities.save_teams(recording,body,self.catalog(recording))
        @r.post('/matches/{recording}/players/identities')
        def confirm(recording:str,body:IdentityConfirmation):return self.identities.confirm(recording,body,self.catalog(recording),self.portraits.public(recording).get('players'))
        @r.delete('/matches/{recording}/players/identities/{player_ref}')
        def remove(recording:str,player_ref:str,session_id:uuid.UUID):return self.identities.remove(recording,session_id,player_ref,self.catalog(recording))
        return r
