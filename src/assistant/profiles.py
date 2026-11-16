"""Presentation-only scorecards and player maps over validated source evidence."""
import json,math,threading
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from .evidence import PlayerIntelligenceService,finite
from src.highlights.render import file_hash

class MatchPresentation:
    def __init__(self,assistant):self.assistant=assistant;self.positions={};self.lock=threading.Lock()
    def score(self,recording):
        a=self.assistant;t=a.loaded.get(recording)
        if not t:raise HTTPException(404,'Football match evidence unavailable')
        match=str(t.summary['match_id']);halves={int(v.summary['half']):(key,v) for key,v in a.loaded.items() if str(v.summary.get('match_id'))==match and v.summary.get('scope')=='half'}
        sources=list(halves.values()) if halves else [(recording,t)];goals={};unknown=0
        for key,tools in sources:
            for e in tools.events:
                if e.source=='bas' and e.type.value=='GOAL' and e.validated_by_bas:
                    identity=e.bas_annotation_id if e.bas_annotation_id is not None else (tools.summary['half'],e.frame,e.actor_id)
                    goals[identity]={'team':e.team,'timestamp':e.timestamp,'recording_id':key,'event_id':e.event_id}
        counts={team:sum(g['team']==team for g in goals.values()) for team in ['left','right']};unknown=sum(g['team'] not in counts for g in goals.values())
        return {'recording_id':recording,'left':counts['left'],'right':counts['right'],'scope':'full_match' if set(halves)=={1,2} else 'half' if halves else 'recording','label':'Recorded match score' if set(halves)=={1,2} else 'Goals in available footage','unassigned_goals':unknown,'goals':list(goals.values()),'definition':'Distinct validated goal annotations, attributed to their recorded team; not a live or externally verified final score.'}
    def trajectory(self,recording,actor):
        cat=self.assistant.catalog(recording);t=cat.tools
        if t is None:
            positions=getattr(cat,'metric_positions',{}).get(actor,{})
            if not positions:return None
            frames=np.array(sorted(positions),int);points=np.array([positions[int(f)] for f in frames],float)
            return points,frames,cat.source_signature
        summary=t.summary
        if summary.get('player_coordinate_source')!='gsr':return None
        half='1st' if summary['half']==1 else '2nd';path=Path('data/soccertrack/gsr')/str(summary['match_id'])/(half+'_compact.npz');expected=next((f['sha256'] for f in summary.get('source_files',[]) if Path(f['path']).name==path.name),None)
        if not expected:return None
        with self.lock:
            key=(recording,expected)
            if key not in self.positions:
                if not path.is_file() or file_hash(path)!=expected:return None
                with np.load(path) as data:
                    offset=int(summary['source_offset_frames']);self.positions[key]=(data['ids'][offset:offset+cat.frames].copy(),data['pitch'][offset:offset+cat.frames].copy())
            ids,pitch=self.positions[key];frames,slots=np.where(ids==actor)
            if len(set(frames))!=len(frames):return None
            points=pitch[frames,slots].astype(float);valid=np.isfinite(points).all(axis=1)&(points[:,0]>=0)&(points[:,0]<=105)&(points[:,1]>=0)&(points[:,1]<=68)
            return points[valid],frames[valid],expected
    def profile(self,recording,ref,session,pipeline='m14'):
        a=self.assistant;cat=a.catalog(recording)
        if ref not in cat.by_ref:raise HTTPException(404,'Choose a player from this recording')
        player=cat.by_ref[ref];identity=a.identities.load(recording,session,cat);index=a.index(recording) if pipeline=='m14' else a.index(recording,pipeline);metrics=PlayerIntelligenceService(index,a.loaded.get(recording),cat).statistics(ref);m={r['name']:r for r in metrics}
        stats=[]
        mapping=[('shots','Shots','Recorded shots'),('goals','Goals','Recorded goals'),('passes','Passes completed','Completed pass estimates'),('pass_attempts','Pass attempts','Recorded pass attempts'),('pass_accuracy','Pass accuracy','Estimated pass completion'),('tackles','Tackles','Recorded tackles'),('interceptions','Interceptions','Interception estimates'),('receptions','Passes received','Reception estimates'),('crosses','Crosses','Recorded crosses'),('distance','Distance covered','Tracked distance'),('peak_speed','Peak speed','Peak estimated speed')]
        for key,label,title in mapping:
            row=m.get(title);stats.append({'key':key,'label':label,'value':row['value'] if row else None,'unit':row['unit'] if row else '', 'support':row['support'] if row else 'unavailable','definition':row['definition'] if row else 'No reliable evidence for this metric.'})
        heatmap=None;observed=None
        trajectory=self.trajectory(recording,player['actor_id'] if player['actor_id'] is not None else ref)
        if trajectory is not None:
            points,frames,source_hash=trajectory;grid=np.histogram2d(points[:,1],points[:,0],bins=[21,32],range=[[0,68],[0,105]])[0]/cat.fps
            observed=len(points)/cat.fps;heatmap={'width':32,'height':21,'seconds':grid.tolist(),'observed_samples':len(points),'observed_seconds':observed,'source_sha256':source_hash,'definition':'Observed metric pitch-position occupancy; missing/out-of-pitch samples excluded. No movement or playing time is invented.'}
        passes=[]
        t=a.loaded.get(recording)
        if t:
            for row in t.transfers:
                if int(row.get('sender_id') or -1)!=player['actor_id'] or (finite(row.get('observed_step_fraction')) or 0)<.9:continue
                vals=[finite(row.get(k)) for k in ['start_x_m','start_y_m','end_x_m','end_y_m']]
                if any(v is None for v in vals) or not all(0<=v<=limit for v,limit in zip(vals,[105,68,105,68])):continue
                start=finite(row.get('start_seconds'));start=start if start is not None else float(row['start_frame'])/cat.fps
                passes.append({'start':vals[:2],'end':vals[2:],'outcome':row.get('outcome'),'timestamp':start,'pass_id':row.get('pass_id'),'definition':'Straight start-to-finish illustration; not the measured ball trajectory.'})
        cv_layer=getattr(index,'cv_layer',None)
        if cv_layer:
            for e in cv_layer.events:
                if e['status']=='estimated' and e['sender_ref']==ref and e['start_pitch'] is not None:
                    passes.append({'start':e['start_pitch'],'end':e['end_pitch'],'outcome':'completed','timestamp':e['start'],'pass_id':e['event_id'],'definition':'Experimental player-foot endpoint illustration, not a measured ball trajectory.'})
        pictures=a.portraits.public(recording).get('players',{}).get(ref,[])
        return {'recording_id':recording,'player_ref':ref,'label':a.identities.label(ref,identity,cat),'temporary_label':player['label'],'team':identity['players'].get(ref,{}).get('team',player['team']),'identity_confirmed':ref in identity['players'],'photo_url':pictures[0]['url'] if pictures else None,'attack_direction':(cat.tools.summary.get('config',{}).get('left_attack_direction',1)*(1 if player['team']=='left' else -1)) if cat.tools else None,'stats':stats,'heatmap':heatmap,'tracked_minutes':observed/60 if observed is not None else None,'pass_map':passes,'window':[player['start'],player['end']],'notes':(['Experimental video-only pass estimates; sparse ball visibility limits the counts.','Goals, shots, tackles, pass attempts and accuracy are unavailable, not zero.','Possible transfers are excluded from completed passes and can be reviewed separately.','Heatmaps use current low-uncertainty player positions only; camera and identity accuracy remain unverified.'] if cv_layer else [])+['Only actions in the available records are counted.','Shots and goals are separate annotation categories.','Passes, completion, interceptions, speed and distance include tracking-derived estimates.','Tracked minutes describe observed position coverage, not verified playing time.'],'metrics':metrics}
