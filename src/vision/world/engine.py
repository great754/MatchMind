"""Persistent entities, missing-state semantics and inspectable global reassociation.

Every threshold is a provisional prototype setting. Histogram evidence is weak;
no roster identities, learned ReID or calibrated confidence probabilities exist.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .schema import Config,PlayerEntity,WorldState,UnassignedTrack,IdentityHypothesis,AssociationEvidence
from .tracking import similarity
from .kits import update_team
from .schema import RosterState,EntityEvent

class WorldModel:
    def __init__(self,recording_id,config=None):
        self.recording_id=recording_id;self.config=config or Config();self.entities={};self.links={};self.next_id=1;self.last_frame=-1;self.last_timestamp=-1;self.pending_since={}
    def compare(self,track,entity,timestamp,camera):
        age=timestamp-entity.last_seen;appearance=similarity(track.appearance,entity.appearance)
        compatible=not(track.team and entity.team and track.team!=entity.team)
        distance=prior=feasible=trajectory=None
        if camera.calibration_valid and camera.calibration_source_kind=='direct_calibrated' and track.pitch is not None and entity.last_reliable_pitch is not None and entity.last_localized_at is not None:
            elapsed=timestamp-entity.last_localized_at
            distance=float(np.linalg.norm(np.array(track.pitch)-entity.last_reliable_pitch))
            feasible=distance<=self.config.max_player_speed_mps*elapsed+self.config.movement_slack_m
            prior=(1/(1+elapsed/self.config.position_prior_seconds))/(1+distance)
            if entity.last_velocity is not None:
                estimate=np.array(entity.last_reliable_pitch)+np.array(entity.last_velocity)*elapsed
                trajectory=float(np.linalg.norm(np.array(track.pitch)-estimate))
        image_error=image_feasible=box_consistency=region_compatible=None
        if entity.last_image_bbox is not None and entity.last_image_foot is not None:
            previous=np.array(entity.last_image_bbox);box=np.array(track.bbox)
            box_consistency=float(np.exp(-np.abs(np.log(box[2:]/previous[2:])).mean()))
            estimate=np.array(entity.last_image_foot)+(np.array(entity.last_image_velocity)*age if entity.last_image_velocity is not None else 0)
            image_error=float(np.linalg.norm(np.array(track.foot)-estimate)/max(previous[3],1))
            image_feasible=(0<age<=self.config.track_grace_seconds and camera.continuity=='continuous' and entity.last_image_velocity is not None and image_error<=.35 and box_consistency>=.8 and track.kit is not None and entity.last_kit is not None and entity.team_memory.evidence_count>=3)
        if camera.visible_pitch_region and entity.last_reliable_pitch is not None:
            import cv2
            region_compatible=cv2.pointPolygonTest(np.array(camera.visible_pitch_region,np.float32),entity.last_reliable_pitch,False)>=0
        score=appearance if appearance is not None else 0.
        if image_feasible and appearance is not None:score=.55*appearance+.3*max(0,1-image_error)+.15*box_consistency
        # Spatial evidence can reject impossible motion; it cannot establish identity.
        eligible=camera.continuity!='cut' and compatible and camera.epoch==entity.camera_epoch and feasible is not False and appearance is not None and appearance>=self.config.appearance_gate
        reason='appearance/team/movement compatible' if eligible else 'camera epoch changed' if camera.epoch!=entity.camera_epoch else 'team mismatch' if not compatible else 'impossible movement' if feasible is False else 'appearance missing or below gate'
        return AssociationEvidence(entity_id=entity.entity_id,appearance=appearance,team_compatible=compatible,elapsed_seconds=age,
            image_trajectory_error=image_error,image_motion_feasible=image_feasible,bbox_consistency=box_consistency,camera_region_compatible=region_compatible,continuity_source='direct metric movement' if feasible is True else 'bounded image trajectory + kit/shape' if image_feasible else 'insufficient independent continuity',
            displacement_m=distance,movement_feasible=feasible,position_prior=prior,trajectory_error_m=trajectory,score=score,eligible=eligible,reason=reason)
    def update(self,tracks,camera):
        if camera.frame<=self.last_frame or camera.timestamp<=self.last_timestamp:raise ValueError('World clock must increase')
        if len({t.track_id for t in tracks})!=len(tracks):raise ValueError('Duplicate temporary tracks')
        self.last_frame=camera.frame;self.last_timestamp=camera.timestamp;now=camera.timestamp
        if camera.continuity=='cut':self.links={}
        tracks=[t if camera.calibration_valid else t.model_copy(update={'pitch':None,'localization_confidence':None,'localization_source':'unknown'}) for t in tracks]
        decisions=[];assigned={};unassigned=[];reconciliations=[]
        # Identity can persist through calibration loss; location never does.
        for t in tracks:
            entity=self.entities.get(self.links.get(t.track_id))
            if entity and entity.camera_epoch==camera.epoch and camera.continuity!='cut':assigned[t.track_id]=(entity,None)
        used={e.entity_id for e,_ in assigned.values()}
        candidates=[e for e in self.entities.values() if e.entity_id not in used]
        pending=[t for t in tracks if t.track_id not in assigned]
        evidence=[[self.compare(t,e,now,camera) for e in candidates] for t in pending]
        costs=np.full((len(pending),len(candidates)+len(pending)),1.)
        for i,row in enumerate(evidence):
            for j,e in enumerate(row):
                if e.eligible:costs[i,j]=1-e.score
        assignment=dict(zip(*linear_sum_assignment(costs))) if pending else {}
        for i,t in enumerate(pending):
            self.pending_since.setdefault(t.track_id,camera.frame)
            row=evidence[i];ranked=sorted((e for e in row if e.eligible),key=lambda e:(-e.score,e.entity_id))
            hypotheses=[IdentityHypothesis(entity_id=e.entity_id,support_score=e.score,evidence=e) for e in sorted(row,key=lambda e:(-e.score,e.entity_id))[:5]]
            hypotheses.append(IdentityHypothesis(entity_id=None,support_score=1-(ranked[0].score if ranked else 0)))
            j=assignment.get(i,len(candidates));selected=row[j] if j<len(candidates) else None
            margin=ranked[0].score-(ranked[1].score if len(ranked)>1 else 0) if ranked else 0
            competitors=[evidence[k][j].score for k in range(len(pending)) if k!=i and j<len(candidates) and evidence[k][j].eligible]
            column_margin=selected.score-max(competitors,default=0) if selected else 0
            sufficient=selected is not None and (selected.movement_feasible is True or (selected.image_motion_feasible is True and (selected.appearance or 0)>=.95) or self.config.allow_appearance_only_reassociation)
            accept=sufficient and selected and selected.eligible and selected.score>=self.config.association_threshold and margin>=self.config.ambiguity_margin and column_margin>=self.config.ambiguity_margin and ranked and selected.entity_id==ranked[0].entity_id
            decision=dict(track_id=t.track_id,frame=camera.frame,evidence_sufficient=sufficient,acceptance_blocker=None if sufficient else 'No direct metric or bounded image-trajectory continuity; kit/color alone is insufficient',evidence=[e.model_dump() for e in row],assignment=selected.entity_id if accept else None,row_margin=margin,column_margin=column_margin)
            if accept:
                assigned[t.track_id]=(self.entities[selected.entity_id],selected);decision['decision']='reassociated'
            elif ranked:
                unassigned.append(UnassignedTrack(observation=t,status='ambiguous',hypotheses=hypotheses));decision['decision']='ambiguous'
            elif t.hits>=self.config.min_track_hits:
                key=f'ENTITY_{self.next_id:05d}';self.next_id+=1
                e=PlayerEntity(entity_id=key,visibility='visible',first_seen=now,last_seen=now,last_seen_frame=camera.frame,camera_epoch=camera.epoch,
                    evidence=['Birth of an anonymous entity; not a roster identity or verified football player.'])
                assigned[t.track_id]=(e,None);decision['decision']='new_provisional_entity';decision['assignment']=key
            else:
                unassigned.append(UnassignedTrack(observation=t,status='unknown',hypotheses=hypotheses));decision['decision']='unknown_unconfirmed_track'
            decision['segment_start_frame']=self.pending_since[t.track_id]
            if decision['assignment'] and self.pending_since[t.track_id]<camera.frame:
                reconciliations.append(dict(track_id=t.track_id,entity_id=decision['assignment'],segment_start_frame=self.pending_since[t.track_id],
                    resolved_at_frame=camera.frame,evidence=decision,original_observations_preserved=True))
            if decision['assignment']:self.pending_since.pop(t.track_id,None)
            decisions.append(decision)
        seen=set();new_links={}
        for t in tracks:
            if t.track_id not in assigned:continue
            entity,association=assigned[t.track_id];seen.add(entity.entity_id);new_links[t.track_id]=entity.entity_id
            localized=t.pitch is not None and camera.calibration_valid
            reliable=localized and camera.calibration_source_kind=='direct_calibrated'
            velocity=entity.last_velocity
            if reliable and entity.last_reliable_pitch is not None and entity.last_localized_at is not None and now>entity.last_localized_at:
                velocity=tuple((np.array(t.pitch)-entity.last_reliable_pitch)/(now-entity.last_localized_at))
            profile=t.appearance or entity.appearance
            if t.kit is not None and t.appearance is not None and entity.appearance is not None and len(t.appearance)==len(entity.appearance):
                vector=.8*np.array(entity.appearance)+.2*np.array(t.appearance);profile=tuple(vector/max(np.linalg.norm(vector),1e-12))
            team,team_confidence,team_memory,team_events=update_team(entity,t,camera)
            history=list(entity.history)+team_events
            if entity.visibility!='visible':history.append(EntityEvent(frame=camera.frame,timestamp=now,kind='recovered',reason='Current detection and accepted continuity',previous=entity.visibility,current='visible'))
            role=(team+'_outfield') if team and team_memory.stable and t.kit is not None and t.kit.field_context>=.3 else entity.role
            updated=entity.model_copy(update=dict(team=team,team_confidence=team_confidence,team_memory=team_memory,current_kit=t.kit,last_kit=t.kit or entity.last_kit,role=role,history=history[-64:],
                last_image_bbox=t.bbox,last_image_foot=t.foot,last_image_velocity=t.image_velocity,
                memory_state='visible_tracked',memory_reason='Current image detection supports the track/entity',projection_reason=t.projection_reason,size_evidence=t.size_evidence,
                visibility='visible',track_id=t.track_id,last_track_id=t.track_id,last_seen=now,last_seen_frame=camera.frame,
                image_bbox=t.bbox,image_foot=t.foot,detection_confidence=t.detection_confidence,current_pitch=t.pitch if localized else None,
                position_source=camera.calibration_source_kind if localized else 'unknown',localization_confidence=t.localization_confidence if localized else None,
                last_reliable_pitch=t.pitch if reliable else entity.last_reliable_pitch,last_localized_at=now if reliable else entity.last_localized_at,
                track_history=entity.track_history+([t.track_id] if t.track_id not in entity.track_history else []),
                last_reliable_camera_epoch=camera.epoch if reliable else entity.last_reliable_camera_epoch,
                last_velocity=velocity,appearance=profile,appearance_evidence_count=entity.appearance_evidence_count+(t.appearance is not None),camera_epoch=camera.epoch,
                identity_status='associated' if association else entity.identity_status,
                identity_confidence=association.score if association else entity.identity_confidence,
                hypotheses=[IdentityHypothesis(entity_id=entity.entity_id,support_score=association.score,evidence=association)] if association else entity.hypotheses,
                evidence=entity.evidence+([f'Reassociated at frame {camera.frame}; appearance support {association.score:.4f}; uncalibrated score.'] if association else [])))
            self.entities[entity.entity_id]=updated
        for key,entity in list(self.entities.items()):
            if key in seen:continue
            outside=False
            if camera.calibration_valid and camera.visible_pitch_region and entity.last_reliable_pitch is not None:
                import cv2
                # Only call an area outside when even the conservative motion
                # radius cannot reach the current visible region.
                distance=cv2.pointPolygonTest(np.array(camera.visible_pitch_region,np.float32),entity.last_reliable_pitch,True)
                outside=distance<-(self.config.max_player_speed_mps*max(0,now-(entity.last_localized_at or 0))+self.config.movement_slack_m)
            visibility='off_camera' if outside or now-entity.last_seen>=self.config.off_camera_seconds else 'temporarily_missing'
            memory='lost' if camera.epoch!=entity.camera_epoch else 'off_camera' if visibility=='off_camera' else 'temporarily_occluded'
            reason='Camera epoch changed; identity continuity unsupported' if memory=='lost' else 'Last reliable area outside current region even with movement slack' if outside else 'No current detection after visibility grace' if visibility=='off_camera' else 'Short detection gap; occlusion is a hypothesis, not confirmed'
            history=list(entity.history)
            if memory!=entity.memory_state:history.append(EntityEvent(frame=camera.frame,timestamp=now,kind='visibility',reason=reason,previous=entity.memory_state,current=memory))
            self.entities[key]=entity.model_copy(update=dict(visibility=visibility,team_memory=entity.team_memory.model_copy(update={'current_observation':None}),current_kit=None,memory_state=memory,memory_reason=reason,history=history[-64:],track_id=None,image_bbox=None,image_foot=None,projection_reason='NOT_CURRENTLY_VISIBLE',
                current_pitch=None,position_source='unknown',localization_confidence=None,detection_confidence=None))
        # Keep short tracker grace links, but never export their old positions.
        self.links={**{k:v for k,v in self.links.items() if v not in set(new_links.values()) and now-self.entities[v].last_seen<=self.config.track_grace_seconds},**new_links}
        roster=RosterState(team_a=[e.entity_id for e in self.entities.values() if e.team=='team_a'],team_b=[e.entity_id for e in self.entities.values() if e.team=='team_b'],unknown=[e.entity_id for e in self.entities.values() if e.team is None],candidate_people=len(self.entities))
        return WorldState(roster=roster,recording_id=self.recording_id,frame=camera.frame,timestamp=now,camera=camera,
            entities=sorted(self.entities.values(),key=lambda e:e.entity_id),unassigned=unassigned,decisions=decisions,reconciliations=reconciliations)
