"""Short-occlusion association evidence without rewriting M9 identities or confirmations."""
import math
from collections import Counter
import numpy as np
from src.vision.world.player_identity import build_segments
from src.vision.world.player_appearance import cosine


def short_links(features,embeddings,worlds,fps,config):
    segments,_=build_segments(features,worlds,embeddings,fps)
    sources={}
    for world in worlds:
        for p in world['player_world']['players']:
            ref=p.get('current_segment_id')
            if ref:sources.setdefault(ref,[]).append((p['entity_id'],p['team']['label'],p['team']['confidence']))
    links=[];ambiguous=[]
    for b in segments:
        options=[]
        for a in segments:
            gap=(b.start-a.end)/fps
            if a.epoch!=b.epoch or not 0<gap<=config['maximum_gap_seconds']:continue
            app=cosine(a.embedding,b.embedding);hist=cosine(a.histogram,b.histogram)
            if min(len(a.evidence_frames),len(b.evidence_frames))<config['minimum_embedding_samples'] or min(a.quality,b.quality)<config['minimum_quality']:continue
            if app is None or hist is None or app<config['minimum_appearance_cosine'] or hist<config['minimum_histogram_cosine']:continue
            old,new=a.observations[-1],b.observations[0];distance=float(np.linalg.norm(np.asarray(old['foot'])-new['foot'])/max(old['bbox'][3],new['bbox'][3],1));shape=float(np.exp(-np.abs(np.log(np.array(new['bbox'][2:])/old['bbox'][2:])).mean()))
            if distance>config['maximum_foot_displacement_heights'] or shape<config['minimum_shape_similarity']:continue
            teams=[];entities=[]
            for s in [a,b]:
                rows=sources.get(s.key,[]);entity=Counter(r[0] for r in rows).most_common(1);team=Counter(r[1] for r in rows if r[1] in ['TEAM_A','TEAM_B'] and r[2]>=.65).most_common(1);entities.append(entity[0][0] if entity else None);teams.append(team[0][0] if team else None)
            if teams[0] is not None and teams[1] is not None and teams[0]!=teams[1]:continue
            score=.5*app+.3*hist+.2*shape
            options.append({'from_segment':a.key,'to_segment':b.key,'from_end_frame':a.end,'to_start_frame':b.start,'gap_seconds':gap,'appearance_cosine':app,'histogram_cosine':hist,'shape_similarity':shape,'foot_displacement_heights':distance,'support':score,'same_m9_entity':entities[0] is not None and entities[0]==entities[1],'team_compatible':True,'from_embedding_frames':a.evidence_frames,'to_embedding_frames':b.evidence_frames})
        options.sort(key=lambda r:(-r['support'],r['from_segment']))
        if not options:continue
        if len(options)>1 and options[0]['support']-options[1]['support']<config['minimum_assignment_margin']:
            ambiguous.extend({**o,'status':'AMBIGUOUS_NOT_MERGED'} for o in options);continue
        links.append(options[0])
    # One predecessor cannot be assigned to competing successors.
    collisions=Counter(r['from_segment'] for r in links);result=[]
    for link in links:
        if collisions[link['from_segment']]>1:ambiguous.append({**link,'status':'COMPETING_SUCCESSORS_NOT_MERGED'});continue
        result.append({**link,'status':'SUPPORTED_EXISTING_M9_CONTINUITY' if link['same_m9_entity'] else 'REVIEW_REQUIRED_NO_IDENTITY_MERGE','manual_identity_propagated':False})
    return {'links':result,'ambiguous':ambiguous,'source_segment_count':len(segments),'supported_existing_identity_links':sum(r['same_m9_entity'] for r in result),'cross_entity_review_proposals':sum(not r['same_m9_entity'] for r in result),'actual_m9_identity_rewrites':0,'verified_identity_switch_rate':None,'limitation':'Association support is heuristic; cross-entity proposals are never silently merged or given confirmed names.'}
