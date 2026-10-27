"""Inspectable kit evidence and temporal team beliefs; no roster/color labels."""
import cv2
import numpy as np
from .schema import KitObservation,TeamMemory,TeamVotes,KitSample,EntityEvent

def torso_features(image,bbox):
    x,y,w,h=bbox;height,width=image.shape[:2]
    x1=max(0,int(x+.22*w));x2=min(width,int(x+.78*w));y1=max(0,int(y+.18*h));y2=min(height,int(y+.48*h))
    if x2-x1<3 or y2-y1<4:return None,None
    crop=image[y1:y2,x1:x2];lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB);hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    # Learn local field color from a ring BELOW the player, not a fixed hue
    # exclusion. A bright green/yellow shirt can differ strongly from that grass.
    ring=image[max(0,min(height-1,int(y+h))):min(height,int(y+h+.22*h)+1),max(0,int(x-.5*w)):min(width,int(x+1.5*w))]
    background=None;context=0.
    if ring.size:
        rh=cv2.cvtColor(ring,cv2.COLOR_BGR2HSV);field=(rh[:,:,0]>=25)&(rh[:,:,0]<=95)&(rh[:,:,1]>40)
        context=float(field.mean())
        if field.sum()>=8:background=np.median(cv2.cvtColor(ring,cv2.COLOR_BGR2LAB)[field],axis=0)
    mask=(hsv[:,:,2]>30)&(hsv[:,:,1]>45)
    if background is not None:mask &= np.linalg.norm(lab.astype(float)-background,axis=2)>22
    fraction=float(mask.mean())
    if mask.sum()<10:return KitObservation(torso_bbox=(x1,y1,x2-x1,y2-y1),background_lab=tuple(background) if background is not None else None,foreground_fraction=fraction,field_context=context,reason='Insufficient foreground chromatic torso pixels'),None
    pixels=lab[mask].astype(float)
    # Dominant chromatic population, robust against white numbers and skin/background.
    bins=(pixels[:,1:]//16).astype(int);keys=bins[:,0]*16+bins[:,1];dominant=np.bincount(keys,minlength=256).argmax()
    center=np.median(pixels[keys==dominant],axis=0);near=np.linalg.norm(pixels[:,1:]-center[1:],axis=1)<24
    color=np.median(pixels[near],axis=0);purity=float(near.mean())
    quality=float(np.clip(fraction*purity*min(1,h/45),0,1))
    hist=cv2.calcHist([hsv],[0,1],mask.astype(np.uint8),[18,3],[0,180,0,256]).ravel()
    descriptor=tuple(map(float,hist/max(np.linalg.norm(hist),1e-12)))
    bgr=tuple(map(int,cv2.cvtColor(np.uint8(np.clip(color,0,255)).reshape(1,1,3),cv2.COLOR_LAB2BGR)[0,0]))
    return KitObservation(torso_bbox=(x1,y1,x2-x1,y2-y1),lab=tuple(color),dominant_bgr=bgr,background_lab=tuple(background) if background is not None else None,foreground_fraction=fraction,field_context=context,quality=quality,reason='Dominant foreground LAB chroma in upper torso; local field-color contrast'),descriptor

class VideoTeams:
    def __init__(self,centers=None,diagnostic=None):self.centers=centers;self.diagnostic=diagnostic or {}
    @classmethod
    def fit(cls,profiles):
        from sklearn.cluster import KMeans
        # One aggregate per temporary segment; one long visible person cannot
        # dominate merely by supplying more frames. No fixed colors or sides.
        usable=[np.median(v,axis=0) for v in profiles.values() if len(v)>=3]
        if len(usable)<8:return cls(diagnostic={'status':'insufficient independent track segments','profiles':len(usable)})
        points=np.array(usable,float);scaled=points*np.array([.35,1,1]);model=KMeans(n_clusters=2,n_init=10,random_state=42).fit(scaled)
        centers=np.array([np.median(points[model.labels_==i],axis=0) for i in range(2)])
        populations=[int(sum(model.labels_==i)) for i in range(2)]
        if min(populations)<3 or np.linalg.norm(centers[0,1:]-centers[1,1:])<25:return cls(diagnostic={'status':'insufficient two-population support','populations':populations})
        centers=np.array(sorted(centers.tolist(),key=tuple))
        return cls(centers,{'status':'two provisional outfield kit populations','profiles':len(usable),'populations':populations,'centers_lab':centers.tolist(),'team_labels':'anonymous sorted prototypes; not defending sides','source':'offline multi-frame segment aggregates; not independent team ground truth'})
    def classify(self,detection):
        kit=detection.kit
        if kit is None or kit.lab is None or self.centers is None or kit.quality<.2 or kit.field_context<.2:return detection.model_copy(update={'team':None,'team_confidence':None})
        distances=np.linalg.norm((self.centers-np.array(kit.lab))*[.35,1,1],axis=1);order=np.argsort(distances);margin=float((distances[order[1]]-distances[order[0]])/max(distances[order[1]],1))
        kit=kit.model_copy(update={'cluster_distances':tuple(map(float,distances))})
        accepted=distances[order[0]]<=40 and margin>=.35
        return detection.model_copy(update={'kit':kit,'team':('team_a' if order[0]==0 else 'team_b') if accepted else None,'team_confidence':margin if accepted else None})

def update_team(entity,track,camera):
    """Accumulate independent frames; unknown evidence never erases learned kit."""
    old=entity.team_memory;votes=old.votes.model_copy();team=entity.team;confidence=entity.team_confidence;events=[]
    kit=track.kit;observed=track.team
    if kit is None:return track.team or team,track.team_confidence if track.team else confidence,old,events # backward-compatible supplied evidence
    recent=list(old.recent);historical=old.historical_lab;count=old.evidence_count
    if kit.lab is not None and kit.quality>=.2:
        recent=(recent+[KitSample(timestamp=camera.timestamp,lab=kit.lab,quality=kit.quality)])[-8:]
        historical=tuple((np.array(historical)*count+kit.lab)/(count+1)) if historical is not None else kit.lab
        count+=1
    streak=old.contradictory_streak
    if observed:
        weight=float(track.team_confidence or 0)*kit.quality
        votes=votes.model_copy(update={'team_a':votes.team_a*.98,'team_b':votes.team_b*.98,observed:getattr(votes,observed)*.98+weight})
        winner='team_a' if votes.team_a>=votes.team_b else 'team_b';support=getattr(votes,winner)/max(votes.team_a+votes.team_b,1e-9)
        streak=streak+1 if team is not None and observed!=team and (track.team_confidence or 0)>=.6 else 0
        candidate=winner if getattr(votes,winner)>=1.2 and support>=.8 and count>=3 else None
        if team is None and candidate:team=candidate
        elif team and candidate!=team and candidate is not None and streak>=6:team=candidate
        if team:confidence=float(getattr(votes,team)/max(votes.team_a+votes.team_b,1e-9))
    else:votes=votes.model_copy(update={'unknown':votes.unknown+1});streak=0
    if team!=entity.team:
        events.append(EntityEvent(frame=camera.frame,timestamp=camera.timestamp,kind='team_acquired' if entity.team is None else 'team_changed',previous=entity.team,current=team,reason=f'Multi-frame kit evidence; count={count}, votes A={votes.team_a:.3f}, B={votes.team_b:.3f}, contradiction streak={streak}'))
    memory=TeamMemory(votes=votes,evidence_count=count,current_observation=observed,contradictory_streak=streak,recent=recent,historical_lab=historical,stable=team is not None)
    return team,confidence,memory,events
