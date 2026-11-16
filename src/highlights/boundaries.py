"""Frame-bounded context windows, with explicit fallback rather than invented buildup."""
import math
class ClipBoundaryEngine:
    def build(self,events,plan,duration,fps):
        budget=plan.duration or 120.;budget_frames=math.floor(budget*30);segments=[]
        for e in events[:30]:
            pre,post={'goal':(15,5),'shot':(6,4),'pass':(3,3)}.get(e['event_type'],(4,3))
            if plan.buildup=='more':pre=20 if e['event_type']=='goal' else pre+5
            if plan.buildup=='short':pre,post=2,2
            start=max(0,math.floor((e['start']-pre)*fps)/fps);end=min(duration,math.ceil((e['end']+post)*fps)/fps)
            if end<=start:continue
            per_event=budget/min(max(1,len(events)),30)
            core=e['end']-e['start']+1/fps
            if end-start>per_event and core<=per_event:
                # Preserve the requested lead-in/aftermath balance when a
                # compilation budget is shared; centering used to discard buildup.
                before=(per_event-core)*pre/(pre+post)
                start=max(start,math.floor((e['start']-before)*fps)/fps);end=min(duration,start+per_event)
            segment={'start':start,'end':end,'event_ids':[e['event_id']],'event_start':e['start'],'event_end':e['end'],'period':e['timebase']['period'],'reason':f"{e['verification']} {e['event_type']}; fixed context window, not a reconstructed possession sequence"}
            if segments and start<=segments[-1]['end'] and segment['period']==segments[-1]['period']:
                segments[-1]['end']=max(end,segments[-1]['end']);segments[-1]['event_ids']+=segment['event_ids'];segments[-1]['event_end']=max(e['end'],segments[-1]['event_end'])
            else:segments.append(segment)
        # A deterministic output-duration cap; preserve whole essential event
        # intervals. Omit events that cannot fit, never cut off the referenced event.
        result=[];remaining=budget_frames/30
        for s in segments:
            if remaining<=0:break
            core=s['event_end']-s['event_start']+1/fps
            if core>remaining:continue
            if s['end']-s['start']>remaining:
                fraction=(s['event_start']-s['start'])/max(1e-8,s['end']-s['start']-core)
                s={**s,'start':max(s['start'],s['event_start']-max(0,remaining-core)*min(1,fraction))};s['end']=min(duration,s['start']+remaining)
                if s['end']<s['event_end']:s['end']=s['event_end'];s['start']=max(0,s['end']-remaining)
            # Output is 30fps; round down to prevent duration-cap overshoot.
            length=math.floor((s['end']-s['start'])*30)/30
            if length<=0 or s['start']+length<s['event_end']:continue
            s={**s,'end':s['start']+length,'output_start':sum(r['end']-r['start'] for r in result)};s['output_end']=s['output_start']+length;result.append(s);remaining-=length
        return result
