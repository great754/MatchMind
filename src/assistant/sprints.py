"""New sustained-run summaries from existing metric trajectories, not new event/CV inference."""
import json,math
from pathlib import Path
import numpy as np
from src.highlights.render import file_hash
from src.highlights.index import digest

def qualifying_runs(motion,fps,cfg):
    """Qualify continuous metric runs; never join missing/rejected derivatives."""
    from src.analytics.motion import segments
    minimum=max(1,math.ceil(cfg['minimum_sprint_seconds']*fps))
    window=max(1,math.ceil(cfg['sustained_window_seconds']*fps))
    eligible=np.isfinite(motion.smoothed_speed_mps)&motion.observed&motion.valid&~motion.rejected_speed_flag
    for start,end in segments(eligible&(motion.smoothed_speed_mps>=cfg['sprint_threshold_mps'])):
        if end-start<max(minimum,window):continue
        speeds=motion.smoothed_speed_mps[start:end]
        path_length=float(np.nansum(motion.step_distance_m[start:end]))
        net=float(np.linalg.norm(motion.positions[end-1]-motion.positions[start]))
        if path_length<=0 or net/path_length<.8:continue
        sustained=np.convolve(speeds,np.ones(window)/window,mode='valid')
        yield start,end,float(max(sustained)),start+int(np.argmax(sustained)),net/path_length

def fastest_measured_run(motion,fps,cfg):
    """Best observed one-second straight run, even when below sprint threshold."""
    from src.analytics.motion import segments
    window=max(1,math.ceil(cfg['sustained_window_seconds']*fps));best=None
    inside=(motion.positions[:,0]>=0)&(motion.positions[:,0]<=105)&(motion.positions[:,1]>=0)&(motion.positions[:,1]<=68)
    eligible=np.isfinite(motion.smoothed_speed_mps)&motion.observed&motion.valid&~motion.rejected_speed_flag&inside
    for start,end in segments(eligible):
        if end-start<window+1:continue
        speeds=np.convolve(motion.smoothed_speed_mps[start+1:end],np.ones(window)/window,mode='valid')
        for offset in np.argsort(-speeds):
            score=float(speeds[offset])
            if best and score<=best[2]:break
            a=int(start)+int(offset);b=a+window
            path=float(np.sum(motion.step_distance_m[a+1:b+1]));net=float(np.linalg.norm(motion.positions[b]-motion.positions[a]));ratio=net/path if path>0 else 0
            if ratio>=.8 and score>=.5:
                best=(a,b,score,ratio);break
    return best

class SprintService:
    def __init__(self,catalog,config=None):
        self.catalog=catalog;self.config=config or json.loads(Path('configs/assistant/m18.json').read_text());self.intervals=[];self.runs=[];self.diagnostic=None;self.sensitivity=[]
        if not catalog.tools:self.diagnostic='I don’t have reliable pitch-position measurements to compare sprints in this footage.';return
        try:self.build()
        except (ValueError,OSError,KeyError,StopIteration,ImportError):self.diagnostic='I can’t verify continuous pitch-position measurements for a sustained sprint ranking.'
    def build(self):
        from src.analytics.motion import calculate_motion,segments
        cat=self.catalog;s=cat.tools.summary
        if s.get('player_coordinate_source')!='gsr':raise ValueError('Unverified pitch source')
        suffix='1st' if s['half']==1 else '2nd';p=Path('data/soccertrack/gsr')/str(s['match_id'])/(suffix+'_compact.npz');expected=next(f['sha256'] for f in s['source_files'] if Path(f['path']).name==p.name)
        if file_hash(p)!=expected:raise ValueError('Trajectory source changed')
        with np.load(p) as a:
            off=int(s['source_offset_frames']);coords=a['pitch'][off:off+cat.frames].astype(float);ids=a['ids'][off:off+cat.frames]
        if len(coords)!=cat.frames:raise ValueError('Clock mismatch')
        cfg=self.config;fps=cat.fps;minimum=max(1,math.ceil(cfg['minimum_sprint_seconds']*fps));window=max(1,math.ceil(cfg['sustained_window_seconds']*fps))
        for player in cat.players:
            actor=player['actor_id'];positions=np.full((cat.frames,2),np.nan);frames,slots=np.where(ids==actor)
            if len(np.unique(frames))!=len(frames):continue
            positions[frames]=coords[frames,slots];motion=calculate_motion(positions,fps,smoothing_frames=int(cfg['smoothing_frames']))
            for start,end,score,peak,net_ratio in qualifying_runs(motion,fps,cfg):
                identifier='run_'+digest([self.catalog.recording,actor,int(start),int(end),cfg,expected])[:24]
                self.intervals.append({'event_id':identifier,'player_ref':player['player_ref'],'actor_id':actor,'player':player['mot_id'],'team':player['team'],'start':int(start)/fps,'end':int(end)/fps,'speed_mps':score,'peak_window_start':peak/fps,'peak_window_end':(peak+window)/fps,'samples':end-start,'source_hash':expected,'definition':f"Highest {cfg['sustained_window_seconds']:g}-second mean speed in a continuous run above {cfg['sprint_threshold_mps']:g} m/s lasting at least {cfg['minimum_sprint_seconds']:g} seconds; speed-limit-rejected derivatives excluded and net/path consistency at least 0.8; raw annotation jumps are audited, not interpreted as sprint peaks.",'support':'estimated','raw_speed_flag_fraction':float(motion.raw_speed_flag[start:end].mean()),'net_to_path_ratio':net_ratio})
            measured=fastest_measured_run(motion,fps,cfg)
            if measured:
                a,b,score,ratio=measured
                self.runs.append({'event_id':'run_'+digest(['measured-run-v1',cat.recording,actor,a,b,cfg,expected])[:24],'player_ref':player['player_ref'],'actor_id':actor,'player':player['mot_id'],'team':player['team'],'start':a/fps,'end':b/fps,'speed_mps':score,'samples':b-a,'source_hash':expected,'definition':f"Highest {cfg['sustained_window_seconds']:g}-second mean speed over continuous observed, valid in-pitch motion, net/path at least 0.8; not necessarily a sprint.",'support':'estimated','qualifies_as_sprint':False,'net_to_path_ratio':ratio})
            alternate=calculate_motion(positions,fps,smoothing_frames=25)
            self.sensitivity.append({'player_ref':player['player_ref'],'max_smoothed_15_mps':float(np.nanmax(motion.smoothed_speed_mps)) if np.isfinite(motion.smoothed_speed_mps).any() else None,'max_smoothed_25_mps':float(np.nanmax(alternate.smoothed_speed_mps)) if np.isfinite(alternate.smoothed_speed_mps).any() else None})
        self.intervals.sort(key=lambda r:(-r['speed_mps'],r['start'],r['player_ref']));self.runs.sort(key=lambda r:(-r['speed_mps'],r['start'],r['player_ref']))
    def best(self,players=None):return next((s for s in self.intervals if not players or s['player_ref'] in players),None)
