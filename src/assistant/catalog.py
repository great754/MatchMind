"""Explicit source mappings. No nearest-player attribution or cross-segment ReID."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from src.highlights.render import file_hash
from src.highlights.index import digest

def letter(i):
    result=''
    while True:
        result=chr(65+i%26)+result;i=i//26-1
        if i<0:return result

class PlayerCatalog:
    def __init__(self,recording,tools=None,cv=None):
        self.recording=recording;self.tools=tools;self.cv=cv;self.players=[];self.boxes=defaultdict(dict);self.diagnostic=None;self.fps=0;self.frames=0;self.source_signature='';self.by_ref={};self.point_coords=None;self.point_ids=None;self.point_source_hash=None;self.metric_positions=defaultdict(dict)
        try:
            if tools:self.structured(tools)
            elif cv:self.video_cv(cv)
        except (OSError,ValueError,KeyError,TypeError) as e:self.diagnostic='Player pictures are unavailable because the source mapping could not be checked.';self.boxes=defaultdict(dict);self.point_coords=None;self.point_ids=None;self.point_source_hash=None;self.metric_positions=defaultdict(dict)
        self.by_ref={p['player_ref']:p for p in self.players}
    def structured(self,tools):
        s=tools.summary;self.fps=float(s['fps']);self.frames=int(s['frames']);match=str(s['match_id']);root=Path('data/soccertrack');sync=root/'mot'/f'{match}_sync.json';mot=root/'mot'/f'{match}.txt'
        if not mot.exists():mot=root/'mot'/f'{match}.txt.txt'
        expected={Path(x['path']).name:x['sha256'] for x in s.get('source_files',[])}
        self.source_signature=digest([self.recording,expected.get(sync.name),self.frames,self.fps])
        # Canonical actor metadata comes from the validated bundle even when the
        # optional local video-to-track mapping is absent. Never guess that map.
        for i,p in enumerate(sorted(tools.players,key=lambda p:int(p['player_id']))):
            actor=int(p['player_id']);ref='actor:'+str(actor)
            self.players.append({'player_ref':ref,'label':'Player '+letter(i),'actor_id':actor,'mot_id':None,'entity_id':None,'segment_id':None,'team':p['team'],'start':0.,'end':self.frames/self.fps,'source_signature':self.source_signature,'continuity':'source_identity_mapping','appearance_note':'Number not confirmed'})
        if sync.name not in expected or file_hash(sync)!=expected[sync.name]:raise ValueError('Unverified sync')
        mapping=json.loads(sync.read_text());actor_to_mot={int(v['player_id']):int(k) for k,v in mapping['players'].items()}
        for p in self.players:p['mot_id']=actor_to_mot.get(p['actor_id'])
        if s.get('scope')=='half':
            self.load_image_points(s,root,expected)
            self.diagnostic='Full-half markers use source image foot points; player portrait boxes are unavailable.'
            return
        if not mot.exists():return
        data=np.loadtxt(mot,delimiter=',',usecols=range(6));ids=set(actor_to_mot.values());reverse={m:a for a,m in actor_to_mot.items()}
        if len(reverse)!=len(actor_to_mot) or np.any(data[:,0]<0) or np.any(data[:,0]>=self.frames):raise ValueError('Invalid MOT mapping')
        seen=set()
        for frame,track,x,y,w,h in data:
            key=(int(frame),int(track))
            if key in seen:raise ValueError('Duplicate track/frame')
            seen.add(key)
            if int(track) in ids and np.isfinite([x,y,w,h]).all() and w>0 and h>0:self.boxes['actor:'+str(reverse[int(track)])][int(frame)]=[float(x),float(y),float(w),float(h)]
        self.source_signature=digest([self.source_signature,file_hash(mot)])
        for p in self.players:p['source_signature']=self.source_signature
    def load_image_points(self,summary,root,expected):
        """Use the compact GSR's original image foot points, never pitch projection."""
        suffix='1st' if summary['half']==1 else '2nd'
        path=root/'gsr'/str(summary['match_id'])/(suffix+'_compact.npz')
        if path.name not in expected or file_hash(path)!=expected[path.name]:raise ValueError('Unverified image points')
        with np.load(path) as data:
            offset=int(summary['source_offset_frames'])
            coords=data['coords'][offset:offset+self.frames].copy();ids=data['ids'][offset:offset+self.frames].copy()
        if ids.ndim!=2 or coords.ndim!=3 or coords.shape!=(self.frames,ids.shape[1],2) or len(ids)!=self.frames:raise ValueError('Image point clock mismatch')
        self.point_coords=coords;self.point_ids=ids;self.point_source_hash=expected[path.name]
    def image_point(self,ref,frame):
        if self.point_coords is None or not 0<=frame<self.frames:return None
        actor=self.by_ref.get(ref,{}).get('actor_id')
        slots=np.flatnonzero(self.point_ids[frame]==actor)
        if len(slots)!=1:return None
        point=self.point_coords[frame,slots[0]]
        return point if np.isfinite(point).all() else None
    def has_marker(self,ref):
        return bool(self.boxes.get(ref)) or self.point_coords is not None and ref in self.by_ref
    def video_cv(self,cv):
        self.fps=cv.manifest['fps'];self.frames=cv.manifest['frames'];self.source_signature=cv.manifest['frames_sha256'];groups={}
        def rows():
            # CVReview validated the complete immutable source at startup. Read
            # only the needed JSON fields here instead of revalidating and
            # serializing every historical identity field a second time.
            if hasattr(cv,'data'):
                if hasattr(cv,'check_source'):
                    cv.check_source();stamps=(cv.data.stat().st_mtime_ns,)
                else:
                    stamps=(cv.data.stat().st_mtime_ns,cv.video.stat().st_mtime_ns)
                    if stamps!=(cv.data_stamp,cv.video_stamp):raise ValueError('CV source changed')
                with cv.data.open('rb') as stream:
                    for line in stream:yield json.loads(line)
                if hasattr(cv,'check_source'):
                    cv.check_source()
                    if (cv.data.stat().st_mtime_ns,)!=stamps:raise ValueError('CV source changed')
                elif (cv.data.stat().st_mtime_ns,cv.video.stat().st_mtime_ns)!=stamps:raise ValueError('CV source changed')
            else:
                for i in range(self.frames):yield cv.frame((i+.001)/self.fps).model_dump()
        for i,w in enumerate(rows()):
            observations={e['track_id']:e['image_bbox'] for e in w['entities'] if e['track_id'] and e['image_bbox']}
            observations.update({u['observation']['track_id']:u['observation']['bbox'] for u in w['unassigned']})
            for p in w['player_world']['players']:
                segment=p.get('current_segment_id');track=p.get('current_track_id')
                if p['visibility_state']!='VISIBLE' or p['identity_status'] in ['AMBIGUOUS','UNKNOWN'] or not segment or track not in observations:continue
                ref='segment:'+segment
                if ref not in groups:groups[ref]={'entity_id':p['entity_id'],'segment_id':segment,'team':{'TEAM_A':'left','TEAM_B':'right'}.get(p['team']['label'],'unknown'),'first':i,'last':i}
                g=groups[ref];g['last']=i
                if g['entity_id']!=p['entity_id']:continue
                self.boxes[ref][i]=observations[track]
                point=p.get('current_position');uncertainty=p.get('position_uncertainty_m')
                if point is not None and p.get('position_provenance') not in [None,'UNKNOWN'] and uncertainty is not None and uncertainty<=8 and 0<=point[0]<=105 and 0<=point[1]<=68:self.metric_positions[ref][i]=point
        for ref,g in sorted(groups.items(),key=lambda pair:pair[1]['first']):
            if len(self.boxes[ref])<max(3,int(self.fps*.3)):continue
            self.players.append({'player_ref':ref,'label':'Player '+letter(len(self.players)),'actor_id':None,'mot_id':None,'entity_id':g['entity_id'],'segment_id':g['segment_id'],'team':g['team'],'start':g['first']/self.fps,'end':(g['last']+1)/self.fps,'source_signature':self.source_signature,'continuity':'single_segment_only','appearance_note':'Separate appearances need separate confirmation'})
    def public(self):return {'recording_id':self.recording,'players':self.players,'diagnostic':self.diagnostic,'fps':self.fps,'frames':self.frames,'source_signature':self.source_signature}
    def at(self,ref,frame):return self.boxes.get(ref,{}).get(frame)
