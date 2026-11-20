"""Development-only assisted review, separate from all original evaluation labels."""
import json
import math
import os
from pathlib import Path
import threading
from typing import Literal
from pydantic import Field,model_validator
from src.query.schema import Closed
from src.serving.cv import digest
from src.vision.events.passes import signature


class DevelopmentLabel(Closed):
    task_id:str=Field(min_length=1,max_length=120)
    reviewer_id:str=Field(min_length=2,max_length=80)
    state:Literal['visible','not_visible','uncertain']
    bbox:tuple[float,float,float,float]|None=None
    reviewed_from_source:Literal[True]
    origin:Literal['human']='human'
    model_assisted:Literal[True]=True
    accepted_proposal:str|None=Field(default=None,max_length=150)
    rejected_proposals:list[str]=Field(default_factory=list,max_length=12)
    reviewed_proposals:list[str]=Field(default_factory=list,max_length=12)
    notes:str=Field(default='',max_length=500)

    @model_validator(mode='after')
    def visibility(self):
        if len(self.reviewer_id.strip())<2:raise ValueError('Use at least two non-space reviewer characters')
        if (self.state=='visible')!=(self.bbox is not None):raise ValueError('Only visible balls require a tight source box')
        if self.bbox is not None:
            x,y,w,h=self.bbox
            if min(x,y)<0 or min(w,h)<=0:raise ValueError('A positive box inside source pixels is required')
        return self


class DevelopmentReviewStore:
    def __init__(self,manifest='configs/training/cv_m22_tasks.json',root='outputs/m22/human_review'):
        self.path=Path(manifest);self.dataset=json.loads(self.path.read_text());self.root=Path(root);self.lock=threading.RLock()
        if signature({k:v for k,v in self.dataset.items() if k!='fingerprint'})!=self.dataset['fingerprint']:
            raise ValueError('M22 task fingerprint changed')
        if self.dataset.get('reference_role')!='development_only' or self.dataset.get('model_proposals_are_truth') is not False:
            raise ValueError('Assisted tasks cannot be independent references')
        if set(self.dataset['sources'])!={'test_match','team_goal','southampton'}:raise ValueError('Only approved short clips may be reviewed')
        self.tasks={t['task_id']:t for t in self.dataset['tasks']}
        if len(self.tasks)!=len(self.dataset['tasks']):raise ValueError('Duplicate development task')
        self.stamp=self.path.stat().st_mtime_ns

    def labels(self):
        path=self.root/'labels.json'
        if not path.exists():return []
        data=json.loads(path.read_text())
        if data['dataset_fingerprint']!=self.dataset['fingerprint']:raise ValueError('M22 labels belong to another task version')
        return data['annotations']

    def validate(self,recording,label):
        task=self.tasks.get(label.task_id)
        if task is None or task['recording_id']!=recording:raise ValueError('Task belongs to another source')
        source=self.dataset['sources'][recording]
        if self.path.stat().st_mtime_ns!=self.stamp:raise ValueError('Task version changed; revalidate')
        if label.bbox is not None:
            x,y,w,h=label.bbox
            if x+w>source['width'] or y+h>source['height']:raise ValueError('Box outside source image')
        proposals={p['proposal_id']:p for p in task['proposals']}
        if label.reviewed_proposals and (len(set(label.reviewed_proposals))!=len(label.reviewed_proposals) or set(label.reviewed_proposals)!=set(proposals)):
            raise ValueError('Completed proposal review must include each source proposal exactly once')
        if label.accepted_proposal is not None and label.accepted_proposal not in proposals:raise ValueError('Unknown proposal')
        if len(set(label.rejected_proposals))!=len(label.rejected_proposals) or any(p not in proposals for p in label.rejected_proposals):raise ValueError('Unknown or duplicate hard-negative proposal')
        if label.accepted_proposal in label.rejected_proposals:raise ValueError('One proposal cannot be accepted and rejected')
        hard_negatives=[]
        for pid in label.rejected_proposals:
            box=proposals[pid]['bbox']
            if label.bbox is not None:
                from src.evaluation.cv_m19 import iou
                if iou(box,label.bbox)>.1:raise ValueError('Rejected region overlaps the labelled ball; correct the proposal instead')
            hard_negatives.append({'proposal_id':pid,'bbox':box,'scope':'human-rejected proposal region only; not whole-frame absence'})
        return {**label.model_dump(),'source_video_sha256':source['video_sha256'],'frame':task['frame'],
            'timestamp':task['frame']/source['fps'],'reference_role':'development_only','hard_negative_regions':hard_negatives}

    def save(self,recording,label):
        value=self.validate(recording,label)
        with self.lock:
            labels=[r for r in self.labels() if (r['task_id'],r['reviewer_id'])!=(label.task_id,label.reviewer_id)]
            labels.append(value);self.root.mkdir(parents=True,exist_ok=True);path=self.root/'labels.json';temp=self.root/'labels.tmp'
            temp.write_text(json.dumps({'dataset_fingerprint':self.dataset['fingerprint'],'reference_role':'development_only','annotations':labels},indent=2)+'\n')
            temp.chmod(0o600);os.replace(temp,path)
        return {'saved':True,'task_id':label.task_id,'saved_tasks':len({r['task_id'] for r in labels})}

    def public(self,recording):
        if recording not in self.dataset['sources']:raise ValueError('Unknown approved source')
        labels=self.labels()
        return {'dataset_fingerprint':self.dataset['fingerprint'],'reference_role':'development_only','model_proposals_are_truth':False,
            'supports_reviewed_proposals':True,
            'source':self.dataset['sources'][recording],'tasks':[t for t in self.tasks.values() if t['recording_id']==recording],
            'labels':[r for r in labels if self.tasks[r['task_id']]['recording_id']==recording],
            'sources':[{'recording_id':name,'tasks':sum(t['recording_id']==name for t in self.tasks.values()),
                'saved_tasks':len({r['task_id'] for r in labels if self.tasks[r['task_id']]['recording_id']==name})} for name in self.dataset['sources']],
            'source_suitability':self.dataset['source_suitability'],'selection':self.dataset['selection']}
