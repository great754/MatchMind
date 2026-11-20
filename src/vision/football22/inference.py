"""Bounded temporal-network inference with a separate actual-current-image check."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from .model import ARCHITECTURE,TemporalFootballNet,preprocess,decode
from src.evaluation.cv_m19 import iou
from src.serving.cv import digest
from src.vision.advanced21.detection import tile_regions
from src.vision.events.passes import signature


class FootballDetector:
    def __init__(self,checkpoint,expected_sha256,config,device=None,batch_size=4):
        if digest(checkpoint)!=expected_sha256:raise ValueError('M22 checkpoint hash mismatch')
        self.checkpoint_sha256=expected_sha256;self.config=config;self.batch_size=batch_size
        self.device=device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        state=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        if state['architecture']!=ARCHITECTURE or state['split']['target_labels_used_for_fitting_or_early_stopping'] is not False:
            raise ValueError('Unsupported M22 architecture or development provenance')
        self.model=TemporalFootballNet();self.model.load_state_dict(state['model'],strict=True);self.model.to(self.device).eval()
        self.metadata={k:state[k] for k in ['architecture','epoch','split','training_manifest_sha256','dataset_sha256','original_annotations_sha256']}
        self.metadata['weak_training']=state.get('weak_training')

    def infer(self,images,index):
        started=time.perf_counter();h,w=images[-1].shape[:2]
        regions=[(0,0,w,h)]+tile_regions(w,h,self.config['tile_source_size'],self.config['tile_overlap'])
        # Whole-source search is independent of tracker state on EVERY frame.
        groups={};coarse=np.zeros((24,40),np.float32);raw=[];current=[]
        for region in dict.fromkeys(regions):
            hw=tuple(self.config['input_hw']) if region==(0,0,w,h) else (384,384)
            x,mapping=preprocess(images,hw,region);groups.setdefault(hw,[]).append((x,mapping))
        views=0
        with torch.inference_mode():
            for group in groups.values():
                for start in range(0,len(group),self.batch_size):
                    chunk=group[start:start+self.batch_size];x=torch.stack([r[0] for r in chunk]).to(self.device)
                    temporal=self.model(x);now=self.model(x[:,-3:].repeat(1,3,1,1));views+=2*len(chunk)
                    for i,(_,mapping) in enumerate(chunk):
                        a={k:v[i:i+1] for k,v in temporal.items()};b={k:v[i:i+1] for k,v in now.items()}
                        values=decode(a,mapping,self.config['candidate_threshold'],5);present=decode(b,mapping,self.config['candidate_threshold'],5)
                        for c in values:c['source_mapping']=mapping
                        raw.extend(values);current.extend(present)
                        hm=a['heatmap'][0,0].sigmoid().detach().cpu().numpy()
                        gx=(np.arange(40,dtype=np.float32)+.5)*w/40;gy=(np.arange(24,dtype=np.float32)+.5)*h/24
                        mx,my=np.meshgrid((gx-mapping['x'])*mapping['scale_x']/4,(gy-mapping['y'])*mapping['scale_y']/4)
                        sample=cv2.remap(hm,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                        mask=(gx[None,:]>=mapping['x'])&(gx[None,:]<mapping['x']+mapping['width'])&(gy[:,None]>=mapping['y'])&(gy[:,None]<mapping['y']+mapping['height'])
                        coarse=np.maximum(coarse,sample*mask)
        kept=[]
        for candidate in sorted(raw,key=lambda c:-c['confidence']):
            if any(iou(candidate['bbox'],r['bbox'])>.4 for r in kept):continue
            matches=[c for c in current if iou(c['bbox'],candidate['bbox'])>=self.config['current_verification_match_iou']]
            current_score=max((c['confidence'] for c in matches),default=0.)
            candidate['candidate_id']=f'm22_{index}_'+signature([candidate['bbox'],candidate['confidence']])[:16]
            candidate['current_only_confidence']=current_score
            accepted=candidate['confidence']>=self.config['observation_threshold'] and current_score>=self.config['current_only_verification_threshold']
            candidate['observation_eligible']=accepted
            candidate['decision_reason']='ACTUAL_CURRENT_PIXELS_AND_TEMPORAL_NETWORK_AGREE' if accepted else 'BELOW_FIXED_TEMPORAL_VISUAL_THRESHOLD' if candidate['confidence']<self.config['observation_threshold'] else 'CURRENT_ONLY_VISUAL_CHECK_DID_NOT_CONFIRM'
            candidate['confidence_is_calibrated_probability']=False
            kept.append(candidate)
            if len(kept)>=self.config['maximum_candidates_per_frame']:break
        if self.device=='mps':torch.mps.synchronize()
        return {'frame':index,'candidates':kept,'heatmap':np.round(coarse,4).tolist(),
            'heatmap_definition':'Maximum diagnostic activation across source-mapped views; not a calibrated probability or an observation.',
            'inference_views':views,'seconds':time.perf_counter()-started,
            'checkpoint_sha256':self.checkpoint_sha256,'current_frame_pixels_sha256':__import__('hashlib').sha256(images[-1].tobytes()).hexdigest()}
