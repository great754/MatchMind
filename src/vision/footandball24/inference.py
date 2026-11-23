"""Single-image inference, explicit coordinate mapping and faithful strict local maxima."""
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.serving.cv import digest
from .model import FootAndBall

CHECKPOINT_SHA='42fbc1f457fc34d804b403e26138524fc40facde303ccd7b4be73121e8da4cd5'


def preprocess(image,scale=1):
    if scale not in [1,2] or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected a BGR source image and frozen scale 1 or 2')
    height,width=image.shape[:2]
    resized=cv2.resize(image,(width*scale,height*scale),interpolation=cv2.INTER_LINEAR) if scale!=1 else image
    rgb=np.ascontiguousarray(resized[:,:,::-1]);tensor=torch.from_numpy(rgb).permute(2,0,1).float().div(255.)
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    tensor=(tensor-mean)/std;h,w=tensor.shape[1:];bottom=(-h)%32;right=(-w)%32
    tensor=F.pad(tensor,(0,right,0,bottom),value=0.).unsqueeze(0)
    return tensor,{'source_width':width,'source_height':height,'scale':scale,'pad_right':right,'pad_bottom':bottom,
        'input_width':w+right,'input_height':h+bottom,'stride':4,'cell_offset':1.5}


def peaks(heatmap,mapping,limit=100):
    """Strict > every neighbouring cell, including replicate-border ties, as upstream."""
    heatmap=np.asarray(heatmap,dtype=np.float32)
    if heatmap.ndim!=2 or not np.isfinite(heatmap).all() or heatmap.min()<0 or heatmap.max()>1:raise ValueError('Invalid ball confidence heatmap')
    kernel=np.ones((3,3),np.uint8);kernel[1,1]=0
    other=cv2.dilate(heatmap,kernel,borderType=cv2.BORDER_REPLICATE)
    ys,xs=np.where(heatmap>other);scores=heatmap[ys,xs]
    order=np.lexsort((xs,ys,-scores));candidates=[];padding=[]
    for index in order:
        x,y=int(xs[index]),int(ys[index]);px=(4*x+1.5)/mapping['scale'];py=(4*y+1.5)/mapping['scale']
        row={'center':[px,py],'confidence':float(scores[index]),'cell':[x,y]}
        if px>=mapping['source_width'] or py>=mapping['source_height']:
            if len(padding)<100:padding.append(row)
            continue
        row['rank']=len(candidates)+1
        if limit is None or len(candidates)<limit:candidates.append(row)
    return candidates,padding


class Detector:
    def __init__(self,checkpoint,device='auto'):
        started=time.perf_counter()
        if digest(checkpoint)!=CHECKPOINT_SHA:raise ValueError('Official FootAndBall checkpoint hash mismatch')
        self.device=('mps' if torch.backends.mps.is_available() else 'cpu') if device=='auto' else device
        if self.device not in ['cpu','mps']:raise ValueError('M24 supports CPU or Apple MPS')
        self.fallbacks=[];self.model=FootAndBall()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):raise ValueError('Expected a safe official state dictionary')
        self.model.load_state_dict(state,strict=True);self.model.eval();self.model.to(self.device)
        self.parameters=sum(p.numel() for p in self.model.parameters())
        if self.parameters!=198840:raise ValueError('Unexpected FootAndBall architecture')
        self.initialization_seconds=time.perf_counter()-started

    def infer(self,image,scale=1):
        started=time.perf_counter();tensor,mapping=preprocess(image,scale);pre=time.perf_counter()-started
        def forward():
            x=tensor.to(self.device)
            if self.device=='mps':torch.mps.synchronize()
            tick=time.perf_counter()
            with torch.inference_mode():output=self.model(x)
            if self.device=='mps':torch.mps.synchronize()
            neural=time.perf_counter()-tick
            return output,neural
        try:output,neural=forward()
        except RuntimeError as exc:
            if self.device!='mps' or not any(s in str(exc).lower() for s in ['mps','not implemented','out of memory']):raise
            self.fallbacks.append({'from':'mps','to':'cpu','reason':str(exc)[:300]});self.device='cpu';self.model.to('cpu');output,neural=forward()
        tick=time.perf_counter()
        expected=(mapping['input_height']//4,mapping['input_width']//4)
        if tuple(output['ball'].shape)!=(1,2,*expected) or tuple(output['player'].shape)!=(1,2,mapping['input_height']//16,mapping['input_width']//16):raise ValueError('FootAndBall output stride/shape mismatch')
        if any(not torch.isfinite(v).all().item() for v in output.values()):raise ValueError('Nonfinite neural output')
        heatmap=output['ball'][0,1].detach().cpu().numpy().copy();candidates,padding=peaks(heatmap,mapping)
        return {'mapping':mapping,'candidates':candidates,'padding_rejected':padding,
            'selected':candidates[0] if candidates and candidates[0]['confidence']>=.7 else None,
            'detection_threshold':.7,'model_version':'official_fb1_20201019_1416_final','device':self.device,
            'heatmap':heatmap,'timing':{'preprocessing_seconds':pre,'neural_seconds':neural,
                'postprocessing_seconds':time.perf_counter()-tick,'inference_seconds':time.perf_counter()-started}}
