"""Three-frame ResNet-18/FPN CenterNet football detector.

Architecture informed by CourtVision's MIT student detector. Football heads,
source geometry, visibility masks and training supervision are separate here.
No CourtVision or basketball-trained weights are loaded.
"""
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'm22-three-frame-resnet18-fpn64-centernet-stride4-v1'
MEAN=np.array([.485,.456,.406],np.float32)
STD=np.array([.229,.224,.225],np.float32)


class TemporalFootballNet(nn.Module):
    def __init__(self, imagenet_path=None):
        super().__init__()
        from torchvision.models import resnet18
        backbone=resnet18(weights=None)
        if imagenet_path is not None:
            backbone.load_state_dict(torch.load(Path(imagenet_path),map_location='cpu',weights_only=True),strict=True)
        rgb=backbone.conv1.weight.detach().clone()
        backbone.conv1=nn.Conv2d(9,64,7,stride=2,padding=3,bias=False)
        with torch.no_grad():backbone.conv1.weight.copy_(torch.cat([rgb/3]*3,dim=1))
        self.backbone=backbone
        self.lateral=nn.ModuleList([nn.Conv2d(c,64,1) for c in [64,128,256,512]])
        self.refine=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU())
        def head(channels):return nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,channels,1))
        self.heatmap=head(1);self.offset=head(2);self.size=head(2)
        nn.init.constant_(self.heatmap[-1].bias,-4.6)

    def forward(self, images):
        if images.ndim!=4 or images.shape[1]!=9:raise ValueError('Oldest/current RGB input must have nine channels')
        b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(images))))
        levels=[]
        for layer in [b.layer1,b.layer2,b.layer3,b.layer4]:x=layer(x);levels.append(x)
        pyramid=self.lateral[-1](levels[-1])
        for i in [2,1,0]:pyramid=self.lateral[i](levels[i])+F.interpolate(pyramid,size=levels[i].shape[-2:],mode='nearest')
        feature=self.refine(pyramid)
        return {'heatmap':self.heatmap(feature),'offset':self.offset(feature),'size':self.size(feature)}

    def freeze_early_backbone(self):
        for name,parameter in self.backbone.named_parameters():
            parameter.requires_grad=name.startswith(('conv1.','layer4.'))
        # ImageNet batch statistics remain stable with tiny local batches.
        for module in self.backbone.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()


def spacing(fps, reference_fps=25):
    if not math.isfinite(fps) or fps<=0:raise ValueError('Finite positive source frame rate required')
    return max(1,int(math.floor(fps/reference_fps+.5)))


def context_indices(index,fps,epochs):
    step=spacing(fps);epoch=epochs[index]
    first=max(0,index-2*step)
    while first<index and epochs[first]!=epoch:first+=1
    return [max(first,i) for i in [index-2*step,index-step,index]]


def preprocess(images, output_hw=(384,640), region=None):
    """Letterbox every whole image/crop with one shared reversible source transform."""
    if len(images)!=3 or any(x.shape!=images[-1].shape for x in images):raise ValueError('Three aligned source-image geometries required')
    height,width=images[-1].shape[:2]
    x,y,w,h=(0,0,width,height) if region is None else region
    if min(x,y)<0 or min(w,h)<=0 or x+w>width or y+h>height:raise ValueError('Region outside source pixels')
    oh,ow=output_hw
    if oh%32 or ow%32:raise ValueError('Network dimensions must be multiples of 32')
    scale=min(ow/w,oh/h);nw,nh=round(w*scale),round(h*scale)
    # Use actual rounded resize factors; a single ideal scale can drift a source pixel.
    mapping={'x':x,'y':y,'width':w,'height':h,'scale_x':nw/w,'scale_y':nh/h,'resized_width':nw,'resized_height':nh,
        'output_height':oh,'output_width':ow,'padding':'right_bottom','stride':4}
    planes=[]
    for image in images:
        canvas=np.zeros((oh,ow,3),np.uint8)
        resized=cv2.resize(image[int(y):int(y+h),int(x):int(x+w)],(nw,nh))
        canvas[:nh,:nw]=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        planes.append(((canvas.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))
    return torch.from_numpy(np.concatenate(planes,axis=0).copy()),mapping


def targets(state,bbox,mapping,weight=1.,negative_regions=None):
    """Uncertain/unannotated frames have ZERO ball loss, including background loss."""
    h,w=mapping['output_height']//4,mapping['output_width']//4
    heatmap=np.zeros((1,h,w),np.float32);offset=np.zeros((2,h,w),np.float32);size=offset.copy();mask=np.zeros((1,h,w),np.float32)
    valid=np.zeros((1,h,w),np.float32)
    if state in ['visible','not_visible']:
        valid[:,:math.ceil(mapping['resized_height']/4),:math.ceil(mapping['resized_width']/4)]=weight
    # A rejected proposal certifies only its exact region. Supervise only cells
    # fully inside that region; surrounding unknown source pixels stay masked.
    for region in negative_regions or []:
        x,y,bw,bh=region
        left=max(0,math.ceil((x-mapping['x'])*mapping['scale_x']/4))
        top=max(0,math.ceil((y-mapping['y'])*mapping['scale_y']/4))
        right=min(w,math.floor((x+bw-mapping['x'])*mapping['scale_x']/4))
        bottom=min(h,math.floor((y+bh-mapping['y'])*mapping['scale_y']/4))
        if right>left and bottom>top:valid[:,top:bottom,left:right]=weight
    if state=='visible':
        if bbox is None:raise ValueError('Visible supervision requires a verified current source box')
        x,y,bw,bh=bbox
        cx=((x+bw/2-mapping['x'])*mapping['scale_x'])/4;cy=((y+bh/2-mapping['y'])*mapping['scale_y'])/4
        ix,iy=int(cx),int(cy)
        if not 0<=ix<w or not 0<=iy<h:raise ValueError('Crop removed the verified ball')
        yy,xx=np.mgrid[:h,:w];sigma=max(.7,min(2.,min(bw*mapping['scale_x'],bh*mapping['scale_y'])/12))
        heatmap[0]=np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma*sigma));heatmap[0,iy,ix]=1
        offset[:,iy,ix]=[cx-ix,cy-iy];size[:,iy,ix]=np.log(np.maximum([bw*mapping['scale_x']/4,bh*mapping['scale_y']/4],.1));mask[:,iy,ix]=weight
    elif state not in ['not_visible','uncertain','unannotated']:raise ValueError('Unknown visibility semantics')
    return {k:torch.from_numpy(v) for k,v in {'heatmap':heatmap,'offset':offset,'size':size,'mask':mask,'valid':valid}.items()}


def detection_loss(output,target):
    p=output['heatmap'].sigmoid().clamp(1e-5,1-1e-5);gt=target['heatmap'];valid=target['valid']
    positive=(gt==1).float()*valid;negative=(gt<1).float()*valid*(1-gt).pow(4)
    norm=positive.sum().clamp(min=1)
    heat=-(positive*(1-p).pow(2)*p.log()+negative*p.pow(2)*(1-p).log()).sum()/norm
    mask=target['mask'];count=mask.sum().clamp(min=1)
    off=(F.smooth_l1_loss(output['offset'],target['offset'],reduction='none')*mask).sum()/count
    size=(F.smooth_l1_loss(output['size'],target['size'],reduction='none')*mask).sum()/count
    return heat+off+.2*size,{'heatmap':float(heat.detach()),'offset':float(off.detach()),'size':float(size.detach())}


def decode(output,mapping,threshold=.05,top_k=5):
    hm=output['heatmap'][0].sigmoid().detach().cpu().numpy()[0]
    offsets=output['offset'][0].detach().cpu().numpy();sizes=output['size'][0].detach().cpu().numpy()
    maxima=(hm>=cv2.dilate(hm,np.ones((3,3),np.uint8)))&(hm>=threshold)
    yy,xx=np.nonzero(maxima);order=np.argsort(-hm[yy,xx],kind='stable')
    candidates=[]
    for j in order:
        ix,iy=int(xx[j]),int(yy[j]);off=offsets[:,iy,ix]
        nx,ny=(ix+float(off[0]))*4,(iy+float(off[1]))*4
        if not 0<=nx<mapping['resized_width'] or not 0<=ny<mapping['resized_height']:continue
        cx=mapping['x']+nx/mapping['scale_x'];cy=mapping['y']+ny/mapping['scale_y']
        bw,bh=np.exp(np.clip(sizes[:,iy,ix],-2.3,4))*4/[mapping['scale_x'],mapping['scale_y']]
        left=max(mapping['x'],cx-bw/2);top=max(mapping['y'],cy-bh/2)
        right=min(mapping['x']+mapping['width'],cx+bw/2);bottom=min(mapping['y']+mapping['height'],cy+bh/2)
        if right<=left or bottom<=top:continue
        candidates.append({'bbox':[float(left),float(top),float(right-left),float(bottom-top)],'confidence':float(hm[iy,ix]),
            'center':[float(cx),float(cy)],'heatmap_cell':[ix,iy],'inference_view':'M22_ACTUAL_CURRENT_AND_TWO_PAST_FRAMES',
            'source':'VIDEO_CV_DETECTOR'})
        if len(candidates)>=top_k:break
    return candidates
