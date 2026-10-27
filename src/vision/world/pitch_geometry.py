"""One configured, stationary metric pitch. Semantic IDs are evidence, not poses."""
from dataclasses import dataclass
import cv2,numpy as np
from src.vision.pitch import PITCH_LENGTH,PITCH_WIDTH,PENALTY_AREA_DEPTH,PENALTY_AREA_WIDTH,GOAL_AREA_DEPTH,GOAL_AREA_WIDTH,CENTER_CIRCLE_RADIUS

@dataclass(frozen=True)
class PitchGeometry:
    length: float = PITCH_LENGTH
    width: float = PITCH_WIDTH
    penalty_depth: float = PENALTY_AREA_DEPTH
    penalty_width: float = PENALTY_AREA_WIDTH
    goal_area_depth: float = GOAL_AREA_DEPTH
    goal_area_width: float = GOAL_AREA_WIDTH
    circle_radius: float = CENTER_CIRCLE_RADIUS
    penalty_spot_distance: float = 11.
    @property
    def center(self):return (self.length/2,self.width/2)
    @property
    def segments(self):
        L,W=self.length,self.width;p=self.penalty_depth;g=self.goal_area_depth;py=(W-self.penalty_width)/2;gy=(W-self.goal_area_width)/2
        return {'Side line top':[(0,0),(L,0)],'Side line bottom':[(0,W),(L,W)],'Side line left':[(0,0),(0,W)],'Side line right':[(L,0),(L,W)],'Middle line':[(L/2,0),(L/2,W)],
        'Big rect. left main':[(p,py),(p,W-py)],'Big rect. left top':[(0,py),(p,py)],'Big rect. left bottom':[(0,W-py),(p,W-py)],
        'Big rect. right main':[(L-p,py),(L-p,W-py)],'Big rect. right top':[(L-p,py),(L,py)],'Big rect. right bottom':[(L-p,W-py),(L,W-py)],
        'Small rect. left main':[(g,gy),(g,W-gy)],'Small rect. left top':[(0,gy),(g,gy)],'Small rect. left bottom':[(0,W-gy),(g,W-gy)],
        'Small rect. right main':[(L-g,gy),(L-g,W-gy)],'Small rect. right top':[(L-g,gy),(L,gy)],'Small rect. right bottom':[(L-g,W-gy),(L,W-gy)]}
    @property
    def circle_centers(self):return {'Circle central':self.center,'Circle left':(self.penalty_spot_distance,self.width/2),'Circle right':(self.length-self.penalty_spot_distance,self.width/2)}
    @property
    def points(self):
        L,W=self.length,self.width
        result={'center':self.center,'top_left':(0,0),'top_right':(L,0),'bottom_right':(L,W),'bottom_left':(0,W),'penalty_left':self.circle_centers['Circle left'],'penalty_right':self.circle_centers['Circle right'],'halfway_circle_top':(L/2,W/2-self.circle_radius),'halfway_circle_bottom':(L/2,W/2+self.circle_radius)}
        for name,segment in self.segments.items():
            result[name+'_start']=segment[0];result[name+'_end']=segment[1]
        return result
    def curves(self):
        angle=np.linspace(0,2*np.pi,257);curves={}
        for name,center in self.circle_centers.items():
            points=np.column_stack([center[0]+self.circle_radius*np.cos(angle),center[1]+self.circle_radius*np.sin(angle)])
            if name=='Circle left':points=points[points[:,0]>=self.penalty_depth]
            if name=='Circle right':points=points[points[:,0]<=self.length-self.penalty_depth]
            curves[name]=points
        return curves
    def residuals(self,h,groups,width,height,heldout=False):
        """Signed line/conic image residuals, source pixels; dependent diagnostics."""
        errors=[];names=[]
        for name,raw in groups.items():
            if name not in self.segments and name not in self.circle_centers:continue
            raw=np.asarray(raw);n=min(80,len(raw));ids=np.linspace(0,len(raw)-1,n,dtype=int);points=raw[ids[1::2] if heldout else ids[::2]]*[width,height]
            if not len(points):continue
            p=np.column_stack([points,np.ones(len(points))])
            if name in self.segments:
                a,b=self.segments[name];line=np.cross([*a,1],[*b,1]);image_line=h.T@line;denom=np.linalg.norm(image_line[:2]);residual=p@image_line/max(denom,1e-12)
            else:
                x,y=self.circle_centers[name];q=np.array([[1,0,-x],[0,1,-y],[-x,-y,x*x+y*y-self.circle_radius**2]],float);q=h.T@q@h
                gradient=2*(p@q.T)[:,:2];denom=np.linalg.norm(gradient,axis=1);residual=np.sum((p@q)*p,axis=1)/np.maximum(denom,1e-12)
            errors.extend(residual);names.extend([name]*len(residual))
        return np.asarray(errors,float),names
    def overlay(self,h,width,height):
        try:inverse=np.linalg.inv(h)
        except np.linalg.LinAlgError:return []
        result=[]
        for name,points in {**self.segments,**self.curves()}.items():
            p=np.column_stack([points,np.ones(len(points))])@inverse.T
            if np.any(np.abs(p[:,2])<1e-9) or not np.isfinite(p).all():continue
            pixels=p[:,:2]/p[:,2:]
            # Break offscreen/horizon segments; never draw a misleading long line.
            for a,b in zip(pixels,pixels[1:]):
                if np.max(np.abs([*a,*b]))>1e6:continue
                ok,start,end=cv2.clipLine((0,0,width,height),tuple(np.round(a).astype(int)),tuple(np.round(b).astype(int)))
                if ok:result.append({'kind':name,'start':list(start),'end':list(end)})
        return result
