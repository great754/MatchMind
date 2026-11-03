"""Small causal image-motion baselines; camera warps precede history prediction."""
import math
import numpy as np

def features(points,times,confidence,diagonal):
 p=np.asarray(points,float);t=np.asarray(times,float);dt=np.maximum(np.diff(t,prepend=t[0]-.04),1e-3);v=np.vstack([np.zeros(2),np.diff(p,axis=0)])/dt[:,None]*.04;acc=np.vstack([np.zeros(2),np.diff(v,axis=0)]);return np.column_stack([(p-p[-1])/diagonal*1000,v/diagonal*1000,acc/diagonal*1000,dt*25,np.asarray(confidence)]).astype('float32')

def extrapolate(points,times,future,mode='velocity'):
 p=np.asarray(points,float);t=np.asarray(times,float);n=min(len(p),5 if mode=='acceleration' else 3);degree=2 if mode=='acceleration' and n>=4 else 1
 if len(p)<2:return p[-1].copy()
 x=t[-n:]-t[-1];coeff=np.polynomial.polynomial.polyfit(x,p[-n:],degree);return np.polynomial.polynomial.polyval(future-t[-1],coeff)

class MotionHistory:
 def __init__(self,maximum=8):self.maximum=maximum;self.points=[];self.times=[];self.confidence=[]
 def reset(self):self.points=[];self.times=[];self.confidence=[]
 def append(self,point,timestamp,confidence):
  if self.times and timestamp<=self.times[-1]:raise ValueError('Motion history must increase')
  self.points=(self.points+[tuple(point)])[-self.maximum:];self.times=(self.times+[timestamp])[-self.maximum:];self.confidence=(self.confidence+[confidence])[-self.maximum:]
 def predict(self,timestamp,mode='velocity'):
  if not self.points:return None
  return extrapolate(self.points,self.times,timestamp,mode)
