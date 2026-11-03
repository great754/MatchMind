import numpy as np

def pool(vectors,qualities=None):
 a=np.asarray(vectors,float)
 if a.ndim!=2 or len(a)==0:raise ValueError('Nonempty tracklet embeddings required')
 w=np.ones(len(a)) if qualities is None else np.clip(np.asarray(qualities,float),.01,1)
 if len(w)!=len(a) or not np.isfinite(a).all():raise ValueError('Invalid tracklet evidence')
 v=np.average(a,axis=0,weights=w);return v/max(np.linalg.norm(v),1e-12)
