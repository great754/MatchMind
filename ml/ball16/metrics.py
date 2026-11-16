"""Same frozen IoU metrics, with genuinely measured source sharpness covariates."""
from functools import lru_cache
import math,cv2
from ml.ball14.evaluate import detection_metrics as original
@lru_cache(maxsize=10000)
def sharpness(path,bbox):
 image=cv2.imread(path);x,y,w,h=bbox;crop=image[max(0,math.floor(y)):min(image.shape[0],math.ceil(y+h)),max(0,math.floor(x)):min(image.shape[1],math.ceil(x+w))]
 return float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if crop.size else 0.
def detection_metrics(rows,predictions,threshold=.35,overlap=.25):
 enriched=[{**r,'balls':[{**b,'sharpness_proxy':sharpness(r['path'],tuple(b['bbox']))} for b in r['balls']]} for r in rows];result=original(enriched,predictions,threshold,overlap);result['sharpness_covariate']='Measured source-box Laplacian variance; proxy, not independently labelled motion blur';return result
