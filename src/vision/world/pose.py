"""Optional offline semantic line/curve camera fitting, with explicit rejection.

No inference imports in web serving. Scores/residual gates are provisional,
not independently validated calibration accuracy or probabilities.
"""
import numpy as np
import cv2
from .semantic_pitch import CLASSES,GEOMETRY

def residual_gate(means,p90):
    return means.get('lines') is not None and all(means.get(k) is None or (p90.get(k) is not None and np.isfinite(means[k]) and np.isfinite(p90[k]) and means[k]<=10 and p90[k]<=20) for k in ('lines','circles'))

class PoseFitter:
    def __init__(self,width,height,steps=2000,starts=3):
        import torch
        from .tvcalib.utils.objects_3d import SoccerPitchSN,SoccerPitchLineCircleSegments
        self.torch=torch;torch.set_num_threads(4);self.width=width;self.height=height;self.steps=steps;self.starts=starts
        torch.manual_seed(42);self.field=SoccerPitchLineCircleSegments(base_field=SoccerPitchSN(),device='cpu')
    @staticmethod
    def groups(mask,confidence):
        result={}
        for i,name in enumerate(CLASSES,1):
            yy,xx=np.where((mask==i)&(confidence>=.5))
            if len(xx)>=12:result[name]=np.column_stack([(xx+.5)/mask.shape[1],(yy+.5)/mask.shape[0]])
        return result
    @staticmethod
    def sufficient(groups):
        ground=[name for name in groups if name in GEOMETRY]
        circles=[name for name in groups if name.startswith('Circle')]
        return (len(ground)>=3 and len({GEOMETRY[n][0] for n in ground})==2) or (len(ground)>=2 and bool(circles))
    def batch(self,samples):
        from .tvcalib.module import TVCalibModule
        from .tvcalib.cam_distr.tv_main_center import get_cam_distr
        torch=self.torch;expanded=[g for g in samples for _ in range(self.starts)];x={};held={}
        for kind,names in [('lines',self.field.line_segments_names),('circles',self.field.circle_segments_names)]:
            batch=[];masks=[];hold_batch=[]
            for groups in expanded:
                points=np.zeros((3,len(names),16),np.float32);points[2]=1;valid=np.zeros((1,len(names),16),bool);hold=points.copy()
                for j,name in enumerate(names):
                    if name not in groups:continue
                    p=groups[name];n=min(16,len(p)//2);ids=np.linspace(0,len(p)-1,2*n,dtype=int)
                    points[:2,j,:n]=(p[ids[::2]]*2-1).T;hold[:2,j,:n]=(p[ids[1::2]]*2-1).T;valid[0,j,:n]=True
                batch.append(points);masks.append(valid);hold_batch.append(hold)
            x[kind+'__ndc_projected_selection_shuffled']=torch.tensor(np.stack(batch))[:,None]
            x[kind+'__is_keypoint_mask']=torch.tensor(np.stack(masks))[:,None]
            held[kind+'__ndc_projected_selection_shuffled']=torch.tensor(np.stack(hold_batch))[:,None]
            held[kind+'__is_keypoint_mask']=x[kind+'__is_keypoint_mask']
        optimizer=TVCalibModule(self.field,get_cam_distr(1.96,len(expanded),1),None,(self.height,self.width),self.steps,'cpu',tqdm_kwqargs={'disable':True})
        original=optimizer.cam_param_dict.initialize
        def initialize(*args,**kwargs):
            original(*args,**kwargs)
            with torch.no_grad():
                for parameter in optimizer.cam_param_dict.param_dict.values():
                    noise=torch.randn_like(parameter)*.5;noise[::self.starts]=0;parameter.copy_(noise)
        optimizer.cam_param_dict.initialize=initialize
        _,camera,_=optimizer.self_optim_batch(x)
        matrices=camera.get_homography_raster().detach().cpu().numpy()
        distances,_=optimizer(held);errors={};p90={}
        # NDC has different x/y scaling for non-square frames. Use the larger
        # pixel scale as a conservative residual bound, never metric accuracy.
        scale=max(self.width,self.height)/2
        for kind in ['lines','circles']:
            raw=distances['loss_ndc_'+kind].detach().cpu().numpy();valid=held[kind+'__is_keypoint_mask'].numpy()
            errors[kind]=[float(raw[i][valid[i]].mean()*scale) if valid[i].any() else None for i in range(len(expanded))]
            p90[kind]=[float(np.quantile(raw[i][valid[i]],.9)*scale) if valid[i].any() else None for i in range(len(expanded))]
        output=[]
        for i,groups in enumerate(samples):
            ids=list(range(i*self.starts,(i+1)*self.starts));diagnostic={'method':'TVCalib line/curve camera optimization','starts':self.starts,'steps':self.steps,
                'heldout_mean_px':{k:[errors[k][j] for j in ids] for k in errors},'heldout_p90_px':{k:[p90[k][j] for j in ids] for k in errors},
                'thresholds':{'mean_px':10,'p90_px':20,'pose_p90_disagreement_m':4},'source_classes':list(groups),'accepted':False,'solver_attempted':True}
            diagnostic['residual_pass_count']=sum(residual_gate({k:errors[k][j] for k in errors},{k:p90[k][j] for k in p90}) for j in ids)
            nonfinite=any(v is not None and not np.isfinite(v) for values in (errors,p90) for k in values for j in ids for v in [values[k][j]])
            # JSON remains strict even when an optimizer produces nonfinite loss.
            for key in ('heldout_mean_px','heldout_p90_px'):
                diagnostic[key]={k:[v if v is None or np.isfinite(v) else None for v in values] for k,values in diagnostic[key].items()}
            if nonfinite:diagnostic['nonfinite_optimizer_residual']=True
            good=[j for j in ids if residual_gate({k:errors[k][j] for k in errors},{k:p90[k][j] for k in p90})]
            if len(good)<2:
                diagnostic['outcome']='POSE_OPTIMIZATION_FAILED' if nonfinite else 'POSE_RESIDUAL_TOO_HIGH';diagnostic['reason']='Insufficient agreeing low-residual camera hypotheses';output.append((None,diagnostic));continue
            h=[np.array([[1,0,52.5],[0,1,34],[0,0,1]])@matrices[j] for j in good]
            image_grid=np.array([(x,y) for y in np.linspace(self.height*.3,self.height*.9,5) for x in np.linspace(self.width*.1,self.width*.9,5)],np.float64)[None]
            projections=np.array([cv2.perspectiveTransform(image_grid,m)[0] for m in h]);spread=np.linalg.norm(projections-projections[0],axis=2)
            diagnostic['pose_p90_disagreement_m']=float(np.quantile(spread,.9)) if np.isfinite(spread).all() else None
            if not np.isfinite(projections).all() or diagnostic['pose_p90_disagreement_m'] is None or diagnostic['pose_p90_disagreement_m']>4:
                diagnostic['outcome']='POSE_HYPOTHESES_DISAGREE';diagnostic['reason']='Metric projection varies between optimization starts';output.append((None,diagnostic));continue
            best=min(good,key=lambda j:sum(errors[k][j] or 0 for k in errors));matrix=np.array([[1,0,52.5],[0,1,34],[0,0,1]])@matrices[best]
            region=visible_region(matrix,self.width,self.height)
            if region is None:
                diagnostic['outcome']='INVALID_VISIBLE_REGION';diagnostic['reason']='Invalid projected pitch or visible region';output.append((None,diagnostic));continue
            diagnostic.update(accepted=True,outcome='DIRECT_ACCEPTED',reason='Held-out pixel residual and local pose-stability gates passed; heuristic, not independent accuracy')
            output.append(({'homography':(matrix/matrix[2,2]).tolist(),'image_region':region[0],'pitch_region':region[1]},diagnostic))
        return output

def visible_region(h,width,height):
    """Clip the projected *fixed* canonical pitch to the image rectangle."""
    pitch=np.array([[0,0],[105,0],[105,68],[0,68]],np.float64)
    try:
        inverse=np.linalg.inv(h);v=np.column_stack([pitch,np.ones(4)])@inverse.T
        if not np.isfinite(v).all() or np.any(np.abs(v[:,2])<1e-9) or np.any(np.sign(v[:,2])!=np.sign(v[0,2])):return None
        image=(v[:,:2]/v[:,2:]).astype(np.float32)
        if not cv2.isContourConvex(image):return None
        area,clipped=cv2.intersectConvexConvex(image,np.array([[0,0],[width,0],[width,height],[0,height]],np.float32))
        if clipped is None or area<width*height*.02:return None
        clipped=clipped.reshape(-1,2);mapped=cv2.perspectiveTransform(clipped[None].astype(np.float64),h)[0]
        if not np.isfinite(mapped).all() or (mapped<[-.01,-.01]).any() or (mapped>[105.01,68.01]).any():return None
        return clipped.tolist(),np.clip(mapped,[0,0],[105,68]).tolist()
    except (np.linalg.LinAlgError,cv2.error):return None
