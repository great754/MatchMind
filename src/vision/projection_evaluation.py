"""Reproducible spatial/temporal audit of the unchanged MOT fallback against GSR."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.vision.coordinate_transform import PitchCoordinateTransformer
from src.vision.data_loading import DatasetPaths,load_mot
from src.vision.tactical_tracking import mapped_gsr_samples


def error_statistics(errors,eligible=None):
    """Coverage is finite pairs divided by all eligible detections, including failures."""
    a=np.asarray(errors,float);mask=np.ones(len(a),bool) if eligible is None else np.asarray(eligible,bool)
    values=a[mask & np.isfinite(a)];total=int(mask.sum())
    result=dict(samples=len(values),detections_total=total,coverage_pct=100*len(values)/total if total else None)
    for name,fn in [('mean_m',np.mean),('median_m',np.median),('max_m',np.max),('rmse_m',lambda x:np.sqrt(np.mean(x*x)))]:
        result[name]=float(fn(values)) if len(values) else None
    result.update({f'p{q}_m':float(np.percentile(values,q)) if len(values) else None for q in (50,75,90,95,99)})
    return result


def region_grid(points,errors,x_edges,y_edges):
    """Actual paired counts and mean errors; empty cells remain NaN, not zero."""
    points=np.asarray(points,float);errors=np.asarray(errors,float)
    valid=np.isfinite(points).all(axis=1)&np.isfinite(errors)
    counts,_,_=np.histogram2d(points[valid,0],points[valid,1],bins=(x_edges,y_edges))
    sums,_,_=np.histogram2d(points[valid,0],points[valid,1],bins=(x_edges,y_edges),weights=errors[valid])
    means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    return counts,means


def evaluate(paths=None,output=Path('outputs/projection_evaluation')):
    paths=paths or DatasetPaths();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    mot=load_mot(paths.mot);sync_path=paths.root/'mot'/f'{paths.match_id}_sync.json';sync=json.loads(sync_path.read_text())
    transformer=PitchCoordinateTransformer(paths.keypoints);feet=mot[['foot_x','foot_y']].to_numpy()
    projected=transformer.transform_points(feet);suffix='1st' if sync['half']==1 else '2nd'
    compact=paths.root/'gsr'/paths.match_id/f'{suffix}_compact.npz'
    with np.load(compact) as raw:
        ids,pitch,sides=raw['ids'],raw['pitch'],raw['sides']
        image_coords=raw['coords'] if 'coords' in raw.files else None
    offset=int(sync['clip_start_frame_zero_based'])
    reference,_=mapped_gsr_samples(mot,ids,pitch,sides,sync['players'],offset)
    foot_diagnostics=None
    if image_coords is not None:
        gsr_feet,_=mapped_gsr_samples(mot,ids,image_coords,sides,sync['players'],offset)
        gsr_projected=transformer.transform_points(gsr_feet)
        native_errors=np.linalg.norm(gsr_projected-reference,axis=1)
        common=np.isfinite(projected).all(axis=1)&np.isfinite(gsr_projected).all(axis=1)&np.isfinite(reference).all(axis=1)
        image_distances=np.linalg.norm(feet-gsr_feet,axis=1)
        image_mapping=[]
        for f in range(0,int(mot.frame.max())+1,25):
            indices=np.flatnonzero((mot.frame.to_numpy()==f)&np.isfinite(gsr_feet).all(axis=1))
            if len(indices)<2:continue
            distances=np.linalg.norm(feet[indices,None,:]-gsr_feet[None,indices,:],axis=2)
            image_mapping.extend((distances.argmin(axis=1)==np.arange(len(indices))).tolist())
        foot_diagnostics=dict(image_foot_disagreement={key.replace('_m','_px'):value for key,value in error_statistics(image_distances).items()},
            mean_image_foot_vector_px=np.nanmean(feet-gsr_feet,axis=0).tolist(),
            native_gsr_foot_projection=error_statistics(native_errors),
            common_pairs_mot_foot=error_statistics(np.linalg.norm(projected-reference,axis=1),common),
            common_pairs_gsr_foot=error_statistics(native_errors,common),
            image_nearest_actor_agreement_fraction=float(np.mean(image_mapping)) if image_mapping else None,
            interpretation='Diagnostic input comparison only: project the provider image foot point through the unchanged transform. Neither transform nor mapping is fitted or replaced; this is not independent accuracy.')
    valid=np.isfinite(projected).all(axis=1)&np.isfinite(reference).all(axis=1)
    delta=projected-reference;errors=np.linalg.norm(delta,axis=1);errors[~valid]=np.nan
    cap=cv2.VideoCapture(str(paths.video))
    try: width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally: cap.release()
    if width<=0 or height<=0: raise ValueError('Readable panorama required to define image regions')
    nearest=cKDTree(transformer.image_points).query(feet)[0]
    simplex=transformer.triangulation.find_simplex(feet)
    triangles=transformer.image_points[transformer.triangulation.simplices]
    edges=np.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,1],triangles[:,0]-triangles[:,2]),axis=1)
    lengths=np.linalg.norm(edges,axis=2)
    areas=np.abs(np.linalg.det(np.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)))/2
    aspect=lengths.max(axis=1)**2/np.maximum(2*areas,1e-12)
    image_region=np.where((feet[:,0]>=width*.25)&(feet[:,0]<=width*.75),'center_half','outer_quarters')
    near=(reference[:,0]<5)|(reference[:,0]>100)|(reference[:,1]<5)|(reference[:,1]>63)
    outside=(reference[:,0]<0)|(reference[:,0]>105)|(reference[:,1]<0)|(reference[:,1]>68)
    boundary=np.where(outside,'outside_pitch',np.where(near,'within_5m_of_boundary','interior'))
    boundary[~np.isfinite(reference).all(axis=1)]='unavailable'
    rows=mot[['frame','player_id','foot_x','foot_y']].copy()
    for name,values in dict(error_m=errors,dx_m=delta[:,0],dy_m=delta[:,1],gsr_x_m=reference[:,0],gsr_y_m=reference[:,1],
        projected_x_m=projected[:,0],projected_y_m=projected[:,1],nearest_landmark_px=nearest,
        simplex=simplex,triangle_aspect=np.where(simplex>=0,aspect[np.maximum(simplex,0)],np.nan),image_region=image_region,pitch_boundary_region=boundary).items(): rows[name]=values
    rows['seconds']=rows.frame/sync['fps'];rows['paired']=valid
    def groups(column):
        return {str(value):error_statistics(errors,rows[column].to_numpy()==value) for value in sorted(rows[column].unique())}
    baseline=error_statistics(errors)
    shifts={}
    for shift in range(-2,3):
        ref,_=mapped_gsr_samples(mot,ids,pitch,sides,sync['players'],offset+shift)
        e=np.linalg.norm(projected-ref,axis=1)
        shifts[str(shift)]=error_statistics(e)
    # Diagnostics only: neither a different orientation nor a fitted correction is deployed.
    orientation={}
    for name,xy in {'original':reference,'x_reversed':np.column_stack((105-reference[:,0],reference[:,1])),
                    'y_reversed':np.column_stack((reference[:,0],68-reference[:,1])),
                    'both_reversed':np.column_stack((105-reference[:,0],68-reference[:,1]))}.items():
        orientation[name]=error_statistics(np.linalg.norm(projected-xy,axis=1))
    # Actor mapping check on a shared finite roster: assigned error versus nearest actor.
    mapping_checks=[]
    for frame in range(0,int(mot.frame.max())+1,25):
        indices=np.flatnonzero((mot.frame.to_numpy()==frame)&valid)
        if len(indices)<2: continue
        distances=np.linalg.norm(projected[indices,None,:]-reference[None,indices,:],axis=2)
        mapping_checks.extend((distances.argmin(axis=1)==np.arange(len(indices))).tolist())
    frame_errors=rows.groupby('frame').error_m.agg(['count','mean','median','max']).reset_index()
    def binned(values,edges):
        return {f'{lo:g}–{hi:g}':error_statistics(errors,(values>=lo)&(values<hi)) for lo,hi in zip(edges[:-1],edges[1:])}
    pitch_x=np.linspace(0,105,13);pitch_y=np.linspace(0,68,9)
    image_x=np.linspace(0,width,17);image_y=np.linspace(0,height,9)
    pc,pm=region_grid(reference,errors,pitch_x,pitch_y);ic,im=region_grid(feet,errors,image_x,image_y)
    def grid(counts,means,x,y):return dict(x_edges=x.tolist(),y_edges=y.tolist(),counts=counts.astype(int).tolist(),mean_error_m=np.where(np.isfinite(means),means,None).tolist())
    report=dict(schema_version='1.0',before=baseline,after=baseline.copy(),method_changed=False,
        evaluation_method='All finite paired MOT foot-point projections and synchronized GSR actors; no fitting, filtering by error, or calibration changes.',
        units={'errors':'meters','landmark_distance':'pixels','time':'recording-local seconds','triangle_aspect':'longest_edge_squared / twice_area'},
        fps=sync['fps'],frames=int(mot.frame.nunique()),source_offset_frames=offset,image_size=[width,height],
        per_player=groups('player_id'),image_regions=groups('image_region'),pitch_boundary_regions=groups('pitch_boundary_region'),
        per_triangle=groups('simplex'),landmark_distance_bins=binned(nearest,[0,100,200,400,800,1600,1e9]),
        triangle_aspect_bins=binned(rows.triangle_aspect.to_numpy(),[0,2,5,10,20,50,1e9]),
        synchronization_shifts=shifts,orientation_experiments=orientation,
        foot_point_diagnostics=foot_diagnostics,
        mean_error_vector_m=np.mean(delta[valid],axis=0).tolist(),median_error_vector_m=np.median(delta[valid],axis=0).tolist(),
        nearest_actor_agreement_fraction=float(np.mean(mapping_checks)) if mapping_checks else None,
        pitch_grid=grid(pc,pm,pitch_x,pitch_y),image_grid=grid(ic,im,image_x,image_y),
        source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (paths.mot,paths.keypoints,sync_path,compact)],
        limitations=['GSR is a quantized reference; disagreement is not independently surveyed absolute error.',
            'Nearest-actor agreement cannot verify or refute IDs when projection error exceeds inter-player spacing.',
            'Image location, landmark spacing and triangle shape are correlated; association does not establish a distortion cause.',
            'Foot-point error cannot be separated from calibration error without independently labeled image ground contacts.'])
    rows.to_csv(output/'projection_pairs.csv',index=False,float_format='%.6f')
    frame_errors.to_csv(output/'projection_error_by_frame.csv',index=False)
    (output/'projection_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matchmind-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));ax.hist(errors[valid],bins=60);ax.set(xlabel='MOT projection vs GSR disagreement (m)',ylabel='Paired detections',title='All paired detections; unchanged calibration');fig.tight_layout();fig.savefig(output/'error_distribution.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(frame_errors.frame/sync['fps'],frame_errors['mean'],label='Mean');ax.plot(frame_errors.frame/sync['fps'],frame_errors['median'],label='Median');ax.legend();ax.set(xlabel='Clip time (s)',ylabel='Disagreement (m)');fig.tight_layout();fig.savefig(output/'error_over_time.png',dpi=130);plt.close(fig)
    for name,means,x,y,xlabel,ylabel in [('pitch',pm,pitch_x,pitch_y,'GSR pitch x (m)','GSR pitch y (m)'),('image',im,image_x,image_y,'Foot x (px)','Foot y (px)')]:
        fig,ax=plt.subplots(figsize=(10,4));plot=ax.pcolormesh(x,y,np.ma.masked_invalid(means.T),cmap='magma',vmin=0,vmax=float(np.nanmax(means)));fig.colorbar(plot,ax=ax,label='Mean disagreement (m)');ax.set(xlabel=xlabel,ylabel=ylabel,title='Actual paired detections; blank cells have no samples');ax.invert_yaxis();fig.tight_layout();fig.savefig(output/f'{name}_error_heatmap.png',dpi=130);plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('outputs/projection_evaluation'));parser.add_argument('--match-id',default='118575');parser.add_argument('--data-root',type=Path,default=Path('data/soccertrack'));args=parser.parse_args()
    report=evaluate(DatasetPaths(args.match_id,args.data_root),args.output)
    print(json.dumps({k:report[k] for k in ('before','after','method_changed','image_regions','mean_error_vector_m','synchronization_shifts')},indent=2))

if __name__=='__main__':main()
