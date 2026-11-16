"""Conditional M8 mapper evaluation against leave-one-view-out published triangulation.
This validates projection math with supplied external cameras, NOT M8 camera estimation or a ground classifier.
"""
import ast,csv,json
from collections import defaultdict,Counter
import numpy as np
from src.vision.ball.geometry import plane_intersection
from .common import ROOT,save,sha

def projection(cal):
 R=np.asarray(cal['rotation_matrix'],float);C=np.asarray(cal['position_meters'],float);pp=cal['principal_point'];K=np.array([[cal['x_focal_length'],0,pp[0]],[0,cal['y_focal_length'],pp[1]],[0,0,1.]]);return K@np.column_stack([R,-R@C])
def image(P,X):
 q=P@np.r_[X,1.];return q[:2]/q[2] if abs(q[2])>1e-8 else None
def triangulate(views):
 A=[]
 for v in views:
  P=v['P'];u,w=v['point'];A.extend([u*P[2]-P[0],w*P[2]-P[1]])
 _,_,vh=np.linalg.svd(np.asarray(A));X=vh[-1]
 return X[:3]/X[3] if abs(X[3])>1e-8 else None


def visible_field_polygon(P,width,height):
 """Clip the world pitch against positive depth and the image frustum before projection."""
 H=P[:,[0,1,3]];polygon=[np.array(p,float) for p in [(-52.5,-34),(52.5,-34),(52.5,34),(-52.5,34)]]
 for boundary in [H[2],H[0],(width-1)*H[2]-H[0],H[1],(height-1)*H[2]-H[1]]:
  clipped=[]
  if not polygon:break
  for a,b in zip(polygon,polygon[1:]+polygon[:1]):
   fa=float(boundary@np.r_[a,1.]);fb=float(boundary@np.r_[b,1.]);ina=fa>=1e-8;inb=fb>=1e-8
   if ina:clipped.append(a)
   if ina!=inb:clipped.append(a+(b-a)*(fa/(fa-fb)))
  polygon=clipped
 return [image(P,np.array([*p,0.])).tolist() for p in polygon]

def run():
 cfg={'minimum_views':3,'minimum_calibration_JaC005':.75,'max_other_view_reprojection_px':3.,'near_ground_abs_z_m':.15,'airborne_abs_z_m':.5,'scope':'Near-ground triangulated reference, not verified physical contact; held-out image point excluded from reference triangulation; supplied cameras, not estimated M8 pose.'};save('metric_geometry_protocol.json',cfg);groups=defaultdict(list);source=ROOT/'metric_reference/SNv3D.csv'
 for r in csv.DictReader(source.open()):
  try:
   if r['is_ball']!='True' or float(r['JaC@0.005'])<.75:continue
   b=ast.literal_eval(r['ball_bbox']);cal=ast.literal_eval(r['calibration']);point=np.array([b[0]+b[2]/2,b[1]+b[3]/2]);P=projection(cal)
   if not np.isfinite(P).all() or not np.isfinite(point).all():continue
  except (ValueError,TypeError,SyntaxError):continue
  key=(r['league'],r['season'],r['match'],r['main_action']);groups[key].append({'P':P,'point':point,'id':r[''],'width':int(float(r['img_w'])),'height':int(float(r['img_h']))})
 counters=Counter();examples=[];errors=[];airerrors=[];unique=set()
 for key,views in groups.items():
  if len(views)<3:continue
  counters['multiview_groups']+=1
  for i,v in enumerate(views):
   others=views[:i]+views[i+1:];X=triangulate(others)
   if X is None or not np.isfinite(X).all():continue
   residual=max(np.linalg.norm(image(o['P'],X)-o['point']) for o in others)
   if residual>3 or not(-52.5<=X[0]<=52.5 and -34<=X[1]<=34):continue
   counters['quality_reference_views']+=1;near=abs(X[2])<=.15;air=abs(X[2])>=.5
   if not near and not air:continue
   P=v['P'];H=P[:,[0,1,3]]
   if abs(np.linalg.det(H))<1e-8:continue
   offset=np.array([[1,0,52.5],[0,1,34],[0,0,1.]])@np.linalg.inv(H);region=visible_field_polygon(P,v['width'],v['height']);camera={'calibration_valid':True,'calibration':{'image_to_pitch':offset.flatten().tolist(),'validity_region':region}};position,reason=plane_intersection(v['point'],camera);truth=X[:2]+[52.5,34]
   if near:
    counters['near_ground_reference_views']+=1;unique.add(key)
    if position is None:counters['mapper_rejected_'+reason]+=1
    if position is not None:
     error=float(np.linalg.norm(np.asarray(position)-truth));errors.append(error);examples.append({'source_row':v['id'],'match':key[2],'action':key[3],'heldout_source_point':v['point'].tolist(),'reference_from_other_views_xyz_m':X.tolist(),'other_view_max_reprojection_px':float(residual),'mapped_pitch_xy':position,'reference_pitch_xy':truth.tolist(),'error_m':error,'source':'published SoccerNet-v3D annotations; held-out input excluded from DLT reference'})
   if air and position is not None:airerrors.append(float(np.linalg.norm(np.asarray(position)-truth)))
 save('metric_geometry_results.json',{'protocol':cfg,'source_sha256':sha(source),'counts':dict(counters),'unique_near_ground_actions':len(unique),'near_ground_mapped_views':len(errors),'near_ground_mapping_coverage':len(errors)/counters['near_ground_reference_views'] if counters['near_ground_reference_views'] else None,'median_error_m':float(np.median(errors)) if errors else None,'p90_error_m':float(np.quantile(errors,.9)) if errors else None,'ungated_airborne_plane_projection_examples':len(airerrors),'ungated_airborne_median_error_m':float(np.median(airerrors)) if airerrors else None,'examples':examples,'region_note':'World pitch clipped against positive camera depth and image frustum; raw full-pitch quadrilaterals can cross the projective horizon and are invalid support polygons. Mapper thresholds unchanged; initial diagnostic retained separately.','limits':['Triangulation-derived reference, not surveyed ground truth or verified physical ground contact.','Supplied PnLCalib camera geometry shared by reference construction, not evaluation of M8 camera estimation.','Multiple images per action and reused calibration are correlated.','No raw images/footage downloaded; cannot train/validate a visual ground classifier or transfer labels to the three user videos.','These external reference mappings are not served as user-video observations.']});print('METRIC_REFERENCE',dict(counters),'mapped',len(errors),'median',np.median(errors) if errors else None,'p90',np.quantile(errors,.9) if errors else None,flush=True)
if __name__=='__main__':run()
