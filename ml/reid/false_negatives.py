"""Lowest-scoring true pairs from cached held-out embeddings; no model selection."""
import json
import numpy as np,cv2
from .protocol import ROOT,retrieval_sets
from .evaluate import embedding_cache
from src.vision.world.player_appearance import digest

def run():
 p=ROOT/'curated_manifest.json';rows=[r for r in json.loads(p.read_text())['samples'] if r['split']=='test'];index={r['sample_id']:i for i,r in enumerate(rows)};q,g=retrieval_sets(rows);q=[x[0] for x in q];g=[x[0] for x in g];lock=json.loads((ROOT/'selection_lock.json').read_text());report=[]
 for name,cp,strategy in [('generic',None,'m9_body'),('soccer',lock['selected_checkpoint'],lock['strategy'])]:
  v,_=embedding_cache(rows,digest(p),cp,strategy,'test');pairs=[]
  for a in q:
   for b in g:
    if a['player_identity']==b['player_identity']:pairs.append((float(v[index[a['sample_id']]]@v[index[b['sample_id']]]),a,b))
  tiles=[]
  for score,a,b in sorted(pairs,key=lambda x:x[0])[:12]:
   tile=np.full((235,360,3),25,np.uint8)
   for j,r in enumerate([a,b]):
    image=cv2.imread(r['crop_path']);tile[25:195,j*180+10:j*180+140]=cv2.resize(image,(130,170),interpolation=cv2.INTER_NEAREST);cv2.putText(tile,r['player_identity'],(j*180+5,215),cv2.FONT_HERSHEY_SIMPLEX,.34,(230,230,230),1)
   cv2.putText(tile,f'{name} SAME GT identity cosine {score:.3f}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(230,230,230),1);tiles.append(tile);report.append({'model':name,'query':a['sample_id'],'gallery':b['sample_id'],'cosine':score,'same_ground_truth_identity':True,'time_separation_seconds':b['timestamp']-a['timestamp'],'query_quality':a['quality'],'gallery_quality':b['quality'],'interpretation':'Inspect pose,scale,blur,occlusion,lighting; low similarity alone does not prove a source label error.'})
  cv2.imwrite(str(ROOT/f'false_negatives_{name}.jpg'),np.vstack(tiles))
 (ROOT/'false_negative_analysis.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':run()
