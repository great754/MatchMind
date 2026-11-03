"""Standalone curves, failure contact sheets and interactive retrieval explorer."""
import json,os,html
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .protocol import ROOT,retrieval_sets
from .evaluate import embedding_cache
from src.vision.world.player_appearance import digest

def run():
 report=json.loads((ROOT/'heldout_comparison.json').read_text());lock=json.loads((ROOT/'selection_lock.json').read_text());history=json.loads((ROOT/'epoch_history.json').read_text());fig,axes=plt.subplots(2,3,figsize=(15,8))
 for aug in ['none','moderate']:
  h=[r for r in history if r['augmentation']==aug];x=[r['epoch'] for r in h]
  for ax,key,title in [(axes[0,0],'train_loss','Train CE + triplet'),(axes[0,1],'validation_loss','Validation triplet loss')]:ax.plot(x,[r[key] for r in h],marker='o',label=aug);ax.set_title(title)
  for ax,key,title in [(axes[0,2],'rank1','Validation crop Rank-1'),(axes[1,0],'map','Validation crop mAP'),(axes[1,1],'same_team_map','Validation same-team mAP'),(axes[1,2],'tracklet_map','Validation same-team tracklet mAP')]:ax.plot(x,[r['validation_metrics'][key] for r in h],marker='o',label=aug);ax.set_title(title)
 for ax in axes.ravel():ax.set_xlabel('epoch');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
 manifest=ROOT/'curated_manifest.json';rows=[r for r in json.loads(manifest.read_text())['samples'] if r['split']=='test'];by={r['sample_id']:r for r in rows};queries,galleries=retrieval_sets(rows);queries=[x[0] for x in queries];galleries=[x[0] for x in galleries];lookup={r['sample_id']:i for i,r in enumerate(rows)};retrieval={};failure_records=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
 for col,(name,checkpoint,strategy) in enumerate([('Generic OSNet',None,'m9_body'),('Soccer-ReID v1',lock['selected_checkpoint'],lock['strategy'])]):
  vectors,_=embedding_cache(rows,digest(manifest),checkpoint,strategy,'test');qv=vectors[[lookup[r['sample_id']] for r in queries]];gv=vectors[[lookup[r['sample_id']] for r in galleries]];sims=qv@gv.T;output=[];positive=[];neg=[]
  for i,q in enumerate(queries):
   choices=[j for j,g in enumerate(galleries) if g['match_id']==q['match_id']];order=sorted(choices,key=lambda j:-sims[i,j]);items=[]
   for j in order[:10]:
    g=galleries[j];items.append({'id':g['sample_id'],'similarity':float(sims[i,j]),'same_identity':g['player_identity']==q['player_identity'],'same_team':g['team_identity']==q['team_identity'],'team':g['team_identity'],'match':g['match_id'],'quality':g['quality'],'image':os.path.relpath(g['crop_path'],ROOT)})
   output.append({'id':q['sample_id'],'identity':q['player_identity'],'team':q['team_identity'],'match':q['match_id'],'quality':q['quality'],'image':os.path.relpath(q['crop_path'],ROOT),'candidates':items})
   for j in choices:
    if galleries[j]['player_identity']==q['player_identity']:positive.append(float(sims[i,j]))
    elif galleries[j]['team_identity']==q['team_identity']:neg.append(float(sims[i,j]))
  retrieval[name]=output;axes[col].hist(positive,bins=35,alpha=.5,density=True,label='same player');axes[col].hist(neg,bins=35,alpha=.5,density=True,label='same-team negative');axes[col].set_title(name);axes[col].set_xlabel('cosine similarity');axes[col].legend()
  cases=report['generic_m9_body' if col==0 else 'soccer_reid_v1']['same_team_crop']['hard_cases'];tiles=[]
  for k,c in enumerate(cases[:12]):
   a,b,z=[by[c[key]] for key in ['query','wrong','lowest_positive']];panel=np.full((245,540,3),25,np.uint8)
   for i,r in enumerate([a,b,z]):
    im=cv2.imread(r['crop_path']);im=cv2.resize(im,(130,170),interpolation=cv2.INTER_NEAREST);panel[30:200,10+i*180:140+i*180]=im;cv2.putText(panel,['query','WRONG same team','positive'][i],(5+i*180,20),cv2.FONT_HERSHEY_SIMPLEX,.38,(235,235,235),1);cv2.putText(panel,r['player_identity'][-14:],(5+i*180,217),cv2.FONT_HERSHEY_SIMPLEX,.35,(235,235,235),1)
   cv2.putText(panel,f'negative {c["wrong_similarity"]:.3f} / weakest positive {c["positive_similarity"]:.3f}',(5,237),cv2.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1);tiles.append(panel);flags=['same-team appearance'];flags+=['tiny crop'] if min(a['quality']['height'],b['quality']['height'])<45 else [];flags+=['blur proxy'] if min(a['quality']['sharpness'],b['quality']['sharpness'])<50 else [];flags+=['occlusion proxy'] if max(a['quality']['occlusion_proxy'],b['quality']['occlusion_proxy'])>.1 else [];failure_records.append({'model':name,**c,'measured_categories':flags,'manual_categories':'NOT_YET_REVIEWED','view_direction':'unavailable','incorrect_source_label':'not inferred from similarity'})
  cv2.imwrite(str(ROOT/f'failures_{col}.jpg'),np.vstack(tiles))
 fig.tight_layout();fig.savefig(ROOT/'heldout_similarity_distributions.png',dpi=160);plt.close(fig);(ROOT/'hard_negative_analysis.json').write_text(json.dumps(failure_records,indent=2));(ROOT/'retrieval_data.json').write_text(json.dumps(retrieval))
 payload=json.dumps(retrieval).replace('</','<\\/');doc='''<!doctype html><meta charset="utf-8"><title>TactiqAI Soccer-ReID v1 retrieval explorer</title><style>body{font:15px system-ui;background:#10151d;color:#e4ebee;margin:30px}h1{color:#9fd7bd}.models{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.models>section{min-width:0}.candidates{display:flex;gap:10px;overflow-x:auto;max-width:100%}a{color:#9fd7bd}.card{background:#202936;padding:12px;border-radius:8px;min-width:150px}img{height:160px;max-width:130px;object-fit:contain;image-rendering:auto}select{padding:10px;max-width:90vw}.good{color:#9fd7bd}.bad{color:#ffada4}</style><h1>TactiqAI · Soccer-ReID v1</h1><p>Held-out sequence-identity retrieval. Same match galleries; facial region masked. Similarity is evidence, not identity certainty. TEST was not used for model selection.</p><p><strong>Research checkpoint only · not integrated into active M9.</strong> See integration_decision.json for the verified deployment decision.</p><label>Query <select id="query"></select></label><div id="result" class="models"></div><script>const data=PAYLOAD;const keys=Object.keys(data),q=document.getElementById('query');data[keys[0]].forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.id;q.append(o)});function card(r,isQuery=false){return `<div class="card"><img src="${r.image}"><p>${r.id}</p><p>match ${r.match} · team ${r.team}</p><p>quality ${r.quality.score.toFixed(3)} · height ${r.quality.height}px</p>${isQuery?'':`<strong class="${r.same_identity?'good':'bad'}">${r.similarity.toFixed(3)} · ${r.same_identity?'SAME':'DIFFERENT'} GT identity</strong>`}</div>`}function draw(){document.getElementById('result').innerHTML=keys.map(k=>{let r=data[k][Number(q.value)];return `<section><h2>${k}</h2>${card(r,true)}<h3>Top10</h3><div class="candidates">${r.candidates.map(c=>card(c)).join('')}</div></section>`}).join('')}q.onchange=draw;draw();</script>'''.replace('PAYLOAD',payload);(ROOT/'retrieval_explorer.html').write_text(doc)
if __name__=='__main__':run()
