"""Audit published multi-view triangulation references; never infer ground truth from our predictions."""
import csv,json,requests,ast
from pathlib import Path
from .common import ROOT,save,load,sha

def download():
 out=ROOT/'metric_reference';out.mkdir(exist_ok=True);records=[]
 for a in load('metric_source_inventory.json')['assets']:
  if a['name'] not in ['SNv3D.csv','ISSIA-3D.csv','issia_calibration.json']:continue
  p=out/a['name']
  if not p.exists():
   r=requests.get(a['url'],timeout=60);r.raise_for_status()
   if len(r.content)!=a['bytes']:raise ValueError('Metric annotation size mismatch')
   p.write_bytes(r.content)
  records.append({'name':a['name'],'bytes':p.stat().st_size,'sha256':sha(p),'url':a['url']})
 save('metric_reference_download.json',records)
 for n in ['SNv3D.csv','ISSIA-3D.csv']:
  with (out/n).open() as f:
   rows=list(csv.DictReader(f));print(n,'rows',len(rows),'keys',list(rows[0]),'first',rows[0],flush=True)
 print('calibration',json.loads((out/'issia_calibration.json').read_text())['1'] if '1' in json.loads((out/'issia_calibration.json').read_text()) else (out/'issia_calibration.json').read_text()[:1200])
if __name__=='__main__':download()
