"""Restore and verify sealed source-only annotations; never generate labels from predictions."""
import json
from pathlib import Path
from .common import ROOT,save,sha

def run():
 source=Path('configs/ml/ball-m15-demo-annotations.json');target=ROOT/'demo_annotations.json'
 if target.exists() and sha(target)!=sha(source):raise ValueError('Sealed manual annotation mismatch')
 if not target.exists():target.write_bytes(source.read_bytes())
 data=json.loads(target.read_text())
 for row in data['rows']:
  if sha(row['path'])!=row['source_sha256']:raise ValueError('Manual source frame changed')
 print('Verified source-only annotation frames',len(data['rows']),'scorable',sum(r['scorable'] for r in data['rows']))
if __name__=='__main__':run()
