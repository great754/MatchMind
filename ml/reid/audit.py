"""Audit actual annotations; never read cloud-offloaded multi-GB GSR placeholders."""
import csv,json,os,argparse,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from src.vision.world.player_appearance import digest
MATCHES=['117092','117093','118575','118576','118577','118578','128057','128058','132831','132877']
ROOT=Path(os.environ.get('TACTIQAI_REID_RUN_ROOT','outputs/ml/soccer_reid_m10'))
def audit():
 results={};group={'train':MATCHES[:6],'validation':['128058','132831'],'test':['128057','132877']}
 for n in MATCHES:
  p=ROOT/'annotations';mot=p/'mot'/f'{n}.txt';xml=p/'raw'/n/f'{n}_tracker_box_metadata.xml';tree=ET.parse(xml);roster={int(x.attrib['id']):x.attrib for x in tree.findall('.//players/player')};cache=Path('/tmp/matchmind-azure-startup-cache')/(n+'_clip');bundle=json.loads((cache/'bundle.json').read_text());csvpath=cache/'player_summary.csv'
  if digest(csvpath)!=bundle['files']['player_summary.csv']:raise ValueError('Artifact roster mapping hash mismatch')
  mapped={int(r['mot_id']):int(r['player_id']) for r in csv.DictReader(csvpath.open())};boxes=np.loadtxt(mot,delimiter=',');ids=sorted(set(boxes[:,1].astype(int)));people={}
  for tid in ids:
   pid=mapped.get(tid)
   if pid not in roster:continue
   player=roster[pid];people[str(tid)]={'annotation_identity':f'{n}:MOT:{tid}','roster_player_id':pid,'team_id':player['teamId'],'jersey_metadata':player.get('shirtNumber'),'role':player.get('position'),'mapping_source':'Hash-validated existing artifact player_summary MOT-to-GSR synchronization; inferred, not independent annotation','identity_source':'Organizer MOT ground-truth persistent sequence ID; NOT generated M9 tracker ID'}
  frame_min=int(boxes[:,0].min());frame_max=int(boxes[:,0].max());results[n]={'annotation_sha256':digest(mot),'metadata_sha256':digest(xml),'mapping_sha256':digest(csvpath),'rows':len(boxes),'frame_min':frame_min,'frame_max':frame_max,'frame_base':frame_min,'ids':len(ids),'last_columns_unique':[np.unique(boxes[:,i]).tolist() for i in range(6,10)],'players':people,'annotation_revision':'325bd4c57ee8e135ab652f3921d274526ada496b'}
 # Stronger than match separation: reserve whole roster teams and registered IDs for held-out splits.
 used_teams=set();used_players=set();allowed={};excluded={}
 for split in ['test','validation','train']:
  current_teams=set();current_players=set()
  for n in group[split]:
   good={k:v for k,v in results[n]['players'].items() if v['team_id'] not in used_teams and v['roster_player_id'] not in used_players};allowed[n]=good;excluded[n]=len(results[n]['players'])-len(good);current_teams.update(p['team_id'] for p in good.values());current_players.update(p['roster_player_id'] for p in good.values())
  used_teams.update(current_teams);used_players.update(current_players)
 manifest={'version':'soccer-reid-dataset-v1','seed':20261102,'groups':group,'allowed_players':allowed,'excluded_roster_players':excluded,'annotations':results,'evaluation_video_exclusions':['test_match','team_goal','southampton'],'policy':'Match-level split plus conservative held-out team/registered-player exclusion; test roster metadata used only for leakage prevention, never model selection','identity_scope':'Ground-truth MOT ID within released clip, not a global identity assertion. No cross-match positive pairs. Mapping to GSR metadata is inferred and retained separately.'}
 ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'identity_audit.json').write_text(json.dumps(manifest,indent=2));Path('configs/ml').mkdir(exist_ok=True,parents=True);Path('configs/ml/soccer-reid-v1-split.json').write_text(json.dumps(manifest,indent=2));print({s:{n:len(allowed[n]) for n in ns} for s,ns in group.items()},flush=True)
 return manifest
def download_annotations():
 from huggingface_hub import hf_hub_download,HfApi
 for n in MATCHES:
  for name in [f'raw/{n}/{n}_tracker_box_metadata.xml',f'mot/{n}.txt']:
   hf_hub_download('atomscott/soccertrack-v2',name,repo_type='dataset',revision='325bd4c57ee8e135ab652f3921d274526ada496b',local_dir=ROOT/'annotations')
 metadata=HfApi().get_paths_info('atomscott/soccertrack-v2',[f'mot/clips/{n}.mp4' for n in MATCHES],repo_type='dataset',revision='325bd4c57ee8e135ab652f3921d274526ada496b');(ROOT/'official_video_metadata.json').write_text(json.dumps({Path(x.path).stem:{'sha256':x.lfs.sha256,'size':x.size} for x in metadata},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
 if args.download:download_annotations()
 audit()
