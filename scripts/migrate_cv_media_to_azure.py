"""Create-only, SHA-verified archive of the four explicitly approved CV source clips.

Deletion is a separate opt-in phase, only after the running app has been migrated.
No source manifest, human annotation, model or derived result is modified.
"""
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
from src.serving.cv import digest
from src.vision.intelligent23.inputs import write

NAMES=('giroud','southampton','team_goal','test_match')


def deletion_candidates(data,registry,allowed):
    """Preflight every path and its archive mapping before deleting any file."""
    for name in NAMES:
        expected=data['expected'][name];remote=registry['recordings'][name]
        if any(expected[k]!=remote[k] for k in ['sha256','bytes']):raise ValueError('Inventory and Azure archive differ')
    paths=[]
    for item in data['copies']:
        path=Path(item['path']);remote=registry['recordings'][item['recording_id']]
        if any(item[k]!=remote[k] for k in ['sha256','bytes']):raise ValueError('Local copy is not mapped to its verified archive')
        if not any(path.resolve().is_relative_to(root.resolve()) for root in allowed) or path.is_symlink():raise ValueError('Unapproved deletion path')
        if not path.exists():continue
        stat=path.stat()
        if (stat.st_dev,stat.st_ino,stat.st_size)!=(item['device'],item['inode'],item['bytes']) or digest(path)!=item['sha256']:raise ValueError('Local source changed; preserve it')
        paths.append(item)
    return paths


def inventory():
    expected={name:{'sha256':digest(Path('data/raw')/(name+'.mp4')),'bytes':(Path('data/raw')/(name+'.mp4')).stat().st_size} for name in NAMES}
    by_size={v['bytes']:name for name,v in expected.items()};copies=[];seen=set()
    roots=[Path.cwd(),Path.home()/'Library/Caches/TactiqAI']
    for root in roots:
        for parent,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ['.git','node_modules','backend-m17','node_modules.cloud-backup']]
            for filename in files:
                if Path(filename).suffix.lower() not in ['.mp4','.mov','.mkv','.m4v','.webm','.avi']:continue
                path=Path(parent)/filename
                if path.is_symlink() or str(path) in seen:continue
                stat=path.stat();name=by_size.get(stat.st_size)
                if name and digest(path)==expected[name]['sha256']:
                    seen.add(str(path));copies.append({'path':str(path),'recording_id':name,'sha256':expected[name]['sha256'],'bytes':stat.st_size,'device':stat.st_dev,'inode':stat.st_ino,'allocated_bytes':stat.st_blocks*512})
    return {'created_at_utc':datetime.now(timezone.utc).isoformat(),'expected':expected,'copies':copies}


def verified_bytes(blob,entry):
    from azure.core import MatchConditions
    props=blob.get_blob_properties()
    if props.size!=entry['bytes']:raise ValueError('Azure source size mismatch')
    h=hashlib.sha256();total=0
    for chunk in blob.download_blob(etag=props.etag,match_condition=MatchConditions.IfNotModified).chunks():h.update(chunk);total+=len(chunk)
    if total!=entry['bytes'] or h.hexdigest()!=entry['sha256']:raise ValueError('Azure source full-content SHA mismatch')
    return props


def run(args):
    from dotenv import load_dotenv
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient,ContentSettings
    from azure.core.exceptions import ResourceExistsError
    load_dotenv(override=False)
    account=os.environ['MATCHMIND_BLOB_ACCOUNT_URL'];container=os.environ['MATCHMIND_MEDIA_BLOB_CONTAINER']
    client=ContainerClient(account,container,credential=DefaultAzureCredential(),max_single_get_size=1024*1024,max_chunk_get_size=1024*1024)
    inventory_path=Path(args.inventory)
    if args.delete_verified_copies:
        data=json.loads(inventory_path.read_text());registry=json.loads(Path(args.registry).read_text())
        if registry['account_url']!=account or registry['container']!=container:raise ValueError('Azure destination changed')
        # Re-read and hash every remote source before deleting any local path.
        for name in NAMES:
            entry=registry['recordings'][name];props=verified_bytes(client.get_blob_client(entry['blob']),entry)
            if props.etag!=entry['etag']:raise ValueError('Pinned Azure revision changed')
        deleted=[]
        allowed=[Path.cwd().resolve(),(Path.home()/'Library/Caches/TactiqAI').resolve()]
        if args.include_downloads:allowed.append((Path.home()/'Downloads').resolve())
        candidates=deletion_candidates(data,registry,allowed)
        for item in candidates:
            path=Path(item['path'])
            if path.exists():path.unlink();deleted.append(item)
        write(args.receipt,{'verified_at_utc':datetime.now(timezone.utc).isoformat(),'account_url':account,'container':container,'registry':args.registry,'deleted':deleted,'logical_bytes_removed':sum(x['bytes'] for x in deleted),'remote_sha256_reverified_before_deletion':True})
        print('Deleted',len(deleted),'verified local copies; Azure originals preserved.',flush=True)
        return
    if inventory_path.exists():raise ValueError('Preserve existing migration inventory; use a new path')
    data=inventory();write(inventory_path,data);entries={}
    for name in NAMES:
        entry=data['expected'][name];blob_name=f'cv/clips/{name}/{entry["sha256"]}.mp4';blob=client.get_blob_client(blob_name)
        try:
            with (Path('data/raw')/(name+'.mp4')).open('rb') as stream:
                blob.upload_blob(stream,overwrite=False,content_settings=ContentSettings(content_type='video/mp4'),metadata={'sha256':entry['sha256'],'recording_id':name})
            action='uploaded'
        except ResourceExistsError:action='existing'
        props=verified_bytes(blob,entry)
        if props.metadata.get('sha256')!=entry['sha256'] or props.metadata.get('recording_id')!=name:raise ValueError('Azure source metadata mismatch')
        entries[name]={'blob':blob_name,**entry,'etag':props.etag};print(name,action,'full SHA-256 verified',entry['bytes'],'bytes',flush=True)
    write(args.registry,{'schema_version':'cv-azure-media-v1','account_url':account,'container':container,'recordings':entries})
    print('Archive verified; no local deletion yet. Inventory:',len(data['copies']),'copies.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',default='outputs/m24/azure_media_inventory_v1.json');p.add_argument('--registry',default='configs/serving/cv_media.json');p.add_argument('--receipt',default='outputs/m24/azure_media_deletion_v1.json');p.add_argument('--delete-verified-copies',action='store_true');p.add_argument('--include-downloads',action='store_true',help='Also permit explicitly inventoried, hash-matching copies in Downloads');run(p.parse_args())
