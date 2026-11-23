"""Read-only live verification after Azure migration and local source deletion."""
import argparse,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from src.vision.intelligent23.inputs import write


def verify(base,output,receipt):
    registry=json.loads(Path('configs/serving/cv_media.json').read_text());removed=json.loads(Path(receipt).read_text())
    if not removed['remote_sha256_reverified_before_deletion'] or {r['recording_id'] for r in removed['deleted']}!=set(registry['recordings']):raise ValueError('Expected verified migration receipt for all four sources')
    for item in removed['deleted']:
        if Path(item['path']).exists():raise ValueError('Persistent local source reappeared')
    def get(path,headers=None):
        req=urllib.request.Request(base.rstrip('/')+path,headers=headers or {})
        with urllib.request.urlopen(req,timeout=90) as response:return response.status,{k.lower():v for k,v in response.headers.items()},response.read()
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'recordings':{},'persistent_video_copies_removed':len(removed['deleted']),'logical_bytes_removed':removed['logical_bytes_removed'],'original_source_hashes_preserved_in_azure':True}
    for path in ['/health','/ready','/recordings']:
        status,_,body=get(path);assert status==200;result[path]=json.loads(body)
    catalog={r['recording_id']:r for r in result['/recordings']['recordings']}
    for name,entry in registry['recordings'].items():
        _,_,raw=get(f'/matches/{name}/cv?ball_pipeline=m23');meta=json.loads(raw)
        assert meta['media_storage']=='azure' and catalog[name]['media_storage']=='azure' and catalog[name]['status']=='available'
        status,headers,raw=get(f'/matches/{name}/video.mp4',{'Range':'bytes=0-15'})
        assert status==206 and len(raw)==16 and headers['etag']==entry['etag']
        _,_,benchmark=get(f'/cv/m24/matches/{name}/metadata');assert json.loads(benchmark)['read_only']
        status,_,jpeg=get(f'/cv/m24/matches/{name}/source.jpg?frame=0');assert status==200 and jpeg.startswith(b'\xff\xd8')
        status,_,heat=get(f'/cv/m24/matches/{name}/heatmap.png?frame=0');assert status==200 and heat.startswith(b'\x89PNG')
        status,headers,video=get(f'/cv/m24/matches/{name}/video.mp4',{'Range':'bytes=0-15'});assert status==206 and video==raw and headers['etag']==entry['etag']
        prefix='/cv/m23' if name=='giroud' else '/cv/m22'
        status,_,jpeg=get(prefix+f'/matches/{name}/review/frame.jpg?frame=0');assert status==200 and jpeg.startswith(b'\xff\xd8')
        if name!='giroud':
            status,_,jpeg=get(f'/matches/{name}/cv/review/frame.jpg?frame=0');assert status==200 and jpeg.startswith(b'\xff\xd8')
        result['recordings'][name]={'media_storage':'azure','blob':entry['blob'],'bytes':entry['bytes'],'sha256':entry['sha256'],'video_range_status':206,'video_etag_matches_registry':True,'m24_metadata_source_heatmap_video_status':200,'review_source_frame_status':200,'frames':meta['frames']}
    for item in removed['deleted']:assert not Path(item['path']).exists()
    result['original_local_source_paths_still_absent_after_requests']=True
    write(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8000');p.add_argument('--output',required=True);p.add_argument('--receipt',default='outputs/m24/azure_media_deletion_v1.json');a=p.parse_args();verify(a.base,a.output,a.receipt)
