"""Read-only live checks plus grounded Local Ask; never sends a hosted-LLM payload."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
import uuid


def verify(base, output, recordings=None):
    def request(path, payload=None, video=False):
        headers={'Range':'bytes=0-15'} if video else {}
        if payload is not None:headers['Content-Type']='application/json'
        req=urllib.request.Request(base.rstrip('/')+path,
            data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
        with urllib.request.urlopen(req,timeout=180) as response:
            raw=response.read()
            return response.status, raw if video else json.loads(raw)
    evidence={'verification_started_at_utc':datetime.now(timezone.utc).isoformat(),'backend':base,'provider':'local'}
    for route in ['/health','/ready']:
        status,data=request(route);assert status==200
        evidence[route]={'status_code':status,'body':data}
    _,catalog=request('/recordings');rows=catalog['recordings']
    evidence['recordings']={'total':len(rows),'ready_azure_analytics':sum(r.get('artifact_ready') is True for r in rows),
        'local_cv':sum(r.get('source')=='cv_experimental' for r in rows)}
    evidence['cv']=[]
    for name in recordings or ['test_match','team_goal','southampton']:
        _,meta=request(f'/matches/{name}/cv?ball_pipeline=m23')
        _,report=request(f'/matches/{name}/cv/intelligent')
        navigation=report['metrics']['navigation']
        case=next((n for n in navigation if n['kind']=='improved_reviewed_frame'),navigation[0] if navigation else {'frame':0})
        timestamp=(case['frame']+.001)/meta['fps']
        _,frame=request(f'/matches/{name}/cv/frame?timestamp={timestamp}&ball_pipeline=m23')
        _,debug=request(f'/matches/{name}/cv/intelligent/frame?timestamp={timestamp}')
        _,reference=request(f'/matches/{name}/cv/intelligent/reference?timestamp={timestamp}')
        assert debug['frame']==frame['frame']==reference['frame']==case['frame']
        assert debug['ball_revision']==meta['m23_ball_revision']==report['ball_revision']
        assert debug['model_set_sha256']==meta['m23_model_set_sha256']
        assert reference['ball']['version']==('m23' if report['reference_role']=='unlabelled_inference' else 'm21')
        status,video=request(f'/matches/{name}/video.mp4',video=True)
        assert status==206 and len(video)==16
        _,answer=request(f'/matches/{name}/assistant',{'question':'How many passes were detected?',
            'provider':'local','session_id':str(uuid.uuid4()),'cv_pipeline':'m23'})
        assert answer['cv_pipeline']=='m23' and meta['m23_ball_revision'] in answer['cv_model_revision']
        assert all(e['safe_for_automatic'] for e in answer['events'])
        evidence['cv'].append({'recording_id':name,'metadata_frame_reference_status':200,
            'source_frame':case['frame'],'video_status':status,'video_bytes':len(video),
            'grounded_ask_status':200,'event_count':len(answer['events']),
            'cv_model_revision':answer['cv_model_revision'],'reference_role':report['reference_role'],
            'answer_text':answer['text'],'engineering':report['metrics']['engineering'],
            'new_development_metrics':report['metrics'].get('new_150',{}).get('ball')})
    _,view=request('/matches/118575_clip/view')
    _,answer=request('/matches/118575_clip/ask',{'question':'How many passes were there altogether?','provider':'local'})
    status,video=request('/matches/118575_clip/video.mp4',video=True)
    assert status==206 and len(video)==16
    evidence['azure_118575_clip']={'view_status':200,'events':len(view['events']),
        'ask_status':200,'evidence_items':len(answer['evidence']),'video_status':status,'video_bytes':len(video)}
    evidence['verified_at_utc']=datetime.now(timezone.utc).isoformat()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8000')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--recording',action='append',help='Explicit sources; defaults to the three original M23 clips')
    args=parser.parse_args();verify(args.base,args.output,args.recording)
