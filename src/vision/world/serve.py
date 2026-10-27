"""Read-only local debug viewer with video byte ranges; no Azure or source copy."""
import argparse,hashlib,json,re
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse,FileResponse

ASSETS={'diagnostics.json','world.jsonl','observations.jsonl','transitions.jsonl','associations.jsonl','reconciliations.jsonl','frame_diagnostics.json','annotated.mp4','environment.txt','validation.json'}

def create_debug_app(output):
    output=Path(output).resolve();metadata=json.loads((output/'diagnostics.json').read_text());video=Path(metadata['input']['path']).resolve()
    with video.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=metadata['input']['sha256']:raise ValueError('Source video changed since world experiment')
    app=FastAPI(title='TactiqAI local world-model debug',docs_url=None,redoc_url=None,openapi_url=None)
    @app.get('/')
    def index():
        page=(output/'debug.html').read_text()
        return HTMLResponse(re.sub(r'(<video id="video" src=")[^"]*(")',r'\1/video.mp4\2',page,count=1))
    @app.get('/video.mp4')
    def source_video():return FileResponse(video,media_type='video/mp4')
    @app.get('/{asset}')
    def artifact(asset:str):
        if asset not in ASSETS:raise HTTPException(404)
        path=output/asset
        if not path.is_file() or path.is_symlink():raise HTTPException(404)
        return FileResponse(path)
    return app

def main():
    import uvicorn
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--port',type=int,default=18765);a=p.parse_args()
    uvicorn.run(create_debug_app(a.output),host='127.0.0.1',port=a.port,access_log=False)
if __name__=='__main__':main()
