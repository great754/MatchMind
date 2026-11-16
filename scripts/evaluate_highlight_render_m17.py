"""Controlled source-color and tone references for boundary/concat/A-V checks."""
import json,threading,time
from pathlib import Path
import numpy as np,cv2
from src.highlights.render import run,HighlightRenderer,probe
ROOT=Path('outputs/m17_evaluation/render_corpus');ROOT.mkdir(parents=True,exist_ok=True)
def tone(path,at):
 raw=run(['ffmpeg','-nostdin','-v','error','-ss',str(at),'-i',str(path),'-t','0.15','-vn','-ar','48000','-ac','1','-f','f32le','-'],threading.Event(),10)
 a=np.frombuffer(raw,dtype=np.float32);freqs=np.fft.rfftfreq(len(a),1/48000);return float(freqs[np.argmax(abs(np.fft.rfft(a*np.hanning(len(a))))[1:])+1])
def color(path,at):
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,at*1000);ok,image=cap.read();cap.release()
 if not ok:raise ValueError('Missing decoded frame')
 radius=min(image.shape[:2])//10;cy,cx=image.shape[0]//2,image.shape[1]//2
 return image[cy-radius:cy+radius,cx-radius:cx+radius].mean(axis=(0,1)).tolist()
def run_all():
 results=[]
 for fps in [24,25,30]:
  for audio in [False,True]:
   for width,height in [(160,90),(90,160)]:
    name=f'{fps}_{int(audio)}_{width}x{height}';d=ROOT/name;d.mkdir(exist_ok=True);source=d/'source.mp4';cancel=threading.Event();started=time.monotonic()
    command=['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i',f"nullsrc=size={width}x{height}:rate={fps},geq=r='if(lt(T,1),255,0)':g='if(between(T,1,2),255,0)':b='if(gte(T,2),255,0)'" ]
    if audio:command+=['-f','lavfi','-i','aevalsrc=sin(2*PI*(440+220*floor(t))*t):s=48000']
    command+=['-t','3','-c:v','libx264','-pix_fmt','yuv420p','-threads','1']
    if audio:command+=['-c:a','aac']
    command+=[str(source)]
    try:
     run(command,cancel,20);result=HighlightRenderer().render(source,[{'start':.4,'end':1.2},{'start':2,'end':2.8}],d/'out',cancel)
     checks=[]
     for out_time,source_time in [(.1,.5),(.7,1.1),(.9,2.1),(1.5,2.7)]:
      expected=color(source,source_time);actual=color(d/'out/highlight.mp4',out_time);error=max(abs(a-b) for a,b in zip(expected,actual));checks.append({'output_seconds':out_time,'expected_source_seconds':source_time,'max_channel_error':error,'correct':error<10})
     video=next(s for s in result['output_streams'] if s['codec_type']=='video');a=next(s for s in result['output_streams'] if s['codec_type']=='audio');av_start=abs(float(video['start_time'])-float(a['start_time']));av_duration=abs(float(video['duration'])-float(a['duration']));tones=[]
     if audio:
      for output_at,source_at in [(.15,.55),(.72,1.12),(1,2.2)]:
       expected=tone(source,source_at);actual=tone(d/'out/highlight.mp4',output_at);tones.append({'output_at':output_at,'source_at':source_at,'expected_hz':expected,'actual_hz':actual,'correct':abs(expected-actual)<15})
     results.append({'case':name,'success':all(c['correct'] for c in checks) and all(t['correct'] for t in tones) and av_start<.04 and av_duration<.08,'frame_checks':checks,'tone_checks':tones,'av_start_delta_seconds':av_start,'av_duration_delta_seconds':av_duration,'render':result,'wall_seconds':time.monotonic()-started})
    except Exception as e:results.append({'case':name,'success':False,'error':type(e).__name__+':'+str(e)})
    Path('outputs/m17_evaluation/render.json').write_text(json.dumps({'protocol':'12 controlled synthetic media fixtures:3frame rates x2audio modes x2aspect ratios. Original color/tone transitions independently identify segment order and time. Not a99%population reliability claim.','successes':sum(r['success'] for r in results),'total':len(results),'results':results},indent=2)+'\n')
    print('RENDER',name,results[-1]['success'],flush=True)


def extra_cases():
 result=[];root=Path('outputs/m17_evaluation/render_origins');root.mkdir(exist_ok=True);base=ROOT/'25_1_160x90/source.mp4'
 for name,args in [('nonzero_origin',['-c','copy','-output_ts_offset','7']),('rotated',['-c','copy','-metadata:s:v:0','rotate=90']),('rational_fps',['-vf','fps=30000/1001','-c:v','libx264','-threads','1','-c:a','copy'])]:
  c=threading.Event();source=root/(name+'.mp4');run(['ffmpeg','-nostdin','-v','error','-y','-i',str(base),*args,str(source)],c);r=HighlightRenderer().render(source,[{'start':.4,'end':1.2},{'start':2,'end':2.8}],root/name,c);checks=[]
  for output_at,source_at in [(.1,.5),(.7,1.1),(.9,2.1),(1.5,2.7)]:
   error=max(abs(a-b) for a,b in zip(color(source,source_at),color(root/name/'highlight.mp4',output_at)));checks.append(error)
  tones=[abs(tone(source,s)-tone(root/name/'highlight.mp4',o)) for o,s in [(.15,.55),(.72,1.12),(1,2.2)]]
  result.append({'case':name,'success':max(checks)<10 and max(tones)<15,'max_color_error':max(checks),'max_tone_error_hz':max(tones),'render':r});print('EXTRA',name,result[-1]['success'],flush=True)
 Path('outputs/m17_evaluation/render_extra.json').write_text(json.dumps({'results':result,'successes':sum(r['success'] for r in result),'total':len(result)},indent=2)+'\n')
if __name__=='__main__':run_all();extra_cases()
