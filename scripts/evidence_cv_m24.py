"""Render a deterministic small set of real source-linked benchmark examples."""
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.vision.intelligent23.inputs import write


def run(inputs,evaluation,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);m=json.loads((Path(inputs)/'manifest.json').read_text());r=json.loads((Path(evaluation)/'report.json').read_text());selected=[]
    for name in m['sources']:
        rows=[json.loads(x) for x in (Path(evaluation)/(name+'.jsonl')).read_text().splitlines()]
        for label,predicate in [('correct',lambda x: any(t['status']['footandball']['correct'] for t in x['references'].values())),
            ('miss',lambda x:any(t['state']=='visible' and t['status']['footandball']['missed'] for t in x['references'].values())),
            ('m23_false',lambda x:any(t['status']['m23']['false'] for t in x['references'].values())),
            ('ranked_confusion',lambda x:any(t.get('candidate_availability',{}).get('nearby_best',{}).get('rank',0)>1 for t in x['references'].values() if t.get('candidate_availability',{}).get('nearby_best')) )]:
            case=next((x for x in rows if predicate(x)),None)
            if case is not None and (name,case['frame']) not in [(c['recording_id'],c['frame']) for c in selected]:selected.append({'recording_id':name,'category':label,**case})
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14);captions=[]
    for i,row in enumerate(selected):
        name=row['recording_id'];source=m['sources'][name];frame=row['frame'];cap=cv2.VideoCapture(source['video'])
        try:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        finally:cap.release()
        if not ok:raise ValueError('Example source missing')
        ref=next((t for t in row['references'].values() if t['state']=='visible'),None);canvas=Image.new('RGB',(1000,410),'#15202b');draw=ImageDraw.Draw(canvas)
        scale=min(640/source['width'],360/source['height']);w=round(source['width']*scale);h=round(source['height']*scale);preview=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).resize((w,h));canvas.paste(preview,(0,40))
        f=row['footandball'];heat=np.load(f['heatmap_file'])['ball'];p=f['candidates'][0] if f['candidates'] else None
        for color,center in [('#ffbb22',p['center'] if p else None),('#ee77ff',row['m23']['selected']['center'] if row['m23']['selected'] else None)]:
            if center:x,y=center[0]*scale,center[1]*scale+40;draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=2)
        if ref:
            x,y,bw,bh=ref['bbox'];draw.rectangle((x*scale,y*scale+40,(x+bw)*scale,(y+bh)*scale+40),outline='#44ff88',width=2)
            cx=x+bw/2;cy=y+bh/2;half=24;left=max(0,int(cx)-half);top=max(0,int(cy)-half);crop=image[top:min(source['height'],top+48),left:min(source['width'],left+48)];canvas.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((192,192),Image.Resampling.NEAREST),(650,55));draw.text((650,250),'Human-ball crop 4x',font=font,fill='white')
        color=cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_INFERNO);canvas.paste(Image.fromarray(cv2.cvtColor(color,cv2.COLOR_BGR2RGB)).resize((330,110)),(650,280));draw.text((650,392),'Heatmap absolute [0,1]',font=font,fill='white')
        draw.text((8,10),f'{name} frame {frame} | {row["category"]} | rank1 {p["confidence"]:.4g}' if p else f'{name} frame {frame} | no peaks',font=font,fill='white')
        path=root/f'{name}_{frame:04d}.jpg';canvas.save(path,quality=90);captions.append({'file':str(path),'recording_id':name,'frame':frame,'timestamp':frame/source['fps'],'category':row['category'],'source_video_sha256':source['video_sha256'],'browser_url':f'http://127.0.0.1:5173/?benchmark=m24&recording={name}', 'visual_cause':'uncertain_pending_actual_inspection'})
    write(root/'examples.json',captions);print(json.dumps(captions,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.inputs,a.evaluation,a.output)
