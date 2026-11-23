"""Evaluate the frozen M24 experiment without changing labels or any detector."""
import argparse,json,math
from pathlib import Path
import numpy as np
from src.serving.cv import digest
from src.vision.intelligent23.inputs import write
from src.vision.footandball24.evaluation import baseline,center,match,score
from src.vision.footandball24.inference import peaks


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def run(inputs,perception,output,m22=None):
    inputs=Path(inputs);perception=Path(perception);root=Path(output)
    if root.exists():raise ValueError('Preserve scored experiments; use a new output directory')
    root.mkdir(parents=True);manifest=json.loads((inputs/'manifest.json').read_text());runtime=json.loads((perception/'report.json').read_text());sources=manifest['sources']
    if runtime['inputs_manifest_sha256']!=digest(inputs/'manifest.json'):raise ValueError('Perception source snapshot mismatch')
    models={'footandball':{},'m21':{},'m23':{},'m22':{}};upscale={};offsets={}
    for name,source in sources.items():
        path=perception/name/'native/frames.jsonl'
        if digest(path)!=runtime['clips'][name+'/native']['frames_sha256']:raise ValueError('Perception output changed')
        models['footandball'][name]=load(path)
        for model in ['m21','m23','m22']:
            path=inputs/f'baselines/{name}/{model}.jsonl'
            if path.exists():
                if digest(path)!=manifest['files'][f'baselines/{name}/{model}.jsonl']['sha256']:raise ValueError('Frozen baseline changed')
                models[model][name]=[baseline(r) for r in load(path)]
        for model in models:
            if name in models[model] and (len(models[model][name])!=source['frames'] or any(r['frame']!=i for i,r in enumerate(models[model][name]))):raise ValueError('Comparison clock mismatch')
    if m22:
        reference=Path(m22);r=json.loads((reference/'report.json').read_text())
        if digest(reference/'giroud.jsonl')!=r['output_sha256'] or r['source_sha256']!=sources['giroud']['video_sha256']:raise ValueError('M22 reference mismatch')
        models['m22']['giroud']=load(reference/'giroud.jsonl')
    up_path=perception/'giroud/upscale2/frames.jsonl'
    if digest(up_path)!=runtime['clips']['giroud/upscale2']['frames_sha256']:raise ValueError('Resolution diagnostic changed')
    upscale['giroud']=load(up_path)
    if len(upscale['giroud'])!=sources['giroud']['frames'] or ('giroud' in models['m22'] and len(models['m22']['giroud'])!=sources['giroud']['frames']):raise ValueError('Secondary comparison frame count mismatch')
    result={'schema_version':'cv-m24-evaluation-v1','inputs_manifest_sha256':digest(inputs/'manifest.json'),
        'protocol_sha256':manifest['protocol_sha256'],'perception_report_sha256':digest(perception/'report.json'),
        'center_tolerance_fraction':.01,'groups':{},'resolution_diagnostic':{},'sizes':{},'diagnostic_counts':{},
        'uncertainty':'Wilson95 on precision/recall only; small correlated development samples, one reviewer, assisted proposals. No independent final test.',
        'm22_giroud_reference':json.loads((Path(m22)/'report.json').read_text()) if m22 else None,'frame_files':{}}
    diagnostics=[];group_frames={}
    for group,refs in manifest['references'].items():
        result['groups'][group]={};group_frames[group]={}
        for model,rows in models.items():
            if any(r['recording_id'] not in rows for r in refs):continue
            result['groups'][group][model]={'primary':score(refs,rows,sources),
                'sensitivity':{str(f):score(refs,rows,sources,fraction=f) for f in [.005,.015]},
                'native_iou':score(refs,rows,sources,native_iou=True) if model!='footandball' else None,
                'per_recording':{name:score([r for r in refs if r['recording_id']==name],rows,sources) for name in sorted({r['recording_id'] for r in refs})}}
        result['groups'][group]['footandball']['official_all_peaks_secondary']=score(refs,models['footandball'],sources,multi=True)
        for ref in refs:
            name=ref['recording_id'];frame=ref['frame'];source=sources[name];tol=math.hypot(source['width'],source['height'])*.01
            truth=[center(ref['bbox'])] if ref['state']=='visible' else [];status={}
            for model in ['footandball','m23']:
                prediction=models[model][name][frame]['selected'];correct=bool(match(truth,[prediction['center']] if prediction else [],tol))
                status[model]={'correct':correct,'missed':bool(truth and not correct),'false':bool(ref['state']!='uncertain' and prediction and not correct),'prediction':prediction}
            row={'group':group,**ref,'tolerance_pixels':tol,'status':status,'visual_cause':'uncertain_not_visually_classified'}
            group_frames[group][(name,frame)]=row
            if ref['state']=='visible':
                box=ref['bbox'];row['size']={'width_pixels':box[2],'height_pixels':box[3],'sqrt_area_pixels':math.sqrt(box[2]*box[3]),'relative_sqrt_area_diagonal':math.sqrt(box[2]*box[3])/math.hypot(source['width'],source['height'])}
                if not status['footandball']['correct']:
                    f=models['footandball'][name][frame];heatpath=Path(f['heatmap_file'])
                    if digest(heatpath)!=f['heatmap_sha256']:raise ValueError('Heatmap changed')
                    heat=np.load(heatpath)['ball'];allpeaks,padding=peaks(heat,f['mapping'],None)
                    nearby=[c for c in allpeaks if match(truth,[c['center']],tol)];best=nearby[0] if nearby else None
                    yy,xx=np.indices(heat.shape);distance=np.hypot((xx*4+1.5)/f['mapping']['scale']-truth[0][0],(yy*4+1.5)/f['mapping']['scale']-truth[0][1]);local=heat[distance<=tol]
                    category=('higher_ranked_object' if best and best['confidence']>=.7 else 'weak_nearby_candidate' if best else 'no_strict_peak_near_reference')
                    if best and best['rank']>100:category='top100_postprocessing_cap'
                    row['candidate_availability']={'category':category,'nearby_best':best,'heat_max_within_tolerance':float(local.max()) if local.size else None,
                        'nearby_peak_count':len(nearby),'global_peak_count':len(allpeaks),'padding_peaks_rejected':len(padding),
                        'postprocessing_note':'Strict local maxima only; a positive response without a peak is not a detection. No-response means no strict candidate, not zero activation.'}
                    diagnostics.append(row)
        bins={}
        for label,low,high in [('le6',0,6),('6to10',6,10),('gt10',10,float('inf'))]:
            subset=[r for r in refs if r['state']=='visible' and low<math.sqrt(r['bbox'][2]*r['bbox'][3])<=high]
            bins[label]={model:score(subset,rows,sources) for model,rows in models.items() if all(r['recording_id'] in rows for r in subset)}
        result['sizes'][group]=bins
    result['resolution_diagnostic']={'giroud_upscale2':score(manifest['references']['giroud_completed'],upscale,sources)}
    for name in sources:
        path=root/f'{name}.jsonl';idx=[]
        with path.open('wb') as stream:
            for frame in range(sources[name]['frames']):
                idx.append(stream.tell());f=models['footandball'][name][frame]
                record={'frame':frame,'timestamp':frame/sources[name]['fps'],'footandball':f,'m23':models['m23'][name][frame],
                    'references':{g:rows[(name,frame)] for g,rows in group_frames.items() if (name,frame) in rows}}
                stream.write(json.dumps(record,separators=(',',':'),allow_nan=False).encode()+b'\n')
        result['frame_files'][name]={'path':str(path),'sha256':digest(path),'offsets':idx}
    for group in group_frames:
        values=[r for r in diagnostics if r['group']==group];result['diagnostic_counts'][group]={k:sum(r['candidate_availability']['category']==k for r in values) for k in sorted({r['candidate_availability']['category'] for r in values})}
    write(root/'missed_visible_diagnostics.json',diagnostics);result['missed_diagnostics_sha256']=digest(root/'missed_visible_diagnostics.json')
    write(root/'report.json',result)
    for group,systems in result['groups'].items():
        print(group,{m:{k:v['primary'][k] for k in ['tp','fp','fn','precision','recall','f1']} for m,v in systems.items()},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--perception',required=True);p.add_argument('--output',required=True);p.add_argument('--m22');a=p.parse_args();run(a.inputs,a.perception,a.output,a.m22)
