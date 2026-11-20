"""Train the single requested three-frame ResNet-18/FPN CenterNet architecture."""
import argparse,json,os,resource,time,random
from pathlib import Path
import numpy as np
from src.serving.cv import digest
from src.vision.football22.data import NAMES,write


def run(data_root,output,epochs=20,resume=False,pretrained='outputs/m22/pretrained/resnet18-f37072fd.pth',validation_layout='source'):
    if validation_layout not in ['source','temporal_groups']:raise ValueError('Unknown validation layout')
    import torch
    from torch.utils.data import DataLoader
    from src.vision.football22.model import ARCHITECTURE,TemporalFootballNet,detection_loss
    from src.vision.football22.training import TemporalSamples,split_sources,split_temporal_groups,matched_validation,HARD_NEGATIVE_CROP_FRACTION
    torch.set_num_threads(4);torch.manual_seed(20261119);random.seed(20261119);np.random.seed(20261119)
    initial=Path(pretrained)
    if not initial.exists():raise ValueError('Explicitly approved ImageNet checkpoint required; automatic downloads disabled')
    initial_sha=digest(initial)
    if initial_sha!='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':raise ValueError('Official ImageNet ResNet-18 hash mismatch')
    dataset_path=Path(data_root)/'dataset.json';dataset=json.loads(dataset_path.read_text());root=Path(output)
    if root.exists() and not resume:raise ValueError('Fresh training root required; use --resume only for this campaign')
    root.mkdir(parents=True,exist_ok=True)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    config={'architecture':ARCHITECTURE,'maximum_epochs':epochs,'batch':2,'lr_heads':.001,'lr_backbone':.0001,
        'weight_decay':.0005,'seed':20261119,'patience':6,'mixed_precision':False,'training_size':[384,384],
        'validation_size':[384,640],'repeats_per_verified_frame':6,'head_width':64,'temporal_input_filters_trainable':True,
        'strong_source_crop_sizes':[192,256,320,384],'weak_source_crop_sizes':[384,640,960,1080],
        'selection':'maximum source-separated matched-detector validation F1, then precision, then minimum loss',
        'hard_negative_crop_fraction':HARD_NEGATIVE_CROP_FRACTION,
        'hard_negative_crop_loss':'only explicitly human-rejected, non-ball-overlapping regions; surrounding pixels unknown',
        'validation_layout':validation_layout}
    detector_config=json.loads(Path('configs/vision/cv_m22.json').read_text())
    config['validation_detector_config_sha256']=digest('configs/vision/cv_m22.json')
    manifest={'schema_version':'cv-m22-temporal-training-v1','dataset_sha256':digest(dataset_path),
        'imagenet_checkpoint_sha256':initial_sha,'imagenet_checkpoint':str(initial),'config':config,'device':device,
        'reference_role':'development_only','independent_test_used':False,'weak_labels_used':bool(dataset.get('weak_or_pseudo_labels_used')),
        'weak_training':dataset.get('weak_training'),
        'm22_annotation_snapshot_sha256':dataset.get('m22_annotation_snapshot_sha256'),
        'human_frame_count':sum(not r.get('weak_supervision') for r in dataset['rows']),
        'human_visible_frames':sum(r['state']=='visible' for r in dataset['rows']),
        'human_negative_regions':sum(len(r.get('negative_regions',[])) for r in dataset['rows']),
        'training_source_sha256':digest(__file__),'sample_source_sha256':digest('src/vision/football22/training.py'),
        'model_source_sha256':digest('src/vision/football22/model.py'),
        'small_data_warning':'Source-separated development data only; no independent generalization claim.'}
    meta=root/'manifest.json'
    if meta.exists() and json.loads(meta.read_text())!=manifest:raise ValueError('Resume dataset/model/config mismatch')
    write(meta,manifest)
    for excluded in NAMES:
        directory=root/excluded;directory.mkdir(exist_ok=True);result_path=directory/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if digest(result['checkpoint'])!=result['checkpoint_sha256']:raise ValueError('Frozen completed model changed')
            print('M22 verified completed fold',excluded,flush=True);continue
        fold_seed=20261119+list(NAMES).index(excluded)
        torch.manual_seed(fold_seed);random.seed(fold_seed);np.random.seed(fold_seed)
        train_rows,val_rows,split=split_sources(dataset['rows'],excluded) if validation_layout=='source' else split_temporal_groups(dataset['rows'],excluded,dataset['sources'])
        train=TemporalSamples(train_rows,True);val=TemporalSamples(val_rows,False)
        loader=DataLoader(train,batch_size=2,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(20261119))
        validation=DataLoader(val,batch_size=1,num_workers=0)
        model=TemporalFootballNet(initial).to(device);model.freeze_early_backbone()
        backbone=[];heads=[]
        for name,p in model.named_parameters():
            if p.requires_grad:(backbone if name.startswith('backbone.') else heads).append(p)
        optimizer=torch.optim.AdamW([{'params':backbone,'lr':.0001},{'params':heads,'lr':.001}],weight_decay=.0005)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
        start_epoch=0;best=float('inf');best_key=(-1.,0.,-float('inf'));stale=0;history=[];last=directory/'last.pt';best_path=directory/'best.pt'
        if resume and last.exists():
            state=torch.load(last,map_location=device,weights_only=False)
            if state['training_manifest_sha256']!=digest(meta):raise ValueError('Resume checkpoint provenance mismatch')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            start_epoch=state['epoch']+1;best=state['best_validation_loss'];stale=state['stale'];history=state['history']
            best_key=tuple(state['best_validation_key'])
            torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            if 'loader_rng' in state:loader.generator.set_state(state['loader_rng'].cpu())
        started=time.perf_counter()
        for epoch in range(start_epoch,epochs):
            tick=time.perf_counter();train.epoch=epoch;model.train();model.freeze_early_backbone();losses=[]
            for x,target in loader:
                x=x.to(device);target={k:v.to(device) for k,v in target.items()};optimizer.zero_grad(set_to_none=True)
                loss,_=detection_loss(model(x),target)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach().cpu()))
            model.eval();values=[]
            with torch.inference_mode():
                for x,target in validation:
                    loss,_=detection_loss(model(x.to(device)),{k:v.to(device) for k,v in target.items()});values.append(float(loss.cpu()))
            valid=float(np.mean(values));detection=matched_validation(model,val_rows,detector_config,device)
            key=(detection['f1'] or 0.,detection['precision'] or 0.,-valid);improved=key>best_key
            if improved:best=valid;best_key=key;stale=0
            else:stale+=1
            scheduler.step();history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'validation_loss':valid,'validation_detection':detection,'seconds':time.perf_counter()-tick})
            common={'architecture':ARCHITECTURE,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,
                'training_manifest_sha256':digest(meta),'dataset_sha256':manifest['dataset_sha256'],'split':split,
                'best_validation_loss':best,'stale':stale,'history':history,'original_annotations_sha256':dataset['original_annotations_sha256'],
                'weak_training':dataset.get('weak_training'),'best_validation_key':list(best_key),
                'm22_annotation_snapshot_sha256':dataset.get('m22_annotation_snapshot_sha256')}
            if improved:torch.save(common,best_path)
            state={**common,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'loader_rng':loader.generator.get_state()}
            temp=directory/'last.tmp';torch.save(state,temp);os.replace(temp,last)
            write(directory/'history.json',history)
            print('M22 TRAIN',excluded,history[-1],flush=True)
            if stale>=config['patience']:break
        if not best_path.exists():raise ValueError('No finite validated checkpoint')
        write(result_path,{'excluded_recording':excluded,'checkpoint':str(best_path),'checkpoint_sha256':digest(best_path),
            'split':split,'verified_training_frames':sum(not r.get('weak_supervision') for r in train_rows),'verified_validation_frames':len(val_rows),
            'weak_training_frames':sum(bool(r.get('weak_supervision')) for r in train_rows),
            'training_visible_frames':sum(r['state']=='visible' for r in train_rows),'validation_visible_frames':sum(r['state']=='visible' for r in val_rows),
            'epochs_run':len(history),'wall_seconds_this_process':time.perf_counter()-started,
            'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_allocated_bytes':torch.mps.current_allocated_memory() if device=='mps' else None,
            'reference_role':'development_only','architecture':ARCHITECTURE})
        del model,optimizer
        if device=='mps':torch.mps.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--resume',action='store_true');p.add_argument('--pretrained',default='outputs/m22/pretrained/resnet18-f37072fd.pth');p.add_argument('--validation-layout',choices=['source','temporal_groups'],default='source');a=p.parse_args()
    if not 1<=a.epochs<=40:raise ValueError('Local training supports 1–40 bounded epochs')
    run(a.data,a.output,a.epochs,a.resume,a.pretrained,a.validation_layout)
