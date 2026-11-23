"""M24's consequential invariants: mapping, matching, exclusion and isolation."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.vision.footandball24.model import FootAndBall
from src.vision.footandball24.inference import preprocess,peaks,Detector
from src.vision.footandball24.evaluation import match,score,baseline
from src.serving.footandball24 import create_router,Benchmark
from src.serving.cv import digest


class M24Tests(unittest.TestCase):
    def test_official_architecture_and_strides(self):
        torch.set_num_threads(2);m=FootAndBall().eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()),198840)
        with torch.inference_mode():values=m(torch.zeros(1,3,64,96))
        for key,shape in [('ball',(1,2,16,24)),('player',(1,2,4,6)),('player_bbox',(1,4,4,6))]:
            self.assertEqual(tuple(values[key].shape),shape);self.assertTrue(torch.isfinite(values[key]).all())

    def test_aspect_padding_and_imagenet_normalization(self):
        x,m=preprocess(np.zeros((361,641,3),np.uint8));self.assertEqual(tuple(x.shape),(1,3,384,672))
        self.assertEqual((m['pad_right'],m['pad_bottom']),(31,23));self.assertAlmostEqual(x[0,0,0,0].item(),-.485/.229,places=5)
        self.assertEqual(x[0,:,361:,641:].abs().sum(),0)
        x,m=preprocess(np.zeros((360,640,3),np.uint8),2);self.assertEqual(tuple(x.shape),(1,3,736,1280))

    def test_strict_nms_topk_and_inverse_mapping(self):
        heat=np.zeros((12,12),np.float32);heat[2,2]=.8;heat[7,7]=.9;heat[0,5]=.99
        m={'scale':2,'source_width':20,'source_height':20};c,p=peaks(heat,m)
        self.assertEqual(len(c),2);self.assertEqual(c[0]['center'],[14.75,14.75]);self.assertGreater(c[0]['confidence'],c[1]['confidence'])
        heat[2,3]=.8;self.assertEqual(len(peaks(heat,m)[0]),1)
        self.assertNotIn('bbox',c[0])

    def test_padding_candidate_rejected_not_selected(self):
        heat=np.zeros((10,10),np.float32);heat[8,8]=.9;heat[2,2]=.8
        c,p=peaks(heat,{'scale':1,'source_width':24,'source_height':24})
        self.assertEqual(len(c),1);self.assertEqual(len(p),1);self.assertEqual(c[0]['center'],[9.5,9.5])

    def test_nonfinite_heatmaps_rejected(self):
        with self.assertRaises(ValueError):peaks(np.full((3,3),np.nan),{})

    def test_checkpoint_hash_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.pth';p.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):Detector(p,'cpu')

    def test_one_to_one_maximum_cardinality(self):
        # Greedy nearest would consume p0, leaving r1 unmatched.
        result=match([[0,0],[1.9,0]],[[.9,0],[-1,0]],1.1)
        self.assertEqual(len(result),2);self.assertEqual(len({r[1] for r in result}),2)
        self.assertEqual(len(match([[0,0]],[[0,0],[.1,0]],1)),1)

    def test_unknown_exclusion_wrong_center_and_candidate_recall(self):
        refs=[{'recording_id':'x','frame':0,'state':'visible','bbox':[0,0,2,2]},
              {'recording_id':'x','frame':1,'state':'uncertain','bbox':None},
              {'recording_id':'x','frame':2,'state':'not_visible','bbox':None}]
        bad={'center':[50,50],'confidence':.9};good={'center':[1,1],'confidence':.01}
        rows={'x':[{'selected':bad,'candidates':[bad,good]},{'selected':bad,'candidates':[bad]},{'selected':bad,'candidates':[bad]}]}
        r=score(refs,rows,{'x':{'width':100,'height':100}})
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['uncertain_excluded']),(0,2,1,1));self.assertEqual(r['candidate_recall']['3']['hits'],1)
        self.assertEqual(r['reviewed_frames'],2)

    def test_predicted_tracking_position_never_scored_as_observed(self):
        row={'directly_observed':False,'image_position':None,'image_bbox':None,'detection_confidence':None,'frame':0,'candidates':[], 'predicted_image_position':[1,1]}
        self.assertIsNone(baseline(row)['selected'])

    def test_missing_benchmark_optional_and_read_only(self):
        app=FastAPI();app.include_router(create_router(lambda:None,config='/nonexistent/m24.json'))
        @app.get('/health')
        def health():return {'status':'ok'}
        client=TestClient(app);self.assertEqual(client.get('/health').status_code,200)
        self.assertEqual(client.get('/cv/m24/matches/giroud/metadata').status_code,503)
        self.assertEqual(client.post('/cv/m24/matches/giroud/metadata',json={}).status_code,405)

    def test_bad_report_never_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/'report.json';report.write_text('{}');config=root/'config.json';config.write_text(json.dumps({'report':str(report),'report_sha256':'0'*64}))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):Benchmark(config)

    def test_evidence_changes_fail_closed_after_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'video.mp4';video.write_bytes(b'fixture, not decoded')
            frames=root/'frames.jsonl';frames.write_text('{"frame":0}\n');inputs=root/'inputs.json';inputs.write_text(json.dumps({'sources':{'x':{'video':str(video),'video_sha256':digest(video),'frames':1}}}))
            report=root/'report.json';report.write_text(json.dumps({'inputs_manifest_sha256':digest(inputs),'frame_files':{'x':{'path':str(frames),'sha256':digest(frames),'offsets':[0]}}}))
            config=root/'config.json';config.write_text(json.dumps({'report':str(report),'report_sha256':digest(report),'inputs':str(inputs)}))
            b=Benchmark(config);self.assertEqual(b.frame('x',0)['frame'],0)
            frames.write_text('{"frame":123}\n')
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as e:b.frame('x',0)
            self.assertEqual(e.exception.status_code,503)

    def test_mps_unsupported_operation_falls_back_to_cpu_with_diagnostic(self):
        detector=object.__new__(Detector);detector.device='mps';detector.fallbacks=[]
        class UnsupportedOnce:
            def __init__(self):self.calls=0
            def to(self,device):return self
            def __call__(self,x):
                self.calls+=1
                if self.calls==1:raise RuntimeError('MPS operation not implemented')
                return {'ball':torch.zeros(1,2,16,16),'player':torch.zeros(1,2,4,4),'player_bbox':torch.zeros(1,4,4,4)}
        detector.model=UnsupportedOnce()
        original_to=torch.Tensor.to
        def cpu_tensor(tensor,*args,**kwargs):return tensor if args and args[0]=='mps' else original_to(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,'to',cpu_tensor),patch('torch.mps.synchronize'):
            output=detector.infer(np.zeros((64,64,3),np.uint8))
        self.assertEqual(output['device'],'cpu');self.assertIn('not implemented',detector.fallbacks[0]['reason']);self.assertIsNone(output['selected'])


if __name__=='__main__':unittest.main()
