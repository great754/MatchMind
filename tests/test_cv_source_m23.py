"""New-source review remains unlabelled, isolated and fail-closed."""
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import uuid
import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from scripts.run_cv_source_m23 import analyze_source
from src.serving.cv import CVReview, digest
from src.serving.intelligent23 import IntelligentRegistry
from src.vision.events.passes import signature
from src.vision.intelligent23.inputs import write
from src.vision.world.schema import WorldState, CameraState
from src.vision.world.player_state import PlayerWorldFrame


class NewSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);world=self.root/'world';world.mkdir()
        video=self.root/'new.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(80,64))
        for i in range(3):writer.write(np.full((64,80,3),i*30,np.uint8))
        writer.release()
        offsets=[];worlds=[]
        with (world/'frames.jsonl').open('wb') as stream:
            for i in range(3):
                c=CameraState(frame=i,timestamp=i/25)
                row=WorldState(recording_id='new',frame=i,timestamp=i/25,camera=c,entities=[],unassigned=[],decisions=[],reconciliations=[],
                    player_world=PlayerWorldFrame(frame=i,timestamp=i/25,players=[],unresolved_detections=[]))
                offsets.append(stream.tell());stream.write(row.model_dump_json().encode()+b'\n')
                worlds.append({'frame':i,'timestamp':i/25,'camera':c.model_dump(),'entities':[],'unassigned':[],'player_world':{'players':[]}})
        meta={'recording_id':'new','frames':3,'fps':25,'width':80,'height':64,'source_video':str(video),
            'video_sha256':digest(video),'frames_sha256':digest(world/'frames.jsonl'),'frame_offsets':offsets}
        write(world/'manifest.json',meta)
        config=json.loads(Path('configs/vision/cv_m23.json').read_text())
        config['offline']['primary_anchored_complement_only']=True
        model=self.root/'model';model.write_bytes(b'synthetic fixture only, no neural inference')
        perception={'components':[{'role':'synthetic_test','path':str(model),'sha256':digest(model)}],
            'config':config,'config_sha256':signature(config),'processing_fps':25.,'recorded_processing_seconds':.12,
            'timing':{'extra_crop_views':0},'peak_rss_bytes_macos':1024}
        p=self.root/'perception';p.mkdir();capture=cv2.VideoCapture(str(video))
        with (p/'inputs.jsonl').open('w') as stream:
            for i in range(3):
                ok,image=capture.read();self.assertTrue(ok)
                stream.write(json.dumps({'frame':i,'timestamp':i/25,'source_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                    'primary_candidates':[],'methods':{'full_m23':{'candidates':[],'memory':None}},
                    'causal_camera':worlds[i]['camera'],'m22_context_frames':[i,i,i]})+'\n')
        capture.release()
        self.report=analyze_source(self.root,perception,meta,worlds)
        self.path=self.root/'registry.json';self.refresh_registry()

    def refresh_registry(self):
        write(self.path,{'new':{'root':str(self.root/'analysis'),'report_sha256':digest(self.root/'analysis/report.json')}})
        self.registry=IntelligentRegistry({'new':CVReview(self.root/'world')},self.path)

    def test_new_clip_has_no_fabricated_accuracy_or_passes(self):
        meta=self.registry.metadata('new')
        self.assertEqual(meta['reference_role'],'unlabelled_inference')
        self.assertIsNone(meta['accuracy']);self.assertEqual(meta['comparisons'],[])
        self.assertEqual(meta['metrics']['estimated_passes'],0)
        self.assertEqual(meta['metrics']['engineering']['states'],{'UNKNOWN':3})
        self.assertFalse(self.report['human_labels_read'])
        for i in range(3):
            self.assertEqual(self.registry.reference('new',i).frame,i)
            self.assertFalse(self.registry.reference('new',i).directly_observed)
        self.assertEqual(self.registry.load('new').event_review.events,[])

    def test_new_source_is_ready_queryable_and_locally_grounded(self):
        from src.serving.app import create_app
        with patch.object(IntelligentRegistry,'entries',return_value=self.registry.entries()):
            client=TestClient(create_app({},cv_registry={'new':self.root/'world'}))
            self.assertEqual(client.get('/ready').json(),{'status':'ready','recordings':1})
            row=client.get('/recordings').json()['recordings'][0]
            self.assertEqual(row['recording_id'],'new');self.assertFalse(row['artifact_ready'])
            self.assertTrue(row['capabilities']['cv_world_model_available'])
            for suffix in ['/cv?ball_pipeline=m23','/cv/frame?ball_pipeline=m23',
                '/cv/intelligent','/cv/intelligent/frame','/cv/intelligent/reference']:
                self.assertEqual(client.get('/matches/new'+suffix).status_code,200,suffix)
            self.assertEqual(client.get('/matches/new/video.mp4',headers={'Range':'bytes=0-15'}).status_code,206)
            answer=client.post('/matches/new/assistant',json={'question':'How many passes were detected?',
                'provider':'local','session_id':str(uuid.uuid4()),'cv_pipeline':'m23'})
            self.assertEqual(answer.status_code,200,answer.text)
            self.assertEqual(answer.json()['events'],[])
            self.assertEqual(answer.json()['cv_pipeline'],'m23')

    def test_corrupt_control_is_rejected_without_invalidating_primary(self):
        (self.root/'analysis/new/m21_temporal_offline/manifest.json').write_text('{}')
        with self.assertRaises(HTTPException) as error:self.registry.reference('new',0)
        self.assertEqual(error.exception.status_code,503)
        self.assertTrue(self.registry.available('new'))

    def test_corrupt_primary_never_becomes_available(self):
        with (self.root/'analysis/new/full_m23/ball_frames.jsonl').open('ab') as stream:stream.write(b'corrupt')
        self.assertFalse(self.registry.available('new'))

    def test_unrelated_recording_cannot_be_used_as_the_control(self):
        self.report['reference']['path']=str(self.root/'analysis/other/m21_temporal_offline')
        write(self.root/'analysis/report.json',self.report);self.refresh_registry()
        with self.assertRaises(HTTPException):self.registry.reference('new',0)

    def test_source_mismatch_is_not_reported_as_this_clips_metadata(self):
        self.report['source']['video_sha256']='a'*64
        write(self.root/'analysis/report.json',self.report);self.refresh_registry()
        with self.assertRaises(HTTPException):self.registry.metadata('new')


if __name__=='__main__':unittest.main()
