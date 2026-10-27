import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from fastapi.testclient import TestClient
from src.vision.world.pitch import PitchPerception,PitchEvidence,TemporalCamera
from src.vision.world.schema import WorldState
from src.vision.world.provider import canonical_observation
from src.vision.world.review import digest
from src.serving.app import create_app

class PitchTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((400,600,3),np.uint8);self.image[:]=(40,130,40)
        self.evidence=PitchEvidence(status='partial',grass_fraction=.9,reason='test')
        self.points=[([20,20],[0,0]),([580,20],[105,0]),([580,380],[105,68]),([20,380],[0,68]),([300,20],[52.5,0]),([300,380],[52.5,68])]
    def fit(self,camera,frame=0,points=None):
        return camera.update(self.image,frame,frame/25,self.evidence,points or self.points,source='synthetic identified points')
    def test_known_projection_and_canonical_region(self):
        c=self.fit(TemporalCamera());self.assertTrue(c.calibration_valid)
        np.testing.assert_allclose(TemporalCamera.project((300,200),c),(52.5,34),atol=1e-5)
        row=canonical_observation(WorldState(recording_id='test',frame=0,timestamp=0,camera=c,entities=[],unassigned=[],decisions=[],reconciliations=[]),25,'test')
        self.assertEqual(row.visible_pitch_region,c.visible_pitch_region)
    def test_unidentified_lines_cannot_calibrate(self):
        cv2.line(self.image,(20,200),(550,200),(255,255,255),4)
        e=PitchPerception().detect(self.image);self.assertGreater(len(e.lines),0)
        c=TemporalCamera().update(self.image,0,0,e);self.assertFalse(c.calibration_valid)
        self.assertIsNone(TemporalCamera.project((300,200),c))
    def test_loss_never_uses_last_transform_and_relocalizes(self):
        camera=TemporalCamera();self.fit(camera)
        lost=camera.update(self.image,1,.04,self.evidence)
        self.assertFalse(lost.calibration_valid);self.assertIsNotNone(lost.last_reliable_transform)
        self.assertIsNone(TemporalCamera.project((300,200),lost))
        self.assertTrue(self.fit(camera,2).calibration_valid)
    def test_pan_changes_visible_region_fixed_world(self):
        camera=TemporalCamera();a=self.fit(camera)
        # A different identified visible portion uses the same canonical system.
        points=[(p,[10+q[0]*.8,q[1]]) for p,q in self.points]
        b=self.fit(camera,1,points)
        self.assertNotEqual(a.visible_pitch_region,b.visible_pitch_region)
        np.testing.assert_allclose(TemporalCamera.project((300,200),b),(52,34),atol=1e-5)
    def test_hard_cut_discards_current_transform(self):
        camera=TemporalCamera();self.fit(camera)
        red=np.zeros_like(self.image);red[:]=(0,0,255)
        cut=camera.update(red,1,.04,self.evidence)
        self.assertEqual(cut.continuity,'cut');self.assertEqual(cut.epoch,1)
        self.assertFalse(cut.calibration_valid);self.assertIsNone(cut.visible_pitch_region)
    def test_collinear_fit_rejected(self):
        points=[([i*10,20],[i,0]) for i in range(6)]
        self.assertFalse(self.fit(TemporalCamera(),points=points).calibration_valid)
    def test_blank_frame_no_fabricated_lines(self):
        self.assertEqual(PitchPerception().detect(np.zeros_like(self.image)).status,'unavailable')

class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.video=self.root/'source.mp4';self.video.write_bytes(b'0123456789')
        c=TemporalCamera().update(np.zeros((40,40,3),np.uint8),0,0,PitchEvidence(status='unavailable',grass_fraction=0,reason='No evidence'))
        rows=[WorldState(recording_id='cv_test',frame=i,timestamp=i/25,camera=c.model_copy(update={'frame':i,'timestamp':i/25}),entities=[],unassigned=[],decisions=[],reconciliations=[]) for i in range(3)]
        offsets=[]
        with (self.root/'frames.jsonl').open('wb') as f:
            for row in rows:offsets.append(f.tell());f.write((row.model_dump_json()+'\n').encode())
        m=dict(recording_id='cv_test',source_video=str(self.video),video_sha256=digest(self.video),frames_sha256=digest(self.root/'frames.jsonl'),frame_offsets=offsets,frames=3,fps=25,width=40,height=40,transitions=[])
        (self.root/'manifest.json').write_text(json.dumps(m))
        self.client=TestClient(create_app({},cv_registry={'cv_test':self.root}))
    def test_precomputed_cv_startup_and_requests_never_invoke_models(self):
        from unittest.mock import patch
        with patch('src.vision.world.detection.YoloXDetector.__init__',side_effect=AssertionError('Serving must not detect')),patch('src.vision.world.pose.PoseFitter.__init__',side_effect=AssertionError('Serving must not fit camera')),patch('src.vision.world.semantic_pitch.SemanticPitch.__init__',side_effect=AssertionError('Serving must not segment')):
            client=TestClient(create_app({},cv_registry={'cv_test':self.root}))
            self.assertEqual(client.get('/recordings').status_code,200)
            self.assertEqual(client.get('/matches/cv_test/cv/frame').status_code,200)
            self.assertEqual(client.get('/matches/cv_test/video.mp4',headers={'Range':'bytes=0-2'}).status_code,206)

    def test_discovery_capabilities_and_ready(self):
        row=self.client.get('/recordings').json()['recordings'][0]
        self.assertTrue(row['capabilities']['cv_world_model_available']);self.assertFalse(row['capabilities']['ask_available'])
        self.assertFalse(row['artifact_ready']);self.assertEqual(self.client.get('/ready').status_code,200)
        self.assertEqual(self.client.post('/matches/cv_test/ask',json={'question':'Who scored?'}).status_code,404)
    def test_frame_is_preceding_bounded_and_recording_scoped(self):
        for t,index in ((0,0),(.079,1),(.08,2),(50,2)):
            row=self.client.get('/matches/cv_test/cv/frame',params={'timestamp':t})
            self.assertEqual(row.status_code,200);self.assertEqual(row.json()['frame'],index)
        self.assertEqual(self.client.get('/matches/other/cv/frame').status_code,404)
        self.assertEqual(self.client.get('/matches/cv_test/cv/frame?timestamp=-1').status_code,422)
    def test_media_ranges_no_source_path_disclosed(self):
        self.assertNotIn(str(self.video),self.client.get('/matches/cv_test/cv').text)
        r=self.client.get('/matches/cv_test/video.mp4',headers={'Range':'bytes=2-4'})
        self.assertEqual(r.status_code,206);self.assertEqual(r.content,b'234')
    def test_corrupt_bundle_remains_unavailable(self):
        (self.root/'frames.jsonl').write_text('corrupt')
        client=TestClient(create_app({},cv_registry={'cv_test':self.root}))
        self.assertFalse(client.get('/recordings').json()['recordings'][0]['capabilities']['cv_world_model_available'])
        self.assertEqual(client.get('/ready').status_code,503)
    def test_modified_source_rejected_until_revalidation(self):
        self.video.write_bytes(b'changed')
        self.assertEqual(self.client.get('/matches/cv_test/cv/frame').status_code,503)

    def test_shared_recording_has_both_capabilities_and_one_ready_count(self):
        from tests.helpers import write_query_artifacts
        analytics=self.root/'analytics';analytics.mkdir();write_query_artifacts(analytics)
        summary=json.loads((analytics/'summary.json').read_text());summary['fps']=25;(analytics/'summary.json').write_text(json.dumps(summary))
        base=json.loads((self.root/'frames.jsonl').read_text().splitlines()[0]);offsets=[]
        with (self.root/'frames.jsonl').open('wb') as f:
            for i in range(summary['frames']):
                row={**base,'frame':i,'timestamp':i/25,'camera':{**base['camera'],'frame':i,'timestamp':i/25}}
                offsets.append(f.tell());f.write((json.dumps(row)+'\n').encode())
        m=json.loads((self.root/'manifest.json').read_text());m.update(frames=summary['frames'],frame_offsets=offsets,frames_sha256=digest(self.root/'frames.jsonl'));(self.root/'manifest.json').write_text(json.dumps(m))
        client=TestClient(create_app({'cv_test':analytics},cv_registry={'cv_test':self.root}))
        caps=client.get('/recordings').json()['recordings'][0]['capabilities']
        self.assertTrue(caps['analytics_available']);self.assertTrue(caps['cv_world_model_available']);self.assertTrue(caps['ask_available'])
        self.assertEqual(client.get('/ready').json()['recordings'],1)
        self.assertEqual(client.get('/matches/cv_test/cv/frame').status_code,200)
        self.assertEqual(client.get('/matches/cv_test/summary').status_code,200)

    def test_optional_cv_clock_mismatch_keeps_analytics_usable(self):
        from tests.helpers import write_query_artifacts
        analytics=self.root/'analytics';analytics.mkdir();write_query_artifacts(analytics)
        client=TestClient(create_app({'cv_test':analytics},cv_registry={'cv_test':self.root}))
        row=client.get('/recordings').json()['recordings'][0]
        self.assertTrue(row['artifact_ready']);self.assertFalse(row['capabilities']['cv_world_model_available'])
        self.assertEqual(client.get('/ready').status_code,200);self.assertEqual(client.get('/matches/cv_test/summary').status_code,200)

class SemanticGeometryTests(unittest.TestCase):
    def test_four_identified_lines_fit_without_neural_inference(self):
        from src.vision.world.semantic_pitch import semantic_geometry,CLASSES
        mask=np.zeros((256,455),np.uint8)
        for name,a,b in [('Side line top',(20,20),(430,20)),('Side line bottom',(20,235),(430,235)),('Side line left',(20,20),(20,235)),('Side line right',(430,20),(430,235))]:
            cv2.line(mask,a,b,CLASSES.index(name)+1,3)
        e,points=semantic_geometry(mask,np.ones_like(mask,dtype=float),455,256)
        self.assertEqual(len(points),4)
        c=TemporalCamera().update(np.zeros((256,455,3),np.uint8),0,0,e,points,'synthetic semantic labels')
        self.assertTrue(c.calibration_valid)
        np.testing.assert_allclose(TemporalCamera.project((225,127.5),c),(52.5,34),atol=.05)
        self.assertIsNone(c.calibration_confidence)
    def test_curves_and_goal_structure_never_become_planar_lines(self):
        from src.vision.world.semantic_pitch import semantic_geometry,CLASSES
        mask=np.full((256,455),CLASSES.index('Circle central')+1,np.uint8)
        e,points=semantic_geometry(mask,np.ones_like(mask,dtype=float),455,256)
        self.assertEqual(points,[]);self.assertEqual(e.lines,[])
    def test_unsupported_intersections_do_not_create_landmarks(self):
        from src.vision.world.semantic_pitch import semantic_geometry,CLASSES
        mask=np.zeros((256,455),np.uint8)
        cv2.line(mask,(50,20),(50,60),CLASSES.index('Side line left')+1,3)
        cv2.line(mask,(200,200),(400,200),CLASSES.index('Side line bottom')+1,3)
        _,points=semantic_geometry(mask,np.ones_like(mask,dtype=float),455,256)
        self.assertEqual(points,[])

class DegenerateGeometryTests(unittest.TestCase):
    def test_three_collinear_points_do_not_define_a_homography(self):
        e=PitchEvidence(status='partial',grass_fraction=None,reason='semantic hypotheses')
        points=[([20,20],[105,0]),([100,100],[105,13.84]),([200,200],[105,24.84]),([100,200],[99.5,24.84])]
        c=TemporalCamera().update(np.zeros((400,400,3),np.uint8),0,0,e,points,'degenerate synthetic constraints')
        self.assertFalse(c.calibration_valid);self.assertIsNone(c.visible_pitch_region)

class PoseGeometryTests(unittest.TestCase):
    def test_visible_region_clips_a_fixed_pitch_and_pan(self):
        from src.vision.world.pose import visible_region
        a=visible_region(np.array([[.1,0,0],[0,.1,0],[0,0,1.]]),400,300)
        b=visible_region(np.array([[.1,0,20],[0,.1,0],[0,0,1.]]),400,300)
        self.assertIsNotNone(a);self.assertIsNotNone(b)
        self.assertAlmostEqual(min(p[0] for p in a[1]),0,places=4)
        self.assertAlmostEqual(max(p[0] for p in a[1]),40,places=4)
        self.assertAlmostEqual(min(p[0] for p in b[1]),20,places=4)
        self.assertAlmostEqual(max(p[0] for p in b[1]),60,places=4)
    def test_invalid_pose_never_generates_a_region(self):
        from src.vision.world.pose import visible_region
        self.assertIsNone(visible_region(np.zeros((3,3)),400,300))
    def test_insufficient_semantics_cannot_seed_camera(self):
        from src.vision.world.pose import PoseFitter
        self.assertFalse(PoseFitter.sufficient({'Side line top':np.ones((10,2))}))
        self.assertTrue(PoseFitter.sufficient({k:np.ones((10,2)) for k in ['Side line top','Middle line','Circle central']}))

class PoseGateTests(unittest.TestCase):
    def test_high_or_missing_heldout_line_evidence_is_rejected(self):
        from src.vision.world.pose import residual_gate
        self.assertFalse(residual_gate({'lines':None,'circles':1},{'lines':None,'circles':2}))
        self.assertFalse(residual_gate({'lines':11,'circles':1},{'lines':12,'circles':2}))
        self.assertFalse(residual_gate({'lines':5,'circles':1},{'lines':21,'circles':2}))
        self.assertFalse(residual_gate({'lines':float('nan'),'circles':1},{'lines':1,'circles':2}))
    def test_curve_disagreement_cannot_hide_behind_good_lines(self):
        from src.vision.world.pose import residual_gate
        self.assertFalse(residual_gate({'lines':2,'circles':11},{'lines':3,'circles':12}))
        self.assertTrue(residual_gate({'lines':2,'circles':3},{'lines':4,'circles':5}))

class TemporalLocalizationTests(unittest.TestCase):
    def setUp(self):
        from src.vision.world.localization import CameraLocalizer
        from src.vision.world.pose import visible_region
        from src.vision.world.pitch import LineEvidence
        from src.vision.world.schema import CameraState
        self.CameraState=CameraState;self.localizer=CameraLocalizer(1050,680)
        h=np.diag([.1,.1,1]);image,pitch=visible_region(h,1050,680)
        self.fit={'homography':h.tolist(),'image_region':image,'pitch_region':pitch}
        self.e=PitchEvidence(status='partial',grass_fraction=.9,reason='synthetic',lines=[LineEvidence(start=(0,0),end=(1050,0),kind='Side line top',support=1),LineEvidence(start=(0,680),end=(1050,680),kind='Side line bottom',support=1)])
        self.motion={'image_affine':[[1,0,10*320/1050],[0,1,0]],'feature_matches':100,'affine_inliers':90,'affine_median_error_small_px':.1}
    def initial(self):return self.localizer.apply(self.CameraState(frame=0,timestamp=0),self.e,self.fit)
    def test_propagation_is_bounded_and_never_direct(self):
        direct=self.initial();c=self.localizer.apply(self.CameraState(frame=1,timestamp=.04,continuity='continuous',motion_evidence=self.motion),self.e)
        self.assertTrue(c.calibration_valid);self.assertEqual(c.calibration_source_kind,'camera_propagated')
        self.assertEqual(c.calibration.provenance.quality,'propagated');self.assertEqual(c.last_reliable_timestamp,0)
        np.testing.assert_allclose(TemporalCamera.project((520,340),c),(51,34),atol=1e-5)
        expired=self.localizer.apply(self.CameraState(frame=8,timestamp=.32,continuity='continuous',motion_evidence=self.motion),self.e)
        self.assertFalse(expired.calibration_valid);self.assertIsNone(expired.visible_pitch_region)
    def test_missing_landmarks_and_cut_cannot_propagate(self):
        self.initial();empty=PitchEvidence(status='unavailable',grass_fraction=None,reason='Close-up')
        c=self.localizer.apply(self.CameraState(frame=1,timestamp=.04,continuity='continuous',motion_evidence=self.motion),empty)
        self.assertFalse(c.calibration_valid)
        self.initial();cut=self.localizer.apply(self.CameraState(frame=1,timestamp=.04,epoch=1,continuity='cut',motion_evidence=self.motion),self.e)
        self.assertFalse(cut.calibration_valid)
    def test_propagated_positions_preserve_provenance_and_last_reliable(self):
        from src.vision.world.engine import WorldModel
        from src.vision.world.schema import TrackObservation
        world=WorldModel('test');direct=self.initial();track=TrackObservation(bbox=(500,300,20,40),detection_confidence=.9,track_id='t',hits=2,pitch=(51,34),appearance=(1.,0.),team='team_a')
        a=world.update([track],direct);c=self.localizer.apply(self.CameraState(frame=1,timestamp=.04,continuity='continuous',motion_evidence=self.motion),self.e)
        b=world.update([track.model_copy(update={'pitch':(52,34)})],c)
        self.assertEqual(b.entities[0].current_pitch,(52,34));self.assertEqual(b.entities[0].last_reliable_pitch,(51,34));self.assertEqual(b.entities[0].last_localized_at,0)
        row=canonical_observation(b,25,'synthetic');self.assertEqual(row.players[0].localization_kind,'camera_propagated');self.assertEqual(row.players[0].position.quality,'propagated')
        candidate=world.compare(track,b.entities[0],.04,c);self.assertIsNone(candidate.movement_feasible)
        from src.vision.world.provider import analytics_recording
        bridge=analytics_recording([canonical_observation(a,25,'synthetic'),row],{'team_a':'left'},1)
        self.assertTrue(np.isnan(bridge.positions[1]).all())
