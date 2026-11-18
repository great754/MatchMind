"""Engineering/provenance regressions; synthetic trajectories are not ball ground truth."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from pydantic import ValidationError
from src.vision.ball.schema import BallFrame
from src.vision.ball19.tracker import VisualBallTracker, AssociationConfig
from src.vision.ball20.motion import MotionBank, jacobian
from src.vision.ball20.tracker import TrajectoryBallTracker
from src.vision.ball20.detector import YoloCropAdapter, merge_candidates, source_candidate
from src.vision.ball20.schema import TrajectoryFrame

CONFIG = json.loads(Path('configs/vision/cv_m20.json').read_text())
CAMERA = {'epoch': 0, 'continuity': 'continuous', 'calibration_valid': False, 'calibration': None, 'motion_evidence': {}}

def candidate(x=100, y=100, confidence=.9, key='c'):
    return {'candidate_id': key, 'bbox': [x-2, y-2, 4, 4], 'confidence': confidence, 'inference_view': 'global'}

class MotionM20Tests(unittest.TestCase):
    def bank(self): return MotionBank([100, 100], 0, 1000, CONFIG['motion'])
    def test_mixture_probabilities_normalize_and_covariance_stays_positive(self):
        b = self.bank()
        for i in range(1, 20):
            b.predict(i*.04); b.update([100+i*2, 100+.2*i*i], i*.04)
            self.assertAlmostEqual(b.mu.sum(), 1); self.assertTrue((b.mu>=0).all()); self.assertGreater(np.linalg.eigvalsh(b.combined()[1]).min(), 0)
    def test_actual_variable_timestamps_change_prediction_distance(self):
        a = self.bank(); b = self.bank(); a.x[:, 2] = b.x[:, 2] = 100
        a.predict(.1); b.predict(.2); self.assertAlmostEqual(a.combined()[0][0], 110, places=4); self.assertAlmostEqual(b.combined()[0][0], 120, places=4)
    def test_uncertainty_expands_without_new_visual_evidence(self):
        b = self.bank(); initial = b.public()['radius90_px']; b.predict(.1); one = b.public()['radius90_px']; b.predict(.3); two = b.public()['radius90_px']; self.assertLess(initial, one); self.assertLess(one, two); self.assertAlmostEqual(b.public()['age_seconds'], .3)
    def test_camera_pan_moves_image_state_without_inventing_ball_speed(self):
        b = self.bank(); h = np.array([[1,0,20],[0,1,10],[0,0,1]],float); b.predict(.04,h,'PAN')
        np.testing.assert_allclose(b.combined()[0][:2], [120,110]); np.testing.assert_allclose(b.combined()[0][2:4], [0,0]); self.assertEqual(b.camera_status,'PAN')
    def test_zoom_transforms_covariance(self):
        b = self.bank(); h=np.diag([2.,2.,1.]); b.predict(.04,h,'ZOOM'); np.testing.assert_allclose(b.combined()[0][:2],[200,200]); self.assertGreater(b.public()['covariance'][0][0], 4*64)
    def test_bad_homography_is_explicitly_rejected(self):
        b=self.bank(); b.predict(.04,np.diag([100.,100.,1.]),'BAD'); self.assertEqual(b.camera_status,'UNKNOWN_OR_REJECTED_CAMERA_WARP'); self.assertLess(b.combined()[0][0],101)
    def test_upward_motion_does_not_classify_physical_airborne_height(self):
        b=self.bank(); b.x[:,3]=-100; b.predict(.1); self.assertLess(b.public()['mean'][1],100); self.assertNotIn('height',b.public()); self.assertNotIn('airborne',b.public())
    def test_accelerated_and_deflected_paths_remain_finite(self):
        b=self.bank()
        for i in range(1,30):
            time=i*.04; b.predict(time,contact=i in [12,13]); x=100+20*i if i<12 else 340-15*(i-12); b.update([x,100+.4*(i-15)**2],time)
        self.assertTrue(np.isfinite(b.combined()[0]).all()); self.assertAlmostEqual(sum(b.public()['motion_probabilities'].values()),1)
    def test_increasing_clock_required(self):
        for time in [0,-1,float('nan')]:
            with self.assertRaises(ValueError): self.bank().predict(time)
    def test_measurement_must_match_prediction_clock(self):
        with self.assertRaises(ValueError): self.bank().update([100,100], .1)

class TrackerM20Tests(unittest.TestCase):
    def setUp(self): self.image=np.zeros((480,640,3),np.uint8); self.tracker=TrajectoryBallTracker(640,480,copy.deepcopy(CONFIG))
    def step(self, i, cs=None, camera=None, time=None):
        regions,global_needed=self.tracker.prepare(i,i*.04 if time is None else time,camera or CAMERA)
        ball,debug=self.tracker.consume(self.image,cs if cs is not None else [candidate(100+i)])
        return ball,debug,regions
    def test_two_frame_visual_birth_and_distinct_unknown_observed_states(self):
        a,d,_=self.step(0); b,e,_=self.step(1); self.assertEqual(d.state,'UNKNOWN'); self.assertEqual(e.state,'OBSERVED'); self.assertTrue(b.directly_observed); self.assertIsNone(b.pitch_position); self.assertEqual(b.version,'m20')
    def test_crop_plan_is_made_before_current_candidates_and_expands_through_occlusion(self):
        self.step(0); self.step(1); b,d,a=self.step(2,[]); c,e,z=self.step(3,[],time=.25)
        self.assertTrue(a); self.assertGreaterEqual(z[0]['bbox'][2],a[0]['bbox'][2]); self.assertEqual(d.state,'PREDICTED'); self.assertEqual(e.state,'PREDICTED'); self.assertIsNone(b.image_position); self.assertIsNotNone(b.predicted_image_position)
    def test_prediction_does_not_confirm_reacquisition(self):
        self.step(0); self.step(1); a,d,_=self.step(2,[]); self.assertNotIn('REACQUIRED_WITH_FRESH_VISUAL_EVIDENCE',d.events)
        b,e,_=self.step(3); self.assertTrue(b.directly_observed); self.assertIn('REACQUIRED_WITH_FRESH_VISUAL_EVIDENCE',e.events)
    def test_competing_candidates_remain_ambiguous_without_current_point(self):
        self.step(0,[candidate(100,key='a'),candidate(400,key='b')]); b,d,_=self.step(1,[candidate(101,key='a'),candidate(401,key='b')]); self.assertEqual(d.state,'AMBIGUOUS'); self.assertIsNone(b.image_position); self.assertIsNone(b.predicted_image_position); self.assertIsNone(d.selected_candidate_id)
    def test_confidence_threshold_unchanged_even_inside_predicted_region(self):
        self.step(0); self.step(1); b,d,_=self.step(2,[candidate(102,confidence=.34)]); self.assertFalse(b.directly_observed); self.assertEqual(d.candidates[0].reason,'BELOW_FROZEN_VISUAL_THRESHOLD')
    def test_epoch_cut_discards_previous_hypotheses_and_crops(self):
        self.step(0); self.step(1); b,d,regions=self.step(2,[],{**CAMERA,'epoch':1,'continuity':'cut'}); self.assertEqual(regions,[]); self.assertEqual(d.hypotheses,[]); self.assertIn('CAMERA_EPOCH_RESET',d.events); self.assertIsNone(b.predicted_image_position)
    def test_expired_track_has_unknown_position(self):
        self.step(0); self.step(1); b,d,r=self.step(2,[],time=.6); self.assertEqual(d.state,'UNKNOWN'); self.assertEqual(r,[]); self.assertIsNone(b.predicted_image_position)
    def test_new_global_detection_can_seed_unexpected_deflection(self):
        self.step(0); self.step(1); self.step(2,[candidate(500)]); b,d,_=self.step(3,[candidate(501)]); self.assertTrue(any(h['point'][0]>400 for h in self.tracker.core.hypotheses)); self.assertIsNone(b.pitch_position)
    def test_no_silent_frame_dimension_change_or_duplicate_prepare(self):
        self.tracker.prepare(0,0,CAMERA)
        with self.assertRaises(ValueError): self.tracker.prepare(1,.04,CAMERA)
        with self.assertRaises(ValueError): self.tracker.consume(np.zeros((20,20,3),np.uint8),[])
    def test_no_prediction_as_current_visual_coordinate_schema(self):
        self.step(0); self.step(1); b,d,_=self.step(2,[]); record=b.model_dump(); record['image_position']=[100,100]
        with self.assertRaises(ValidationError): BallFrame.model_validate(record)
    def test_observed_debug_cannot_be_forged_without_selected_visual_candidate(self):
        self.step(0); _,d,_=self.step(1); record=d.model_dump(); record['selected_candidate_id']=None
        with self.assertRaises(ValidationError): TrajectoryFrame.model_validate(record)
    def test_appearance_disagreement_cannot_be_resolved_by_proximity_alone(self):
        self.image[96:110,96:110]=255
        self.step(0);self.step(1)
        self.image[96:110,96:110]=[255,0,0]
        ball,debug,_=self.step(2)
        self.assertFalse(ball.directly_observed)
        self.assertTrue(any(c.reason=='INNOVATION_OR_APPEARANCE_GATE' for c in debug.candidates))
    def test_epoch_restarts_do_not_claim_cross_cut_reacquisition(self):
        self.step(0);self.step(1)
        self.step(2,camera={**CAMERA,'epoch':1,'continuity':'cut'})
        ball,debug,_=self.step(3,camera={**CAMERA,'epoch':1})
        self.assertTrue(ball.directly_observed);self.assertNotIn('REACQUIRED_WITH_FRESH_VISUAL_EVIDENCE',debug.events)
    def test_default_m19_does_not_consult_m20_priors(self):
        a=VisualBallTracker(640,480,.35); b=VisualBallTracker(640,480,.35)
        for i in range(10):
            cs=[candidate(100+i)]; one=a.step(i,i*.04,cs,CAMERA); two=b.step(i,i*.04,cs,CAMERA,association_priors=None,association_cost=None); self.assertEqual(one.model_dump(),two.model_dump())

class CropM20Tests(unittest.TestCase):
    def test_non_square_crop_offset_and_internal_resize_coordinates(self):
        c=source_candidate([10,20,14,26],.8,[100,50,160,100],(640,480),'c',960)
        self.assertEqual(c['bbox'],(110.,70.,4.,6.)); self.assertEqual(c['confidence'],.8)
    def test_out_of_crop_padding_clipped_and_huge_objects_rejected(self):
        self.assertIsNone(source_candidate([0,0,160,100],.9,[0,0,160,100],(640,480),'c',640)); self.assertIsNone(source_candidate([200,0,205,5],.9,[0,0,160,100],(640,480),'c',640))
    def test_real_predict_called_with_current_pixels_and_requested_resolution(self):
        class Tensor:
            def __init__(self,a): self.a=np.asarray(a)
            def cpu(self): return self
            def numpy(self): return self.a
        output=SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[10,20,14,26]]),conf=Tensor([.8])))
        model=SimpleNamespace(predict=unittest.mock.Mock(return_value=[output])); adapter=YoloCropAdapter(model,960)
        image=np.zeros((480,640,3),np.uint8); image[50:150,100:260]=123
        cs,t=adapter.infer(image,[{'bbox':[100,50,160,100]}]); self.assertEqual(model.predict.call_count,1); self.assertEqual(model.predict.call_args.kwargs['imgsz'],960); self.assertEqual(model.predict.call_args.args[0][0].shape,(100,160,3)); self.assertTrue((model.predict.call_args.args[0][0]==123).all()); self.assertEqual(cs[0]['bbox'],(110.,70.,4.,6.))
        adapter.infer(image,[{'bbox':[100,50,160,100]}]); self.assertEqual(model.predict.call_count,2)
    def test_no_regions_no_detector_invocation(self):
        model=SimpleNamespace(predict=unittest.mock.Mock()); result,t=YoloCropAdapter(model).infer(np.zeros((20,20,3),np.uint8),[]); self.assertEqual(result,[]); model.predict.assert_not_called()
    def test_crop_outside_actual_source_fails(self):
        with self.assertRaises(ValueError): YoloCropAdapter(None).infer(np.zeros((20,20,3),np.uint8),[{'bbox':[10,10,30,30]}])
    def test_merge_does_not_raise_visual_confidence(self):
        result=merge_candidates([candidate(confidence=.5)],[candidate(101,confidence=.6,key='local')]); self.assertEqual(len(result),1); self.assertEqual(result[0]['confidence'],.6)


class ServingM20Tests(unittest.TestCase):
    def make_ball(self):
        import tempfile
        from scripts.run_cv_m20 import write_ball
        from src.serving.cv import digest
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name)
        tracker=TrajectoryBallTracker(640,480,CONFIG);image=np.zeros((480,640,3),np.uint8);balls=[];debug=[]
        for i in range(6):
            tracker.prepare(i,i*.04,CAMERA);ball,row=tracker.consume(image,[candidate(100+i)] if i<4 else []);balls.append(ball);debug.append(row)
        offsets=[];path=root/'trajectory_frames.jsonl'
        with path.open('wb') as target:
            for row in debug:offsets.append(target.tell());target.write((row.model_dump_json()+'\n').encode())
        source={'recording_id':'r','frames':6,'fps':25.,'width':640,'height':480,'video_sha256':'v'*64,'frames_sha256':'w'*64}
        from src.vision.events.passes import signature
        write_ball(root/'ball_m20','r',source,balls,'d'*64,'m20',{'config':CONFIG,'config_sha256':signature(CONFIG),'trajectory_sha256':digest(path),'trajectory_frame_offsets':offsets})
        return root,source
    def test_valid_sidecar_keeps_observation_prediction_and_debug_synchronized(self):
        from src.serving.ball import BallReview
        root,source=self.make_ball();layer=BallReview(root/'ball_m20',source);self.assertEqual(layer.trajectory.frame(1).state,'OBSERVED');self.assertEqual(layer.trajectory.frame(4).state,'PREDICTED');self.assertFalse(layer.frame(4).directly_observed)
    def test_corrupt_trajectory_is_rejected(self):
        from src.serving.ball import BallReview
        root,source=self.make_ball();(root/'trajectory_frames.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):BallReview(root/'ball_m20',source)
    def test_rehashed_visual_confidence_forgery_is_rejected(self):
        from src.serving.ball import BallReview
        from src.serving.cv import digest
        root,source=self.make_ball();path=root/'trajectory_frames.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()];rows[1]['candidates'][0]['visual_confidence']=.2
        offsets=[]
        with path.open('wb') as target:
            for row in rows:offsets.append(target.tell());target.write((json.dumps(row)+'\n').encode())
        manifest=root/'ball_m20/manifest.json';m=json.loads(manifest.read_text());m.update(trajectory_sha256=digest(path),trajectory_frame_offsets=offsets);manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'confidence'):BallReview(root/'ball_m20',source)
    def test_changed_debug_is_unavailable_after_startup(self):
        from src.serving.ball import BallReview
        from fastapi import HTTPException
        root,source=self.make_ball();layer=BallReview(root/'ball_m20',source);(root/'trajectory_frames.jsonl').write_text('{}\n')
        with self.assertRaises(HTTPException):layer.trajectory.frame(0)
    def test_optional_m20_missing_does_not_create_fallback_or_readiness(self):
        from fastapi.testclient import TestClient
        from src.serving.app import create_app
        client=TestClient(create_app({},cv_registry={}))
        self.assertEqual(client.get('/health').status_code,200);self.assertEqual(client.get('/matches/missing/cv/trajectory').status_code,503);self.assertEqual(client.get('/matches/missing/cv/events?ball_pipeline=m20').status_code,503)
    def test_trajectory_route_obeys_existing_auth(self):
        from fastapi.testclient import TestClient
        from src.serving.app import create_app
        client=TestClient(create_app({},token='unit-secret',cv_registry={}))
        self.assertEqual(client.get('/matches/missing/cv/trajectory').status_code,401)

    def test_corrupt_optional_m20_keeps_valid_baseline_and_ready(self):
        from tests import test_cv_ball_m15 as fixtures
        from fastapi.testclient import TestClient
        from src.serving.app import create_app
        fixture=fixtures.VersionedServingTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        good=fixture.sidecar('m14');bad=fixture.sidecar('m20')
        def registry(path):
            if path.name=='cv_ball.json':return {'cv_test':str(good)}
            if path.name=='cv_ball_m20.json':return {'cv_test':str(bad)}
            return {}
        with patch('src.serving.ball.read_ball_registry',side_effect=registry):client=TestClient(create_app({},cv_registry={'cv_test':fixture.root}))
        self.assertEqual(client.get('/ready').status_code,200)
        self.assertEqual(client.get('/matches/cv_test/cv/frame').status_code,200)
        self.assertEqual(client.get('/matches/cv_test/cv?ball_pipeline=m20').status_code,503)
        self.assertEqual(client.get('/matches/cv_test/cv/trajectory').status_code,503)
        self.assertNotIn('m20',client.get('/matches/cv_test/cv').json()['ball_pipeline_versions'])


class ReferencePreparationM20Tests(unittest.TestCase):
    def test_unlabelled_localization_is_unavailable(self):
        from scripts.evaluate_cv_m20 import localization
        self.assertIsNone(localization([],[],{}, {'ball_box_match_iou':.25})['median_pixel_error'])
    def test_center_error_requires_a_matched_direct_human_reference(self):
        from scripts.evaluate_cv_m20 import localization
        tasks=[{'task_id':'a','kind':'ball','recording_id':'r','frame':0}]
        labels=[{'task_id':'a','state':'visible','bbox':[10,10,10,10]}]
        balls={'r':[{'directly_observed':True,'image_bbox':[12,10,10,10],'image_position':[17,15]}]}
        self.assertEqual(localization(tasks,labels,balls,{'ball_box_match_iou':.25})['median_pixel_error'],2)
        balls['r'][0]['directly_observed']=False
        self.assertIsNone(localization(tasks,labels,balls,{'ball_box_match_iou':.25})['median_pixel_error'])

if __name__=='__main__':unittest.main()
