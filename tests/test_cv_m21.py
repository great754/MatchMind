"""Engineering regressions. Synthetic sequences are not independent football accuracy labels."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np

from src.vision.advanced21.camera import causal_camera, refine_motion
from src.vision.advanced21.detection import DetectorBackend, tile_regions, past_frame_fusion, merge_candidates
from src.vision.advanced21.offline import viterbi, dag_link, reconstruct
from src.vision.advanced21.temporal import visual_features, batch_visual_features, deterministic_verdict, fit_linear, linear_score, FEATURE_NAMES
from src.vision.advanced21.players import PlayerTracker
from src.vision.advanced21.physics import fit_ballistic, image_quadratic
from src.vision.advanced21.passes import verify_events
from src.vision.advanced21.tracking import offline_rows

def candidate(x=10, confidence=.9, key='a'):
    return {'candidate_id': key, 'bbox': [x, 20, 8, 8], 'confidence': confidence, 'inference_view': 'current', 'source': 'VIDEO_CV_DETECTOR'}

def camera(epoch=0, cut=False):
    return {'epoch': epoch, 'continuity': 'cut' if cut else 'continuous', 'calibration_valid': False, 'calibration': None, 'motion_evidence': {}}


class DetectionM21Tests(unittest.TestCase):
    def test_dense_regions_cover_source_and_stay_in_bounds(self):
        for width, height in [(640, 360), (1080, 1080), (1280, 720), (31, 17)]:
            covered = np.zeros((height, width), bool)
            for x, y, w, h in tile_regions(width, height, 192):
                self.assertGreaterEqual(min(x, y), 0)
                self.assertLessEqual(x+w, width); self.assertLessEqual(y+h, height)
                covered[y:y+h, x:x+w] = True
            self.assertTrue(covered.all())
    def test_invalid_tiles_rejected(self):
        for args in [(0, 10, 5, .2), (10, 10, 5, 1), (10, 10, 0, 0)]:
            with self.assertRaises(ValueError): tile_regions(*args)
    def test_fusion_keeps_current_when_history_is_unsupported(self):
        image = np.full((12, 15, 3), 123, np.uint8)
        fused, count = past_frame_fusion(image, [np.zeros_like(image)], [None])
        np.testing.assert_array_equal(fused, image); self.assertEqual(count, 0)
    def test_past_warp_maps_into_current_source_coordinates(self):
        current = np.zeros((20, 20, 3), np.uint8); previous = current.copy(); previous[10, 5] = 255
        fused, count = past_frame_fusion(current, [previous], [np.array([[1,0,3],[0,1,0],[0,0,1]])])
        self.assertGreater(fused[10, 8].sum(), 0); self.assertEqual(count, 1)
    def test_merge_never_inflates_neural_confidence(self):
        rows = merge_candidates([candidate(confidence=.6)], [candidate(confidence=.8)])
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['confidence'], .8)
    def test_crop_detections_map_back_without_resizing_twice(self):
        import torch
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[2.,3.,10.,11.]]), conf=torch.tensor([.7])))]
        backend = DetectorBackend(model, 'test', batch_size=1)
        rows, _ = backend.infer(np.zeros((200, 300, 3), np.uint8), [(100, 50, 80, 80)])
        self.assertEqual(rows[0]['bbox'], [102., 53., 8., 8.])
        self.assertAlmostEqual(rows[0]['confidence'], .7, places=6)
        with self.assertRaises(ValueError): backend.infer(np.zeros((20,20,3), np.uint8), [(10,10,30,30)])


class TemporalM21Tests(unittest.TestCase):
    def test_textureless_history_abstains_without_nan(self):
        image = np.zeros((60, 60, 3), np.uint8)
        features, _ = visual_features(image, image, candidate())
        self.assertIsNone(features['patch_correlation']); json.dumps(features, allow_nan=False)
        self.assertTrue(deterministic_verdict(features, .9)['accepted'])
    def test_batch_features_and_missing_history_are_finite(self):
        image = np.random.default_rng(3).integers(0, 255, (80, 80, 3), dtype=np.uint8)
        features = batch_visual_features(image, image, [candidate()], np.eye(3))
        self.assertTrue(features[0]['history_available'])
        self.assertLess(features[0]['forward_backward_error_px'], .1)
        self.assertFalse(batch_visual_features(None, image, [candidate()])[0]['history_available'])
    def test_classifier_requires_both_training_classes(self):
        self.assertFalse(fit_linear([[0]*len(FEATURE_NAMES)], [1])['available'])
    def test_small_classifier_has_finite_scores_and_preserves_schema(self):
        x = [[float(i)]*len(FEATURE_NAMES) for i in range(12)]
        model = fit_linear(x, [0]*6+[1]*6)
        self.assertTrue(model['available']); self.assertFalse(model['score_calibrated'])
        features, _ = visual_features(None, np.zeros((80,80,3), np.uint8), candidate())
        self.assertTrue(0 <= linear_score(model, features, .7) <= 1)
        model['feature_names'] = ['wrong']
        with self.assertRaises(ValueError): linear_score(model, features, .7)


class OfflineM21Tests(unittest.TestCase):
    def test_both_linkers_use_observations_and_reset_at_cuts(self):
        sets = [[candidate(10+i)] for i in range(8)]
        cameras = [camera() for _ in range(4)]+[camera(1, cut=True)]+[camera(1) for _ in range(3)]
        for linker in [viterbi, dag_link]:
            selected = linker(sets, cameras, 25, 200)
            self.assertTrue(all(c is None or c in sets[i] for i,c in enumerate(selected)))
            self.assertGreater(sum(c is not None for c in selected), 4)
    def test_unsupported_low_score_and_lone_candidates_never_become_observed(self):
        for linker in [viterbi, dag_link]:
            self.assertEqual(linker([[candidate(confidence=.1)]]*4, [camera()]*4, 25, 200), [None]*4)
            self.assertEqual(linker([[candidate()]], [camera()], 25, 200), [None])
    def test_reconstruction_is_future_inference_and_never_crosses_cut(self):
        selected = [candidate(10), None, candidate(14)]
        rows = reconstruct(selected, [camera()]*3, 25)
        self.assertFalse(rows[1]['directly_observed']); self.assertTrue(rows[1]['uses_future'])
        self.assertIsNone(reconstruct(selected, [camera(),camera(1,True),camera(1)], 25)[1])
    def test_reconstruction_cannot_become_ball_image_or_metric_evidence(self):
        selected = [candidate(10), None, candidate(14)]
        cameras = [camera()]*3
        metadata = {'fps':25,'width':100,'height':100}
        rows = offline_rows(selected, reconstruct(selected,cameras,25), [[candidate(10)],[],[candidate(14)]], [{'camera':c} for c in cameras], metadata)
        self.assertIsNone(rows[1].image_position); self.assertIsNone(rows[1].pitch_position)
        self.assertIsNotNone(rows[1].predicted_image_position)
        self.assertTrue(rows[0].directly_observed)


class CameraPhysicsM21Tests(unittest.TestCase):
    def test_causal_camera_strips_offline_absolute_calibration(self):
        c = camera(); c.update(calibration_valid=True,calibration={'image_to_pitch':np.eye(3).tolist()})
        c['motion_evidence']['temporal_camera'] = {'uses_future':True}
        output = causal_camera(c)
        self.assertFalse(output['calibration_valid']); self.assertIsNone(output['calibration'])
        self.assertNotIn('temporal_camera',output['motion_evidence'])
    def test_cut_prevents_relative_flow_refinement(self):
        image = np.zeros((50,50,3),np.uint8)
        self.assertFalse(refine_motion(image,image,camera(),camera(1,True))['accepted'])
    def test_plane_homography_is_insufficient_for_ball_height(self):
        result = fit_ballistic([0,1],[[2,3],[4,5]],[np.eye(3)]*2,calibration_verified=False)
        self.assertFalse(result['accepted']); self.assertFalse(result['eligible_for_pass_gates'])
        self.assertIsNone(result['authoritative_ball_height'])
    def test_physical_fit_on_known_synthetic_projection_and_impossible_fit_rejection(self):
        times = np.linspace(0,1.2,12)
        xyz = np.array([10,10,2])+times[:,None]*[5,3,7]+np.c_[times*0,times*0,-4.905*times**2]
        projection = np.array([[800,0,0,0],[0,800,0,0],[0,0,1,40]],float)
        q = np.c_[xyz,np.ones(len(times))]@projection.T
        pixels = q[:,:2]/q[:,2:]
        result = fit_ballistic(times,pixels,[projection]*len(times),calibration_verified=True)
        self.assertTrue(result['accepted']); self.assertLess(result['rms_px'],1e-4)
        self.assertFalse(result['directly_observed']); self.assertFalse(result['eligible_for_pass_gates'])
        result = fit_ballistic(times[:4],pixels[:4],[projection]*4,calibration_verified=True)
        self.assertFalse(result['accepted'])


class PlayersPassesM21Tests(unittest.TestCase):
    def test_deep_continuation_needs_multiple_observations_and_resets_at_cuts(self):
        config = json.loads(Path('configs/vision/cv_m21.json').read_text())['players']
        tracker = PlayerTracker(25,config,deep=True)
        d = {'bbox':[10,10,10,20],'confidence':.9,'histogram':[1,0], 'embedding':[1,0], 'team':'TEAM_A','team_confidence':.9}
        first = tracker.step(0,camera(),[d])[0]
        second = tracker.step(1,camera(),[d])[0]
        self.assertEqual(second['embedding_observations'],2)
        moved = {**d,'bbox':[18,10,10,20]}
        third = tracker.step(2,camera(),[moved])[0]
        self.assertEqual(third['entity'],first['entity'])
        after_cut = tracker.step(3,camera(1,True),[moved])[0]
        self.assertNotEqual(after_cut['entity'],third['entity'])
        self.assertEqual(after_cut['embedding_observations'],1)

    def test_alternative_player_ids_reset_and_never_propagate_manual_identity(self):
        config = json.loads(Path('configs/vision/cv_m21.json').read_text())['players']
        tracker = PlayerTracker(25,config)
        d = {'bbox':[10,10,10,20],'confidence':.9,'histogram':[1,0], 'team':'TEAM_A','team_confidence':.9}
        a = tracker.step(0,camera(),[d])[0]
        b = tracker.step(1,camera(),[d])[0]
        self.assertEqual(a['entity'],b['entity']); self.assertFalse(b['manual_identity_propagated'])
        c = tracker.step(2,camera(1,True),[d])[0]
        self.assertNotEqual(b['entity'],c['entity'])
    def test_pass_second_stage_can_reject_but_never_promote_candidates(self):
        candidate_event = {'event_id':'x','status':'candidate','event_type':'transfer_candidate','reasons':['TEAM_UNCERTAIN'],
            'start_frame':0,'end_frame':2,'sender_entity':'a','receiver_entity':'b','camera_epoch':0,'ball_frames':[0,1,2]}
        evidence = [{'frame':i,'epoch':0,'observed':True,'owner':'b' if i>=2 else 'a','people':{'a':{'entity':'a'},'b':{'entity':'b'}}} for i in range(5)]
        out, _ = verify_events([candidate_event],evidence,25)
        self.assertEqual(out[0]['status'],'candidate'); self.assertEqual(out[0]['reasons'],['TEAM_UNCERTAIN'])
        estimated = {**candidate_event,'status':'estimated','event_type':'pass','reasons':[]}
        out, diagnostics = verify_events([estimated],evidence[:3],25)
        self.assertEqual(out[0]['status'],'candidate')
        self.assertIn('SECOND_STAGE_RECEIVER_CONTINUATION_UNCONFIRMED',out[0]['reasons'])
        self.assertTrue(diagnostics[0]['future_evidence_used'])
        self.assertEqual(estimated['status'],'estimated')


if __name__ == '__main__': unittest.main()
