"""Current-pixel reacquisition, uncertainty, camera resets and frozen-default reuse."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
from src.vision.intelligent23.memory import BallMemory
from src.vision.intelligent23.verification import verify, combine
from src.vision.advanced21.offline import dag_link


def camera(epoch=0, cut=False, motion=None):
    return {'epoch': epoch, 'continuity': 'cut' if cut else 'continuous', 'calibration_valid': False,
            'calibration': None, 'motion_evidence': {'field_motion': {
                'accepted': motion is not None, 'interval_frames': 1, 'homography': motion}}}


def candidate(frame, x=50., confidence=.9, origin='primary'):
    return {'candidate_id': 'c'+str(frame), 'bbox': [x, 50., 6., 6.], 'confidence': confidence,
            'inference_view': 'ACTUAL_CURRENT_PIXELS', 'source': 'VIDEO_CV_DETECTOR', 'origin': origin,
            'temporal_features': {'aspect_ratio': 1., 'edge_coherence': .1, 'contrast': .2,
                'camera_residual_px': 2., 'forward_backward_error_px': .1,
                'patch_correlation': .8, 'history_available': True},
            'deterministic': {'accepted': True, 'score': None, 'reason': 'NO_CONCLUSIVE_VISUAL_VETO'}}


class MemoryM23Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/vision/cv_m23.json').read_text())
        self.m20 = json.loads(Path('configs/vision/cv_m20.json').read_text())
        self.memory = BallMemory(320, 180, self.m20, self.config)
        self.image = np.zeros((180, 320, 3), np.uint8)

    def observed(self):
        for i in range(3):
            self.memory.prepare(i, i/25, camera(), [])
            ball, state = self.memory.consume(self.image, [candidate(i, 50.+i*2)])
        self.assertTrue(ball.directly_observed)
        return state

    def test_no_label_or_prediction_seed_and_regions_are_planned_before_current_candidates(self):
        plan = self.memory.prepare(0, 0., camera(), [])
        self.assertTrue(all(r['kind'] == 'global_discovery' for r in plan['regions']))
        with self.assertRaises(ValueError):self.memory.prepare(1, .04, camera(), [])
        ball, state = self.memory.consume(self.image, [candidate(0)])
        self.assertFalse(ball.directly_observed)
        self.memory.prepare(1, .04, camera(), [])
        ball, _ = self.memory.consume(self.image, [candidate(1)])
        self.assertTrue(ball.directly_observed)
        plan = self.memory.prepare(2, .08, camera(), [])
        self.assertTrue(plan['regions'])
        self.assertTrue(all(r['planned_before_current_candidates'] for r in plan['regions']))

    def test_memory_survives_short_matcher_expiry_but_never_becomes_a_visual_measurement(self):
        self.observed()
        radii = []
        for i in range(3, 14):
            plan = self.memory.prepare(i, i/25, camera(), [])
            if plan['predictions']:radii.append(max(v['radius90_px'] for v in plan['predictions'].values()))
            ball, state = self.memory.consume(self.image, [])
            self.assertFalse(ball.directly_observed)
            self.assertIsNone(ball.image_position)
            self.assertEqual(state['last_direct_observation']['frame'], 2)
        self.assertTrue(self.memory.archive)
        self.assertGreater(radii[-1], radii[0])
        self.assertEqual(state['state'], 'UNKNOWN')
        self.memory.prepare(40, 1.6, camera(), [])
        _, state = self.memory.consume(self.image, [])
        self.assertEqual(self.memory.archive, {})
        self.assertTrue(all(r['kind'] == 'global_discovery' for r in state['search_regions']))

    def test_camera_cut_clears_archive_appearance_and_last_observation(self):
        self.observed()
        plan = self.memory.prepare(3, .12, camera(1, True), [])
        self.assertEqual(plan['predictions'], {})
        self.assertTrue(all(r['kind'] == 'global_discovery' for r in plan['regions']))
        self.assertEqual(self.memory.archive, {})
        _, state = self.memory.consume(self.image, [])
        self.assertIsNone(state['last_direct_observation'])

    def test_camera_pan_transports_search_center_and_covariance(self):
        self.observed()
        old = next(iter(self.memory.core.banks.values())).combined()[0][:2].copy()
        h = [[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]]
        plan = self.memory.prepare(3, .12, camera(motion=h), [])
        predicted = next(iter(plan['predictions'].values()))
        self.assertGreater(predicted['mean'][0]-old[0], 18)
        values = np.linalg.eigvalsh(predicted['covariance'])
        self.assertTrue(np.all(values > 0))
        self.assertEqual(plan['camera_homography'], h)

    def test_incompatible_camera_transform_discards_old_memory(self):
        self.observed()
        plan = self.memory.prepare(3, .12, camera(motion=[[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]), [])
        self.assertEqual(plan['predictions'], {})
        self.assertIsNone(self.memory.last_observation)
        self.assertTrue(plan['cut'])

    def test_one_tentative_visual_hit_can_request_search_but_cannot_be_observed(self):
        self.memory.prepare(0, 0., camera(), [])
        ball, _ = self.memory.consume(self.image, [candidate(0)])
        self.assertFalse(ball.directly_observed)
        plan = self.memory.prepare(1, .04, camera(), [])
        self.assertEqual(sum(r['kind'] == 'tentative_visual' for r in plan['regions']), 1)
        ball, _ = self.memory.consume(self.image, [])
        self.assertFalse(ball.directly_observed)

    def test_competing_visual_births_do_not_force_a_current_ball(self):
        for i in range(3):
            self.memory.prepare(i, i/25, camera(), [])
            a, b = candidate(i, 40.), candidate(i+20, 180.)
            ball, state = self.memory.consume(self.image, [a, b])
        self.assertFalse(ball.directly_observed)
        self.assertEqual(state['state'], 'AMBIGUOUS')
        self.assertIsNone(ball.image_position)


class VerificationM23Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/vision/cv_m23.json').read_text())
        self.image = np.zeros((180, 320, 3), np.uint8)
        class Memory:
            last_observation = {'timestamp': 0.}
            plan = {'timestamp': .04}
            value = {'track_id': 1, 'mahalanobis_squared': 1., 'appearance_similarity': .9,
                     'size_ratio': 1., 'velocity': [100., 0.], 'stationary_frames': 5}
            def evidence(self, image, c):return self.value
        self.memory = Memory()

    def test_trajectory_cannot_promote_low_neural_confidence(self):
        c = candidate(1, confidence=.2, origin='guided')
        result = verify(c, self.memory, self.image, self.config, additional_verification=True)
        self.assertFalse(result['observation_eligible'])
        self.assertEqual(result['confidence'], .2)

    def test_guided_wrong_object_outside_motion_or_appearance_gate_is_rejected(self):
        original = dict(self.memory.value)
        for key, value in [('mahalanobis_squared', 100.), ('appearance_similarity', .1), ('size_ratio', 10.)]:
            self.memory.value = {**original, key: value}
            result = verify(candidate(1, origin='guided'), self.memory, self.image, self.config, additional_verification=True)
            self.assertFalse(result['observation_eligible'])

    def test_stationary_line_is_not_reinforced_by_high_detector_confidence(self):
        c = candidate(1)
        c['temporal_features'].update(aspect_ratio=4., edge_coherence=.99, camera_residual_px=.1)
        result = verify(c, self.memory, self.image, self.config, additional_verification=True)
        self.assertFalse(result['observation_eligible'])
        self.assertIn('SCENE_LOCKED', result['decision_reason'])

    def test_blurred_streak_can_pass_with_new_visual_motion_and_appearance_support(self):
        c = candidate(1, confidence=.6, origin='guided')
        c['bbox'] = [50., 50., 16., 4.]
        c['temporal_features'].update(aspect_ratio=4., edge_coherence=.99, camera_residual_px=4., forward_backward_error_px=8.)
        c['deterministic'] = {'accepted': False, 'reason': 'LINE_LIKE_WITH_TEMPORAL_CONTRADICTION'}
        result = verify(c, self.memory, self.image, self.config, additional_verification=True)
        self.assertTrue(result['blur_supported'])
        self.assertTrue(result['observation_eligible'])
        self.assertEqual(result['confidence'], .6)

    def test_m22_disagreement_abstains_and_agreement_changes_only_association_score(self):
        c = candidate(1)
        r = {'candidate_id': 'm22', 'bbox': c['bbox'], 'confidence': .8, 'current_only_confidence': .1, 'observation_eligible': False}
        a = verify(c, self.memory, self.image, self.config, additional_verification=True, m22_candidates=[r])
        self.assertEqual(a['m22_support'], [])
        r.update(current_only_confidence=.8, observation_eligible=True)
        b = verify(c, self.memory, self.image, self.config, additional_verification=True, m22_candidates=[r])
        self.assertGreater(b['association_adjustment'], a['association_adjustment'])
        self.assertEqual(b['confidence'], c['confidence'])

    def test_crop_merge_cannot_remove_an_independently_valid_primary_observation(self):
        a, b = candidate(1, confidence=.6), candidate(2, confidence=.7, origin='guided')
        b['temporal_features']['forward_backward_error_px'] = 10.
        c = combine([a], [b], 2)[0]
        self.assertEqual(c['origin'], 'guided')
        result = verify(c, self.memory, self.image, self.config, additional_verification=True)
        self.assertTrue(result['observation_eligible'])
        self.assertEqual(result['confidence'], .7)

    def test_duplicate_crops_retain_an_actual_winning_box_and_confidence(self):
        a, b = candidate(0, confidence=.6), candidate(1, confidence=.8, origin='guided')
        combined = combine([a], [b], 2)
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]['bbox'], b['bbox'])
        self.assertEqual(combined[0]['confidence'], .8)
        self.assertEqual({v['origin'] for v in combined[0]['support_views']}, {'primary', 'guided'})


class OfflineM23Tests(unittest.TestCase):
    def test_complement_can_bridge_but_cannot_replace_primary_observations(self):
        from src.vision.intelligent23.offline import anchored_complements
        before,a,b,after=candidate(0,48.),candidate(1,50.),candidate(4,56.),candidate(5,58.)
        good=candidate(2,52.,origin='complement');good['observation_eligible']=True
        bad=candidate(3,250.,origin='complement');bad['observation_eligible']=True
        primary=[before,a,None,None,b,after]
        pool=[[before],[a],[good],[bad],[b],[after]]
        cameras=[camera(motion=np.eye(3).tolist()) for _ in pool]
        descriptors={(i,c['candidate_id']):np.ones(32)/32 for i,row in enumerate(pool) for c in row}
        result=anchored_complements(primary,pool,cameras,25,400,descriptors)
        self.assertIs(result[1],a);self.assertIs(result[4],b)
        self.assertIs(result[2],good);self.assertIsNone(result[3])
        self.assertEqual(anchored_complements([None]*6,pool,cameras,25,400,descriptors),[None]*6)

    def test_complement_gap_does_not_cross_camera_cut_or_unsupported_motion(self):
        from src.vision.intelligent23.offline import anchored_complements,reconstruct_camera
        a,b=candidate(0),candidate(2)
        c=candidate(1,origin='complement');c['observation_eligible']=True
        for cameras in [[camera(),camera(),camera()], [camera(),camera(1,True),camera(1)]]:
            result=anchored_complements([a,None,b],[[a],[c],[b]],cameras,25,400,{})
            self.assertIsNone(result[1])
            self.assertIsNone(reconstruct_camera([a,None,b],cameras,25,320,180)[1])

    def test_camera_relative_reconstruction_and_history_remain_inferred(self):
        from src.vision.intelligent23.offline import reconstruct_camera,history
        cameras=[camera(motion=[[1,0,10],[0,1,0],[0,0,1]]) for _ in range(3)]
        primary=[candidate(0,20.),None,candidate(2,40.)]
        row=reconstruct_camera(primary,cameras,25,320,180)[1]
        self.assertFalse(row['directly_observed']);self.assertTrue(row['uses_future'])
        self.assertEqual(row['support_frames'],[0,2]);np.testing.assert_allclose(row['image'],[33.,53.])
        old=history(primary,cameras,2,25)
        self.assertFalse(old[0]['current_observation']);np.testing.assert_allclose(old[0]['current_camera_image'],[43.,53.])

    def test_optional_joint_score_preserves_frozen_default_and_neural_values(self):
        candidates = [[candidate(i, 50.+i)] for i in range(4)]
        before = copy.deepcopy(candidates)
        cameras = [camera() for _ in candidates]
        a = dag_link(candidates, cameras, 25, 400)
        b = dag_link(candidates, cameras, 25, 400, node_adjustment=lambda c, i: 0.)
        self.assertEqual(a, b)
        self.assertEqual(candidates, before)
        with self.assertRaises(ValueError):dag_link(candidates, cameras, 25, 400, node_adjustment=lambda c, i: float('nan'))

    def test_camera_cut_prevents_future_link_or_false_visual_birth_across_cut(self):
        candidates = [[candidate(0)], [candidate(1)]]
        selected = dag_link(candidates, [camera(0), camera(1, True)], 25, 400)
        self.assertEqual(selected, [None, None])
