"""Joint current-pixel verification; trajectory support never becomes a detector score."""
import math
import numpy as np
from src.evaluation.cv_m19 import iou
from src.vision.advanced21.detection import merge_candidates
from src.vision.advanced21.temporal import deterministic_verdict
from src.vision.events.passes import signature


def combine(primary, local, frame):
    raw = [*primary, *local]
    kept = merge_candidates(primary, local)
    for c in kept:
        c['candidate_id'] = 'm23_'+str(frame)+'_'+signature([c['bbox'], c['confidence'], c['inference_view']])[:16]
        c['support_views'] = [{'origin': v.get('origin', 'primary'), 'bbox': v['bbox'],
            'confidence': v['confidence'], 'source_region': v.get('source_region'),
            'independently_eligible_primary': v.get('origin') == 'primary' and v['confidence'] >= .35 and v.get('deterministic', {}).get('accepted', False)}
            for v in raw if iou(v['bbox'], c['bbox']) > .3]
    return kept


def verify(candidate, memory, image, config, *, additional_verification, m22_candidates=()):
    c = dict(candidate)
    features = c['temporal_features']
    evidence = memory.evidence(image, c)
    primary = c.get('origin') == 'primary' or any(v.get('independently_eligible_primary') for v in c.get('support_views', []))
    eligible = c['confidence'] >= config['visual_threshold']
    reason = 'CURRENT_NEURAL_PIXELS_MEET_FIXED_THRESHOLD' if eligible else 'BELOW_FIXED_CURRENT_VISUAL_THRESHOLD'
    original_check = c.get('deterministic') or deterministic_verdict(features, c['confidence'])
    if eligible and not original_check['accepted']:
        eligible, reason = False, original_check['reason']
    aspect = features['aspect_ratio']
    velocity = np.asarray(evidence['velocity']) if evidence else np.zeros(2)
    axis = np.array([1., 0.]) if c['bbox'][2] >= c['bbox'][3] else np.array([0., 1.])
    alignment = abs(float(axis@velocity))/max(float(np.linalg.norm(velocity)), 1e-9)
    moving = features.get('camera_residual_px') is not None and features['camera_residual_px'] >= 1
    blur = bool(aspect > 2 and moving and (evidence is None or alignment >= config['verification']['blur_direction_cosine']))
    if additional_verification and not primary and blur and evidence and c['confidence'] >= config['visual_threshold']:
        if evidence['mahalanobis_squared'] <= config['verification']['maximum_mahalanobis_squared'] and (
            evidence['appearance_similarity'] is not None and evidence['appearance_similarity'] >= config['verification']['minimum_appearance_similarity']):
            eligible, reason = True, 'BLURRED_CURRENT_CROP_WITH_MOTION_AND_APPEARANCE_SUPPORT'
    if eligible and additional_verification:
        v = config['verification']
        if not primary and c.get('origin') != 'discovery':
            if evidence is None:
                eligible, reason = False, 'GUIDED_CANDIDATE_HAS_NO_RETAINED_VISUAL_MEMORY'
            else:
                ratio_max = config['memory']['maximum_blurred_size_ratio'] if blur else config['memory']['maximum_size_ratio']
                if evidence['mahalanobis_squared'] > v['maximum_mahalanobis_squared']:
                    eligible, reason = False, 'OUTSIDE_MEMORY_UNCERTAINTY_GATE'
                elif evidence['appearance_similarity'] is not None and evidence['appearance_similarity'] < v['minimum_appearance_similarity']:
                    eligible, reason = False, 'APPEARANCE_CONTRADICTS_LAST_VISUAL_EVIDENCE'
                elif not config['memory']['minimum_size_ratio'] <= evidence['size_ratio'] <= ratio_max:
                    eligible, reason = False, 'SIZE_CHANGE_UNSUPPORTED_BY_VISUAL_MEMORY'
                elif not blur and features.get('forward_backward_error_px') is not None and features['forward_backward_error_px'] > v['maximum_forward_backward_error_px']:
                    eligible, reason = False, 'CURRENT_CROP_TEMPORAL_FLOW_CONTRADICTION'
        if evidence and aspect > 3 and features['edge_coherence'] > .9 and not moving and evidence['stationary_frames'] >= config['memory']['static_line_minimum_frames']:
            eligible, reason = False, 'PERSISTENT_SCENE_LOCKED_LINE_NOT_REINFORCED_BY_MEMORY'
    matches = [r for r in m22_candidates if c.get('origin') != 'complement' and r.get('observation_eligible') and
        r['confidence'] >= config['verification']['m22_temporal_threshold'] and
        r['current_only_confidence'] >= config['verification']['m22_current_threshold'] and
        iou(r['bbox'], c['bbox']) >= config['verification']['m22_match_iou']]
    adjustment = 0.
    if additional_verification and evidence and moving:
        distance = evidence['mahalanobis_squared']
        if distance <= config['verification']['maximum_mahalanobis_squared']:
            adjustment += .15*math.exp(-distance/2)
            adjustment -= config['verification']['appearance_penalty']*(1-(evidence['appearance_similarity'] or .5))
        elif memory.last_observation and memory.plan['timestamp']-memory.last_observation['timestamp'] < .12:
            adjustment -= config['verification']['motion_penalty']
    if matches:
        adjustment += config['verification']['m22_agreement_bonus']
    if c.get('origin') == 'complement' and (c.get('current_only_confidence', 0) < .35 or c['confidence'] < .35):
        eligible, reason = False, 'M22_COMPLEMENT_REQUIRES_TEMPORAL_AND_CURRENT_PIXEL_GATES'
    c.update(observation_eligible=bool(eligible), decision_reason=reason,
        joint_memory_evidence=evidence, blur_supported=blur,
        blur_definition='Appearance/motion compatibility proxy, not a human blur label',
        m22_support=[{'candidate_id': r['candidate_id'], 'bbox': r['bbox'], 'confidence': r['confidence'],
                     'current_only_confidence': r['current_only_confidence']} for r in matches],
        association_adjustment=float(adjustment), detector_confidence_unchanged=True,
        confidence_is_calibrated_probability=False)
    return c
