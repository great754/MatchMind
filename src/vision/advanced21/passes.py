"""Second-stage verification only narrows existing M19 completed-pass estimates."""
import copy
from collections import Counter


def verify_events(events, evidence, fps, temporal=None):
    out, diagnostics = [], []
    temporal = temporal or {}
    for original in events:
        event = copy.deepcopy(original)
        start, end = event['start_frame'], event['end_frame']
        sender, receiver = event['sender_entity'], event['receiver_entity']
        third_party = []
        for row in evidence[start:end+1]:
            owner = row['people'].get(row['owner']) if row['owner'] else None
            if owner is not None and owner['entity'] not in [sender, receiver]:
                third_party.append(row['frame'])
        hold = []
        for row in evidence[end:min(len(evidence), end+max(3, round(.16*fps)))]:
            owner = row['people'].get(row['owner']) if row['owner'] else None
            if row['epoch'] == event['camera_epoch'] and row['observed'] and owner is not None and owner['entity'] == receiver:
                hold.append(row['frame'])
        reasons = []
        if third_party:
            reasons.append('SECOND_STAGE_OTHER_PLAYER_CONTACT')
        if len(hold) < 2:
            reasons.append('SECOND_STAGE_RECEIVER_CONTINUATION_UNCONFIRMED')
        scores = [temporal[i] for i in event['ball_frames'] if i in temporal and temporal[i] is not None]
        if len(scores) >= 2 and sum(scores)/len(scores) < .5:
            reasons.append('SECOND_STAGE_VISUAL_VERIFIER_CONTRADICTION')
        if event['status'] == 'estimated' and reasons:
            event.update(status='candidate', event_type='transfer_candidate', reasons=reasons)
        diagnostics.append({'event_id': event['event_id'], 'base_status': original['status'],
                            'verified_status': event['status'], 'reasons': reasons,
                            'receiver_observed_continuation_frames': hold,
                            'alternative_player_contact_frames': third_party,
                            'temporal_visual_samples': len(scores),
                            'future_evidence_used': True, 'unsupported_metrics_added': False})
        out.append(event)
    return out, diagnostics


def verified_summary(summary, events):
    return {**summary, 'completed_pass_estimates': sum(e['status'] == 'estimated' for e in events),
            'transfer_candidates': sum(e['status'] == 'candidate' for e in events),
            'candidate_reasons': dict(Counter(reason for e in events for reason in e['reasons']))}
