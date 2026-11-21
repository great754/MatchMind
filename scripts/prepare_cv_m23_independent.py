"""Prepare an approved unseen source; never auto-download or auto-label footage.

First create a source-only contact sheet. Only freeze a final-test manifest after
a person has reviewed relevance and original-match overlap. Neither stage runs CV.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def prepare(video, approval, output, frozen_analysis, review=None):
    video, output = Path(video), Path(output)
    approval_path = Path(approval)
    approval = json.loads(approval_path.read_text())
    source_hash = digest(video)
    if approval.get('explicit_user_approval') is not True or approval.get('video_sha256') != source_hash:
        raise ValueError('Explicit approval must identify this exact source video')
    match = approval.get('original_match_id', '').strip()
    if not match or not approval.get('approval_reference', '').strip():
        raise ValueError('Original match and approval evidence are required')
    frozen_analysis = Path(frozen_analysis)
    report = json.loads(frozen_analysis.read_text())
    known_hashes = {c['source_video_sha256'] for c in report['perception']['clips'].values()}
    if source_hash in known_hashes:
        raise ValueError('An existing development source is not an independent test')
    if output.exists():raise ValueError('Choose a new immutable output root')
    if review is None:
        import cv2
        from PIL import Image, ImageDraw
        cap = cv2.VideoCapture(str(video))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if count < 12 or fps <= 0:raise ValueError('Cannot inspect the approved footage')
        output.mkdir(parents=True)
        sheet = Image.new('RGB', (960, 4*205), '#151b22')
        draw = ImageDraw.Draw(sheet)
        indices = [round(i*(count-1)/11) for i in range(12)]
        for i, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:raise ValueError('Source preview decode failed')
            tile = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tile.thumbnail((320, 180))
            x,y = (i%3)*320,(i//3)*205
            sheet.paste(tile, (x,y));draw.text((x+4,y+183),f'{index/fps:.2f}s / frame {index}',fill='white')
        cap.release();sheet.save(output/'source_preview.jpg')
        (output/'review_required.json').write_text(json.dumps({'video_sha256':source_hash,
            'original_match_id':match,'football_content_reviewed':False,
            'football_players_and_ball_present':False,'original_match_overlap_audited':False,
            'original_match_overlaps_training_or_development':None,
            'training_source_inventory_sha256':None,'overlap_audit_evidence':None,
            'reviewer':None,'preview_path':str(output/'source_preview.jpg')},indent=2)+'\n')
        return {'status':'source_preview_only_no_tasks_or_model_run','output':str(output)}
    audit = json.loads(Path(review).read_text())
    if audit.get('video_sha256') != source_hash or audit.get('original_match_id') != match:
        raise ValueError('Content/overlap review identifies a different source')
    required = ['football_content_reviewed','football_players_and_ball_present','original_match_overlap_audited']
    if not all(audit.get(k) is True for k in required) or audit.get('original_match_overlaps_training_or_development') is not False:
        raise ValueError('Relevant football and no original-match overlap must be reviewed first')
    for key in ['reviewer','overlap_audit_evidence','training_source_inventory_sha256']:
        if not audit.get(key):raise ValueError('Missing source-overlap review evidence: '+key)
    output.mkdir(parents=True)
    frozen = {'schema_version':'cv-m23-independent-freeze-v1','source_video_sha256':source_hash,
        'original_match_id':match,'approval_sha256':digest(approval_path),
        'content_and_overlap_review_sha256':digest(review),'analysis_report_sha256':digest(frozen_analysis),
        'perception_config_sha256':report['perception']['config_sha256'],
        'runtime_source_sha256':report['perception']['source_sha256'],
        'selected_unseen_source_checkpoint':'must_be_explicitly_frozen_before_inference',
        'matching_iou':.25,'pass_endpoint_tolerance_seconds':.25,
        'labels_reviewed':False,'status':'source_and_protocol_frozen_model_selection_required',
        'promotion_allowed':False,'policy':{'precision_minimum':.85,'recall_minimum':.75,'f1_minimum':.80,
            'preserve_supported_passes':True,'exhaustive_window_false_completed_passes_maximum':0},
        'remaining':['Freeze one existing checkpoint and its training-source inventory before inference',
            'Freeze source-only review task windows before labels',
            'Review independent source frames without model proposals',
            'Run frozen measurements once; do not tune on final-test labels']}
    (output/'freeze.json').write_text(json.dumps(frozen,indent=2)+'\n')
    return frozen


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['video','approval','output','frozen-analysis']:parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.video,args.approval,args.output,args.frozen_analysis,args.review)))
