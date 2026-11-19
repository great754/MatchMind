"""Seal source-disjoint model-blind review tasks from two bounded Azure video excerpts.

No detector or model-generated annotation is used. The full match is not processed
or saved locally. Original M19 labels and tasks remain untouched.
"""
import argparse
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading

from src.serving.cv import digest
from src.vision.events.passes import signature
from src.vision.advanced21.experiment import fresh_directory, write_json


def prepare(output):
    target = Path('configs/evaluation/cv_m21_holdout.json')
    if target.exists():
        raise ValueError('The review protocol is sealed; do not overwrite tasks or annotations')
    root = fresh_directory(output)
    # Fixed before inspecting footage or running any detector. Both windows are one source-match group.
    acquisition = {'source_group': 'Tottenham_Arsenal_external_vibuxer_z4ny0atm95s8',
                   'offsets_seconds': [600, 1800], 'duration_seconds': 5,
                   'output_fps': 25, 'maximum_width': 1280, 'selection_uses_model_candidates': False,
                   'reference_role': 'sealed_final_test_pending_human_labels', 'maximum_served_bytes': 128*1024*1024}
    write_json(root/'acquisition_plan.json', acquisition)
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobClient
    from azure.core import MatchConditions
    verified = json.loads(Path('docs/evidence/external_video_upload_2026_11_13.json').read_text())['azure']
    blob = BlobClient(account_url='https://'+verified['account']+'.blob.core.windows.net',
            container_name=verified['container'], blob_name=verified['blob'], credential=AzureCliCredential(),
            max_single_get_size=1024*1024, max_chunk_get_size=1024*1024)
    props = blob.get_blob_properties()
    if props.size != verified['bytes'] or props.etag != verified['etag'] or props.metadata.get('sha256') != verified['sha256_metadata']:
        raise ValueError('Azure source differs from the previously verified upload')
    served = [0]
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_HEAD(self): self.respond(False)
        def do_GET(self): self.respond(True)
        def respond(self, body):
            if self.path != '/source.mp4':
                self.send_error(404); return
            start, end = 0, props.size-1
            value = self.headers.get('Range')
            if value:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', value)
                if not match:
                    self.send_error(416); return
                start, tail = int(match[1]), match[2]
                end = min(props.size-1, int(tail)) if tail else props.size-1
            if start > end or start >= props.size:
                self.send_error(416); return
            self.send_response(206 if value else 200)
            self.send_header('Content-Type', 'video/mp4'); self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end-start+1))
            if value: self.send_header('Content-Range', f'bytes {start}-{end}/{props.size}')
            self.end_headers()
            if not body: return
            try:
                stream = blob.download_blob(offset=start, length=end-start+1, etag=props.etag,
                    match_condition=MatchConditions.IfNotModified, validate_content=True, max_concurrency=1)
                for chunk in stream.chunks():
                    with lock:
                        served[0] += len(chunk)
                        if served[0] > acquisition['maximum_served_bytes']:
                            raise ValueError('Bounded source-read budget exceeded')
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sources, tasks = {}, []
    try:
        for index, offset in enumerate(acquisition['offsets_seconds']):
            name = 'm21_holdout_'+str(index+1)
            video = root/(name+'.mp4')
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(offset), '-i',
                f'http://127.0.0.1:{server.server_port}/source.mp4', '-t', '5', '-an',
                '-vf', 'scale=min(1280\\,iw):-2,fps=25', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                '-movflags', '+faststart', str(video)]
            subprocess.run(command, check=True, timeout=180)
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames','-of','json',str(video)]))['streams'][0]
            fps, frames = float(Fraction(probe['avg_frame_rate'])), int(probe['nb_read_frames'])
            if fps != 25 or frames != 125:
                raise ValueError('Unexpected sealed review source clock')
            sources[name] = {'video_sha256': digest(video), 'source_video': str(video), 'source_group': acquisition['source_group'],
                'source_offset_seconds': offset, 'source_blob_etag': props.etag, 'source_blob_sha256_metadata': verified['sha256_metadata'],
                'frames': frames, 'fps': fps, 'width': probe['width'], 'height': probe['height'],
                'media_url': f'/cv/m21/matches/{name}/video.mp4'}
            # Twelve uniform frames plus a six-frame continuous ball window, not chosen from model failures.
            frame_list = sorted(set([round(5+i*114/11) for i in range(12)]+list(range(49,55))))
            for frame in frame_list:
                tasks.append({'task_id': f'{name}:ball:{frame}', 'recording_id': name, 'kind': 'ball',
                    'frame': frame, 'timestamp': frame/fps, 'temporal_group': name+':49-54' if 49 <= frame <= 54 else None})
            for j, frame in enumerate([15,40,65,90]):
                tasks.append({'task_id': f'{name}:identity:{frame}', 'recording_id': name, 'kind': 'identity',
                    'frame': frame, 'other_frame': frame+5, 'timestamp': frame/fps,
                    'instruction': 'Compare the same player if recognizable; on alternate tasks also include clearly different-player comparisons. Mark uncertainty when identity is unclear.'})
            for start, end in [(12,49),(65,112)]:
                tasks.append({'task_id': f'{name}:passes:{start}-{end}', 'recording_id': name, 'kind': 'passes',
                    'start_frame': start, 'end_frame': end, 'start': start/fps, 'end': end/fps, 'fps': fps})
    finally:
        server.shutdown(); server.server_close(); blob.close()
    document = {'schema_version':'cv-m21-review-v1', 'protocol':{
        'schema_version':'cv-m21-review-v1', 'ball_box_match_iou':.25, 'pass_endpoint_tolerance_seconds':.25,
        'selection':'predeclared_source_times_without_model_candidate_selection',
        'reference_role':'sealed_final_test_pending_human_labels',
        'source_group_split':'Both excerpts are ONE unseen match group; do not train on either excerpt.',
        'temporal_ball_target':'The in-play match ball, tightly boxed; uncertainty is not a negative.',
        'independent_accuracy_requires_human_review':True}, 'sources':sources, 'tasks':tasks,
        'selection_uses_model_candidates':False, 'human_labels_created':False}
    document['fingerprint'] = signature(document)
    write_json(target, document)
    write_json(root/'acquisition_results.json', {'source_etag_verified':True,'source_metadata_sha256_verified':True,
        'whole_source_independently_rehashed_this_run':False, 'azure_chunks_content_md5_validated':True,
        'served_bytes':served[0], 'full_source_saved_or_processed':False, 'source_match_groups':1,
        'task_count':len(tasks),'model_inference_on_test_sources':False,'tasks_fingerprint':document['fingerprint']})
    print(json.dumps({'tasks':len(tasks),'sources':list(sources),'source_match_groups':1,'labels_created':0}),flush=True)
    return document


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    prepare(parser.parse_args().output)
