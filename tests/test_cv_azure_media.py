"""Azure-only CV playback/decoding preserves integrity without permanent videos."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
from fastapi import FastAPI,HTTPException,Request
from fastapi.testclient import TestClient
from src.serving.cv import CVReview,digest
from src.serving.cv_media import AzureCVSource,load_sources
from src.vision.world.schema import WorldState,CameraState
from src.vision.world.player_state import PlayerWorldFrame
from scripts.migrate_cv_media_to_azure import deletion_candidates,NAMES


class Blob:
    def __init__(self,data):self.data=data;self.etag='"verified"';self.sha=hashlib.sha256(data).hexdigest();self.reads=[]
    def get_blob_properties(self):return SimpleNamespace(size=len(self.data),etag=self.etag,metadata={'sha256':self.sha,'recording_id':'giroud'})
    def download_blob(self,**kwargs):
        self.reads.append(kwargs)
        if kwargs['etag']!=self.etag:raise ValueError('Remote ETag changed')
        start=kwargs.get('offset',0);data=self.data[start:start+kwargs['length']] if 'length' in kwargs else self.data
        return SimpleNamespace(chunks=lambda:iter([data]))


class AzureCVTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.video=self.root/'giroud.mp4';writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'mp4v'),25,(80,64))
        for value in [0,60,120]:writer.write(np.full((64,80,3),value,np.uint8))
        writer.release();self.bytes=self.video.read_bytes();self.blob=Blob(self.bytes);self.client=SimpleNamespace(get_blob_client=lambda name:self.blob)
        self.entry={'blob':f'cv/clips/giroud/{self.blob.sha}.mp4','sha256':self.blob.sha,'bytes':len(self.bytes),'etag':self.blob.etag};self.source=AzureCVSource(self.client,'giroud',self.entry)
        self.world=self.root/'world';self.world.mkdir();offsets=[]
        with (self.world/'frames.jsonl').open('wb') as stream:
            for i in range(3):
                row=WorldState(recording_id='giroud',frame=i,timestamp=i/25,camera=CameraState(frame=i,timestamp=i/25),entities=[],unassigned=[],decisions=[],reconciliations=[],player_world=PlayerWorldFrame(frame=i,timestamp=i/25,players=[],unresolved_detections=[]))
                offsets.append(stream.tell());stream.write(row.model_dump_json().encode()+b'\n')
        m={'recording_id':'giroud','source_video':str(self.video),'video_sha256':self.blob.sha,'frames_sha256':digest(self.world/'frames.jsonl'),'frames':3,'fps':25,'width':80,'height':64,'frame_offsets':offsets}
        (self.world/'manifest.json').write_text(json.dumps(m))

    def test_cv_starts_and_queries_after_local_video_deleted(self):
        self.video.unlink();review=CVReview(self.world,video_source=self.source)
        self.assertEqual(review.frame(.04).frame,1);self.assertEqual(review.metadata()['media_storage'],'azure');self.assertFalse(self.video.exists())

    def test_private_decode_lease_removed_on_success_and_exception(self):
        self.video.unlink();review=CVReview(self.world,video_source=self.source)
        with review.video_path() as path:
            capture=cv2.VideoCapture(str(path));ok,image=capture.read();capture.release();self.assertTrue(ok);self.assertEqual(image.shape,(64,80,3))
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(RuntimeError,'decoder failure'):
            with review.video_path() as path:raise RuntimeError('decoder failure')
        self.assertFalse(path.exists());self.assertFalse(self.video.exists());self.assertEqual(len(self.blob.reads),1)

    def test_corrupt_cloud_bytes_rejected_even_when_local_video_exists(self):
        self.blob.data=b'x'*len(self.bytes)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):CVReview(self.world,video_source=self.source)
        self.assertTrue(self.video.exists())

    def test_wrong_remote_manifest_identity_rejected(self):
        self.source.sha256='0'*64
        with self.assertRaisesRegex(ValueError,'frozen recording'):CVReview(self.world,video_source=self.source)

    def test_changed_remote_revision_rejected(self):
        review=CVReview(self.world,video_source=self.source);self.blob.etag='"changed"';self.source.checked_at=0
        with self.assertRaises(HTTPException) as exc:review.frame(0)
        self.assertEqual(exc.exception.status_code,503)

    def test_video_seeking_reads_azure_with_etag_condition(self):
        self.video.unlink();self.source.validate();app=FastAPI()
        @app.get('/video.mp4')
        def video(request:Request):return self.source.response(request)
        r=TestClient(app).get('/video.mp4',headers={'Range':'bytes=16-31'})
        self.assertEqual(r.status_code,206);self.assertEqual(r.content,self.bytes[16:32]);self.assertEqual(self.blob.reads[-1]['etag'],self.entry['etag']);self.assertFalse(self.video.exists())

    def test_bad_cloud_registry_fails_closed_independently(self):
        p=self.root/'bad.json';p.write_text('{}');sources=load_sources(self.client,'account','container',p)
        self.assertEqual(set(sources),{'giroud','test_match','southampton','team_goal'})
        with self.assertRaises(ValueError):sources['giroud'].validate()
        self.assertEqual(load_sources(self.client,'account','container',self.root/'absent.json'),{})

    def deletion_fixture(self):
        st=self.video.stat();copy={'path':str(self.video),'recording_id':'giroud','sha256':self.blob.sha,'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino}
        expected={name:{'sha256':self.blob.sha,'bytes':st.st_size} for name in NAMES}
        return {'expected':expected,'copies':[copy]},{'recordings':{name:dict(value) for name,value in expected.items()}}

    def test_archive_identity_mismatch_prevents_local_deletion(self):
        data,registry=self.deletion_fixture();registry['recordings']['giroud']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'archive differ'):deletion_candidates(data,registry,[self.root])
        self.assertTrue(self.video.exists())

    def test_changed_local_file_preserved_by_deletion_preflight(self):
        data,registry=self.deletion_fixture();self.video.write_bytes(b'x'*len(self.bytes))
        with self.assertRaisesRegex(ValueError,'Local source changed'):deletion_candidates(data,registry,[self.root])
        self.assertTrue(self.video.exists())


if __name__=='__main__':unittest.main()
