"""Exercise video delivery and seeking without public blobs or signed URLs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
from fastapi.testclient import TestClient
from src.serving.app import create_app
from src.serving.media import AzureClipMedia
from tests.helpers import write_query_artifacts
try:
    import azure.core
except ImportError:
    azure=None


@unittest.skipIf(azure is None,'Install requirements-azure.txt for Azure media tests')
class AzureMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        path=Path(self.temp.name);write_query_artifacts(path)
        self.data=b'0123456789'
        self.blob=Mock()
        self.blob.get_blob_properties.return_value=SimpleNamespace(size=10,etag='"clip-v1"')
        self.blob.download_blob.side_effect=lambda **kw:SimpleNamespace(chunks=lambda:iter([self.data[kw['offset']:kw['offset']+kw['length']]]))
        self.container=Mock();self.container.get_blob_client.return_value=self.blob
        self.media=AzureClipMedia(self.container)
        self.client=TestClient(create_app({'demo':path},media_store=self.media,token='test'))
        self.headers={'Authorization':'Bearer test'}

    def test_view_advertises_private_api_video_without_credentials(self):
        r=self.client.get('/matches/demo/view',headers=self.headers)
        self.assertEqual(r.json()['media_url'],'/matches/demo/video.mp4')
        self.assertNotIn('blob.core',r.text)
        self.container.get_blob_client.assert_called_with('mot/clips/synthetic.mp4')

    def test_full_video_and_seek_ranges(self):
        for value,expected,status,content_range in [(None,b'0123456789',200,None),('bytes=2-5',b'2345',206,'bytes 2-5/10'),('bytes=7-',b'789',206,'bytes 7-9/10'),('bytes=-3',b'789',206,'bytes 7-9/10'),('bytes=8-99',b'89',206,'bytes 8-9/10')]:
            with self.subTest(value=value):
                headers={**self.headers,**({'Range':value} if value else {})}
                r=self.client.get('/matches/demo/video.mp4',headers=headers)
                self.assertEqual(r.status_code,status);self.assertEqual(r.content,expected)
                self.assertEqual(r.headers.get('content-range'),content_range)
                self.assertEqual(r.headers['content-length'],str(len(expected)))
                self.assertEqual(r.headers['accept-ranges'],'bytes')
                self.assertEqual(r.headers['content-type'],'video/mp4')

    def test_invalid_ranges_and_head_do_not_download(self):
        for value in ('bytes=11-','bytes=5-2','bytes=-0','bytes=','bytes=1-2,4-5'):
            r=self.client.get('/matches/demo/video.mp4',headers={**self.headers,'Range':value})
            self.assertEqual(r.status_code,416);self.assertEqual(r.headers['content-range'],'bytes */10')
        r=self.client.head('/matches/demo/video.mp4',headers=self.headers)
        self.assertEqual(r.status_code,200);self.assertEqual(r.content,b'')
        self.blob.download_blob.assert_not_called()

    def test_auth_and_unknown_ids_never_access_video(self):
        self.assertEqual(self.client.get('/matches/demo/video.mp4').status_code,401)
        self.assertEqual(self.client.get('/matches/unknown/video.mp4',headers=self.headers).status_code,404)
        self.blob.download_blob.assert_not_called()
        self.assertFalse(self.media.available('../private'))

    def test_missing_video_keeps_analytics_queryable(self):
        self.blob.get_blob_properties.side_effect=RuntimeError('private credential')
        self.media.metadata.clear()
        r=self.client.get('/matches/demo/view',headers=self.headers)
        self.assertEqual(r.status_code,200);self.assertIsNone(r.json()['media_url'])
        self.assertEqual(self.client.get('/matches/demo/video.mp4',headers=self.headers).status_code,404)

    def test_download_failure_is_safe_and_if_range_mismatch_returns_full(self):
        r=self.client.get('/matches/demo/video.mp4',headers={**self.headers,'Range':'bytes=2-5','If-Range':'"old"'})
        self.assertEqual(r.status_code,200);self.assertEqual(r.content,self.data)
        self.blob.download_blob.side_effect=RuntimeError('credential-secret')
        r=self.client.get('/matches/demo/video.mp4',headers=self.headers)
        self.assertEqual(r.status_code,503);self.assertNotIn('credential-secret',r.text)

    def test_video_only_recording_can_play_without_becoming_analytics_ready(self):
        from tests.test_artifact_azure_workflow import FakeBlob
        from src.artifacts.store import AzureBlobArtifactStore
        store=AzureBlobArtifactStore(FakeBlob(),['new_clip'],Path(self.temp.name)/'cache')
        c=TestClient(create_app(store,media_store=self.media,recording_catalog=[dict(recording_id='new_clip',match_id='new',processed=True)]))
        row=c.get('/recordings').json()['recordings'][0]
        self.assertEqual(row['status'],'unavailable');self.assertFalse(row['artifact_ready'])
        self.assertTrue(row['media_available']);self.assertEqual(row['media_url'],'/matches/new_clip/video.mp4')
        self.assertEqual(c.get(row['media_url'],headers={'Range':'bytes=0-3'}).content,b'0123')
        self.assertEqual(c.get('/matches/new_clip/view').status_code,503)
        self.assertEqual(c.post('/matches/new_clip/ask',json={'question':'How many passes?'}).status_code,503)
        self.assertEqual(c.get('/ready').status_code,503)
