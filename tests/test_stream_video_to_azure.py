import hashlib
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
from azure.core import MatchConditions
from scripts.stream_video_to_azure import stream_video


class StreamUploadTests(unittest.TestCase):
    def setup_stream(self, chunks, length=None, status=200, content_type='video/mp4'):
        response = Mock(status_code=status)
        response.headers = {'Content-Type': content_type}
        if length is not None:
            response.headers['Content-Length'] = str(length)
        response.iter_content.return_value = iter(chunks)
        http = MagicMock()
        http.get.return_value.__enter__.return_value = response
        blob = Mock()
        blob.exists.return_value = False
        blob.get_blob_properties.return_value = SimpleNamespace(size=sum(map(len, chunks)))
        return blob, http

    def test_streams_blocks_and_commits_create_only_with_checksum(self):
        blob, http = self.setup_stream([b'abc', b'', b'def'], 6)
        result = stream_video('https://source.test/video', blob, http, Mock())
        self.assertEqual(result, {'bytes': 6, 'sha256': hashlib.sha256(b'abcdef').hexdigest()})
        self.assertEqual(blob.stage_block.call_count, 2)
        self.assertTrue(blob.stage_block.call_args.kwargs['validate_content'])
        self.assertEqual(blob.commit_block_list.call_args.kwargs['match_condition'], MatchConditions.IfMissing)
        self.assertNotIn('etag', blob.commit_block_list.call_args.kwargs)

    def test_existing_video_is_never_downloaded_or_overwritten(self):
        blob, http = self.setup_stream([b'abc'])
        blob.exists.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            stream_video('https://source.test/video', blob, http, Mock())
        http.get.assert_not_called()
        blob.commit_block_list.assert_not_called()

    def test_incomplete_stream_is_not_committed(self):
        blob, http = self.setup_stream([b'abc'], 4)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete source'):
            stream_video('https://source.test/video', blob, http, Mock())
        blob.commit_block_list.assert_not_called()

    def test_empty_stream_is_not_committed(self):
        blob, http = self.setup_stream([], 0)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete source'):
            stream_video('https://source.test/video', blob, http, Mock())
        blob.commit_block_list.assert_not_called()

    def test_source_error_or_html_is_rejected_before_upload(self):
        for status, content_type in [(403, 'text/html'), (206, 'video/mp4'), (200, 'text/html')]:
            blob, http = self.setup_stream([b'error'], status=status, content_type=content_type)
            with self.assertRaises(RuntimeError):
                stream_video('https://source.test/video', blob, http, Mock())
            blob.stage_block.assert_not_called()
            blob.commit_block_list.assert_not_called()

    def test_stream_without_content_length_is_supported(self):
        blob, http = self.setup_stream([b'abc'])
        self.assertEqual(stream_video('https://source.test/video', blob, http, Mock())['bytes'], 3)

    def test_real_sdk_serializes_create_only_without_leftover_transport_arguments(self):
        from azure.storage.blob._blob_client_helpers import _commit_block_list_options
        blob, http = self.setup_stream([b'abc'], 3)
        stream_video('https://source.test/video', blob, http, Mock())
        args, kwargs = blob.commit_block_list.call_args
        serialized = _commit_block_list_options(*args, **kwargs)
        self.assertEqual(serialized['modified_access_conditions'].if_none_match, '*')
        self.assertNotIn('etag', serialized)
        self.assertNotIn('match_condition', serialized)

    def test_commit_failure_can_resume_without_downloading_or_staging_again(self):
        blob, http = self.setup_stream([b'abc', b'def'], 6)
        blob.commit_block_list.side_effect = RuntimeError('commit temporarily unavailable')
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'upload.json'
            with self.assertRaises(RuntimeError):
                stream_video('https://source.test/video', blob, http, Mock(), checkpoint)
            ids = [c.kwargs['block_id'] for c in blob.stage_block.call_args_list]
            blob.get_block_list.return_value = ([], [SimpleNamespace(id=i,size=3) for i in ids])
            blob.commit_block_list.side_effect = None
            http.reset_mock(); blob.stage_block.reset_mock()
            result = stream_video('https://source.test/video', blob, http, Mock(), checkpoint)
            self.assertEqual(result['bytes'], 6)
            http.get.assert_not_called(); blob.stage_block.assert_not_called()
            self.assertEqual(blob.commit_block_list.call_args.args[0], ids)
