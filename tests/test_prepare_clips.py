import unittest
from unittest.mock import patch
from scripts.prepare_clips import MATCHES,configuration,prepare_one

class PrepareClipsTests(unittest.TestCase):
    def test_only_short_clip_ids_are_configured(self):
        self.assertEqual(len(MATCHES),10)
        for match in MATCHES:
            config=configuration(match);self.assertEqual(config['recording_id'],match+'_clip');self.assertEqual(config['output'],f'outputs/analytics_{match}_tactics')
    def test_ready_clip_is_not_downloaded_or_reprocessed(self):
        with patch('scripts.prepare_clips.already_processed',return_value=True),patch('subprocess.run') as run,patch('huggingface_hub.hf_hub_download') as download:
            self.assertEqual(prepare_one(configuration('118575'),'unused'),'already_processed');run.assert_not_called();download.assert_not_called()
    def test_downloads_exact_missing_files_then_processes(self):
        config=configuration('117092')
        with patch('scripts.prepare_clips.already_processed',side_effect=[False,True]),patch('scripts.prepare_clips.inspect',side_effect=[{'missing':['mot/117092.txt']},{'ready':True}]),patch('huggingface_hub.hf_hub_download') as download,patch('subprocess.run') as run:
            self.assertEqual(prepare_one(config,'config.json'),'processed')
            download.assert_called_once_with('atomscott/soccertrack-v2','mot/117092.txt',repo_type='dataset',local_dir='data/soccertrack')
            self.assertEqual(run.call_args.args[0][-1],'--run');self.assertTrue(run.call_args.kwargs['check'])
    def test_missing_files_after_download_prevent_processing(self):
        with patch('scripts.prepare_clips.already_processed',return_value=False),patch('scripts.prepare_clips.inspect',side_effect=[{'missing':[]},{'ready':False}]),patch('subprocess.run') as run:
            with self.assertRaises(ValueError):prepare_one(configuration('117092'),'unused')
            run.assert_not_called()
