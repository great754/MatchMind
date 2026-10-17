import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from fastapi.testclient import TestClient
from src.vision.data_loading import DatasetPaths
from src.serving.app import create_app
from scripts.prepare_recording import inspect,protect_output,register
from tests.helpers import write_query_artifacts

class RecordingSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.config=dict(recording_id='128058_clip',match_id='128058',data_root=str(self.root/'data'),output=str(self.root/'new'),registry=str(self.root/'registry.json'))
    def test_preflight_names_only_selected_match_and_excludes_existing_video(self):
        p=DatasetPaths('128058',Path(self.config['data_root']));p.video.parent.mkdir(parents=True);p.video.write_bytes(b'video')
        report=inspect(self.config);self.assertFalse(report['ready']);self.assertEqual(len(report['missing']),8)
        self.assertNotIn('--include',report['download_command']);self.assertNotIn('118575',report['download_command']);self.assertNotIn('clips/128058.mp4',report['download_command'])
        self.assertIn('gsr/128058/128058_2nd.json',report['missing'])
    def test_download_suffix_aliases_do_not_cross_match(self):
        p=DatasetPaths('128058',self.root);(self.root/'mot/clips').mkdir(parents=True)
        (self.root/'mot/128058.txt.txt').touch();(self.root/'mot/clips/128058.txt.mp4').touch()
        self.assertEqual(p.mot.name,'128058.txt.txt');self.assertEqual(p.video.name,'128058.txt.mp4')
        self.assertEqual(DatasetPaths('118575',self.root).video.name,'118575.mp4')
    def test_output_cannot_replace_other_recording(self):
        Path(self.config['registry']).write_text(json.dumps({'118575_clip':self.config['output']}))
        with self.assertRaises(ValueError):protect_output(self.config)
    def test_registered_bundles_remain_independent_and_pending_is_not_served(self):
        old=self.root/'old';old.mkdir();write_query_artifacts(old)
        new=self.root/'new';new.mkdir();write_query_artifacts(new)
        for directory,match in [(old,'118575'),(new,'128058')]:
            p=directory/'summary.json';v=json.loads(p.read_text());v['match_id']=match;p.write_text(json.dumps(v))
        c=TestClient(create_app({'118575_clip':old,'128058_clip':new},recording_catalog=[dict(recording_id='other_clip',match_id='other')]))
        items=c.get('/recordings').json()['recordings'];self.assertEqual(len(items),3)
        self.assertEqual(next(r['status'] for r in items if r['recording_id']=='other_clip'),'not_processed')
        self.assertEqual(c.get('/matches/other_clip/view').status_code,404)
        self.assertEqual(c.get('/matches/118575_clip/view').json()['summary']['facts'][0]['value'],'118575')
        self.assertEqual(c.get('/matches/128058_clip/view').json()['summary']['facts'][0]['value'],'128058')
    def test_register_preserves_old_entry_and_rejects_wrong_match(self):
        directory=Path(self.config['output']);directory.mkdir();write_query_artifacts(directory)
        Path(self.config['registry']).write_text(json.dumps({'118575_clip':str(self.root/'old')}))
        with self.assertRaises(ValueError):register(self.config)
        p=directory/'summary.json';v=json.loads(p.read_text());v['match_id']='128058';p.write_text(json.dumps(v))
        register(self.config);values=json.loads(Path(self.config['registry']).read_text())
        self.assertIn('118575_clip',values);self.assertIn('128058_clip',values)
    def test_other_match_annotations_are_rejected_before_processing(self):
        from src.analytics.data import load_recording
        paths=DatasetPaths('128058',self.root)
        paths.sync.parent.mkdir(parents=True);paths.sync.write_text(json.dumps(dict(match_id='128058',fps=25)))
        paths.bas.parent.mkdir(parents=True);paths.bas.write_text(json.dumps(dict(match_id='118575')))
        with self.assertRaisesRegex(ValueError,'BAS match ID'):load_recording(paths)

    def test_invalid_match_cannot_escape_dataset(self):
        with self.assertRaises(ValueError):DatasetPaths('../118575',self.root)
