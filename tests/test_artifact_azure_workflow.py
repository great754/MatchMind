"""No network: real manifests/store/app against an injected Blob client."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
from src.artifacts.__main__ import export,main
from src.artifacts.manifest import parse_manifest,validate_snapshot
from src.artifacts.store import ArtifactError,AzureBlobArtifactStore,LocalArtifactStore
from src.artifacts.azure_workflow import upload,verify,AzureWorkflowError,client_from_environment,diagnostic
from tests.helpers import write_query_artifacts

class MissingBlob(Exception):status_code=404
class FakeBlob:
    def __init__(self):self.data={};self.uploads=[];self.reads=[]
    def download_blob(self,name):
        self.reads.append(name)
        if name not in self.data:raise MissingBlob()
        return SimpleNamespace(readall=lambda:self.data[name])
    def upload_blob(self,*,name,data,**kwargs):
        if name in self.data:raise ValueError('Existing blob')
        assert kwargs['overwrite'] is False and kwargs['blob_type']=='BlockBlob'
        self.data[name]=bytes(data);self.uploads.append(name)
    def list_blobs(self,*,name_starts_with):
        return [SimpleNamespace(name=n) for n in sorted(self.data) if n.startswith(name_starts_with)]
    def get_container_properties(self):return {'public_access':None}
    def close(self):pass

class AzureWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir();write_query_artifacts(self.source)
        self.registry=self.root/'registry.json';self.registry.write_text(json.dumps({'demo':str(self.source)}))
        self.output=self.root/'export';export({'demo':self.source},self.output);self.bundle=self.output/'matches/demo';self.client=FakeBlob()
    def manifest(self):return json.loads((self.bundle/'bundle.json').read_text())
    def write_manifest(self,value): (self.bundle/'bundle.json').write_text(json.dumps(value))
    def test_legacy_export_and_local_store_regression(self):
        self.assertEqual(LocalArtifactStore({'demo':self.source}).resolve('demo'),self.source.resolve())
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['--registry',str(self.registry),'--output',str(self.root/'legacy')]),0)
        self.assertTrue((self.root/'legacy/matches/demo/bundle.json').is_file())
    def test_build_cli_selects_one_recording_without_contacting_azure(self):
        with patch('src.artifacts.azure_workflow.client_from_environment') as client,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['build','--recording','demo','--registry',str(self.registry),'--output',str(self.output)]),0);client.assert_not_called()
    def test_missing_recording_cli_fails_safely(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(['build','--recording','missing','--registry',str(self.registry)]),1)
        self.assertIn('artifact_validation_failed',stderr.getvalue())
    def test_only_allowlisted_manifest_files_are_uploaded(self):
        for name in ('.env','video.mp4','credentials.json','cache.bin'):(self.bundle/name).write_text('never-upload-this')
        result=upload(self.client,'demo',self.bundle)
        self.assertEqual(set(self.client.data),{'matches/demo/'+n for n in self.manifest()['files']}|{'matches/demo/bundle.json'})
        self.assertEqual(self.client.uploads[-1],'matches/demo/bundle.json');self.assertTrue(result['manifest_published'])
    def test_public_container_is_rejected_before_upload(self):
        self.client.get_container_properties=lambda:{'public_access':'blob'}
        with self.assertRaises(AzureWorkflowError):upload(self.client,'demo',self.bundle)
        self.assertFalse(self.client.uploads)

    def test_prefix_isolation(self):
        upload(self.client,'demo',self.bundle);first=dict(self.client.data)
        upload(self.client,'second',self.bundle)
        for name,data in first.items():self.assertEqual(self.client.data[name],data)
        self.assertTrue(any(n.startswith('matches/second/') for n in self.client.data))
        p=AzureBlobArtifactStore(self.client,['demo'],self.root/'cache').resolve('demo')
        self.assertTrue(p.is_dir());self.assertTrue(all(n.startswith('matches/demo/') for n in self.client.reads[-len(self.manifest()['files'])-1:]))
    def test_upload_is_idempotent_and_never_overwrites(self):
        upload(self.client,'demo',self.bundle);count=len(self.client.uploads)
        result=upload(self.client,'demo',self.bundle);self.assertEqual(result['uploaded'],0);self.assertEqual(len(self.client.uploads),count)
    def test_different_remote_manifest_is_rejected(self):
        upload(self.client,'demo',self.bundle);manifest=self.manifest();manifest['files']['summary.json']='0'*64
        self.client.data['matches/demo/bundle.json']=json.dumps(manifest).encode();count=len(self.client.uploads)
        with self.assertRaises(AzureWorkflowError):upload(self.client,'demo',self.bundle)
        self.assertEqual(len(self.client.uploads),count)
    def test_orphaned_conflicting_blob_prevents_any_write(self):
        self.client.data['matches/demo/summary.json']=b'conflict'
        with self.assertRaises(AzureWorkflowError):upload(self.client,'demo',self.bundle)
        self.assertFalse(self.client.uploads)
    def test_failed_upload_does_not_publish_manifest(self):
        self.client.upload_blob=Mock(side_effect=RuntimeError('unavailable'))
        with self.assertRaises(RuntimeError):upload(self.client,'demo',self.bundle)
        self.assertNotIn('matches/demo/bundle.json',self.client.data)
    def test_hash_mismatch_and_missing_required_artifact(self):
        (self.bundle/'summary.json').write_text('{}')
        with self.assertRaises(ArtifactError):validate_snapshot(self.bundle)
        (self.bundle/'summary.json').unlink()
        with self.assertRaises(ArtifactError):upload(self.client,'demo',self.bundle)
        self.assertFalse(self.client.uploads)
    def test_manifest_rejects_unsafe_names_extra_keys_duplicates_and_missing_required(self):
        for change in ('path','digest','required','extra'):
            m=self.manifest()
            if change=='path':m['files']['../.env']='0'*64
            elif change=='digest':m['files']['summary.json']='not-a-hash'
            elif change=='required':del m['files']['events.json']
            else:m['secret']='unexpected'
            with self.subTest(change=change),self.assertRaises(ArtifactError):parse_manifest(json.dumps(m))
        with self.assertRaises(ArtifactError):parse_manifest('{"schema_version":"1.0","schema_version":"1.0","files":{}}')
    def test_schema_failure_is_rejected_even_with_matching_hash(self):
        raw=b'{"events":[{"invented":"claim"}]}'
        (self.bundle/'events.json').write_bytes(raw);m=self.manifest();m['files']['events.json']=hashlib.sha256(raw).hexdigest();self.write_manifest(m)
        with self.assertRaises(ArtifactError):upload(self.client,'demo',self.bundle)
        self.assertFalse(self.client.uploads)
    def test_symlink_artifact_and_cache_directory_are_rejected(self):
        original=self.bundle/'events.json';data=original.read_bytes();original.unlink();outside=self.root/'outside';outside.write_bytes(data);original.symlink_to(outside)
        with self.assertRaises(ArtifactError):upload(self.client,'demo',self.bundle)
        original.unlink();original.write_bytes(data);upload(self.client,'demo',self.bundle)
        cache=self.root/'cache';cache.mkdir();(cache/'demo').symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ArtifactError):AzureBlobArtifactStore(self.client,['demo'],cache).resolve('demo')
    def test_verify_downloads_validates_and_reads_through_real_application(self):
        upload(self.client,'demo',self.bundle);result=verify(self.client,'demo',self.root/'verify-cache')
        for key in ('fresh_download_verified','manifest_verified','hashes_verified','schemas_verified','application_ready','application_view_verified','application_ask_verified'):self.assertTrue(result[key],key)
        self.assertEqual(result['completed_transfers'],0);self.assertTrue(Path(result['cache'],'bundle.json').exists())
    def test_remote_missing_blob_and_hash_corruption_are_rejected(self):
        upload(self.client,'demo',self.bundle);del self.client.data['matches/demo/events.json']
        with self.assertRaises(AzureWorkflowError):verify(self.client,'demo',self.root/'cache')
        self.client.data['matches/demo/events.json']=b'bad'
        with self.assertRaises(ArtifactError):verify(self.client,'demo',self.root/'cache')
    def test_upload_cli_invalid_snapshot_does_not_authenticate(self):
        (self.bundle/'summary.json').unlink()
        with patch('src.artifacts.azure_workflow.client_from_environment') as client,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['upload','--recording','demo','--output',str(self.output)]),1);client.assert_not_called()
    def test_upload_and_verify_cli_use_injected_client_and_close_credentials(self):
        credential=Mock()
        with patch('src.artifacts.azure_workflow.client_from_environment',return_value=(self.client,credential)),contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(['upload','--recording','demo','--output',str(self.output)]),0)
            self.assertEqual(main(['verify','--recording','demo','--cache',str(self.root/'cli-cache')]),0)
        self.assertEqual(credential.close.call_count,2);self.assertIn('application_ask_verified',stdout.getvalue())

    def test_configuration_rejects_sas_in_urls_and_credentials_are_never_printed(self):
        with patch.dict('os.environ',{'MATCHMIND_BLOB_ACCOUNT_URL':'https://example.blob.core.windows.net/?sig=secret','MATCHMIND_BLOB_CONTAINER':'artifacts'}):
            with self.assertRaises(AzureWorkflowError):client_from_environment()
        self.assertEqual(diagnostic(SimpleNamespace(status_code=403)),'azure_forbidden_check_blob_data_role_or_network_rules')
        self.assertNotIn('secret',diagnostic(RuntimeError('secret')))

    def application(self,ids,catalog=None,cache='app-cache'):
        from fastapi.testclient import TestClient
        from src.serving.app import create_app
        return TestClient(create_app(AzureBlobArtifactStore(self.client,ids,self.root/cache),recording_catalog=catalog))

    def test_two_valid_azure_recordings_and_missing_recording_start_and_serve(self):
        for key in ('118575_clip','128058_clip'):upload(self.client,key,self.bundle)
        catalog=[dict(recording_id='118575_clip',match_id='118575',processed=True),
                 dict(recording_id='pending',match_id='pending',processed=False),
                 dict(recording_id='local_only',match_id='local',processed=True)]
        c=self.application(['118575_clip','128058_clip','118575_h1'],catalog)
        self.assertEqual(c.get('/health').status_code,200)
        self.assertEqual(c.get('/ready').json(),dict(status='ready',recordings=2))
        rows={r['recording_id']:r for r in c.get('/recordings').json()['recordings']}
        for key in ('118575_clip','128058_clip'):
            self.assertEqual(rows[key]['status'],'available');self.assertTrue(rows[key]['artifact_ready'])
            for route in ('view','summary','events'):self.assertEqual(c.get(f'/matches/{key}/{route}').status_code,200)
            answer=c.post(f'/matches/{key}/ask',json={'question':'How many completed passes were there altogether?','provider':'local'})
            self.assertEqual(answer.status_code,200);self.assertEqual(answer.json()['answer']['status'],'ok')
        missing=rows['118575_h1']
        self.assertEqual(missing['status'],'unavailable');self.assertFalse(missing['artifact_available']);self.assertFalse(missing['artifact_ready'])
        self.assertIsNone(missing['processed']);self.assertTrue(missing['configured'])
        self.assertEqual(rows['pending']['status'],'not_processed')
        self.assertTrue(rows['local_only']['processed']);self.assertFalse(rows['local_only']['artifact_ready'])
        for route in ('view','summary','events'):self.assertEqual(c.get('/matches/118575_h1/'+route).status_code,503)
        self.assertEqual(c.post('/matches/118575_h1/ask',json={'question':'How many passes?'}).status_code,503)
        self.assertEqual(c.get('/matches/unknown/view').status_code,404)

    def test_missing_azure_bundle_ignores_previously_valid_cache_and_is_not_ready(self):
        upload(self.client,'demo',self.bundle)
        self.assertEqual(self.application(['demo']).get('/ready').status_code,200)
        del self.client.data['matches/demo/bundle.json']
        c=self.application(['demo'])
        self.assertEqual(c.get('/health').status_code,200);self.assertEqual(c.get('/ready').status_code,503)
        self.assertFalse(c.get('/recordings').json()['recordings'][0]['artifact_ready'])
        self.assertEqual(c.get('/matches/demo/view').status_code,503)

    def test_corrupt_azure_bundles_are_isolated_and_never_queryable(self):
        upload(self.client,'good',self.bundle)
        for kind in ('manifest','hash','schema','missing_file'):
            with self.subTest(kind=kind):
                upload(self.client,kind,self.bundle)
                prefix=f'matches/{kind}/'
                if kind=='manifest':self.client.data[prefix+'bundle.json']=b'{}'
                elif kind=='hash':self.client.data[prefix+'events.json']=b'bad'
                elif kind=='missing_file':del self.client.data[prefix+'events.json']
                else:
                    raw=b'{"events":[{"invented":"claim"}]}'
                    self.client.data[prefix+'events.json']=raw
                    m=self.manifest();m['files']['events.json']=hashlib.sha256(raw).hexdigest()
                    self.client.data[prefix+'bundle.json']=json.dumps(m).encode()
                c=self.application(['good',kind],cache=kind+'-cache')
                self.assertEqual(c.get('/ready').status_code,200)
                row=next(r for r in c.get('/recordings').json()['recordings'] if r['recording_id']==kind)
                self.assertEqual(row['status'],'error');self.assertTrue(row['artifact_available']);self.assertFalse(row['artifact_ready'])
                self.assertEqual(c.get(f'/matches/{kind}/view').status_code,503)
                self.assertEqual(c.get('/matches/good/view').status_code,200)

    def test_local_invalid_recording_still_fails_startup(self):
        from src.serving.app import create_app
        with self.assertRaises(ArtifactError):create_app({'missing':self.root/'absent'})

    def test_azure_access_failure_is_unknown_availability_and_safe_diagnostic(self):
        self.client.download_blob=Mock(side_effect=RuntimeError('credential-secret'))
        c=self.application(['demo'])
        r=c.get('/recordings')
        row=r.json()['recordings'][0]
        self.assertEqual(row['status'],'error');self.assertIsNone(row['artifact_available'])
        self.assertEqual(row['diagnostic'],'azure_bundle_download_failed')
        self.assertNotIn('credential-secret',r.text)
        self.assertEqual(c.get('/ready').status_code,503)
