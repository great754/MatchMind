"""Entra-only, allowlisted immutable uploads and application-level read-back."""
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from .store import ArtifactError,AzureBlobArtifactStore,recording_key
from .manifest import parse_manifest,validate_snapshot


class AzureWorkflowError(ArtifactError):
    """Only a fixed diagnostic code crosses the CLI boundary."""


def client_from_environment():
    account=os.environ.get('MATCHMIND_BLOB_ACCOUNT_URL','');container=os.environ.get('MATCHMIND_BLOB_CONTAINER','')
    url=urlsplit(account)
    if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('','/'):
        raise AzureWorkflowError('invalid_blob_configuration')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]',container) or '--' in container:
        raise AzureWorkflowError('invalid_blob_configuration')
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=True)
    return ContainerClient(account_url=account,container_name=container,credential=credential,connection_timeout=10,read_timeout=60,retry_total=2),credential


def _read_if_present(client,name):
    try:return client.download_blob(name).readall()
    except Exception as error:
        if getattr(error,'status_code',None)==404:return None
        raise


def check_private_container(client):
    properties=client.get_container_properties()
    if properties.get('public_access') is not None:raise AzureWorkflowError('container_must_be_private')


def upload(client,recording,directory):
    """Create-only block blobs. A visible manifest is the commit marker.

    A repeat upload of identical bytes is safe. Different content requires a new
    recording/version prefix rather than silently replacing an active snapshot.
    """
    key=recording_key(recording);directory=Path(directory);manifest=validate_snapshot(directory)
    prefix=f'matches/{key}/';raw=(directory/'bundle.json').read_bytes()
    check_private_container(client)
    existing=_read_if_present(client,prefix+'bundle.json')
    if existing is not None and parse_manifest(existing)!=manifest:raise AzureWorkflowError('remote_manifest_conflict')
    # Preflight every remote file before writing anything, including orphaned blobs.
    payloads={name:(directory/name).read_bytes() for name in manifest['files']}
    if any(hashlib.sha256(data).hexdigest()!=manifest['files'][name] for name,data in payloads.items()):raise ArtifactError('Artifact changed after validation')
    missing=[]
    for name,digest in manifest['files'].items():
        data=_read_if_present(client,prefix+name)
        if data is None:missing.append(name)
        elif hashlib.sha256(data).hexdigest()!=digest:raise AzureWorkflowError('remote_artifact_conflict')
    for name in missing:
        client.upload_blob(name=prefix+name,data=payloads[name],blob_type='BlockBlob',overwrite=False,validate_content=True)
    # Never publish a manifest for a partially uploaded recording.
    if existing is None:client.upload_blob(name=prefix+'bundle.json',data=raw,blob_type='BlockBlob',overwrite=False,validate_content=True)
    return dict(recording_id=key,prefix=prefix,files=len(manifest['files']),uploaded=len(missing),manifest_published=True,private_container_verified=True)


def verify(client,recording,cache):
    """List, download, validate and start the real application with this store."""
    key=recording_key(recording);prefix=f'matches/{key}/'
    check_private_container(client)
    raw=client.download_blob(prefix+'bundle.json').readall();manifest=parse_manifest(raw)
    listed={b.name for b in client.list_blobs(name_starts_with=prefix)}
    expected={prefix+n for n in manifest['files']}|{prefix+'bundle.json'}
    if not expected<=listed:raise AzureWorkflowError('remote_blobs_missing')
    # Always exercise fresh network reads, regardless of older cache contents.
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='verify-',dir=cache) as fresh:
        store=AzureBlobArtifactStore(client,[key],fresh);directory=store.resolve(key)
        validate_snapshot(directory)
        from src.serving.app import create_app
        from src.query import MatchMindTools
        # The factory resolves the Azure store itself, rather than a local alias.
        app=create_app(store,default_provider='local')
        # Exercise ASGI without requiring a listening socket or test-only httpx dependency.
        import asyncio
        async def request(path,method='GET',payload=None):
            messages=[]
            received=False
            async def receive():
                nonlocal received
                if not received:
                    received=True
                    return {'type':'http.request','body':json.dumps(payload).encode() if payload is not None else b'','more_body':False}
                await asyncio.Event().wait()
            async def send(message):messages.append(message)
            await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':method,'scheme':'http','path':path,'raw_path':path.encode(),'query_string':b'','root_path':'','headers':[(b'content-type',b'application/json')] if payload is not None else [],'server':('verify',80),'client':('127.0.0.1',1)},receive,send)
            start=next(m for m in messages if m['type']=='http.response.start')
            body=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
            if start['status']!=200:raise AzureWorkflowError('application_read_failed')
            return json.loads(body)
        ready=asyncio.run(request('/ready'));view=asyncio.run(request(f'/matches/{key}/view'))
        answered=asyncio.run(request(f'/matches/{key}/ask','POST',{'question':'How many passes were there altogether?','provider':'local'}))
        if answered.get('answer',{}).get('status')!='ok' or answered.get('conversation',{}).get('recording_id')!=key:raise AzureWorkflowError('application_read_failed')
        if ready.get('status')!='ready' or view.get('recording_id')!=key:raise AzureWorkflowError('application_read_failed')
        tools=MatchMindTools(directory);facts=tools.get_confirmed_pass_share().facts
        total=next((f.value for f in facts if f.field=='total_confirmed_passes'),None)
        # Fresh directory is removed on exit; maintain a normal validated persistent cache too.
        persisted=AzureBlobArtifactStore(client,[key],cache/'recordings').resolve(key)
        validate_snapshot(persisted)
        return dict(recording_id=key,prefix=prefix,files=len(manifest['files']),listed_blobs=len(listed),
            unexpected_blobs=sorted(listed-expected),private_container_verified=True,fresh_download_verified=True,manifest_verified=True,hashes_verified=True,
            schemas_verified=True,application_ready=True,application_view_verified=True,application_ask_verified=True,completed_transfers=total,cache=str(persisted))


def diagnostic(error):
    if isinstance(error,AzureWorkflowError):return str(error)
    if isinstance(error,ArtifactError):return 'artifact_validation_failed'
    status=getattr(error,'status_code',None)
    if status==403:return 'azure_forbidden_check_blob_data_role_or_network_rules'
    if status==401:return 'azure_authentication_failed'
    if status==404:return 'azure_container_or_blob_not_found'
    return 'azure_operation_failed'
