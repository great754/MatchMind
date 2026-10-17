import hashlib
import json
from pathlib import Path
import re
from typing import Protocol

BUNDLE_FILES=('possession.csv','possession_attribution.json','formation_windows.json','pressure_levels.json','review_frames.json','summary.json','events.json','rolling_context.json','commentary.json','tactical_states.json','tactical_events.json','tactical_summary.json','transfer_network.json','pre_event_evidence.json','player_summary.csv','transfers.csv','measurement_definitions.json','report.html','motion.png','passes.png','tactical_pressure.png','tactical_shape.png')
class ArtifactError(ValueError):pass

class AzureArtifactError(ArtifactError):
    """Safe per-recording diagnostics; never expose SDK exceptions or credentials."""
    def __init__(self,diagnostic,bundle_available=None):
        super().__init__('Azure artifact snapshot unavailable or invalid')
        self.diagnostic=diagnostic
        self.bundle_available=bundle_available

def reject_nonfinite(value):raise ValueError('Nonfinite artifact value')

def recording_key(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):raise ArtifactError('Invalid recording ID')
    return value

class ArtifactStore(Protocol):
    def ids(self)->list[str]: ...
    def resolve(self,recording_id:str)->Path: ...


def validate_bundle(directory):
    directory=Path(directory).resolve()
    try:
        for name in BUNDLE_FILES:
            p=directory/name
            if p.exists() and not p.resolve().is_relative_to(directory):raise ArtifactError('Artifact symlink leaves bundle')
            if p.is_file() and p.suffix=='.json':
                json.loads(p.read_text(),parse_constant=reject_nonfinite)
        summary=json.loads((directory/'summary.json').read_text())
        events=json.loads((directory/'events.json').read_text())
        if not isinstance(summary,dict) or not isinstance(events,dict) or not isinstance(events.get('events'),list):raise ValueError()
        from src.query import MatchMindTools
        tools=MatchMindTools(directory)
        from src.serving.review import Replay,Ownership,FormationWindow
        p=directory/'review_frames.json'
        if p.is_file():Replay.model_validate_json(p.read_text())
        for row in tools.ownership:Ownership.model_validate(row)
        for row in tools.formations:FormationWindow.model_validate(row)
        p=directory/'pressure_levels.json'
        if p.is_file():
            from src.tactics.schema import Pressure
            rows=json.loads(p.read_text())
            if not isinstance(rows,list):raise ValueError()
            for row in rows:
                if not isinstance(row,dict) or row.get('level') not in ('low','moderate','high','unknown') or not isinstance(row.get('timestamp'),(int,float)) or row['timestamp']<0:raise ValueError()
                # Older level-only exports remain readable; complete exports use the closed pressure schema.
                Pressure.model_validate({k:v for k,v in row.items() if k not in ('level','timestamp')})
    except Exception:raise ArtifactError('Missing or malformed artifact bundle') from None
    return directory

class LocalArtifactStore:
    def __init__(self,registry):self.registry={recording_key(k):Path(v).resolve() for k,v in registry.items()}
    def ids(self):return sorted(self.registry)
    def resolve(self,recording_id):
        key=recording_key(recording_id)
        if key not in self.registry:raise KeyError(key)
        return validate_bundle(self.registry[key])

class AzureBlobArtifactStore:
    """Download immutable, hash-verified snapshots at startup, never per query.

    client is an azure.storage.blob.ContainerClient; inject a fake in tests.
    A trusted per-recording manifest maps allowlisted filenames to SHA256 hashes.
    """
    def __init__(self,client,recordings,cache):
        self.client=client;self.recordings=[recording_key(k) for k in recordings];self.cache=Path(cache).resolve()
    def ids(self):return sorted(self.recordings)
    def resolve(self,recording_id):
        key=recording_key(recording_id)
        if key not in self.recordings:raise KeyError(key)
        destination=self.cache/key
        bundle_available=None;stage='azure_cache_unavailable_or_unsafe'
        try:
            destination.mkdir(parents=True,exist_ok=True)
            if destination.is_symlink() or not destination.resolve().is_relative_to(self.cache):raise ArtifactError('Cache path leaves root')
            stage='azure_bundle_download_failed'
            raw=self.client.download_blob(f'matches/{key}/bundle.json').readall()
            bundle_available=True;stage='azure_manifest_invalid'
            if len(raw)>65536:raise ValueError()
            from .manifest import parse_manifest
            manifest=parse_manifest(raw);files=manifest['files']
            stage='azure_artifact_download_or_hash_failed'
            for name,digest in files.items():
                if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError()
                p=destination/name
                if p.is_symlink() or not p.resolve().is_relative_to(destination):raise ValueError()
                if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==digest:continue
                data=self.client.download_blob(f'matches/{key}/{name}').readall()
                if len(data)>100_000_000 or hashlib.sha256(data).hexdigest()!=digest:raise ValueError()
                temporary=p.with_suffix(p.suffix+'.tmp')
                if not temporary.resolve().is_relative_to(destination):raise ValueError()
                temporary.write_bytes(data);temporary.replace(p)
            # Remove stale optional artifacts absent from this immutable snapshot.
            for name in set(BUNDLE_FILES)-set(files):
                p=destination/name
                if p.exists():p.unlink()
            stage='azure_artifact_schema_invalid'
            validated=validate_bundle(destination)
            stage='azure_cache_write_failed'
            manifest_path=destination/'bundle.json'
            if manifest_path.is_symlink():raise ArtifactError('Unsafe cache manifest path')
            manifest_path.write_bytes(raw)
            return validated
        except Exception as error:
            if bundle_available is None and getattr(error,'status_code',None)==404:
                bundle_available=False;stage='azure_bundle_missing'
            raise AzureArtifactError(stage,bundle_available) from None
