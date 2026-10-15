import hashlib
import json
from pathlib import Path
import re
from typing import Protocol

BUNDLE_FILES=('summary.json','events.json','rolling_context.json','commentary.json','tactical_states.json','tactical_events.json','tactical_summary.json','transfer_network.json','pre_event_evidence.json','player_summary.csv','transfers.csv','measurement_definitions.json','report.html','motion.png','passes.png','tactical_pressure.png','tactical_shape.png')
class ArtifactError(ValueError):pass

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
        MatchMindTools(directory) # Closed event/tactical schemas and CSV parsing; no calculations.
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
        destination=self.cache/key;destination.mkdir(parents=True,exist_ok=True)
        if not destination.resolve().is_relative_to(self.cache):raise ArtifactError('Cache path leaves root')
        try:
            raw=self.client.download_blob(f'matches/{key}/bundle.json').readall()
            if len(raw)>65536:raise ValueError()
            manifest=json.loads(raw)
            if set(manifest)!= {'schema_version','files'} or manifest['schema_version']!='1.0':raise ValueError()
            files=manifest['files']
            if not {'summary.json','events.json'}<=set(files) or set(files)-set(BUNDLE_FILES):raise ValueError()
            for name,digest in files.items():
                if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError()
                p=destination/name
                if not p.resolve().is_relative_to(destination):raise ValueError()
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
            return validate_bundle(destination)
        except Exception:raise ArtifactError('Azure artifact snapshot unavailable or invalid') from None
