"""Shared closed manifest validation for local exports and Azure snapshots."""
import hashlib
import json
from pathlib import Path
import re
from .store import ArtifactError,BUNDLE_FILES,validate_bundle

REQUIRED_FILES=frozenset(('summary.json','events.json'))
MAX_FILE_BYTES=100_000_000


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate manifest key')
        result[key]=value
    return result


def parse_manifest(raw):
    try:
        if len(raw)>65536:raise ValueError()
        manifest=json.loads(raw,object_pairs_hook=_object)
        if not isinstance(manifest,dict) or set(manifest)!={'schema_version','files'} or manifest['schema_version']!='1.0':raise ValueError()
        files=manifest['files']
        if not isinstance(files,dict) or not REQUIRED_FILES<=set(files) or set(files)-set(BUNDLE_FILES):raise ValueError()
        if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in files.values()):raise ValueError()
        return manifest
    except (ValueError,TypeError,UnicodeError):raise ArtifactError('Invalid artifact manifest') from None


def validate_snapshot(directory):
    """Ignore unrelated files; only a validated manifest may select upload bytes."""
    directory=Path(directory).resolve();p=directory/'bundle.json'
    if p.is_symlink():raise ArtifactError('Manifest symlinks are not accepted')
    try:raw=p.read_bytes()
    except OSError:raise ArtifactError('Artifact manifest missing') from None
    manifest=parse_manifest(raw)
    for name,digest in manifest['files'].items():
        p=directory/name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(directory):raise ArtifactError('Manifest artifact missing or unsafe')
        if p.stat().st_size>MAX_FILE_BYTES or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ArtifactError('Artifact hash mismatch or size limit exceeded')
    # validate_bundle also checks optional local files; exports contain only selected canonical names.
    validate_bundle(directory)
    return manifest
