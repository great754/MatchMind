"""Provider contract shared by hosted and local editorial selectors."""
from typing import Protocol
import json

class CommentaryError(RuntimeError):
    """Safe user-facing failure; never include HTTP bodies or credentials."""

class CommentaryProvider(Protocol):
    @property
    def configuration(self) -> dict: ...
    def generate(self, request: dict) -> str: ...

class TemplateProvider:
    """Explicitly non-AI preview of the grounded statement catalogue."""
    configuration = {'provider':'template','version':'1'}
    def generate(self, request: dict) -> str:
        return json.dumps({'sentence_ids':list(request['sentences'])[:1]})
