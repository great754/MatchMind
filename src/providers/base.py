"""Shared protocols, without SDK dependencies or analytics imports."""
from typing import Protocol
from pydantic import Field
from src.query.schema import Closed,Selection,Statement,QueryResult
from src.intelligence.commentary.base import CommentaryProvider

class ProviderError(RuntimeError):
    """Safe public provider failure, with no upstream body or credential."""

class ToolCall(Closed):
    name:str
    arguments:dict=Field(default_factory=dict)

class ToolPlan(Closed):
    calls:list[ToolCall]=Field(min_length=1,max_length=3)

class MatchQueryProvider(Protocol):
    @property
    def configuration(self)->dict: ...
    def plan(self,question:str,tool_specs:list[dict])->ToolPlan: ...
    def select(self,question:str,mode:str,results:list[QueryResult],statements:list[Statement])->Selection: ...
