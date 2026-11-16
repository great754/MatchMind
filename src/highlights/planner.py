"""Deterministic validation boundary between language and evidence."""
from .schema import HighlightQueryPlan
class HighlightQueryPlanner:
    def validate(self,plan):return HighlightQueryPlan.model_validate(plan)
    def resolve(self,index,plan):return index.retrieve(self.validate(plan))
