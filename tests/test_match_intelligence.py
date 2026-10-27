import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from src.query.intelligence import build_intelligence
from src.query.tools import MatchMindTools
from src.query.schema import QueryResult
from tests.helpers import write_query_artifacts

class MatchIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);write_query_artifacts(self.path)
        self.tools=MatchMindTools(self.path)
    def test_determinism_modes_and_provenance(self):
        a=build_intelligence(self.tools,'a');b=build_intelligence(self.tools,'a')
        self.assertEqual(a,b);self.assertTrue(a.insights)
        for insight in a.insights:
            fan=insight.presentations['fan']
            for answer in insight.presentations.values():
                self.assertEqual(answer.facts,fan.facts)
                self.assertEqual(answer.supporting_fact_ids,fan.supporting_fact_ids)
                self.assertTrue(all(f.source and '#record=' in f.source for f in answer.facts))
                self.assertTrue(all(set(s.fact_ids)<=set(answer.supporting_fact_ids) for s in answer.statements))
    def test_recording_isolation(self):
        a=build_intelligence(self.tools,'a');b=build_intelligence(self.tools,'b')
        self.assertTrue({f.fact_id for i in a.insights for f in i.presentations['fan'].facts}.isdisjoint({f.fact_id for i in b.insights for f in i.presentations['fan'].facts}))
    def test_missing_evidence_does_not_make_insights(self):
        tools=Mock()
        for method in ('get_confirmed_pass_share','get_attributed_control_summary','get_fastest_players','get_distance_leaders','get_difficult_transfers','get_transfer_network','get_pressure_peaks','get_match_summary'):
            getattr(tools,method).return_value=QueryResult(question_type='match_summary',status='unknown')
        self.assertEqual(build_intelligence(tools,'empty').insights,[])
    def test_summary_ask_uses_closed_existing_tool_registry(self):
        from src.query.planning import retrieve_question
        result=retrieve_question(self.tools,'Summarize this match')
        self.assertEqual(result.status,'ok')
        self.assertTrue(result.facts)
        self.assertTrue(all(f.source.split('#')[0] in ('possession.csv','summary.json','player_summary.csv','transfers.csv','events.json','possession_attribution.json','tactical_states.json') for f in result.facts))
        self.assertIsNone(retrieve_question(self.tools,'Summarize why the coach caused this win'))
