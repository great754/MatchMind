"""Grounded paraphrases must retrieve evidence before declining a question."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from src.query import MatchMindTools,query_question
from src.providers.agent import ask
from src.query.planning import semantic_plan
from tests.helpers import write_query_artifacts

QUESTIONS=[
 'How many passes did both teams make?',
 'How many completed passes did each side have?',
 'Who passed more?',
 'Which team completed the most passes?',
 'How many passes were there altogether?',
]

class ParaphraseTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);write_query_artifacts(self.path);self.tools=MatchMindTools(self.path)
        self.tools.transfers=[dict(outcome='completed',sender_team='left')]*16+[dict(outcome='completed',sender_team='right')]*13+[
            dict(outcome='intercepted',sender_team='right'),dict(outcome='unresolved',sender_team='left')]
    def test_required_pass_questions_share_identical_canonical_evidence(self):
        expected=None
        for question in QUESTIONS:
            with self.subTest(question=question):
                answer=ask(self.tools,question).answer;self.assertEqual(answer.status,'ok')
                facts={(f.record_id,f.field):f.value for f in answer.facts}
                self.assertEqual(facts['left','confirmed_passes'],16);self.assertEqual(facts['right','confirmed_passes'],13)
                self.assertEqual(facts['total','total_confirmed_passes'],29)
                self.assertEqual(facts['comparison','team_with_more_confirmed_passes'],'left')
                self.assertTrue(all(f.source.startswith('transfers.csv') for f in answer.facts))
                ids=answer.supporting_fact_ids
                if expected is None:expected=ids
                self.assertEqual(ids,expected)
                self.assertIn('confirmed completed observable transfers',answer.statements[0].text)
    def test_totals_and_comparisons_are_prominent_in_preview(self):
        self.assertIn('29',query_question(self.tools,QUESTIONS[-1]).statements[1].text)
        self.assertIn('Home side completed more',query_question(self.tools,QUESTIONS[2]).statements[1].text)
    def test_successful_passes_ball_transfers_and_pass_share(self):
        for q in ['Count successful passes for each side.','How many ball transfers were successful?','Compare possession and completed passes.','What was the pass share?']:
            with self.subTest(q=q):
                a=query_question(self.tools,q);self.assertEqual(a.status,'ok')
                self.assertEqual(next(f.value for f in a.facts if f.field=='confirmed_passes' and f.record_id=='left'),16)
    def test_two_explicit_teams_do_not_filter_to_first(self):
        a=query_question(self.tools,'Compare completed passes by Team 1 and Team 2.')
        self.assertEqual({f.record_id for f in a.facts if f.field=='confirmed_passes'},{'left','right'})
    def test_team_filter_and_rankings(self):
        a=query_question(self.tools,'How many successful passes did Team 2 make?')
        self.assertEqual([(f.record_id,f.value) for f in a.facts if f.field=='confirmed_passes'],[('right',13)])
        a=query_question(self.tools,'Rank teams by completed passes.')
        self.assertEqual(next(f.value for f in a.facts if f.field=='left_pass_rank'),1)
    def test_synonym_tool_plans(self):
        for a,b in [('Who was fastest?','Who had the highest speed?'),('When was Team 1 under pressure?','When was Team 1 pressed?'),('What formation was detected?','What shape did the teams use?')]:
            self.assertEqual(semantic_plan(a),semantic_plan(b))
        self.assertEqual(semantic_plan('Show the top 3 fastest players.')[0][1]['limit'],3)
    def test_multi_tool_retrieval_works_without_external_provider(self):
        a=ask(self.tools,'Compare completed passes and highest speed.').answer
        self.assertEqual(a.status,'ok');fields={f.field for f in a.facts}
        self.assertIn('confirmed_passes',fields);self.assertIn('max_speed_kmh',fields)
    def test_provider_failure_cannot_hide_known_pass_counts(self):
        p=Mock();p.plan.side_effect=RuntimeError('provider unavailable')
        a=ask(self.tools,QUESTIONS[0],provider=p)
        self.assertTrue(a.fallback);self.assertEqual(a.answer.status,'ok')
        self.assertTrue(any(f.field=='total_confirmed_passes' and f.value==29 for f in a.answer.facts))
    def test_zero_and_tied_counts_are_facts_not_unknown(self):
        self.tools.transfers=[];a=query_question(self.tools,QUESTIONS[-1]);self.assertEqual(a.status,'ok')
        self.assertEqual(next(f.value for f in a.facts if f.field=='total_confirmed_passes'),0)
        self.tools.transfers=[dict(outcome='completed',sender_team=t) for t in ('left','right')]
        a=query_question(self.tools,'Who passed more?');self.assertIn('same number',a.statements[1].text)
    def test_each_team_speed_and_pressure_use_separate_team_evidence(self):
        self.assertEqual(semantic_plan('Which team had the highest speed?')[0][0],'get_team_speed_peaks')
        self.assertEqual(semantic_plan('Compare pressure on both sides.'),[('get_pressure_peaks',{'team':'left'}),('get_pressure_peaks',{'team':'right'})])
        self.tools.players=[dict(player_id=1,team='left',max_speed_kmh=30),dict(player_id=2,team='left',max_speed_kmh=25),dict(player_id=3,team='right',max_speed_kmh=28)]
        result=query_question(self.tools,'What was the highest speed for each side?')
        self.assertEqual({f.value for f in result.facts if f.field=='max_speed_kmh'},{30,28})

    def test_canonical_events_are_checked_when_transfer_csv_is_missing(self):
        (self.path/'transfers.csv').unlink()
        answer=query_question(MatchMindTools(self.path),'How many passes were there altogether?')
        self.assertEqual(answer.status,'ok')
        self.assertEqual(next(f.value for f in answer.facts if f.field=='total_confirmed_passes'),1)
        self.assertTrue(all(f.source.startswith('events.json') for f in answer.facts))

    def test_missing_transfer_source_is_not_reported_as_zero(self):
        (self.path/'transfers.csv').unlink()
        (self.path/'events.json').unlink()
        answer=query_question(MatchMindTools(self.path),'How many passes were there altogether?')
        self.assertEqual(answer.status,'unknown');self.assertFalse(answer.facts)
        self.assertIn('enough evidence',answer.statements[0].text)

    def test_genuinely_unsupported_questions_remain_explicit(self):
        for q in ['What did the coach intend?','What is the referee salary?','What will the weather be tomorrow?','Who will win?']:
            a=ask(self.tools,q).answer;self.assertEqual(a.status,'unsupported',q)
            self.assertFalse(a.facts);self.assertIn('enough evidence',a.statements[0].text)
