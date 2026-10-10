"""Closed-tool filtering, known answers, provenance, unknowns and safe provider selection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pandas as pd
from pydantic import ValidationError
from src.query import MatchMindTools,query_question,explain_event
from src.query.answers import answer_result
from src.query.schema import Fact,QueryResult,Selection
from src.intelligence.schema import make_event,EventType,EventMetrics
from test_tactics import tactical_fixture
from src.tactics.engine import analyze_tactics
from src.analytics.config import AnalyticsConfig

class QueryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        recording,result=tactical_fixture(seconds=2)
        states=analyze_tactics(recording,result,AnalyticsConfig(),events=[])['states']
        events=[make_event('test',frame=25+i*10,fps=25,type=EventType.PASS_COMPLETED,source='transfer',significance=80,group_id=str(i),
            team='left',player_id=2,actor_id=101,receiver_id=3+i,metrics=EventMetrics(start_frame=20+i*10,end_frame=25+i*10,
                distance_m=10+i,difficulty_score_0_100=30+i*10,outcome='completed')) for i in range(2)]
        self.events=events
        summary=dict(duration_seconds=2,limitations=['Test uncertainty.'],team_possession={'left':dict(controlled_seconds=1.,fraction_of_all_frames=.5,fraction_of_controlled_frames=1.),'right':dict(controlled_seconds=0.,fraction_of_all_frames=0.,fraction_of_controlled_frames=0.)})
        for name,value in [('summary.json',summary),('events.json',dict(events=[e.model_dump(mode='json',exclude_none=True) for e in events],player_labels={})),
            ('tactical_states.json',dict(states=[s.model_dump(mode='json') for s in states])),
            ('pre_event_evidence.json',dict(items=[dict(event_id=events[0].event_id,event_start_frame=20,sample_frame=15,sample_age_s=.2,pre_event_context=states[3].model_dump(mode='json'),interpretation='Not causal.')]))]:
            (self.path/name).write_text(json.dumps(value))
        pd.DataFrame([dict(player_id=101,mot_id=2,team='left',max_speed_kmh=20,peak_speed_seconds=1.,valid_motion_fraction=.9),
            dict(player_id=102,mot_id=3,team='left',max_speed_kmh=30,peak_speed_seconds=1.4,valid_motion_fraction=1.)]).to_csv(self.path/'player_summary.csv',index=False)
        pd.DataFrame([dict(transfer_id=i,start_frame=e.metrics.start_frame,sender_id=101,receiver_id=102+i,sender_team='left',outcome='completed',
            distance_m=e.metrics.distance_m,start_seconds=e.metrics.start_frame/25,difficulty_score_0_100=e.metrics.difficulty_score_0_100) for i,e in enumerate(events)]).to_csv(self.path/'transfers.csv',index=False)
        self.tools=MatchMindTools(self.path)

    def test_filter_type_team_and_time(self):
        self.assertFalse(self.tools.get_events('SHOT_CANDIDATE').facts)
        self.assertFalse(self.tools.get_events(team='right').facts)
        facts=self.tools.get_events(start_time=1.2,end_time=1.5).facts
        self.assertEqual({f.event_id for f in facts},{self.events[1].event_id})
        with self.assertRaises(ValueError):self.tools.get_events(start_time=5,end_time=2)

    def test_fastest_and_player_summary(self):
        r=self.tools.get_fastest_players(1)
        self.assertEqual(next(f.value for f in r.facts if f.field=='player_id'),102)
        self.assertEqual(r.facts[0].timestamp,1.4)
        self.assertEqual(self.tools.get_player_summary(101).status,'ok')
        self.assertEqual(self.tools.get_player_summary(999).status,'unknown')
        with self.assertRaises(ValueError):self.tools.get_fastest_players(1000)

    def test_difficult_completed_ranking(self):
        result=self.tools.get_difficult_transfers(limit=1)
        self.assertEqual({f.event_id for f in result.facts},{self.events[1].event_id})
        self.assertIn('heuristic',' '.join(result.limitations))

    def test_tactical_lookup_at_or_before(self):
        r=self.tools.get_tactical_state(.31)
        self.assertEqual({f.timestamp for f in r.facts},{.2})
        self.assertEqual(self.tools.get_tactical_state(10).status,'unknown')

    def test_strict_pre_event_evidence(self):
        r=self.tools.get_pre_event_evidence(self.events[0].event_id)
        self.assertEqual(next(f.value for f in r.facts if f.field=='sample_frame'),15)
        self.assertTrue(all(f.timestamp is None or f.timestamp<.8 for f in r.facts))
        self.assertEqual(self.tools.get_pre_event_evidence('missing').status,'unknown')

    def test_network_and_no_synthetic_pressure_sequence(self):
        r=self.tools.get_transfer_network('left',1.1,1.5)
        self.assertEqual(next(f.value for f in r.facts if f.field=='transfer_count'),1)
        self.assertEqual(self.tools.get_pressure_sequences().status,'unknown')

    def test_unknown_questions(self):
        for question in ('Was he tired?','What did the coach say?','What is the score?','What is his name?','Did he intentionally shoot?'):
            r=query_question(self.tools,question)
            self.assertEqual(r.status,'unsupported',question)
            self.assertIn('cannot determine',r.statements[0].text)

    def test_modes_share_facts(self):
        answers=[query_question(self.tools,'Who was fastest?',mode) for mode in ('beginner','fan','analyst')]
        self.assertEqual(answers[0].supporting_fact_ids,answers[2].supporting_fact_ids)
        self.assertEqual(answers[0].statements[0].fact_ids,answers[1].statements[0].fact_ids)
        self.assertIn('Test uncertainty.',answers[2].limitations)

    def test_numeric_claims_trace_to_fact_fields(self):
        result=self.tools.get_fastest_players(1);answer=answer_result(result)
        ids={f.fact_id for f in result.facts}
        self.assertTrue(all(set(s.fact_ids)<=ids for s in answer.statements))
        self.assertIn('30',answer.statements[0].text)
        self.assertIn('player_summary.csv',result.facts[0].source)
        self.assertNotIn('player name',answer.statements[0].text)

    def test_closed_schema_and_provider_plan(self):
        with self.assertRaises(ValidationError):QueryResult(question_type='unsupported',invented_score=3)
        with self.assertRaises(ValidationError):Fact(fact_id='x',record_id='x',field='x',value=float('nan'),source='x')
        class Bad:
            def select(self,*args):return Selection(statement_ids=[999])
        with self.assertRaises(ValueError):answer_result(self.tools.get_fastest_players(),selector=Bad())
        class Text:
            def select(self,*args):return dict(statement_ids=[0],text='Invented goal')
        with self.assertRaises(ValidationError):answer_result(self.tools.get_fastest_players(),selector=Text())

    def test_explanation_limits_and_validation(self):
        answer=explain_event(self.tools,self.events[0].event_id)
        self.assertIn('not proof',' '.join(answer.limitations))
        self.assertTrue(any('validation state: estimated' in s.text for s in answer.statements))
        self.assertFalse(any('forced' in s.text or 'intentionally' in s.text for s in answer.statements))

    def test_no_invented_formation(self):
        r=query_question(self.tools,'What formation was detected?')
        self.assertTrue(all('estimate: unknown' in s.text for s in r.statements))

    def test_provider_independent_dispatch(self):
        from src.query import dispatch,tool_specifications
        self.assertEqual(dispatch(self.tools,'get_fastest_players',{'limit':1}).question_type,'fastest_players')
        self.assertTrue(all(spec['input_schema'].get('additionalProperties') is False for spec in tool_specifications()))
        with self.assertRaises(ValidationError):dispatch(self.tools,'get_events',{'sql':'SELECT *'})
        with self.assertRaises(ValueError):dispatch(self.tools,'read_file',{'path':'.env'})
        with self.assertRaises(ValidationError):dispatch(self.tools,'get_tactical_state',{'timestamp':float('inf')})

    def test_sequence_strictly_precedes_release(self):
        result=self.tools.get_event_sequence(self.events[1].event_id)
        self.assertEqual({f.event_id for f in result.facts},{self.events[0].event_id})
        self.assertTrue(all(f.timestamp<1.2 for f in result.facts))
