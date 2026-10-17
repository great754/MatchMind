"""Regression tests for presentation semantics, independent of detectors."""
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import Mock
from src.analytics.possession import attribute_possession
from src.tactics.review import formation_windows,pressure_level
from src.tactics.schema import Pressure,Formation
from src.query import MatchMindTools
from src.providers.agent import ask
from src.query.language import sentences
from src.query.schema import Fact
from tests.helpers import write_query_artifacts


def row(frame, state='free', player=None, team=None, candidate=None):
    return dict(frame=frame,state=state,possessing_player_id=player,possessing_team=team,candidate_player_id=candidate)

class PossessionContinuityTests(unittest.TestCase):
    def test_initial_unknown_and_established_control(self):
        result=attribute_possession([row(0),row(1,'controlled',101,'left')],25)
        self.assertIsNone(result[0]['player_id']);self.assertEqual(result[1]['player_id'],101)
        self.assertEqual(result[1]['attribution'],'controlled')
    def test_departure_flight_and_completed_pass(self):
        result=attribute_possession([row(0,'controlled',101,'left'),row(1),row(40),row(41,'controlled',102,'left')],25)
        self.assertEqual([r['player_id'] for r in result],[101,101,101,102])
    def test_interception_switches_only_when_opponent_controls(self):
        result=attribute_possession([row(0,'controlled',101,'left'),row(1,'candidate',candidate=201),row(2,'controlled',201,'right')],25)
        self.assertEqual([r['team'] for r in result],['left','left','right'])
    def test_occlusion_and_noisy_nearest_players_do_not_switch(self):
        result=attribute_possession([row(0,'controlled',101,'left'),row(1,'unknown'),row(2,'contested'),row(3,'candidate',candidate=201),row(4,'candidate',candidate=102)],25)
        self.assertEqual({r['player_id'] for r in result},{101})
    def test_unresolved_timeout_is_not_extended_by_candidates(self):
        result=attribute_possession([row(0,'controlled',101,'left'),row(500,'candidate',candidate=201),row(501)],25)
        self.assertEqual(result[1]['player_id'],101);self.assertIsNone(result[2]['player_id'])
    def test_exported_summary_separates_attributed_possession_from_transfer_share(self):
        import pandas as pd
        from src.analytics.pipeline import team_possession_summary
        possession=pd.DataFrame([row(0,'controlled',101,'left'),row(1),row(2,'controlled',201,'right')])
        transfers=pd.DataFrame([dict(outcome='completed',sender_team='right')])
        summary=team_possession_summary(possession,transfers,25)
        self.assertAlmostEqual(summary['left']['attributed_share'],2/3)
        self.assertEqual(summary['left']['confirmed_pass_share'],0)
        self.assertEqual(summary['right']['confirmed_pass_share'],1)

    def test_out_of_play_clears_and_reacquisition_restores(self):
        result=attribute_possession([row(0,'controlled',101,'left'),row(1,'out_of_play'),row(2),row(3,'controlled',201,'right')],25)
        self.assertEqual([r['player_id'] for r in result],[101,None,None,201])

class FormationWindowTests(unittest.TestCase):
    def states(self,estimates):
        return [SimpleNamespace(timestamp=i,left=SimpleNamespace(formation=Formation(estimate=e,fit_quality='clear' if e!='unknown' else 'unavailable')),right=SimpleNamespace(formation=Formation(reason='poor_fit'))) for i,e in enumerate(estimates)]
    def test_window_dominance_is_stable_despite_short_changes(self):
        result=formation_windows(self.states(['4-3-3']*180+['4-4-2']*30+['unknown']*30))
        self.assertEqual(len(result),2);self.assertEqual(result[0]['estimate'],'4-3-3');self.assertEqual(result[1]['estimate'],'unknown')
    def test_insufficient_or_ambiguous_evidence_stays_unknown(self):
        self.assertEqual(formation_windows(self.states(['4-3-3']*40+['unknown']*200))[0]['estimate'],'unknown')
        self.assertEqual(formation_windows(self.states(['4-3-3']*120+['4-4-2']*120))[0]['estimate'],'unknown')
    def test_new_window_is_independent(self):
        result=formation_windows(self.states(['4-3-3']*240+['4-4-2']*240))
        self.assertEqual([r['estimate'] for r in result if r['team']=='left'],['4-3-3','4-4-2'])

class PressureLevelTests(unittest.TestCase):
    def test_levels_do_not_change_sustained_event(self):
        for near,local,outer,closing,level in [(0,0,0,0,'low'),(0,1,1,0,'moderate'),(1,2,2,1,'high'),(2,2,2,0,'high')]:
            p=Pressure(available_defenders=10,defenders_within_near=near,defenders_within_local=local,defenders_within_outer=outer,closing_defenders=closing)
            self.assertEqual(pressure_level(p),level);self.assertFalse(p.sustained_press);self.assertFalse(p.press_candidate)
        self.assertEqual(pressure_level(Pressure()),'unknown')

class ReviewPressureTests(unittest.TestCase):
    def test_retained_possessor_uses_current_geometry_without_claiming_sustained_press(self):
        from src.serving.review import Replay
        from src.tactics.review import review_pressure
        replay=Replay(fps=25,frames=10,stride=5,ids=list(range(1,11)),teams=['left']+['right']*9,side_colors={},
            positions=[[[20,20]]+[[21,20],[22,20]]+[[90,60]]*7,[[20,20]]+[[90,60]]*9],
            ball=[[20,20],[40,20]],owner=[1,None],states=['controlled','free'])
        owned=attribute_possession([row(0,'controlled',1,'left'),row(5)],25)
        values=review_pressure(replay,owned)
        self.assertEqual([pressure_level(p) for p in values],['high','low'])
        self.assertTrue(all(not p.sustained_press for p in values))

class FreeformGroundingTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);p=Path(self.temp.name);write_query_artifacts(p);self.tools=MatchMindTools(p)
    def provider(self,indices=[0]):
        return SimpleNamespace(configuration={'provider':'microsoft_foundry'},plan=Mock(return_value={'calls':[{'name':'get_players','arguments':{'limit':2}},{'name':'get_match_summary','arguments':{}}]}),select=Mock(return_value={'statement_ids':indices}))
    def test_arbitrary_question_reaches_multi_tool_planning(self):
        provider=self.provider();result=ask(self.tools,'Describe the tracked runners alongside the recording overview.',provider=provider)
        self.assertEqual(result.provider,'microsoft_foundry');self.assertFalse(result.fallback);self.assertEqual(result.tools,['get_players','get_match_summary'])
        ids={f.fact_id for f in result.answer.facts};self.assertTrue(all(set(s.fact_ids)<=ids for s in result.answer.statements))
    def test_overall_pass_possession_is_independent_of_current_owner(self):
        self.tools.transfers=[dict(outcome='completed',sender_team='left')]*2+[dict(outcome='completed',sender_team='right')]
        self.tools.ownership=[dict(frame=0,timestamp=0,player_id=201,team='right',attribution='controlled')]
        self.tools.ownership_times=[0]
        overall=self.tools.get_possession_summary()
        shares={f.record_id:f.value for f in overall.facts if f.field=='confirmed_pass_share'}
        self.assertAlmostEqual(shares['left'],2/3)
        current=self.tools.get_current_possession(0)
        self.assertEqual(next(f.value for f in current.facts if f.field=='team'),'right')

    def test_compound_known_question_can_combine_tools(self):
        provider=self.provider();result=ask(self.tools,'Compare fastest players and distance covered.',provider=provider)
        self.assertEqual(result.tools,['get_players','get_match_summary']);self.assertFalse(result.fallback)
        self.assertEqual(ask(self.tools,'Compare fastest players and distance covered.').answer.status,'ok')

    def test_unsupported_evidence_selection_returns_natural_unknown(self):
        result=ask(self.tools,'Tell me about the referee.',provider=self.provider([]))
        self.assertEqual(result.answer.status,'unsupported');self.assertIn('enough evidence',result.answer.statements[0].text)
    def test_invalid_selection_does_not_leak_retrieved_claims(self):
        result=ask(self.tools,'Describe the tracked runners.',provider=self.provider([999]))
        self.assertTrue(result.fallback);self.assertEqual(result.answer.status,'unsupported');self.assertFalse(result.answer.facts)
    def test_coaching_intent_does_not_call_provider(self):
        provider=self.provider();result=ask(self.tools,'What did the coach intend?',provider=provider)
        self.assertEqual(result.answer.status,'unsupported');provider.plan.assert_not_called()
    def test_english_does_not_serialize_raw_fields(self):
        facts=[Fact(fact_id=str(i),record_id='left',field=k,value=v,source='test') for i,(k,v) in enumerate([('team','left'),('confirmed_passes',16),('confirmed_pass_share',16/29),('start_x_m',29.312),('validation_state','bas_time_actor_match')])]
        text=sentences(facts)
        for raw in ['team:','confirmed_pass_share','start_x_m','29.312','bas_time_actor_match','True','False']:self.assertNotIn(raw,text)
        self.assertIn('Home side',text);self.assertIn('16',text)
