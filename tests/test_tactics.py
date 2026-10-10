"""Synthetic known-answer geometry, temporal gates, grounding and no-lookahead tests."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from pydantic import ValidationError
from src.analytics.config import AnalyticsConfig
from src.intelligence.context import ContextBuilder
from src.intelligence.schema import EventType as T,EventMetrics,make_event
from src.intelligence.commentary.service import request_for,validate_plan
from src.tactics.config import TacticsConfig
from src.tactics.shape import team_shape,block_candidate,players_relative_to_ball
from src.tactics.pressure import local_pressure,causal_velocities
from src.tactics.formations import estimate_formation
from src.tactics.events import PersistentState,PersistentTrigger
from src.tactics.engine import analyze_tactics,phase_comparison
from src.tactics.passing_network import transfer_network
from src.tactics.evidence import pre_event_evidence


def formation_points(counts,direction=1):
    points=[]
    for line,count in enumerate(counts):
        points.extend([[20+line*20,float(y)] for y in np.linspace(10,58,count)] if count>1 else [[20+line*20,34.]])
    points=np.array(points)
    if direction==-1:points[:,0]=105-points[:,0]
    return points


def tactical_fixture(seconds=30):
    n=round(seconds*25)
    left=np.vstack(([5,34],formation_points((4,3,3))))
    right=np.vstack(([100,34],formation_points((4,4,2),-1)))
    positions=np.tile(np.vstack((left,right)),(n,1,1))
    ids=np.arange(100,122)
    recording=SimpleNamespace(positions=positions,ball=np.tile(left[1],(n,1)),fps=25.,player_ids=ids,
        player_teams=['left']*11+['right']*11,mot_ids={int(actor):int(actor-99) for actor in ids},goalkeeper_ids={100,111},
        source_files=[],source_offset=0,half=1,scope='clip',player_source='gsr',sync={'match_id':'synthetic','method':'synthetic'})
    ownership=pd.DataFrame(dict(frame=np.arange(n),possessing_player_id=pd.array([101]*n,dtype='Int64'),possessing_team=['left']*n,
        state=['controlled']*n,control_confidence=[.9]*n))
    motion=[SimpleNamespace(smoothed_speed_mps=np.zeros(n)) for _ in ids]
    ball_motion=SimpleNamespace(segment_id=np.zeros(n),positions=recording.ball)
    result={'possession':ownership,'players':motion,'ball':ball_motion}
    return recording,result


def transfer(frame=100,distance=10,progression=5,difficulty=20,validated=False):
    return make_event('synthetic',frame=frame,fps=25,type=T.PASS_COMPLETED,group_id=f'transfer:{frame}',source='transfer',significance=60,
        team='left',player_id=2,actor_id=101,receiver_id=3,receiver_actor_id=102,
        validation_state='bas_time_actor_match' if validated else 'estimated',validated_by_bas=validated,
        metrics=EventMetrics(start_frame=frame-25,end_frame=frame,distance_m=distance,forward_progression_m=progression,
            difficulty_score_0_100=difficulty,outcome='completed',start_x_m=20,start_y_m=20,end_x_m=30,end_y_m=30))

class TacticalGeometryTests(unittest.TestCase):
    def test_centroid_width_depth_compactness_spread(self):
        points=[[20,10],[20,30],[40,10],[40,30],[0,0]]
        shape=team_shape(points,1,[False]*4+[True],'explicit',min_players=4)
        self.assertEqual((shape.centroid_x_m,shape.centroid_y_m),(30,20))
        self.assertEqual((shape.width_m,shape.depth_m),(20,20))
        self.assertAlmostEqual(shape.compactness_m,np.sqrt(200))
        self.assertAlmostEqual(shape.spread_m,(80+40*np.sqrt(2))/6)
    def test_goalkeeper_exclusion_inclusion_and_unknown(self):
        points=[[20,10],[20,30],[40,10],[40,30],[0,0]];mask=[False]*4+[True]
        out=team_shape(points,1,mask,'explicit',4);all_players=team_shape(points,1,mask,'explicit',4,True)
        self.assertEqual(out.depth_m,20);self.assertEqual(all_players.depth_m,40)
        self.assertEqual(all_players.centroid_x_m,24);self.assertFalse(all_players.goalkeeper_excluded)
        self.assertIsNone(team_shape(points,1,None,min_players=4).width_m)
    def test_missing_or_off_pitch_observations(self):
        points=np.array([[20,10],[20,30],[40,10],[40,30],[0,0]],float);points[0]=np.nan;points[1]=[200,20]
        shape=team_shape(points,1,[False]*4+[True],'explicit',4)
        self.assertIsNone(shape.width_m);self.assertEqual(shape.available_players,2)
    def test_attack_direction_orientation(self):
        a=np.vstack(([[0,34]],formation_points((4,3,3))));b=a.copy();b[:,0]=105-b[:,0]
        x=team_shape(a,1,[True]+[False]*10,'explicit');y=team_shape(b,-1,[True]+[False]*10,'explicit')
        self.assertAlmostEqual(x.longitudinal_m,y.longitudinal_m);self.assertAlmostEqual(x.depth_m,y.depth_m)
    def test_block_bands_and_unknown(self):
        for center,expected in ((20,'low'),(50,'mid'),(90,'high')):
            points=np.vstack(([0,34],np.tile([center,34],(10,1))))
            shape=team_shape(points,1,[True]+[False]*10,'explicit')
            self.assertEqual(block_candidate(shape,'defending'),expected)
            self.assertEqual(block_candidate(shape,'possessing'),'unknown')
        unknown=team_shape([[0,0]],1,None)
        self.assertEqual(block_candidate(unknown,'defending'),'unknown')
    def test_players_relative_to_ball_and_mirror(self):
        a=np.array([[20,30],[40,30],[30,30]]);d=np.array([[40,30],[20,30]]);ball=np.array([30,30])
        answer=players_relative_to_ball(a,d,ball,1)
        self.assertEqual((answer.attacking_ahead,answer.attacking_behind,answer.attacking_level),(1,1,1))
        self.assertEqual(answer.defenders_goal_side,1);self.assertEqual(answer.defenders_behind_ball_to_own_goal,1)
        a[:,0]=105-a[:,0];d[:,0]=105-d[:,0];ball[0]=105-ball[0]
        self.assertEqual(answer,players_relative_to_ball(a,d,ball,-1))
    def test_pressure_radii_closing_and_local_balance(self):
        config=TacticsConfig(pressure_min_observed_defenders=4)
        p=local_pressure(np.array([50,34]),np.array([0,0]),[[52,34],[54,34],[57,34],[60,34]],[[-1,0]]*4,[[50,34],[51,34]],config)
        self.assertEqual((p.defenders_within_near,p.defenders_within_local,p.defenders_within_outer),(1,2,3))
        self.assertEqual(p.nearest_defender_distance_m,2);self.assertEqual(p.closing_defenders,3)
        self.assertEqual(p.local_numerical_balance,0);self.assertTrue(p.press_candidate)
    def test_proximity_without_closing_is_not_pressing(self):
        p=local_pressure(np.array([50,34]),np.zeros(2),[[51,34],[52,34]],np.zeros((2,2)),[[50,34]],TacticsConfig(pressure_min_observed_defenders=2))
        self.assertEqual(p.local_pressure,'multiple_nearby');self.assertFalse(p.press_candidate)
    def test_causal_velocity_and_missing_gap(self):
        positions=np.zeros((51,2,2));positions[:,:,0]=np.arange(51)[:,None]/25*2
        velocity=causal_velocities(positions,25,25,1,14)
        np.testing.assert_allclose(velocity,[[2,0],[2,0]])
        positions[26:]=10000
        np.testing.assert_array_equal(velocity,causal_velocities(positions,25,25,1,14))
        positions[10,0]=np.nan
        self.assertTrue(np.isnan(causal_velocities(positions,25,25,1,14)[0]).all())

class FormationTests(unittest.TestCase):
    def test_synthetic_formations(self):
        config=TacticsConfig();frames=list(range(0,375,5))
        for name,counts in [('4-3-3',(4,3,3)),('4-4-2',(4,4,2)),('4-2-3-1',(4,2,3,1)),('3-5-2',(3,5,2))]:
            for direction in (1,-1):
                estimate=estimate_formation(np.tile(formation_points(counts,direction),(75,1,1)),frames,direction,config)
                self.assertEqual(estimate.estimate,name);self.assertAlmostEqual(estimate.fit_rmse_m,0)
                self.assertEqual(estimate.fit_quality,'clear')
    def test_unknown_formation_conditions(self):
        config=TacticsConfig();history=np.tile(formation_points((4,3,3)),(75,1,1));frames=list(range(0,375,5))
        self.assertEqual(estimate_formation(history,frames,1,config,False).reason,'goalkeeper_unknown')
        self.assertEqual(estimate_formation(history[:2],frames[:2],1,config).reason,'insufficient_history')
        self.assertEqual(estimate_formation(history[:,:9],frames,1,config).reason,'incomplete_roster')
        history[:30,0]=np.nan
        self.assertEqual(estimate_formation(history,frames,1,config).reason,'insufficient_coverage')
    def test_collapsed_shape_not_forced_into_formation(self):
        history=np.tile(np.column_stack((np.full(10,50),np.linspace(10,60,10))),(75,1,1))
        self.assertEqual(estimate_formation(history,list(range(0,375,5)),1,TacticsConfig()).estimate,'unknown')

class TacticalPersistenceTests(unittest.TestCase):
    def test_block_persistence_and_gap(self):
        state=PersistentState(2)
        self.assertEqual(state.update('low',0,25)[0],'unknown')
        self.assertFalse(state.update('low',45,25)[1]);self.assertTrue(state.update('low',50,25)[1])
        self.assertFalse(state.update('low',55,25)[1]);self.assertEqual(state.update('unknown',60,25)[0],'unknown')
        self.assertFalse(state.update('low',65,25)[1])
    def test_block_keeps_confirmed_state_until_new_state_persists(self):
        state=PersistentState(2);state.update('low',0,25);state.update('low',50,25)
        self.assertEqual(state.update('high',55,25)[0],'low')
        self.assertEqual(state.update('high',100,25)[0],'low')
        value,changed,_=state.update('high',105,25)
        self.assertEqual(value,'high');self.assertTrue(changed)
    def test_event_latch_deduplication_and_rearming(self):
        trigger=PersistentTrigger(1,1)
        fired=[frame for frame in range(100) if trigger.update(True,frame,25)[0]]
        self.assertEqual(fired,[25]);trigger.update(False,100,25)
        fired=[frame for frame in range(101,150) if trigger.update(True,frame,25)[0]]
        self.assertEqual(fired,[126])
    def test_engine_pressure_persistence(self):
        r,result=tactical_fixture(8)
        r.positions[:,1]=[50,34];r.ball[:]=[50,34]
        r.positions[:,12]=np.column_stack((54-np.arange(200)/25*.6,np.full(200,34)))
        r.positions[:,13]=np.column_stack((54.5-np.arange(200)/25*.6,np.full(200,35)))
        r.positions[:,14:]=[90,60]
        out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        pressure=[e for e in out['events'] if e.type==T.HIGH_PRESSURE_SEQUENCE]
        self.assertEqual(len(pressure),1);self.assertGreaterEqual(pressure[0].frame,75)
        self.assertGreaterEqual(pressure[0].tactical.duration_s,2)
    def test_direction_change_restarts_evidence(self):
        r,result=tactical_fixture(8);config=TacticsConfig(attack_direction_changes=((100,-1),))
        out=analyze_tactics(r,result,AnalyticsConfig(),config,events=[])
        before=next(s for s in out['states'] if s.frame==95);after=next(s for s in out['states'] if s.frame==100)
        self.assertEqual(before.left.attack_direction,1);self.assertEqual(after.left.attack_direction,-1)
        self.assertEqual(after.right.block,'unknown');self.assertEqual(after.right.block_evidence_s,0)
    def test_no_future_tactical_data_leakage(self):
        r,result=tactical_fixture(8);baseline=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        r.positions[101:]=np.nan;r.ball[101:]=np.nan
        changed=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        before=lambda out:[s.model_dump(mode='json') for s in out['states'] if s.frame<=100]
        self.assertEqual(before(baseline),before(changed))
        events=lambda out:[e.model_dump(mode='json') for e in out['events'] if e.frame<=100]
        self.assertEqual(events(baseline),events(changed))
    def test_shape_events_persist_and_do_not_spam(self):
        r,result=tactical_fixture(45)
        r.positions[500:,1:11,1]=34+(r.positions[500:,1:11,1]-34)*1.3
        out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        widths=[e for e in out['events'] if e.type==T.TEAM_WIDTH_INCREASED and e.team=='left']
        self.assertEqual(len(widths),1);self.assertGreater(widths[0].frame,500)
    def test_formation_is_sustained_and_has_no_future_window(self):
        r,result=tactical_fixture(25);out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        events=[e for e in out['events'] if e.type==T.FORMATION_ESTIMATE_CHANGED]
        self.assertEqual(len(events),2)
        for e in events:
            self.assertGreaterEqual(e.timestamp,19.8)
            self.assertLessEqual(e.tactical.formation_window_end_frame,e.frame)
    def test_insufficient_data_stays_unknown(self):
        r,result=tactical_fixture(8);r.positions[:,12:16]=np.nan
        out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        self.assertTrue(all(s.right.block=='unknown' for s in out['states']))
        self.assertTrue(all(s.right.formation.estimate=='unknown' for s in out['states']))
    def test_configuration_validation(self):
        for values in ({'sample_hz':float('nan')},{'min_outfield_players':11},{'pressure_near_radius_m':6},{'attack_direction_changes':((10,1),(5,-1))}):
            with self.assertRaises(ValueError):TacticsConfig(**values)
        with self.assertRaises(ValueError):TacticsConfig(sample_hz=3).stride(25)

class TransferNetworkTests(unittest.TestCase):
    def test_edge_aggregation_and_bas_provenance(self):
        network=transfer_network([transfer(),transfer(200,30,15,60,True)])
        edge=network['edges'][0]
        self.assertEqual(edge['transfer_count'],2);self.assertEqual(edge['mean_distance_m'],20)
        self.assertEqual(edge['total_progression_m'],20);self.assertEqual(edge['mean_progression_m'],10)
        self.assertEqual(edge['mean_heuristic_difficulty'],40);self.assertEqual(edge['long_transfer_count'],1)
        self.assertEqual(edge['bas_supported_pass_count'],1);self.assertEqual(network['name'],'Transfer Network')
    def test_window_does_not_include_future_receipt(self):
        network=transfer_network([transfer(100),transfer(200)],end_seconds=5)
        self.assertEqual(network['transfer_count'],1)
        self.assertEqual(transfer_network([transfer(100),transfer(200)],start_seconds=4,end_seconds=8)['transfer_count'],1)
    def test_network_pattern_event_deduplication(self):
        r,result=tactical_fixture(10)
        out=analyze_tactics(r,result,AnalyticsConfig(),events=[transfer(25),transfer(50),transfer(75)])
        events=[e for e in out['events'] if e.type==T.TRANSFER_NETWORK_PATTERN]
        self.assertEqual(len(events),1);self.assertEqual(events[0].tactical.transfer_count,3)
        self.assertGreaterEqual(events[0].frame,150)

class TacticalIntelligenceTests(unittest.TestCase):
    def test_tactical_context_serialization_and_prompt(self):
        r,result=tactical_fixture(8);out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        events=sorted(out['events']+[transfer()],key=lambda e:(e.frame,e.event_id))
        builder=ContextBuilder(r,result,events,out['states']);context=builder.at(6,15)
        raw=json.dumps(context,allow_nan=False)
        self.assertIn('tactical_state',raw);self.assertLessEqual(context['tactical_state']['frame'],150)
        request=request_for([transfer()],context,'analyst')
        self.assertIn('context_shape_left',request['sentences'])
        self.assertIn('causal explanations are forbidden',request['instruction'])
    def test_unknown_tactical_fields_never_reach_gemini(self):
        r,result=tactical_fixture(2);out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        context=ContextBuilder(r,result,[],out['states']).at(1,15)
        context['tactical_state']['left']['manager_instruction']='Invented intent'
        with self.assertRaises(ValidationError):request_for([transfer()],context,'analyst')
        with self.assertRaises(ValueError):validate_plan({'sentence_ids':['invented_tactic']},{'sentences':{}})
    def test_pre_event_state_strictly_precedes_release(self):
        r,result=tactical_fixture(8);out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        e=transfer(100);evidence=pre_event_evidence([e],out['states'])[0]
        self.assertEqual(evidence['event_start_frame'],75);self.assertEqual(evidence['sample_frame'],70)
        self.assertLess(evidence['sample_frame'],e.metrics.start_frame)
    def test_phase_comparisons_require_enough_samples(self):
        r,result=tactical_fixture(12);out=analyze_tactics(r,result,AnalyticsConfig(),events=[])
        self.assertFalse(out['summary']['phase_comparison']['left']['comparison_available'])
        self.assertIsNone(out['summary']['phase_comparison']['left']['wider_in_possession'])
        changed=[s.model_copy(update={'left':s.left.model_copy(update={'phase':'defending'})}) for s in out['states'][:60]]
        summary=phase_comparison(out['states']+changed,TacticsConfig())
        self.assertTrue(summary['left']['comparison_available']);self.assertFalse(summary['left']['wider_in_possession'])

if __name__=='__main__':unittest.main()
