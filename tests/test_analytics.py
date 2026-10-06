"""Synthetic known-answer tests: no match annotations are used as expected labels."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from src.analytics.config import AnalyticsConfig
from src.analytics.motion import calculate_motion
from src.analytics.possession import detect_possession
from src.analytics.passes import detect_passes
from src.analytics.quality import pass_features
from src.analytics.shots import detect_shots, annotated_shot_windows, speed_window
from src.analytics.validation import validate_events
from src.analytics.data import Recording, load_recording
from src.analytics.pipeline import analyze_recording


CFG = AnalyticsConfig(player_smoothing_frames=3, ball_smoothing_frames=3)


def ball_motion(xy):
    return calculate_motion(xy,25,3,3,50)


def possession_fixture(receiver_team='left', missing=False, self_recovery=False):
    ball = np.vstack((np.tile([10.,34.],(15,1)),
                      np.column_stack((np.linspace(10,30,26),np.full(26,34.))),
                      np.tile([30.,34.],(15,1))))
    if self_recovery:
        ball = np.vstack((np.tile([10.,34.],(15,1)),
                          np.column_stack((np.r_[np.linspace(10,20,15),np.linspace(20,10,15)],np.full(30,34.))),
                          np.tile([10.,34.],(15,1))))
    if missing:
        ball[25:29] = np.nan
    n = len(ball)
    players = np.tile([[10.,34.],[30.,34.]],(n,1,1))
    velocity = np.zeros_like(players)
    motion = ball_motion(ball)
    possession = detect_possession(players,velocity,[1,2],['left',receiver_team],motion,CFG)
    return players,motion,possession


class MotionTests(unittest.TestCase):
    def test_known_five_meters_per_second_and_distance(self):
        xy = np.column_stack((np.arange(100)*5/25,np.full(100,34.)))
        m = calculate_motion(xy,25,15,3,14)
        np.testing.assert_allclose(m.instantaneous_speed_mps[1:],5,atol=1e-10)
        np.testing.assert_allclose(m.smoothed_speed_mps[1:],5,atol=1e-10)
        self.assertAlmostEqual(m.distance_m[-1],99*5/25)
        self.assertTrue(np.isnan(m.smoothed_speed_mps[0]))

    def test_missing_gap_does_not_add_a_hundred_meters(self):
        xy = np.vstack((np.tile([10.,34.],(10,1)),np.full((4,2),np.nan),np.tile([100.,34.],(10,1))))
        m = calculate_motion(xy)
        self.assertAlmostEqual(m.distance_m[-1],0,places=8)
        self.assertTrue(np.isnan(m.smoothed_speed_mps[14]))
        self.assertTrue(np.isnan(m.positions[10:14]).all())
        self.assertNotEqual(m.segment_id[9],m.segment_id[14])

    def test_isolated_spike_is_flagged_and_smoothed(self):
        xy = np.column_stack((np.arange(100)*.1,np.full(100,34.)))
        xy[50,0] += 40
        m = calculate_motion(xy)
        self.assertTrue(m.raw_speed_flag[50])
        self.assertTrue(m.raw_speed_flag[51])
        self.assertLess(np.nanmax(m.smoothed_speed_mps),4)

    def test_smoothing_reduces_noise(self):
        rng = np.random.default_rng(7)
        xy = np.column_stack((np.arange(200)*.2,np.full(200,34.)))+rng.normal(0,.04,(200,2))
        m = calculate_motion(xy)
        self.assertLess(np.nanstd(m.smoothed_speed_mps[15:-15]),np.nanstd(m.instantaneous_speed_mps[15:-15]))

    def test_teleport_has_no_accepted_impossible_speed(self):
        xy = np.tile([10.,34.],(70,1));xy[35:,0]=100
        m = calculate_motion(xy)
        self.assertGreater(m.rejected_speed_flag.sum(),0)
        self.assertTrue(np.nanmax(m.smoothed_speed_mps)<=14)
        self.assertLess(m.distance_m[-1],90)

    def test_short_segments_and_empty_input(self):
        m = calculate_motion(np.array([[1.,2.],[np.nan,np.nan],[2.,2.]]))
        self.assertTrue(np.isnan(m.smoothed_speed_mps).all())
        self.assertEqual(m.distance_m[-1],0)
        self.assertEqual(len(calculate_motion(np.empty((0,2))).positions),0)


class PossessionTests(unittest.TestCase):
    def test_persistence_prevents_one_frame_owner(self):
        n=20;ball=ball_motion(np.tile([10.,34.],(n,1)))
        p=np.tile([[50.,34.],[70.,34.]],(n,1,1));p[10,0]=[10.,34.]
        r=detect_possession(p,np.zeros_like(p),[1,2],['left','right'],ball,CFG)
        self.assertTrue(r.possessing_player_id.isna().all())

    def test_persistent_owner_then_release(self):
        _,_,r=possession_fixture()
        self.assertTrue(r.possessing_player_id.iloc[:5].isna().all())
        self.assertEqual(r.possessing_player_id.iloc[10],1)
        self.assertEqual(r.possessing_player_id.iloc[-1],2)
        self.assertIn('free',r.state.tolist())

    def test_contested_has_no_forced_owner(self):
        n=20;ball=ball_motion(np.tile([10.,34.],(n,1)))
        p=np.tile([[9.,34.],[11.,34.]],(n,1,1))
        r=detect_possession(p,np.zeros_like(p),[1,2],['left','right'],ball,CFG)
        self.assertTrue(r.possessing_player_id.isna().all())
        self.assertTrue((r.state.iloc[1:]=='contested').all())

    def test_fast_flyby_has_no_owner(self):
        xy=np.column_stack((np.linspace(5,45,51),np.full(51,34.)))
        p=np.tile([[25.,34.],[60.,34.]],(len(xy),1,1))
        r=detect_possession(p,np.zeros_like(p),[1,2],['left','right'],ball_motion(xy),CFG)
        self.assertTrue(r.possessing_player_id.isna().all())

    def test_missing_and_off_pitch_ball_clear_owner(self):
        n=30;xy=np.tile([10.,34.],(n,1));xy[15:18]=np.nan;xy[25:]=[-10.,34.]
        p=np.tile([[10.,34.],[40.,34.]],(n,1,1))
        r=detect_possession(p,np.zeros_like(p),[1,2],['left','right'],ball_motion(xy),CFG)
        self.assertTrue((r.state.iloc[15:18]=='unknown').all())
        self.assertTrue(r.possessing_player_id.iloc[15:18].isna().all())
        self.assertTrue((r.state.iloc[26:]=='out_of_play').all())

    def test_relative_motion_allows_fast_dribbling(self):
        n=30;xy=np.column_stack((10+np.arange(n)*.4,np.full(n,34.)))
        p=np.stack((xy,xy+[20,0]),axis=1);v=np.tile([[10.,0.],[10.,0.]],(n,1,1))
        r=detect_possession(p,v,[1,2],['left','right'],ball_motion(xy),CFG)
        self.assertEqual(r.possessing_player_id.iloc[-1],1)


class PassTests(unittest.TestCase):
    def test_completed_pass_has_duration_speed_sender_receiver(self):
        _,b,p=possession_fixture()
        r=detect_passes(p,b,CFG)
        self.assertEqual(len(r),1)
        row=r.iloc[0]
        self.assertEqual(row.outcome,'completed')
        self.assertEqual(row.sender_id,1);self.assertEqual(row.receiver_id,2)
        self.assertGreater(row.distance_m,10);self.assertGreater(row.duration_seconds,.5)
        self.assertAlmostEqual(row.ball_speed_peak_kmh,row.ball_speed_peak_mps*3.6)

    def test_opponent_reception_is_interception_candidate(self):
        _,b,p=possession_fixture('right');r=detect_passes(p,b,CFG)
        self.assertEqual(r.iloc[0].outcome,'intercepted')
        self.assertEqual(r.iloc[0].receiver_team,'right')

    def test_no_completed_pass_across_missing_gap(self):
        _,b,p=possession_fixture(missing=True);r=detect_passes(p,b,CFG)
        self.assertFalse((r.outcome=='completed').any())

    def test_self_recovery_is_not_a_pass(self):
        _,b,p=possession_fixture(self_recovery=True)
        self.assertTrue(detect_passes(p,b,CFG).empty)

    def test_direct_ownership_change_is_not_a_pass(self):
        xy=np.tile([10.,34.],(20,1));b=ball_motion(xy)
        p=pd.DataFrame(dict(frame=np.arange(20),state=['controlled']*20,
                            possessing_player_id=[1]*10+[2]*10,
                            possessing_team=['left']*20,control_start_frame=[0]*10+[10]*10))
        self.assertTrue(detect_passes(p,b,CFG).empty)

    def test_recording_end_is_unresolved(self):
        _,b,p=possession_fixture();p.loc[41:,'possessing_player_id']=pd.NA;p.loc[41:,'state']='free'
        r=detect_passes(p,b,CFG)
        self.assertEqual(r.iloc[-1].outcome,'unresolved')
        self.assertEqual(r.iloc[-1].termination_reason,'end_of_recording')

    def test_timeout_terminates_a_flight(self):
        _,b,p=possession_fixture()
        r=detect_passes(p,b,replace(CFG,pass_timeout_seconds=.3))
        self.assertTrue((r.termination_reason=='timeout').any())
        self.assertTrue((r.outcome=='unresolved').any())


class QualityTests(unittest.TestCase):
    def test_pressure_and_longer_distance_increase_difficulty(self):
        easy=pass_features([10,34],[20,34],[[90,10]],[],1,CFG)
        hard=pass_features([10,34],[50,34],[[12,34],[25,34],[49,34]],[],1,CFG)
        self.assertGreater(hard['difficulty_score_0_100'],easy['difficulty_score_0_100'])
        self.assertGreater(hard['passing_lane_defenders'],0)
        self.assertLess(hard['heuristic_completion_score_0_1'],easy['heuristic_completion_score_0_1'])
        self.assertTrue(0<=hard['difficulty_score_0_100']<=100)

    def test_attack_direction_and_lane_projection(self):
        f=pass_features([20,34],[10,34],[[15,34],[5,34],[25,34],[15,50]],[],-1,CFG)
        self.assertEqual(f['forward_progression_m'],10)
        self.assertEqual(f['passing_lane_defenders'],1)

    def test_missing_defenders_do_not_mean_easy(self):
        f=pass_features([10,34],[30,34],[],[],1,CFG)
        self.assertFalse(f['quality_available'])
        self.assertIsNone(f['difficulty_score_0_100'])


class ShotTests(unittest.TestCase):
    def fixture(self, away=False):
        n=30;xy=np.column_stack((80+np.arange(n)*.8,np.full(n,34.)))
        if away:xy[:,0]=80-np.arange(n)*.8
        b=ball_motion(xy)
        p=pd.DataFrame(dict(frame=np.arange(n),state=['free']*n,
                            possessing_player_id=pd.array([None]*n,dtype='Int64'),possessing_team=[None]*n))
        return b,p

    def test_goal_directed_fast_ball_is_candidate_with_speed(self):
        b,p=self.fixture();s=detect_shots(p,b,CFG)
        self.assertEqual(len(s),1)
        self.assertEqual(s.iloc[0].goal_x_m,105)
        self.assertAlmostEqual(s.iloc[0].speed_median_mps,20,places=6)
        self.assertAlmostEqual(s.iloc[0].speed_peak_kmh,72,places=6)

    def test_ball_going_away_from_nearby_goal_is_not_shot(self):
        b,p=self.fixture(away=True)
        self.assertTrue(detect_shots(p,b,CFG).empty)

    def test_slow_or_wide_trajectory_is_not_shot(self):
        b,p=self.fixture();b.smoothed_speed_mps[:]=2
        self.assertTrue(detect_shots(p,b,CFG).empty)
        b,p=self.fixture();b.positions[:,1]=10
        self.assertTrue(detect_shots(p,b,CFG).empty)

    def test_bas_windows_do_not_create_candidates(self):
        b,p=self.fixture(away=True)
        e=[dict(annotation_id=1,frame=5,seconds=.2,label='SHOT',player_id=1,team='left')]
        windows=annotated_shot_windows(e,b,CFG)
        self.assertEqual(len(windows),1)
        self.assertTrue(detect_shots(p,b,CFG).empty)
        self.assertAlmostEqual(windows.iloc[0].speed_peak_kmh,72,places=6)

    def test_partial_speed_window_reports_missing_coverage(self):
        xy=np.column_stack((10+np.arange(30)*.2,np.full(30,34.)))
        xy[10:15]=np.nan
        w=speed_window(ball_motion(xy),5,20,CFG)
        self.assertEqual(w['end_frame'],9)
        self.assertEqual(w['valid_speed_samples'],5)
        self.assertEqual(w['window_requested_frames'],16)
        self.assertAlmostEqual(w['window_coverage_fraction'],5/16)
        self.assertTrue(w['window_truncated'])


class ValidationTests(unittest.TestCase):
    def test_one_to_one_matching_counts_duplicate_as_false_positive(self):
        predictions=[dict(start_frame=100,sender_id=1),dict(start_frame=101,sender_id=1)]
        refs=[dict(annotation_id=1,frame=100,label='PASS',player_id=1)]
        s,r=validate_events(predictions,refs,25,1,'sender_id')
        self.assertEqual(s['matched'],1);self.assertEqual(s['false_positives'],1)
        self.assertEqual(s['precision'],.5);self.assertEqual(s['recall'],1)

    def test_no_truth_is_not_perfect_validation(self):
        s,_=validate_events([dict(start_frame=100)],[],25)
        self.assertEqual(s['status'],'no_reference_events')
        self.assertIsNone(s['precision']);self.assertIsNone(s['recall'])

    def test_wrong_actor_or_timing_does_not_match(self):
        refs=[dict(annotation_id=1,frame=100,label='PASS',player_id=1)]
        s,_=validate_events([dict(start_frame=100,sender_id=2)],refs,25,1,'sender_id')
        self.assertEqual(s['matched'],0)
        s,_=validate_events([dict(start_frame=130,sender_id=1)],refs,25,1,'sender_id')
        self.assertEqual(s['matched'],0)

    def test_empty_predictions_preserve_false_negatives(self):
        s,r=validate_events([],[dict(annotation_id=1,frame=10,label='SHOT',player_id=1)],25)
        self.assertEqual(s['false_negatives'],1)
        self.assertEqual(s['recall'],0)


class PipelineTests(unittest.TestCase):
    def test_detectors_do_not_depend_on_bas_labels(self):
        positions,b,p=possession_fixture()
        recording=Recording(positions,b.raw_positions,np.ones(len(b.positions),dtype=int),
                            np.array([1,2]),['left','left'],{1:1,2:2},[],25,0,1,'clip','synthetic',[],{})
        first=analyze_recording(recording,CFG)
        self.assertEqual(len(first['passes']),1)
        recording.events=[dict(annotation_id=1,frame=10,seconds=.4,
                               label='SHOT',player_id=1,team='left')]
        second=analyze_recording(recording,CFG)
        pd.testing.assert_frame_equal(first['possession'],second['possession'])
        pd.testing.assert_frame_equal(first['passes'],second['passes'])
        pd.testing.assert_frame_equal(first['shots'],second['shots'])
        self.assertEqual(len(second['bas_shots']),1)

    def test_real_clip_alignment_and_actor_coverage(self):
        recording=load_recording()
        self.assertEqual(recording.positions.shape,(6000,22,2))
        self.assertEqual(recording.source_offset,5999)
        self.assertEqual(recording.fps,25)
        self.assertEqual(len(recording.events),88)
        self.assertTrue(np.isnan(recording.ball[recording.ball_status==0]).all())
        self.assertEqual(set(recording.player_ids),set(recording.mot_ids))

    def test_full_half_accepts_events_without_an_actor(self):
        recording=load_recording(scope='half')
        self.assertEqual(len(recording.ball),71850)
        self.assertEqual(recording.source_offset,0)
        self.assertEqual(sum(e['label']=='SHOT' for e in recording.events),10)
        self.assertTrue(any(e['player_id'] is None for e in recording.events))


class ConfigTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        for values in ({'fps':0},{'ball_smoothing_frames':4},{'possession_frames':0},
                       {'acquisition_radius_m':4,'retention_radius_m':3},{'left_attack_direction':0},{'fps':float('nan')},{'possession_frames':2.5}):
            with self.assertRaises(ValueError):AnalyticsConfig(**values)


if __name__=='__main__':
    unittest.main()
