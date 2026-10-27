"""Deterministic known-answer cases, independent of detector/model downloads."""
import json,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np
from src.vision.world.schema import Config,TrackObservation,CameraState,Detection
from src.vision.world.engine import WorldModel
from src.vision.world.tracking import ShortTermTracker
from src.vision.world.provider import canonical_observation,analytics_recording
from src.vision.world.camera import CameraProvider
from src.vision.world.evaluation import ReferenceEvaluator
from src.observations.schema import RecordingObservation

A=(1.,0.,0.);B=(0.,1.,0.)
def track(key='t1',appearance=A,pitch=(10.,20.),team='team_a',hits=3):
    return TrackObservation(track_id=key,bbox=(20.,20.,10.,20.),detection_confidence=.9,appearance=appearance,team=team,team_confidence=.8,hits=hits,pitch=pitch,localization_confidence=.7)
def camera(frame,valid=True,epoch=0,cut=False):
    return CameraState(frame=frame,timestamp=frame/25,calibration_valid=valid,epoch=epoch,continuity='cut' if cut else 'continuous')

class WorldLogicTests(unittest.TestCase):
    def setUp(self):self.world=WorldModel('synthetic',Config(min_track_hits=1))
    def test_visible_missing_off_camera_keeps_last_not_current(self):
        a=self.world.update([track()],camera(0));key=a.entities[0].entity_id
        b=self.world.update([],camera(5));self.assertEqual(b.entities[0].visibility,'temporarily_missing')
        c=self.world.update([],camera(30));e=c.entities[0]
        self.assertEqual(e.entity_id,key);self.assertEqual(e.visibility,'off_camera');self.assertIsNone(e.current_pitch)
        self.assertEqual(e.last_reliable_pitch,(10.,20.));self.assertEqual(e.last_seen,0)
        self.assertEqual(a.entities[0].visibility,'visible') # earlier snapshot not mutated
    def test_returning_strong_match_is_reassociated(self):
        a=self.world.update([track()],camera(0));self.world.update([],camera(30))
        b=self.world.update([track('return',pitch=(12.,20.))],camera(50))
        self.assertEqual(b.entities[0].entity_id,a.entities[0].entity_id);self.assertEqual(b.entities[0].track_id,'return')
        self.assertEqual(b.decisions[0]['decision'],'reassociated');self.assertTrue(b.decisions[0]['evidence'][0]['movement_feasible'])
    def test_equal_candidates_remain_ambiguous(self):
        self.world.update([track('one'),track('two')],camera(0));self.world.update([],camera(30))
        state=self.world.update([track('return')],camera(50))
        self.assertEqual(state.unassigned[0].status,'ambiguous');self.assertIsNone(canonical_observation(state,25,'test').players[-1].actor_id)
        self.assertEqual(len(state.unassigned[0].hypotheses),3)
    def test_impossible_movement_blocks_reassociation(self):
        initial=self.world.update([track()],camera(0));state=self.world.update([track('new',pitch=(100.,60.))],camera(1))
        ev=state.decisions[0]['evidence'][0];self.assertFalse(ev['movement_feasible']);self.assertIsNone(state.entities[0].current_pitch)
        self.assertNotEqual(state.decisions[0]['assignment'],initial.entities[0].entity_id)
    def test_global_assignment_is_one_to_one(self):
        self.world.update([track()],camera(0));self.world.update([],camera(30))
        state=self.world.update([track('return1'),track('return2')],camera(50))
        visible=[e for e in state.entities if e.visibility=='visible']
        self.assertEqual(len({e.track_id for e in visible}),len(visible));self.assertEqual(len(visible),0)
        self.assertEqual(len(state.unassigned),2)
    def test_global_distinct_appearances_assign_distinct_entities(self):
        self.world.update([track('one',A),track('two',B)],camera(0));self.world.update([],camera(30))
        state=self.world.update([track('newB',B),track('newA',A)],camera(50))
        self.assertEqual(sum(d['decision']=='reassociated' for d in state.decisions),2)
        self.assertEqual(len({e.entity_id for e in state.entities if e.track_id}),2)
    def test_calibration_loss_zoom_keeps_identity_not_metric_position(self):
        a=self.world.update([track()],camera(0));state=self.world.update([track(pitch=(50.,50.))],camera(1,valid=False).model_copy(update={'mode':'close_up'}))
        e=state.entities[0];self.assertEqual(e.entity_id,a.entities[0].entity_id);self.assertEqual(e.visibility,'visible')
        self.assertIsNone(e.current_pitch);self.assertEqual(e.last_reliable_pitch,(10.,20.))
        obs=canonical_observation(state,25,'frame');self.assertIsNone(obs.players[0].pitch);self.assertEqual(obs.players[0].position.quality,'unknown')
    def test_hard_cut_breaks_continuity_and_old_calibration(self):
        old=self.world.update([track()],camera(0));state=self.world.update([track()],camera(1,False,epoch=1,cut=True))
        self.assertEqual(state.entities[0].visibility,'temporarily_missing');self.assertNotEqual(state.entities[-1].entity_id,old.entities[0].entity_id)
        self.assertEqual(state.decisions[0]['evidence'][0]['reason'],'camera epoch changed')
    def test_unconfirmed_track_does_not_invent_identity(self):
        world=WorldModel('x');state=world.update([track(hits=1,pitch=None,team=None)],camera(0,False))
        self.assertEqual(state.entities,[]);obs=canonical_observation(state,25,'x');self.assertIsNone(obs.players[0].actor_id);self.assertIsNone(obs.players[0].team)
    def test_off_camera_not_emitted_as_direct_observation(self):
        self.world.update([track()],camera(0));state=self.world.update([],camera(30));obs=canonical_observation(state,25,'x')
        self.assertIsNone(obs.players[0].pitch);self.assertIsNone(obs.players[0].image_bbox);self.assertIsNone(obs.players[0].position.confidence)
    def test_confidence_provenance_and_schema_roundtrip(self):
        state=self.world.update([track()],camera(0));obs=canonical_observation(state,25,'video_sha_source')
        parsed=RecordingObservation.model_validate_json(obs.model_dump_json());p=parsed.players[0]
        self.assertEqual(p.detection_confidence,.9);self.assertEqual(p.position.confidence,.7);self.assertEqual(p.position.source,'video_sha_source')
        self.assertEqual(p.image_foot,(25.,40.));self.assertIsNone(p.identity.confidence)
    def test_clock_and_duplicate_track_rejected(self):
        with self.assertRaises(ValueError):self.world.update([track(),track()],camera(0))
        self.world.update([],camera(0))
        with self.assertRaises(ValueError):self.world.update([],camera(0))
    def test_retired_tracker_link_cannot_claim_one_entity_twice(self):
        self.world.update([track('old')],camera(0));self.world.update([],camera(1))
        self.world.update([track('new')],camera(2));state=self.world.update([track('new'),track('old')],camera(3))
        assigned=[e.track_id for e in state.entities if e.visibility=='visible'];self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(len(assigned),2) # new provisional entity rather than duplicate claim to first
    def test_analytics_compatibility_explicit_side_mapping(self):
        observations=[canonical_observation(self.world.update([track(pitch=(10+i*.1,20.))],camera(i)),25,'synthetic') for i in range(30)]
        with self.assertRaises(ValueError):analytics_recording(observations,{},1)
        recording=analytics_recording(observations,{'team_a':'left'},1)
        self.assertEqual(recording.positions.shape,(30,1,2));self.assertEqual(recording.player_teams,['left']);self.assertTrue(np.isnan(recording.ball).all())
        from src.analytics.motion import calculate_motion
        motion=calculate_motion(recording.positions[:,0,:],25)
        self.assertAlmostEqual(motion.distance_m[-1],2.9,places=5)
        from src.analytics.pipeline import analyze_recording
        from src.analytics.config import AnalyticsConfig
        result=analyze_recording(recording,AnalyticsConfig())
        self.assertTrue(result['passes'].empty);self.assertTrue(result['shots'].empty)
        self.assertTrue((result['possession'].state=='unknown').all())

class CameraTrackerReferenceTests(unittest.TestCase):
    def test_bottom_center_and_short_tracker_cut(self):
        tracker=ShortTermTracker();d=Detection(bbox=(10,20,30,40),detection_confidence=.9)
        self.assertEqual(d.foot,(25,60));a=tracker.update([d],0,(100,100));b=tracker.update([d],.04,(100,100))
        self.assertEqual(a[0].track_id,b[0].track_id)
        c=tracker.update([d],.08,(100,100),cut=True);self.assertNotEqual(c[0].track_id,b[0].track_id)
    def test_homography_foot_projection_bounds_loss_and_cut(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'video';video.write_bytes(b'test');sha=hashlib.sha256(b'test').hexdigest()
            calibration={'video_sha256':sha,'intervals':[dict(start_frame=0,end_frame=10,image_to_pitch=[1,0,0,0,1,0,0,0,1],validity_region=[[0,0],[100,0],[100,60],[0,60]],provenance=dict(provider='synthetic',source='test',quality='provided'))]}
            (d/'cal.json').write_text(json.dumps(calibration));(d/'cuts.json').write_text(json.dumps({'video_sha256':sha,'events':[{'frame':5,'continuity':'cut'}]}))
            provider=CameraProvider(video,d/'cal.json',d/'cuts.json')
            self.assertEqual(provider.project((25,40),0),(25.,40.));self.assertIsNone(provider.project((120,40),0))
            self.assertFalse(provider.state(5,.2).calibration_valid);self.assertIsNone(provider.project((25,40),11))
            self.assertIsNotNone(provider.state(0,0).visible_pitch_region)
            calibration['video_sha256']='wrong';(d/'cal.json').write_text(json.dumps(calibration))
            with self.assertRaises(ValueError):CameraProvider(video,d/'cal.json')
    def test_reference_metrics_unavailable_without_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v';p.write_bytes(b'x');e=ReferenceEvaluator(p,25);e.update(0,[track()]);r=e.report()
            self.assertIsNone(r['reference_box_coverage']);self.assertIsNone(r['median_pitch_error_m']);self.assertIsNone(r['consecutive_reference_track_switches'])
    def test_reference_matching_is_one_to_one(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'v';p.write_bytes(b'x');reference=dict(video_sha256=hashlib.sha256(b'x').hexdigest(),fps=25,frames=[dict(frame=0,players=[dict(bbox=[20,20,10,20],actor_id='a',pitch=[10,20])])])
            (d/'ref.json').write_text(json.dumps(reference));e=ReferenceEvaluator(p,25,d/'ref.json');e.update(0,[track('one'),track('two')]);r=e.report()
            self.assertEqual(r['matched_person_boxes'],1);self.assertEqual(r['extra_person_boxes'],1);self.assertEqual(r['median_pitch_error_m'],0);self.assertFalse(r['independent_accuracy'])

class ReconciliationAndSoccerTrackTests(unittest.TestCase):
    def test_later_evidence_records_reconciliation_without_rewriting_history(self):
        w=WorldModel('x',Config(min_track_hits=1,appearance_gate=.6,association_threshold=.6))
        w.update([track('a',A),track('b',B)],camera(0));w.update([],camera(30))
        ambiguous=w.update([track('return',(.707,.707,0.))],camera(50))
        self.assertEqual(ambiguous.unassigned[0].status,'ambiguous')
        resolved=w.update([track('return',A)],camera(51))
        self.assertEqual(len(resolved.reconciliations),1)
        self.assertEqual(resolved.reconciliations[0]['segment_start_frame'],50)
        self.assertTrue(resolved.reconciliations[0]['original_observations_preserved'])
        self.assertEqual(ambiguous.unassigned[0].status,'ambiguous')
    def test_existing_soccertrack_piecewise_transform_is_reused(self):
        from src.vision.world.camera import SoccerTrackCameraProvider
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'v';video.write_bytes(b'x')
            (d/'points.json').write_text(json.dumps({'(0, 0)':[0,0],'(100, 0)':[1000,0],'(0, 60)':[0,600],'(100, 60)':[1000,600]}))
            (d/'metadata.json').write_text(json.dumps(dict(video_sha256=hashlib.sha256(b'x').hexdigest(),keypoints='points.json',start_frame=0,end_frame=10)))
            provider=SoccerTrackCameraProvider(video,d/'metadata.json')
            np.testing.assert_allclose(provider.project((250,400),0),(25,40))
            self.assertIsNone(provider.project((250,400),11));self.assertTrue(provider.state(0,0).calibration_valid)
    def test_singular_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'v';p.write_bytes(b'x')
            (d/'cal.json').write_text(json.dumps(dict(video_sha256=hashlib.sha256(b'x').hexdigest(),intervals=[dict(start_frame=0,end_frame=10,image_to_pitch=[0]*9,provenance=dict(provider='test',source='test'))])))
            with self.assertRaises(ValueError):CameraProvider(p,d/'cal.json')

class DebugViewerTests(unittest.TestCase):
    def test_range_serving_and_no_unrelated_files(self):
        from fastapi.testclient import TestClient
        from src.vision.world.serve import create_debug_app
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'source.mp4';video.write_bytes(b'0123456789')
            (d/'diagnostics.json').write_text(json.dumps({'input':dict(path=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest())}))
            (d/'debug.html').write_text('<video id="video" src="old-relative.mp4" controls></video>')
            client=TestClient(create_debug_app(d));response=client.get('/video.mp4',headers={'Range':'bytes=2-4'})
            self.assertEqual(response.status_code,206);self.assertEqual(response.content,b'234')
            self.assertIn('/video.mp4',client.get('/').text);self.assertEqual(client.get('/.env').status_code,404)
    def test_viewer_rejects_changed_video(self):
        from src.vision.world.serve import create_debug_app
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'source.mp4';video.write_bytes(b'changed')
            (d/'diagnostics.json').write_text(json.dumps({'input':dict(path=str(video),sha256='wrong')}))
            with self.assertRaises(ValueError):create_debug_app(d)

class LocalizationSemanticsTests(unittest.TestCase):
    def test_projection_is_inferred_and_last_known_never_becomes_direct(self):
        world=WorldModel('test',Config(min_track_hits=1))
        obs=canonical_observation(world.update([track()],camera(0)),25,'source')
        self.assertEqual(obs.players[0].position.quality,'inferred');self.assertEqual(obs.players[0].localization_kind,'direct_calibrated')
        stale=obs.players[0].model_copy(update={'localization_kind':'last_known'})
        output=analytics_recording([obs.model_copy(update={'players':[stale]})],{'team_a':'left'},1)
        self.assertTrue(np.isnan(output.positions).all())

class EvidenceSufficiencyTests(unittest.TestCase):
    def test_identical_kit_without_calibration_is_not_a_return_identity(self):
        world=WorldModel('x',Config(min_track_hits=1))
        world.update([track(pitch=None)],camera(0,False));world.update([],camera(30,False))
        state=world.update([track('new',pitch=None)],camera(50,False))
        self.assertEqual(state.unassigned[0].status,'ambiguous');self.assertEqual(state.decisions[0]['decision'],'ambiguous')
        self.assertFalse(state.decisions[0]['evidence_sufficient']);self.assertIsNone(state.entities[0].track_id)
