"""Deterministic kit, memory and projection cases; no model downloads."""
import unittest
import numpy as np,cv2
from src.vision.world.schema import Detection,TrackObservation,CameraState,PlayerEntity,KitObservation,Config
from src.vision.world.kits import torso_features,VideoTeams,update_team
from src.vision.world.tracking import ShortTermTracker
from src.vision.world.engine import WorldModel
from src.vision.world.player_geometry import project_player,assess_sizes
from src.observations.schema import PitchCalibrationObservation,Provenance

class KitTests(unittest.TestCase):
    def image(self,color):
        image=np.full((200,180,3),(40,120,40),np.uint8);image[38:85,66:110]=color;return image
    def test_green_kit_survives_local_grass_exclusion(self):
        kit,descriptor=torso_features(self.image((40,230,210)),(50,20,80,140))
        self.assertIsNotNone(kit.lab);self.assertGreater(kit.foreground_fraction,.5);self.assertIsNotNone(descriptor)
        empty,_=torso_features(np.full((200,180,3),(40,120,40),np.uint8),(50,20,80,140));self.assertIsNone(empty.lab)
    def test_two_populations_are_learned_without_color_names(self):
        profiles={str(i):[(95,175,145)]*4 for i in range(5)};profiles.update({str(i):[(190,110,180)]*4 for i in range(5,10)})
        model=VideoTeams.fit(profiles);self.assertIsNotNone(model.centers)
        gray=Detection(bbox=(0,0,20,60),detection_confidence=.9,kit=KitObservation(lab=(150,128,128),quality=.8,field_context=.8))
        self.assertIsNone(model.classify(gray).team)
        self.assertIsNone(VideoTeams.fit({'one':[(1,2,3)]}).centers)
    def test_team_persists_through_unknown_and_single_contradiction(self):
        entity=PlayerEntity(entity_id='E',visibility='visible',last_seen=0,last_seen_frame=0)
        for i in range(7):
            t=TrackObservation(bbox=(0,0,20,60),detection_confidence=.9,track_id='T',hits=3,team='team_a',team_confidence=.95,kit=KitObservation(lab=(90,170,145),quality=.9))
            team,confidence,memory,events=update_team(entity,t,CameraState(frame=i,timestamp=i/25));entity=entity.model_copy(update={'team':team,'team_confidence':confidence,'team_memory':memory})
        self.assertEqual(entity.team,'team_a');self.assertTrue(entity.team_memory.stable)
        for observed in (None,'team_b'):
            updated=update_team(entity,t.model_copy(update={'team':observed}),CameraState(frame=10,timestamp=.4));self.assertEqual(updated[0],'team_a')
    def test_sustained_contradiction_records_evidence_based_change(self):
        e=PlayerEntity(entity_id='E',visibility='visible',last_seen=0,last_seen_frame=0);changes=[]
        for i in range(40):
            observed='team_a' if i<5 else 'team_b'
            t=TrackObservation(bbox=(0,0,20,60),detection_confidence=.9,track_id='T',hits=3,team=observed,team_confidence=.99,kit=KitObservation(lab=(100,180,150),quality=.99))
            team,confidence,memory,events=update_team(e,t,CameraState(frame=i,timestamp=i/25));changes+=events;e=e.model_copy(update={'team':team,'team_confidence':confidence,'team_memory':memory})
        self.assertEqual(e.team,'team_b');self.assertEqual(sum(v.kind=='team_changed' for v in changes),1)

class PlayerMemoryTests(unittest.TestCase):
    def track(self,key='T',x=10,appearance=(1.,0.),team='team_a'):
        return TrackObservation(track_id=key,bbox=(x,20,20,60),detection_confidence=.9,hits=4,appearance=appearance,team=team,team_confidence=.95,image_velocity=(25,0),kit=KitObservation(lab=(90,170,145),quality=.9,field_context=.9))
    def camera(self,i):return CameraState(frame=i,timestamp=i/25,continuity='continuous')
    def test_short_gap_prediction_is_association_only(self):
        tracker=ShortTermTracker();d=Detection(bbox=(10,20,20,60),detection_confidence=.9)
        a=tracker.update([d],0,(200,200));tracker.update([d.model_copy(update={'bbox':(12,20,20,60)})],.04,(200,200));self.assertEqual(tracker.update([],.08,(200,200)),[])
        b=tracker.update([d.model_copy(update={'bbox':(16,20,20,60)})],.12,(200,200));self.assertEqual(a[0].track_id,b[0].track_id);self.assertIsNone(b[0].pitch)
    def test_image_continuity_can_reassociate_without_metric_geometry(self):
        w=WorldModel('one',Config(min_track_hits=1))
        for i in range(6):w.update([self.track(x=10+i)],self.camera(i))
        w.update([],self.camera(6));s=w.update([self.track('return',x=17)],self.camera(7))
        self.assertEqual(s.decisions[0]['decision'],'reassociated');e=s.entities[0];self.assertEqual(e.track_history,['T','return']);self.assertIsNone(e.current_pitch);self.assertEqual(e.team,'team_a')
    def test_equal_appearance_and_motion_candidates_remain_ambiguous(self):
        w=WorldModel('one',Config(min_track_hits=1))
        for i in range(6):w.update([self.track('A',10+i),self.track('B',10+i)],self.camera(i))
        w.update([],self.camera(6));s=w.update([self.track('return',17)],self.camera(7));self.assertEqual(len(s.unassigned),1);self.assertEqual(s.unassigned[0].status,'ambiguous')
    def test_off_camera_memory_keeps_team_bbox_history_but_no_current_position(self):
        w=WorldModel('one',Config(min_track_hits=1))
        for i in range(6):w.update([self.track(x=10+i)],self.camera(i))
        s=w.update([],self.camera(40));e=s.entities[0];self.assertEqual(e.memory_state,'off_camera');self.assertEqual(e.team,'team_a');self.assertIsNone(e.image_bbox);self.assertIsNotNone(e.last_image_bbox);self.assertIsNone(e.current_pitch)
        self.assertIn(e.entity_id,s.roster.team_a)
    def test_camera_region_can_mark_off_camera_with_movement_slack(self):
        w=WorldModel('one',Config(min_track_hits=1));first=self.camera(0).model_copy(update={'calibration_valid':True})
        w.update([self.track().model_copy(update={'pitch':(82,31)})],first)
        moved=self.camera(1).model_copy(update={'calibration_valid':True,'visible_pitch_region':[(0,0),(20,0),(20,68),(0,68)]})
        e=w.update([],moved).entities[0];self.assertEqual(e.memory_state,'off_camera');self.assertIsNone(e.current_pitch);self.assertEqual(e.last_reliable_pitch,(82,31));self.assertIn('movement slack',e.memory_reason)
    def test_epoch_change_marks_identity_lost_without_deleting_entity(self):
        w=WorldModel('one',Config(min_track_hits=1));w.update([self.track()],self.camera(0));s=w.update([],self.camera(1).model_copy(update={'epoch':1,'continuity':'cut'}));self.assertEqual(s.entities[0].memory_state,'lost');self.assertIsNone(s.entities[0].current_pitch)
    def test_background_appearance_cannot_reconnect_after_long_gap(self):
        w=WorldModel('one',Config(min_track_hits=1))
        for i in range(6):w.update([self.track()],self.camera(i))
        w.update([],self.camera(100));s=w.update([self.track('steward')],self.camera(500));self.assertEqual(s.unassigned[0].status,'ambiguous');self.assertFalse(s.decisions[0]['evidence_sufficient'])

class ProjectionAndSizeTests(unittest.TestCase):
    def camera(self):
        cal=PitchCalibrationObservation(start_frame=0,end_frame=1,image_to_pitch=(.1,0,0,0,.1,0,0,0,1),validity_region=[(0,0),(1000,0),(1000,600),(0,600)],provenance=Provenance(provider='test',source='known answer',quality='provided'))
        return CameraState(frame=0,timestamp=0,calibration_valid=True,calibration=cal)
    def test_center_circle_evidence_alone_is_not_accepted_camera_geometry(self):
        c=self.camera().model_copy(update={'calibration_valid':False,'pitch_evidence':{'curves':[{'kind':'Circle central'}]}})
        self.assertEqual(project_player((525,340),c),(None,'NO_CAMERA_GEOMETRY'))
    def test_accepted_center_projection_and_explicit_rejections(self):
        c=self.camera();p,reason=project_player((525,340),c);np.testing.assert_allclose(p,(52.5,34));self.assertEqual(reason,'PROJECTED')
        self.assertEqual(project_player((1050,340),c)[1],'OUTSIDE_CALIBRATION_REGION');self.assertEqual(project_player((np.nan,340),c)[1],'INVALID_FOOT_POINT')
    def test_canonical_contract_preserves_projection_reason_and_unknown_role(self):
        from src.vision.world.provider import canonical_observation
        world=WorldModel('one',Config(min_track_hits=1));c=self.camera();p,reason=project_player((525,340),c)
        t=TrackObservation(track_id='T',hits=3,bbox=(515,280,20,60),detection_confidence=.9,pitch=p,projection_reason=reason)
        row=canonical_observation(world.update([t],c),25,'synthetic');self.assertEqual(row.players[0].projection_reason,'PROJECTED');self.assertEqual(row.players[0].role,'unknown_person');self.assertEqual(row.players[0].pitch,(52.5,34))
    def test_height_is_soft_and_contextual_not_a_universal_limit(self):
        ds=[Detection(bbox=(x,400,30,100),detection_confidence=.9) for x in (10,70,130,190)]+[Detection(bbox=(300,200,80,300),detection_confidence=.9)]
        result=assess_sizes(ds,CameraState(frame=0,timestamp=0),1080,1080);self.assertEqual(len(result),5);self.assertIn('LOCAL_SIZE_OUTLIER',result[-1].size_evidence.flags);self.assertGreater(result[-1].size_evidence.support,0)
