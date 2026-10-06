import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.vision.data_loading import DatasetPaths
from src.vision.pitch import create_pitch, draw_player, pitch_to_screen
from src.vision.tactical_tracking import mapped_gsr_samples, compare_positions, prepare_tactical_tracking
from src.vision.team_classifier import infer_goalkeepers, cluster_team_colors


class TacticalTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths=DatasetPaths('test',Path(self.temp.name))
        (self.paths.root/'mot').mkdir()
        (self.paths.root/'gsr'/'test').mkdir(parents=True)
        self.paths.raw.mkdir(parents=True)
        self.mot=pd.DataFrame({'frame':[0]*4+[1]*4,'player_id':[1,2,3,4]*2,
                               'foot_x':[100,200,300,400]*2,'foot_y':[250]*8})
        self.mapping={str(t):{'player_id':a,'team':0 if t<3 else 1}
                      for t,a in enumerate([101,102,201,202],1)}
        sync=dict(match_id='test',fps=25,half=1,clip_start_frame_zero_based=1,
                  players=self.mapping,side_to_color={'0':1,'1':0})
        (self.paths.root/'mot'/'test_sync.json').write_text(json.dumps(sync))
        self.ids=np.array([[0,0,0,0],[101,102,201,202],[202,201,102,101]])
        self.pitch=np.array([[[np.nan,np.nan]]*4,[[10,34],[30,34],[90,34],[70,34]],
                             [[71,34],[91,34],[31,34],[11,34]]])
        self.sides=np.array([[-1]*4,[0,0,1,1],[1,1,0,0]])
        self.save_compact()
        (self.paths.raw/'test_tracker_box_metadata.xml').write_text(
            '<metadata><players><player id="101" position="GK"/>'
            '<player id="201" position="GK"/></players></metadata>')

    def save_compact(self):
        np.savez(self.paths.root/'gsr'/'test'/'1st_compact.npz',
                 ids=self.ids,pitch=self.pitch,sides=self.sides)

    def prepare(self, fps=25):
        with patch('src.vision.tactical_tracking.PitchCoordinateTransformer') as transform:
            transform.return_value.transform_points.return_value=np.tile([5.,34.],(8,1))
            with patch('src.vision.tactical_tracking.classify_teams',side_effect=AssertionError('Should use GSR labels')):
                return prepare_tactical_tracking(self.paths,self.mot,fps)[0]

    def test_join_uses_sync_offset_and_actor_id_despite_slot_changes(self):
        p,s=mapped_gsr_samples(self.mot,self.ids,self.pitch,self.sides,self.mapping,1)
        np.testing.assert_equal(p[:,0],[10,30,90,70,11,31,91,71])
        np.testing.assert_equal(s,[0,0,1,1,0,0,1,1])

    def test_primary_gsr_teams_include_different_kit_goalkeepers(self):
        tracking=self.prepare()
        self.assertEqual(tracking.teams,{1:1,2:1,3:0,4:0})
        self.assertEqual(tracking.goalkeepers,{1,3})
        self.assertEqual(tracking.comparison['position_source_counts'],{'gsr':8})
        self.assertEqual(tracking.comparison['frames_compared'],2)
        self.assertEqual(tracking.comparison['mean_m'],45.5)

    def test_only_missing_gsr_coordinate_uses_projection(self):
        self.pitch[2,2]=np.nan
        self.save_compact()
        tracking=self.prepare()
        self.assertEqual(tracking.comparison['position_source_counts'],{'gsr':7,'pitch_transform':1})
        row=tracking.rows[(tracking.rows.frame==1)&(tracking.rows.player_id==2)].iloc[0]
        self.assertEqual(row.position_source,'pitch_transform')
        self.assertEqual(row.pitch_x,5)
        self.assertEqual(tracking.teams[2],1)

    def test_present_off_pitch_gsr_is_not_replaced_with_projection(self):
        self.pitch[1,0,0]=-2
        self.save_compact()
        tracking=self.prepare()
        self.assertEqual(tracking.rows.iloc[0].position_source,'gsr')
        self.assertEqual(tracking.rows.iloc[0].pitch_x,-2)

    def test_missing_gsr_file_keeps_sync_team_labels_and_falls_back(self):
        (self.paths.root/'gsr'/'test'/'1st_compact.npz').unlink()
        tracking=self.prepare()
        self.assertEqual(tracking.comparison['position_source_counts'],{'pitch_transform':8})
        self.assertIsNone(tracking.comparison['mean_m'])
        self.assertEqual(tracking.teams[1],1)
        self.assertEqual(tracking.teams[3],0)

    def test_raw_gsr_is_used_when_compact_cache_is_missing(self):
        (self.paths.root/'gsr'/'test'/'1st_compact.npz').unlink()
        annotations=[]
        for frame in (1,2):
            for slot,actor in enumerate(self.ids[frame]):
                x,y=self.pitch[frame,slot]
                annotations.append(dict(image_id=str(frame+1),category_id=1,
                                        attributes=dict(player_id=int(actor),team='left' if self.sides[frame,slot]==0 else 'right'),
                                        bbox_pitch=dict(x_bottom_middle=x-52.5,y_bottom_middle=y-34)))
        raw=self.paths.root/'gsr'/'test'/'test_1st.json'
        raw.write_text(json.dumps({'annotations':annotations},indent=4))
        tracking=self.prepare()
        self.assertEqual(tracking.comparison['position_source_counts'],{'gsr':8})
        self.assertEqual(tracking.comparison['gsr_source'],str(raw))
        np.testing.assert_equal(tracking.rows.pitch_x.to_numpy(),[10,30,90,70,11,31,91,71])

    def test_duplicate_actor_and_conflicting_team_fail_explicitly(self):
        self.ids[1]=[101,101,201,202]
        self.save_compact()
        with self.assertRaises(ValueError):self.prepare()
        self.ids[1]=[101,102,201,202]
        self.sides[2,3]=1
        self.save_compact()
        with self.assertRaises(ValueError):self.prepare()

    def test_wrong_fps_and_missing_mapping_are_rejected(self):
        with self.assertRaises(ValueError):self.prepare(30)
        path=self.paths.root/'mot'/'test_sync.json'
        sync=json.loads(path.read_text());del sync['players']['4'];path.write_text(json.dumps(sync))
        with self.assertRaises(ValueError):self.prepare()

    def test_comparison_excludes_missing_pairs_and_reports_meters(self):
        mot=pd.DataFrame({'frame':[0,1,2],'player_id':[1,1,1]})
        report,frames=compare_positions(mot,np.array([[0,0],[0,0],[np.nan,np.nan]]),
                                       np.array([[3,4],[0,2],[1,1]]))
        self.assertEqual(report['samples'],2)
        self.assertEqual(report['mean_m'],3.5)
        self.assertEqual(report['frames_compared'],2)
        self.assertAlmostEqual(report['comparison_coverage_fraction'],2/3)


class KeeperFallbackTests(unittest.TestCase):
    def test_persistent_depth_not_single_frame_extreme(self):
        mot=pd.DataFrame({'player_id':[1,2,3,4]*10,'pitch_x':[10,40,60,95]*10})
        mot.loc[1,'pitch_x']=0
        self.assertEqual(infer_goalkeepers(mot),{1:'left',4:'right'})

    def test_keeper_jerseys_do_not_affect_outfield_colors(self):
        bgr=np.uint8([[[220,50,30],[230,60,40],[220,220,220],[230,230,230],
                       [0,0,255],[0,255,255]]])
        lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)[0]
        colors={p:c for p,c in enumerate(lab,1)}
        depth={1:70,2:75,3:30,4:35,5:10,6:95}
        assignment=cluster_team_colors(colors,depth,{5:'left',6:'right'})
        self.assertEqual(assignment[1],0);self.assertEqual(assignment[3],1)
        self.assertEqual(assignment[5],1);self.assertEqual(assignment[6],0)

    def test_keeper_marker_keeps_team_color(self):
        pitch=create_pitch();draw_player(pitch,20,34,20,team=1,show_id=False,is_goalkeeper=True)
        x,y=pitch_to_screen(20,34)
        self.assertGreater(pitch[y,x,2],pitch[y,x,0])
        self.assertGreater(pitch[y,x+14,0],200)


if __name__=='__main__':unittest.main()
