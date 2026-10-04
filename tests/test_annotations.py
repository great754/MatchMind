import unittest

import numpy as np

from src.vision.annotations import ACTION_CLASSES, ClipAnnotations
from src.vision.coordinate_transform import PitchCoordinateTransformer
from src.vision.data_loading import DatasetPaths


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = DatasetPaths()
        cls.annotations = ClipAnnotations(cls.paths, PitchCoordinateTransformer(cls.paths.keypoints), 6000, 25)

    def test_half_clock_and_one_based_frame_alignment(self):
        a = self.annotations
        self.assertEqual(a.offset, 5999)
        self.assertEqual(len(a.events), 88)
        first = a.events[0]
        self.assertEqual(first['clip_frame'], int(first['frame'])-1-a.offset)
        self.assertEqual(first['clip_frame'], 37)
        for e in a.events:
            self.assertLessEqual(abs((int(e['frame'])-1)/25-e['half_video_seconds']), 0.04)
            self.assertTrue(0 <= e['clip_frame'] < 6000)
            self.assertTrue(e['gameTime'].startswith('1 -'))

    def test_ball_origin_and_unknown_samples(self):
        a = self.annotations
        with np.load(self.paths.ball_first_half) as raw:
            np.testing.assert_allclose(a.ball[0], [raw['x'][5999]+52.5, raw['y'][5999]+34])
        unknown = np.flatnonzero(a.status == 0)
        self.assertGreater(len(unknown), 0)
        self.assertIsNone(a.ball_position(int(unknown[0])))
        self.assertIsNotNone(a.ball_position(0))

    def test_all_action_classes_onset_and_expiry(self):
        a = object.__new__(ClipAnnotations)
        a.fps = 25
        a.events = [{'label': label, 'clip_frame':100} for label in ACTION_CLASSES]
        self.assertEqual(a.active_events(99), [])
        self.assertEqual(len(a.active_events(100)),12)
        self.assertEqual(len(a.active_events(124)),12)
        self.assertEqual(a.active_events(125),[])

    def test_goalkeepers_and_actor_identity_mapping(self):
        a = self.annotations
        self.assertEqual(sum(t == 0 for t in a.teams.values()),11)
        self.assertEqual(sum(t == 1 for t in a.teams.values()),11)
        self.assertEqual(a.teams[20],1)
        self.assertEqual(a.teams[21],0)
        self.assertEqual(a.actor_tracks['476371'],13)


if __name__ == '__main__':
    unittest.main()
