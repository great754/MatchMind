import unittest
from types import SimpleNamespace
import numpy as np
from pydantic import ValidationError
from src.observations.schema import *
from src.observations.soccertrack import observations,analytics_arrays

class ObservationTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(source_files=['gsr.npz','bas.json'],source_offset=5,fps=25,half=1,
            player_ids=[101,102],player_teams=['left','unknown'],
            positions=np.array([[[1.,2.],[np.nan,np.nan]],[[2.,3.],[4.,5.]]]),ball=np.array([[np.nan,np.nan],[3.,4.]]))
    def test_compatibility_missing_unknown_and_provenance(self):
        r=self.fixture();frames=list(observations(r,'clip'))
        self.assertIsNone(frames[0].players[1].pitch);self.assertIsNone(frames[0].players[1].team)
        self.assertIsNone(frames[0].ball.pitch);self.assertIsNone(frames[0].calibration)
        self.assertEqual(frames[0].half,1)
        self.assertEqual(frames[0].coordinate_system,'pitch_corner_metres')
        self.assertEqual(frames[0].provenance.source,'gsr.npz,bas.json')
        self.assertEqual(frames[0].ball.provenance.quality,'reconstructed')
        positions,ball=analytics_arrays(frames)
        np.testing.assert_equal(positions,r.positions);np.testing.assert_equal(ball,r.ball)
        self.assertIsNone(frames[0].players[0].identity.confidence)
    def test_schema_validation(self):
        f=list(observations(self.fixture(),'clip'))[0].model_dump()
        for changes in ({'fps':0},{'timestamp':2},{'frame':-1},{'unknown':'field'}):
            with self.assertRaises(ValidationError):RecordingObservation.model_validate(dict(f,**changes))
        with self.assertRaises(ValidationError):Provenance(provider='cv',source='x',confidence=1.1)
        with self.assertRaises(ValidationError):PlayerObservation(position=f['provenance'],identity=f['provenance'],pitch=(float('nan'),2))
    def test_unknown_identity_and_duplicate_rejection(self):
        f=list(observations(self.fixture(),'clip'))[0].model_dump();f['players'][0]['actor_id']=None
        self.assertIsNone(RecordingObservation.model_validate(f).players[0].actor_id)
        f['players'][0]['actor_id']=f['players'][1]['actor_id']
        with self.assertRaises(ValidationError):RecordingObservation.model_validate(f)
    def test_clock_and_recording_isolation(self):
        frames=list(observations(self.fixture(),'clip'));frames[1]=frames[1].model_copy(update={'recording_id':'other'})
        with self.assertRaises(ValueError):analytics_arrays(frames)
    def test_calibration_range(self):
        p=Provenance(provider='cv',source='calibration')
        with self.assertRaises(ValidationError):PitchCalibrationObservation(start_frame=5,end_frame=2,provenance=p)

    def test_adapter_preserves_unknown_actor(self):
        r=self.fixture();r.player_ids=[None,102]
        self.assertIsNone(next(observations(r,'clip')).players[0].actor_id)
    def test_adapter_rejects_unaligned_arrays(self):
        r=self.fixture();r.player_teams=['left']
        with self.assertRaises(ValueError):list(observations(r,'clip'))
