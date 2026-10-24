import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from pose_stability import smooth_measured


class PoseStability(unittest.TestCase):
    def setUp(self):
        self.profile=np.array([[100,100],[120,100],[45,160],[100,220],[120,220]],np.float32)
        self.front=np.array([[60,100],[160,100],[110,160],[80,220],[140,220]],np.float32)
    def test_turn_toward_front_does_not_distort_profile(self):
        result=smooth_measured(self.profile,[self.front]*10+[self.profile],[1]*11)
        np.testing.assert_allclose(result,self.profile)
    def test_opposite_profile_cannot_pull_landmarks(self):
        opposite=self.profile.copy();opposite[2,0]=175
        np.testing.assert_allclose(smooth_measured(self.profile,[opposite],[1]),self.profile)
    def test_small_noise_is_reduced_for_compressed_eyes(self):
        noisy=self.profile+np.array([2,1]);result=smooth_measured(noisy,[self.profile]*4,[1]*4)
        self.assertLess(np.linalg.norm(result-self.profile),np.linalg.norm(noisy-self.profile))
        self.assertGreater(np.linalg.norm(result-self.profile),0)
    def test_large_bad_measurement_does_not_drag_anchor(self):
        np.testing.assert_allclose(smooth_measured(self.profile,[self.profile+80],[1]),self.profile)

if __name__=='__main__':unittest.main()
