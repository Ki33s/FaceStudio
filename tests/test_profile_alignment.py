import unittest,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from profile_alignment import profile_matrix

class ProfileAlignment(unittest.TestCase):
 def setUp(self):
  self.five=np.array([[100,100],[120,100],[180,160],[100,220],[120,220]],np.float32)
  self.points=np.zeros((68,2),np.float32)
  self.points[36]=[80,100];self.points[39]=[105,100];self.points[42]=[115,100];self.points[45]=[120,100]
  self.matrix=cv2.getRotationMatrix2D((0,0),30,1.8);self.matrix[:,2]=[20,50]
 def test_profile_roll_corrected_without_stretching(self):
  result,strength=profile_matrix(self.five,self.points,self.matrix)
  self.assertAlmostEqual(strength,1)
  self.assertLess(abs(np.degrees(np.arctan2(result[0,1],result[0,0]))),.05)
  np.testing.assert_allclose(result[:,:2]@result[:,:2].T,np.eye(2)*1.8**2,atol=1e-6)
 def test_nose_and_scale_preserved(self):
  result,_=profile_matrix(self.five,self.points,self.matrix)
  nose=self.five[2]
  np.testing.assert_allclose(result[:,:2]@nose+result[:,2],self.matrix[:,:2]@nose+self.matrix[:,2],atol=1e-5)
 def test_frontal_transform_bitwise_unchanged(self):
  front=self.five.copy();front[2,0]=110
  result,strength=profile_matrix(front,self.points,self.matrix)
  self.assertEqual(strength,0);np.testing.assert_array_equal(result,self.matrix)
 def test_transition_is_continuous(self):
  angles=[]
  for nose in np.linspace(143,174,100):
   points=self.five.copy();points[2,0]=nose
   result,_=profile_matrix(points,self.points,self.matrix);angles.append(np.degrees(np.arctan2(result[0,1],result[0,0])))
  self.assertLess(max(abs(np.diff(angles))),1)
 def test_invalid_landmarks_fall_back(self):
  for points in [np.full((68,2),np.nan),np.zeros((5,2)),np.zeros((68,2))]:
   result,strength=profile_matrix(self.five,points,self.matrix)
   self.assertEqual(strength,0);np.testing.assert_array_equal(result,self.matrix)
 def test_both_profile_directions_are_supported(self):
  points=self.five.copy();points[2,0]=40
  result,strength=profile_matrix(points,self.points,self.matrix)
  self.assertEqual(strength,1);self.assertLess(abs(result[0,1]),.01)
 def test_small_eye_cannot_rotate_crop(self):
  points=self.points.copy();points[39]=points[36]+[1,0];points[45]=points[42]+[1,0]
  result,strength=profile_matrix(self.five,points,self.matrix)
  self.assertEqual(strength,0)

if __name__=='__main__':unittest.main()
