import unittest,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from profile_alignment import profile_matrix,protect_features

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

class ProfileFeatureProtection(unittest.TestCase):
 def setUp(self):
  self.points=np.zeros((68,2),np.float32)
  self.points[36:42]=[70,50];self.points[42:48]=[95,50]
  self.points[27:36]=[[90,60],[90,70],[90,80],[90,90],[80,90],[85,95],[90,95],[95,95],[100,90]]
  angles=np.linspace(0,2*np.pi,20,endpoint=False)
  self.points[48:68]=np.c_[90+15*np.cos(angles),130+5*np.sin(angles)]
  self.mask=np.ones((192,192),np.float32)
 def test_good_geometry_is_untouched(self):
  result,count=protect_features(self.mask,self.points,self.points,1)
  self.assertEqual(count,0);np.testing.assert_array_equal(result,self.mask)
 def test_displaced_mouth_preserves_both_old_and_generated_locations(self):
  pred=self.points.copy();pred[48:68]+=[-28,0]
  result,count=protect_features(self.mask,self.points,pred,.9)
  self.assertEqual(count,1);self.assertEqual(result[130,90],0);self.assertEqual(result[130,62],0)
  self.assertEqual(result[50,80],1);self.assertEqual(result[0,0],1)
 def test_occluder_holes_can_never_be_filled(self):
  pred=self.points.copy();pred[48:68]+=[-28,0];mask=self.mask.copy();mask[100:120,100:120]=0
  result,_=protect_features(mask,self.points,pred,.9)
  self.assertTrue(np.all(result<=mask));self.assertTrue(np.all(result>=0))
 def test_unreliable_generated_measurement_does_not_change_mask(self):
  result,count=protect_features(self.mask,self.points,self.points+50,.2)
  self.assertEqual(count,0);np.testing.assert_array_equal(result,self.mask)
 def test_invalid_measurement_is_ignored(self):
  result,count=protect_features(self.mask,self.points,np.full((68,2),np.nan),1)
  self.assertEqual(count,0)
 def test_soft_mask_has_no_hard_edge(self):
  pred=self.points.copy();pred[48:68]+=[-28,0]
  result,_=protect_features(self.mask,self.points,pred,.9)
  self.assertTrue(np.any((result>0)&(result<1)))

if __name__=='__main__':unittest.main()
