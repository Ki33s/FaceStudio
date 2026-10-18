import sys
from pathlib import Path
from collections import namedtuple
import unittest
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml'),str(ROOT/'runtime/facefusion')]
from difficult_detection import map_face,retry_faces,detector_options,unique_faces
from facefusion import state_manager

Face=namedtuple('MeasuredFace','origin bounding_box landmark_set score_set')
def face(box,score=.9):
    return Face('detect',np.array(box,np.float32),{'5':np.array([[120,140],[260,140],[190,200],[140,250],[240,250]],np.float32)},
                {'detector':score,'landmarker':score})


class DifficultDetection(unittest.TestCase):
    def setUp(self):
        self.old={key:state_manager.get_item(key) for key in ('face_detector_margin','face_detector_score')}
        state_manager.set_item('face_detector_margin',[60]*4);state_manager.set_item('face_detector_score',.35)
        self.addCleanup(self.restore)

    def restore(self):
        for key,value in self.old.items():state_manager.set_item(key,value)

    def test_lost_face_retried_when_other_person_is_already_found(self):
        frame=np.zeros((400,600,3),np.uint8)
        with patch('facefusion.face_creator.get_many_faces',return_value=[face([90,90,290,290])]) as detect:
            result=retry_faces(frame,[[200,100,300,200]],[face([20,20,80,80])])
        self.assertEqual(len(result),2);self.assertEqual(detect.call_count,1)
        np.testing.assert_allclose(result[1].bounding_box,[200,100,300,200],atol=.001)
        self.assertEqual(state_manager.get_item('face_detector_margin'),[60]*4)
        self.assertEqual(state_manager.get_item('face_detector_score'),.35)

    def test_options_restored_when_detector_fails(self):
        with patch('facefusion.face_creator.get_many_faces',side_effect=RuntimeError('detector failed')):
            with self.assertRaises(RuntimeError):retry_faces(np.zeros((400,600,3),np.uint8),[[200,100,300,200]])
        self.assertEqual(state_manager.get_item('face_detector_margin'),[60]*4)
        self.assertEqual(state_manager.get_item('face_detector_score'),.35)

    def test_no_measurement_produces_no_face(self):
        with patch('facefusion.face_creator.get_many_faces',return_value=[]) as detect:
            result=retry_faces(np.zeros((400,600,3),np.uint8),[[200,100,300,200]])
        self.assertEqual(result,[]);self.assertEqual(detect.call_count,3)

    def test_good_measured_region_does_not_need_retry(self):
        with patch('facefusion.face_creator.get_many_faces') as detect:
            result=retry_faces(np.zeros((400,600,3),np.uint8),[[200,100,300,200]],[face([200,100,300,200])])
        self.assertEqual(len(result),1);detect.assert_not_called()

    def test_box_uses_all_four_rotated_corners(self):
        measured=face([0,0,100,50]);matrix=cv2.getRotationMatrix2D((0,0),20,.5)
        result=map_face(measured,matrix)
        corners=cv2.transform(np.array([[[0,0],[100,0],[100,50],[0,50]]],np.float32),matrix)[0]
        np.testing.assert_allclose(result.bounding_box,np.r_[corners.min(axis=0),corners.max(axis=0)])
        np.testing.assert_allclose(result.landmark_set['5'],cv2.transform(measured.landmark_set['5'][None],matrix)[0])

    def test_duplicate_detectors_return_one_measurement(self):
        result=unique_faces([face([100,100,200,200],.5),face([101,101,201,201],.9)])
        self.assertEqual(len(result),1);self.assertEqual(result[0].score_set['detector'],.9)


if __name__=='__main__':unittest.main()
