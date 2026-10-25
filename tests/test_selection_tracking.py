import copy,sys,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from selection_tracking import retain_measurements,measurement_patch,bridge_faces


class Tracking(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        self.gray=cv2.GaussianBlur(rng.integers(20,235,(240,320),dtype=np.uint8),(3,3),0)
        self.box=np.array([70,40,210,210.],np.float32)
        self.points=np.array([[110,90],[165,90],[140,125],[115,170],[162,170]],np.float32)
    def item(self,index,identity=1,strong=False,offset=0):
        box=self.box+[offset,0,offset,0]
        gray=cv2.warpAffine(self.gray,np.float32([[1,0,offset],[0,1,0]]),(320,240))
        face={'bounding_box':box.tolist(),'embedding_norm':[1.,0.],
              'landmark_set':{'5/68':(self.points+[offset,0]).tolist(),'68':np.repeat(self.points[:1]+[offset,0],68,axis=0).tolist()},
              'origin':'detect','score_set':{'detector':.9,'landmarker':.9}}
        return {'id':identity,'group':identity,'ordinal':0,'strong':strong,'sharpness':7.5,
                'vector':np.array([1.,0.]),'scene':cv2.resize(gray,(64,36)),
                'box':box,'points':self.points+[offset,0],'patch':measurement_patch(gray,box,(1,1)),'face':face}
    def test_small_sharpness_dip_holds_for_the_rest_of_same_measured_shot(self):
        samples={i:[self.item(i,strong=i==0,offset=i*.25)] for i in range(25)}
        accepted,_,held,_=retain_measurements(samples,30)
        self.assertEqual(len(accepted),25);self.assertEqual(held,24)
    def test_no_clear_entry_never_activates_weak_faces(self):
        self.assertEqual(retain_measurements({i:[self.item(i)] for i in range(8)},30)[0],{})
    def test_cut_does_not_carry_old_entry_to_a_weak_shot(self):
        left=self.item(0,strong=True);right=self.item(1);right['scene']=255-right['scene']
        right['patch']=(255-right['patch'][0],right['patch'][1])
        self.assertNotIn('1',retain_measurements({0:[left],1:[right]},30)[0])
    def test_different_named_people_never_join(self):
        self.assertNotIn('1',retain_measurements({0:[self.item(0,strong=True)],1:[self.item(1,2)]},30)[0])
    def test_missing_id_is_restored_only_between_named_anchors(self):
        data={i:[self.item(i,0 if i in (1,2) else 1,strong=i in (0,3))] for i in range(4)}
        accepted,promotions,_,identified=retain_measurements(data,30)
        self.assertEqual(accepted['1'],[1]);self.assertEqual(accepted['2'],[1]);self.assertEqual(identified,2)
        self.assertEqual(promotions['1'],[{'ordinal':0,'person_id':1}])
        data.pop(3)
        self.assertNotIn('1',retain_measurements(data,30)[0])
    def test_ambiguous_parallel_people_cannot_take_the_same_unknown(self):
        samples={0:[self.item(0,1,True),self.item(0,2,True)],1:[self.item(1,0)],
                 2:[self.item(2,1,True),self.item(2,2,True)]}
        self.assertNotIn('1',retain_measurements(samples,30)[0])
    def test_short_motion_bridge_uses_both_anchors(self):
        images=[cv2.warpAffine(self.gray,np.float32([[1,0,i],[0,1,0]]),(320,240)) for i in range(4)]
        left=self.item(0,offset=0)['face'];right=self.item(3,offset=3)['face']
        result=bridge_faces(images,left,right,(1.,1.))
        self.assertEqual(len(result),2)
        for i,face in enumerate(result,1):
            np.testing.assert_allclose(face['landmark_set']['5/68'],self.points+[i,0],atol=.3)
            self.assertEqual(face['origin'],'flow-bridge')
    def test_bridge_rejects_different_identity_or_wrong_reappearance(self):
        left=self.item(0)['face'];right=copy.deepcopy(left);right['embedding_norm']=[0.,1.]
        self.assertIsNone(bridge_faces([self.gray]*3,left,right,(1.,1.)))
        right=copy.deepcopy(left);right['landmark_set']['5/68']=(self.points+[40,0]).tolist()
        self.assertIsNone(bridge_faces([self.gray]*3,left,right,(1.,1.)))
    def test_occluder_and_empty_shot_are_not_filled(self):
        face=self.item(0)['face'];blank=np.full_like(self.gray,50)
        self.assertIsNone(bridge_faces([self.gray,blank,self.gray],face,face,(1.,1.)))
        self.assertIsNone(bridge_faces([self.gray]*10,face,face,(1.,1.)))
    def test_textured_occluder_cannot_be_a_face_bridge(self):
        face=self.item(0)['face'];occluded=self.gray.copy()
        occluded[50:200,75:210]=np.random.default_rng(99).integers(20,235,(150,135),dtype=np.uint8)
        self.assertIsNone(bridge_faces([self.gray,occluded,self.gray],face,face,(1.,1.)))

if __name__=='__main__':unittest.main()
