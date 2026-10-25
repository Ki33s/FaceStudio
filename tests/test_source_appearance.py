import sys,tempfile,unittest,base64
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from facial_hair import FacialHair,measure_hair,hair_regions,restore_source_detail
from test_profile_detail import landmarks
from occlusion_context import visible_bridge,project_visibility


class SourceHair(unittest.TestCase):
    def setUp(self):
        self.points=landmarks();self.skin=np.full((256,256,3),[150,180,210],np.uint8)
        region=hair_regions(self.points,(256,256))[0]>0
        self.hairy=self.skin.copy();rng=np.random.default_rng(11)
        self.hairy[region]=rng.integers(10,55,(int(region.sum()),3),dtype=np.uint8)
    def test_visible_moustache_is_preserved_and_clean_skin_is_not(self):
        mask,_,scores=measure_hair(self.hairy,self.points)
        self.assertTrue(scores['mustache']['confident']);self.assertGreater(float(mask.max()),.8)
        renderer=FacialHair();self.assertGreater(renderer.apply(1,0,self.hairy,self.points).max(),.8)
        self.assertEqual(FacialHair().apply(2,0,self.skin,self.points).max(),0)
    def test_connected_moustache_is_not_cut_at_the_landmark_seed(self):
        regions=hair_regions(self.points,(256,256));outside=(regions[0]>0)&(regions[4]==0)
        mask=measure_hair(self.hairy,self.points)[0]
        self.assertGreater(mask[outside].max(),.8)
    def test_global_brightness_change_does_not_create_new_hair(self):
        renderer=FacialHair()
        for index,offset in enumerate([0,-25,15,-10]):
            image=np.clip(self.skin.astype(int)+offset,0,255).astype(np.uint8)
            self.assertEqual(renderer.apply(1,index,image,self.points).max(),0)
    def test_dark_open_mouth_is_not_a_new_moustache(self):
        image=self.skin.copy();opening=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(opening,np.rint(self.points[60:68]).astype(np.int32),1)
        image[opening>0]=15
        _,_,scores=measure_hair(image,self.points)
        self.assertFalse(scores['mustache']['confident'])
    def test_skin_coloured_shadow_does_not_start_a_moustache(self):
        lab=cv2.cvtColor(self.skin,cv2.COLOR_BGR2LAB)
        region=hair_regions(self.points,(256,256))[0]>0
        light=lab[:,:,0].copy();rng=np.random.default_rng(3)
        light[region]=np.clip(light[region].astype(int)-55+rng.integers(-3,4,int(region.sum())),0,255)
        lab[:,:,0]=light;image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        self.assertFalse(measure_hair(image,self.points)[2]['mustache']['confident'])
    def test_state_cannot_move_original_hair_onto_another_person(self):
        renderer=FacialHair();renderer.apply(1,0,self.hairy,self.points)
        self.assertEqual(renderer.apply(2,1,self.skin,self.points).max(),0)
    def test_brief_bad_lip_landmarks_do_not_erase_visible_original_hair(self):
        renderer=FacialHair();first=renderer.apply(1,0,self.hairy,self.points)
        changed=self.points.copy();changed[48:68]+=[50,20]
        second=renderer.apply(1,1,self.hairy,changed)
        reliable=first>.9
        self.assertGreater(float(second[reliable].mean()),.8)
    def test_removed_hair_is_not_carried_onto_a_clean_current_frame(self):
        renderer=FacialHair();renderer.apply(1,0,self.hairy,self.points)
        self.assertEqual(renderer.apply(1,1,self.skin,self.points).max(),0)
    def test_only_opacity_is_kept_not_past_generated_rgb(self):
        renderer=FacialHair();renderer.apply(1,0,self.hairy,self.points)
        mask=renderer.apply(1,1,self.hairy,self.points)
        self.assertEqual(mask.ndim,2)
        self.assertNotIn('crop',renderer.states[1]);self.assertNotIn('image',renderer.states[1])
    def test_snapshot_preserves_same_temporal_mask_at_part_boundary(self):
        renderer=FacialHair();renderer.apply(1,0,self.hairy,self.points)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'hair.npz';renderer.save(path);resumed=FacialHair();resumed.load(path)
        np.testing.assert_allclose(renderer.apply(1,1,self.hairy,self.points),resumed.apply(1,1,self.hairy,self.points))
    def test_repair_can_restore_hair_erased_in_a_previous_result(self):
        previous=np.full_like(self.hairy,[200,100,80]);mask=FacialHair().apply(1,0,self.hairy,self.points)
        result=restore_source_detail(previous,self.hairy,mask,np.float32([[1,0,0],[0,1,0]]))
        pixel=np.unravel_index(mask.argmax(),mask.shape)
        self.assertLess(np.abs(result[pixel].astype(float)-self.hairy[pixel]).max(),2)
        np.testing.assert_array_equal(result[mask==0],previous[mask==0])
        np.testing.assert_array_equal(previous,np.full_like(previous,[200,100,80]))


class OcclusionContext(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        gray=cv2.GaussianBlur(rng.integers(40,220,(240,320),dtype=np.uint8),(3,3),0)
        self.images=[cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) for _ in range(10)]
        self.masks=[]
        for index,image in enumerate(self.images):
            mask=np.zeros((240,320),np.float32);mask[40:210,70:210]=1
            if index not in (0,9):image[65:155,145:205]=35;mask[65:155,145:205]=0
            image[0,0]=index;self.masks.append(mask)
        points=np.array([[110,90],[165,90],[140,125],[115,170],[162,170]],np.float32)
        self.face={'bounding_box':[70,40,210,210],'embedding_norm':[1.,0.],
                   'landmark_set':{'5/68':points.tolist(),'68':np.repeat(points[:1],68,axis=0).tolist()},'origin':'detect'}
    def measure(self,image,face):
        index=int(image[0,0,0]);mask=self.masks[index]
        return mask,np.float32([[1,0,0],[0,1,0]]),mask
    def test_partial_hand_does_not_break_visible_face_tracking(self):
        result=visible_bridge(self.images,self.face,self.face,self.measure,maximum=24)
        self.assertIsNotNone(result);self.assertEqual(len(result),8)
        face,visibility=result[3];self.assertEqual(face['origin'],'occlusion-context')
        mask=project_visibility(visibility,np.float32([[1,0,0],[0,1,0]]),(240,320),(320,240))
        self.assertEqual(mask[100,175],0);self.assertEqual(mask[100,110],1)
    def test_completely_hidden_face_keeps_source_instead_of_guessing(self):
        self.masks[5][:]=0
        self.assertIsNone(visible_bridge(self.images,self.face,self.face,self.measure))
    def test_no_occluder_never_uses_permissive_context_path(self):
        self.masks=[self.masks[0]]*10
        self.assertIsNone(visible_bridge(self.images,self.face,self.face,self.measure))
    def test_other_identity_or_cut_never_carries_a_face(self):
        other={**self.face,'embedding_norm':[0.,1.]}
        self.assertIsNone(visible_bridge(self.images,self.face,other,self.measure))
        self.images[4][:]=255
        self.assertIsNone(visible_bridge(self.images,self.face,self.face,self.measure))
    def test_visibility_projection_follows_output_scale(self):
        mask=self.masks[4];ok,data=cv2.imencode('.png',np.rint(mask*255).astype(np.uint8));self.assertTrue(ok)
        value={'matrix':[[1,0,0],[0,1,0]],'size':[320,240],'png':base64.b64encode(data.tobytes()).decode()}
        output=project_visibility(value,np.float32([[.5,0,0],[0,.5,0]]),(240,320),(640,480))
        np.testing.assert_allclose(output,mask)

if __name__=='__main__':unittest.main()
