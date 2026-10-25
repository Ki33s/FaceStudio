import sys,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from profile_detail import ProfileDetails,detail_mask,anchor_parameters,correction_matrix,smooth_dense,pose_weights,five,profile_strength


def landmarks():
    p=np.zeros((68,2),np.float32)
    theta=np.linspace(0,np.pi,17);p[:17]=np.c_[120-60*np.cos(theta),90+95*np.sin(theta)]
    for start,x,y in [(17,85,65),(22,110,68),(27,135,85),(31,145,110),
                       (36,85,85),(42,110,88),(48,125,155),(60,125,155)]:
        count={17:5,22:5,27:4,31:5,36:6,42:6,48:12,60:8}[start]
        a=np.linspace(0,np.pi*2,count,endpoint=False)
        p[start:start+count]=np.c_[x+10*np.cos(a),y+4*np.sin(a)]
    for start,x,y in [(36,85,85),(42,110,88)]:
        p[start:start+6]=np.array([[-10,0],[-5,-4],[5,-4],[10,0],[5,4],[-5,4]])+[x,y]
    p[30]=[155,115];p[48]=[115,155];p[54]=[135,155]
    return p


class ProfileDetailsTests(unittest.TestCase):
    def test_looking_down_with_wide_eyes_is_not_a_profile(self):
        points=np.array([[80,70],[180,70],[110,103],[98,130],[162,130]],np.float32)
        self.assertEqual(profile_strength(points),0.)
    def test_turn_still_uses_partial_compositing_with_compressed_eyes(self):
        points=np.array([[110,80],[120,80],[145,115],[112,150],[132,150]],np.float32)
        self.assertGreater(profile_strength(points),.95)
    def setUp(self):
        self.points=landmarks();self.mask=np.ones((256,256),np.float32)
        y,x=np.mgrid[:256,:256];gray=((x+y)%180+30).astype(np.uint8)
        self.original=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);self.crop=np.full_like(self.original,[30,90,170])
    def apply(self,renderer,index=0,**kwargs):
        data=dict(identity=1,index=index,original=self.original,crop=self.crop,base_mask=self.mask,
                  points=self.points,predicted=self.points,confidence=.9,gain=1.)
        data.update(kwargs);return renderer.apply(**data)
    def test_partial_region_contains_visible_eye_and_preserves_nose(self):
        mask=detail_mask(self.points,(256,256),0)
        self.assertGreater(mask[85,85],.8);self.assertEqual(mask[115,155],0)
        self.assertEqual(mask[155,125],0);self.assertEqual(mask[0,0],0)
    def test_confidence_crossing_old_point_six_gate_cannot_flip_opacity(self):
        r=ProfileDetails();_,before=self.apply(r,confidence=.599)
        _,after=self.apply(r,1,confidence=.601)
        self.assertLess(np.abs(after-before).max(),.04)
    def test_missing_measurement_holds_alignment_instead_of_full_face(self):
        r=ProfileDetails();self.apply(r)
        _,mask=self.apply(r,1,predicted=None,confidence=0)
        self.assertEqual(mask[155,125],0);self.assertGreater(mask[85,85],.8)
    def test_current_occluder_always_wins(self):
        r=ProfileDetails();self.apply(r);mask=self.mask.copy();mask[70:100,70:105]=0
        _,result=self.apply(r,1,base_mask=mask)
        self.assertEqual(result[85,85],0);self.assertTrue(np.all(result<=mask))
    def test_generated_rgb_is_never_reused(self):
        r=ProfileDetails();self.apply(r);blue=np.full_like(self.crop,[255,0,0])
        output,_=self.apply(r,1,crop=blue)
        np.testing.assert_array_equal(output,blue)
    def test_frame_gap_does_not_transport_mask(self):
        r=ProfileDetails();self.apply(r);self.apply(r,8)
        self.assertEqual(r.stats['transported_masks'],0)
    def test_scene_cut_resets_history(self):
        r=ProfileDetails();self.apply(r);self.apply(r,1,original=np.zeros_like(self.original))
        self.assertEqual(r.stats['transported_masks'],0)
    def test_people_do_not_share_history(self):
        r=ProfileDetails();self.apply(r);self.apply(r,1,identity=2)
        self.assertEqual(r.stats['transported_masks'],0);self.assertEqual(len(r.states),2)
    def test_planned_gain_has_no_causal_delay(self):
        r=ProfileDetails();self.apply(r);self.apply(r,1,gain=0)
        self.assertAlmostEqual(r.states[1]['gain'],0)
    def test_nose_and_visible_eye_pin_with_valid_similarity(self):
        predicted=self.points+[3,-2]
        p=anchor_parameters(self.points,predicted,1,0);matrix=correction_matrix(p,self.points)
        output=cv2.transform(predicted[None].astype(np.float32),matrix)[0]
        np.testing.assert_allclose(output[[30,36]],self.points[[30,36]],atol=1e-5)
    def test_bad_hidden_eye_does_not_move_visible_anchors(self):
        predicted=self.points.copy();predicted[42:48]+=[40,-40]
        result=anchor_parameters(self.points,predicted,1,0)
        np.testing.assert_allclose(result,0,atol=1e-6)
    def test_opposite_pose_cannot_smooth_dense_points(self):
        from profile_detail import five
        front=five(self.points);other=front.copy();other[2,0]=20
        result=smooth_dense(self.points,front,[self.points+20],[other],[1])
        np.testing.assert_array_equal(result,self.points)
    def test_mask_releases_to_frontal_face_over_multiple_frames(self):
        r=ProfileDetails();self.apply(r)
        means=[]
        for index in range(1,13):
            _,mask=self.apply(r,index,gain=0);means.append(float(mask.mean()))
        self.assertLess(means[0],means[-1]);self.assertGreater(means[-1],.97)
        self.assertEqual(r.states[1]['tail'],0)
    def test_snapshot_continues_same_mask_at_a_chunk_boundary(self):
        import tempfile
        r=ProfileDetails();self.apply(r)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.npz';r.save(path);resumed=ProfileDetails();resumed.load(path)
        expected,mask=self.apply(r,1);actual,resumed_mask=self.apply(resumed,1)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(mask,resumed_mask)
        self.assertEqual(resumed.stats['resets'],0)
    def test_compressed_eyes_identify_profile_even_with_small_yaw(self):
        p=np.array([[100,100],[120,105],[85,160],[100,220],[120,220]],np.float32)
        self.assertEqual(profile_strength(p),1)
    def test_context_is_ready_before_profile_turn(self):
        front=five(self.points);front[0,0]=60;front[1,0]=140;front[2,0]=100
        plan=pose_weights([front]*20+[five(self.points)]*10,[True]*29)
        self.assertGreater(plan[19],.95);self.assertEqual(plan[5],0)
    def test_pose_plan_does_not_cross_scene_cut(self):
        front=five(self.points);front[0,0]=60;front[1,0]=140;front[2,0]=100
        links=[True]*29;links[19]=False
        plan=pose_weights([front]*20+[five(self.points)]*10,links)
        self.assertEqual(plan[19],0);self.assertEqual(plan[20],1)
    def test_one_noisy_frontal_measurement_cannot_toggle_profile(self):
        front=five(self.points);front[0,0]=60;front[1,0]=140;front[2,0]=100
        points=[five(self.points)]*20;points[10]=front
        plan=pose_weights(points,[True]*19)
        self.assertEqual(plan[10],1)

if __name__=='__main__':unittest.main()
