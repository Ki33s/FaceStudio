import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from frame_selection import basic_reason,sharpness,ranges,selected_ids,apply_plan,build_plan,keep_short_dips

def face():
    return {'bounding_box':[50,40,180,190],'score_set':{'detector':.9,'landmarker':.9},
            'landmark_set':{'5/68':[[80,80],[140,80],[110,110],[90,145],[135,145]]},'embedding_norm':[1.,0.]}

class Selection(unittest.TestCase):
    def sample(self,identity=1,strong=True):
        return {'id':identity,'group':identity,'ordinal':0,'box':np.array([50,40,180,190],np.float32),
                'vector':np.array([1.,0.],np.float32),'scene':np.zeros((36,64),np.uint8),'strong':strong,'sharpness':30.}
    def test_clear_profile_is_not_rejected_by_frontal_ratios(self):
        f=face();f['landmark_set']['5/68']=[[115,80],[123,80],[145,110],[118,145],[137,145]]
        self.assertIsNone(basic_reason(f,320,240))
    def test_brief_confidence_dip_keeps_existing_identity(self):
        accepted,promotions,held=keep_short_dips({0:[self.sample()],1:[self.sample(strong=False)],2:[self.sample()]},30)
        self.assertEqual(accepted['1'],[1]);self.assertEqual(held,1);self.assertEqual(promotions,{})
    def test_measured_unknown_can_rejoin_same_person_between_anchors(self):
        unknown=self.sample(0,False)
        accepted,promotions,held=keep_short_dips({0:[self.sample()],1:[unknown],2:[self.sample()]},30)
        self.assertEqual(accepted['1'],[1]);self.assertEqual(promotions['1'],[{'ordinal':0,'person_id':1}])
    def test_clear_side_view_uses_both_identity_anchors(self):
        unknown=self.sample(0,False);unknown['landmarker']=.8
        unknown['vector']=np.array([.42,np.sqrt(1-.42**2)],np.float32)
        accepted,promotions,_=keep_short_dips({0:[self.sample()],1:[unknown],2:[self.sample()]},30)
        self.assertEqual(accepted['1'],[1]);self.assertEqual(promotions['1'][0]['person_id'],1)
        unknown['landmarker']=.5
        self.assertNotIn('1',keep_short_dips({0:[self.sample()],1:[unknown],2:[self.sample()]},30)[0])
    def test_empty_frame_is_never_filled_without_measurement(self):
        accepted,_,_=keep_short_dips({0:[self.sample()],2:[self.sample()]},30)
        self.assertNotIn('1',accepted)
    def test_long_absence_does_not_reuse_old_face(self):
        accepted,_,_=keep_short_dips({0:[self.sample()],10:[self.sample(strong=False)],20:[self.sample()]},30)
        self.assertNotIn('10',accepted)
    def test_cut_does_not_bridge_confidence_dip(self):
        unknown=self.sample(0,False);unknown['scene'][:]=255
        accepted,_,_=keep_short_dips({0:[self.sample()],1:[unknown],2:[self.sample()]},30)
        self.assertNotIn('1',accepted)
    def test_clear_measured_face_is_eligible(self):self.assertIsNone(basic_reason(face(),320,240))
    def test_weak_small_and_clipped_faces_are_rejected(self):
        f=face();f['score_set']['landmarker']=.3;self.assertEqual(basic_reason(f,320,240),'confidence')
        f=face();f['bounding_box']=[50,40,70,65];self.assertEqual(basic_reason(f,320,240),'small')
        f=face();f['landmark_set']['5/68'][0]=[-20,80];f['landmark_set']['5/68'][1]=[-1,80]
        self.assertIsNotNone(basic_reason(f,320,240))
    def test_blurred_texture_is_rejected(self):
        gray=np.random.default_rng(5).integers(0,255,(240,320),dtype=np.uint8)
        self.assertGreater(sharpness(gray,face()['bounding_box'],(1,1)),8)
        self.assertLess(sharpness(cv2.GaussianBlur(gray,(0,0),15),face()['bounding_box'],(1,1)),8)
    def test_intervals_keep_exact_frame_indices(self):self.assertEqual(ranges([0,1,2,7,8,11]),[[0,3],[7,9],[11,12]])
    def test_person_aliases_are_selected_together(self):
        self.assertEqual(selected_ids({'assignments':[{'person_id':1,'sources':['x']}],'person_groups':[[1,4]]},[4]),{1,4})
    def test_filter_preserves_unassigned_people_and_original_card_ids(self):
        scan={'characters':[{'id':1},{'id':2}],'frames':{'0':[{'person_id':1},{'person_id':2}], '1':[{'person_id':1}]}}
        apply_plan(scan,{'selected':[1],'accepted':{'1':[1]},'digest':'a'})
        self.assertEqual(scan['frames']['0'],[{'person_id':2}]);self.assertEqual(scan['characters'],[{'id':1},{'id':2}])
    def test_cached_plan_never_decodes_video_again(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'original.mp4';video.write_bytes(b'test')
            scan=root/'people.json';scan.write_text(json.dumps({'meta':{'width':320,'height':240,'fps':25,'frames':3},
                'frames':{'0':[{'person_id':1,'face':face()}],'1':[],'2':[{'person_id':1,'face':face()}]}}),encoding='utf-8')
            project={'folder':folder,'target':str(video),'people_path':str(scan),'assignments':[{'person_id':1,'sources':['photo']}]}
            gray=np.random.default_rng(3).integers(0,255,(240,320),dtype=np.uint8)
            with patch('frame_selection.gray_frames',return_value=iter([(i,gray,(1.,1.)) for i in range(3)])):
                path,plan=build_plan(project,0,3)
            self.assertEqual(plan['processed_frames'],2);self.assertEqual(plan['skipped_frames'],1)
            with patch('frame_selection.gray_frames',side_effect=AssertionError('Repeated decode')):
                self.assertEqual(build_plan(project,0,3)[1],plan)
    def test_duplicate_unknown_cannot_restore_original_over_replacement(self):
        records=[{'person_id':i,'face':face()} for i in [1,0,2]]
        distant={'person_id':0,'face':face()};distant['face']['bounding_box']=[200,40,300,190]
        scan={'frames':{'0':records+[distant]}}
        apply_plan(scan,{'selected':[1],'accepted':{'0':[1]},'digest':'a'})
        self.assertEqual([r['person_id'] for r in scan['frames']['0']],[1,2,0])
    def test_empty_plan_never_decodes_for_clarity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'original.mp4';video.write_bytes(b'test')
            scan=root/'people.json';scan.write_text(json.dumps({'meta':{'width':320,'height':240,'fps':25,'frames':3},'frames':{'0':[],'1':[],'2':[]}}),encoding='utf-8')
            project={'folder':folder,'target':str(video),'people_path':str(scan),'assignments':[{'person_id':1,'sources':['photo']}]}
            with patch('frame_selection.gray_frames',side_effect=AssertionError('Empty part decoded')):
                _,plan=build_plan(project,0,3)
            self.assertEqual(plan['processed_frames'],0);self.assertEqual(plan['skipped_frames'],3)

if __name__=='__main__':unittest.main()
