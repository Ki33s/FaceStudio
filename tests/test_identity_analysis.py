import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from identity_analysis import _components,reconcile,decode_blocks,_photo_labels,good_anchor
from video_tools import FFMPEG,FLAGS,metadata


def vector(index):
    value=np.zeros(512,np.float32);value[index]=1;return value


def track(identity,value):
    return {'gallery':[(identity,value,100.)],'labels':set(),'first':identity,'good':3}


class IdentityAnalysis(unittest.TestCase):
    def components(self,forward,backward,labels=None,old=(1,0,2),cuts=()):
        db=sqlite3.connect(':memory:');self.addCleanup(db.close)
        db.execute('CREATE TABLE nodes(n INTEGER,frame INTEGER,old_id INTEGER,v BLOB,quality REAL,good INTEGER)')
        db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?,?)',[(i,i,identity,vector(0).tobytes(),100.,1) for i,identity in enumerate(old)])
        return _components(db,len(old),{'edges':forward,'cuts':list(cuts)}, {'edges':backward,'cuts':[]}, labels or {})

    def test_forward_link_requires_reverse_confirmation(self):
        mapping,*_=self.components([[0,1,.99,1]],[])
        self.assertNotEqual(mapping[0],mapping[1])

    def test_reciprocal_path_joins_real_detections(self):
        edges=[[0,1,.99,1],[1,2,.99,1]]
        mapping,*_=self.components(edges,edges)
        self.assertEqual(len(set(mapping)),1)

    def test_path_does_not_cross_scene_cut(self):
        edges=[[0,1,.99,1],[1,2,.99,1]]
        mapping,*_=self.components(edges,edges,cuts=(1,))
        self.assertNotEqual(mapping[0],mapping[1])
        self.assertEqual(mapping[1],mapping[2])

    def test_unknown_bridge_cannot_mix_conflicting_photo_choices(self):
        edges=[[0,1,.99,1],[1,2,.99,1]]
        mapping,*_=self.components(edges,edges,{1:('photo-a',),2:('photo-b',)})
        self.assertNotEqual(mapping[0],mapping[2])

    def test_simultaneous_people_never_merge_even_with_identical_vectors(self):
        tracks={i:track(i,vector(0)) for i in (1,2,3)}
        groups,_,_=reconcile(tracks,{(1,2),(1,3),(2,3)})
        self.assertEqual(len(groups),3)

    def test_reappearance_can_join_across_scenes_by_gallery(self):
        tracks={1:track(1,vector(0)),2:track(2,vector(1)),3:track(3,vector(0))}
        groups,_,_=reconcile(tracks,{(1,2)})
        self.assertIn([1,3],groups)
        self.assertIn([2],groups)

    def test_ambiguous_reappearance_is_not_forced_onto_either_person(self):
        a=vector(0);b=a*.7+vector(1)*np.sqrt(.51);c=(a+b)/np.linalg.norm(a+b)
        tracks={1:track(1,a),2:track(2,b),3:track(3,c)}
        groups,_,_=reconcile(tracks,{(1,2)})
        self.assertEqual(len(groups),3)

    def test_explicit_keep_original_blocks_photo_propagation(self):
        tracks={1:track(1,vector(0)),2:track(2,vector(0))}
        tracks[1]['labels']={('photo-a',)};tracks[2]['labels']={()}
        groups,_,_=reconcile(tracks,set())
        self.assertEqual(len(groups),2)

    def test_same_photo_copied_under_different_names_is_one_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            first=Path(temporary)/'person-1.png';second=Path(temporary)/'person-2.png'
            first.write_bytes(b'identical photograph');second.write_bytes(first.read_bytes())
            photos,labels=_photo_labels({'assignments':[{'person_id':1,'sources':[str(first)]},
                                                       {'person_id':2,'sources':[str(second)]}]})
            self.assertEqual(labels[1],labels[2])
            self.assertEqual(len({labels[i]:photos[i] for i in (1,2)}),1)

    def test_collapsed_eyes_on_back_of_head_cannot_seed_a_person(self):
        face={'bounding_box':[0,0,300,600],'score_set':{'detector':.8,'landmarker':.8},
              'landmark_set':{'5':[[200,150],[203,152],[250,300],[195,450],[198,450]]}}
        self.assertFalse(good_anchor(face))
        face['landmark_set']['5']=[[80,150],[220,150],[150,260],[100,360],[200,360]]
        self.assertTrue(good_anchor(face))

    def test_text_commands_use_displayed_person_numbers_after_grouping(self):
        from person_groups import profiles_for_scan,command_people,resolve_people
        scan={'fingerprint':'fixture','characters':[{'id':i,'thumbnail':'x','first_seconds':n,'last_seconds':n+1} for n,i in enumerate((1,5,9,2))]}
        project={'profile_numbering':1,'persons':profiles_for_scan(scan,[[1,5]]),
                 'assignments':[{'person_id':i,'sources':['photo']} for i in (1,5,9)]}
        self.assertEqual(command_people(project),[1,2])
        self.assertEqual(resolve_people(project,[1]),[1,5])
        self.assertEqual(resolve_people(project,[2]),[9])
        with self.assertRaises(RuntimeError):resolve_people(project,[3])
        self.assertEqual(resolve_people({'assignments':[{'person_id':9}]},[9]),[9])

    def test_selected_profile_keeps_weak_appearance_without_a_new_person(self):
        a=vector(0);b=a*.2+vector(1)*np.sqrt(.96)
        tracks={1:track(1,a),2:track(2,b)}
        for t in tracks.values():t['old']={7:1};t['labels']={('selected-photo',)}
        groups,_,_=reconcile(tracks,set())
        self.assertEqual(groups,[[1,2]])

    @unittest.skipUnless(Path(FFMPEG).is_file(),'Bundled FFmpeg required')
    def test_reverse_blocks_match_forward_frames_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'clock.mkv'
            frames=np.empty((205,48,64,3),np.uint8)
            for index in range(len(frames)):frames[index]=[index,index*7%256,index*13%256]
            subprocess.run([FFMPEG,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','64x48','-r','10',
                '-i','pipe:0','-c:v','ffv1',str(path)],input=frames.tobytes(),check=True,creationflags=FLAGS)
            meta=metadata(path);forward={};backward={}
            for start,end,images,_ in decode_blocks(path,meta,block=31):
                forward.update({start+i:image.tobytes() for i,image in enumerate(images)})
            for start,end,images,_ in decode_blocks(path,meta,reverse=True,block=17):
                backward.update({start+i:image.tobytes() for i,image in enumerate(images)})
            self.assertEqual(len(forward),205)
            self.assertEqual(forward,backward)


if __name__=='__main__':unittest.main()
