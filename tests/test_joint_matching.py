import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/directml')]
from joint_matching import assign
from people import Registry


class JointMatching(unittest.TestCase):
    def test_global_solution_resolves_local_competition(self):
        edges=[(.90,0,1),(.83,0,2),(.89,1,1),(.20,1,2)]
        self.assertEqual({(r,c) for _,r,c in assign(edges)}, {(0,2),(1,1)})

    def test_ambiguous_component_does_not_discard_independent_person(self):
        edges=[(.9,0,1),(.7,1,2),(.7,1,3),(.7,2,2),(.7,2,3)]
        self.assertEqual([(r,c) for _,r,c in assign(edges)],[(0,1)])

    def test_unknown_slots_are_not_forced_to_a_person(self):
        self.assertEqual(assign([(.01,0,1)]),[])
        self.assertEqual(assign([(float('nan'),0,1)]),[])

    def test_blocked_people_and_faces_are_protected(self):
        edges=[(.95,0,1),(.91,1,2),(.89,2,3)]
        self.assertEqual([(r,c) for _,r,c in assign(edges,blocked_rows={0},blocked_columns={2})],[(2,3)])

    def test_order_does_not_change_physical_assignment(self):
        edges=[(.9,'a',1),(.83,'a',2),(.89,'b',1),(.2,'b',2),(.5,'a',1)]
        self.assertEqual({(r,c) for _,r,c in assign(edges)}, {(r,c) for _,r,c in assign(list(reversed(edges)))})

    def registry(self):
        registry=Registry()
        for axis in (0,1):
            vector=np.zeros(512,np.float32);vector[axis]=1
            registry.people.append({'vector':vector,'gallery':[vector.tolist()],'count':1,'first':0,'last':0,'best':1})
        return registry

    def face(self,a,b):
        v=np.zeros(512,np.float32);v[:3]=[a,b,np.sqrt(1-a*a-b*b)]
        return SimpleNamespace(embedding_norm=v,bounding_box=np.array([0,0,100,100]),score_set={'detector':.9,'landmarker':.9})

    def test_registry_joint_solution_keeps_second_face(self):
        registry=self.registry();a=self.face(.65,.6);b=self.face(.64,.05)
        self.assertEqual([(identity,face is a) for identity,face in registry.assign([a,b],10)],[(2,True),(1,False)])
        self.assertEqual(len(registry.people),2)

    def test_registry_does_not_guess_identical_candidates(self):
        registry=self.registry()
        self.assertEqual(registry.assign([self.face(.7,.7),self.face(.7,.7)],10),[])
        self.assertEqual(len(registry.people),2)


if __name__=='__main__':unittest.main()
