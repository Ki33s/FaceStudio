"""Persistent face identities and portable scan data, shared by discovery and rendering."""
import hashlib
import json
from pathlib import Path
import numpy as np


def fingerprint(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def pack_face(face):
    def plain(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,dict): return {key:plain(item) for key,item in value.items()}
        if isinstance(value,(list,tuple,range)): return [plain(item) for item in value]
        return value
    return plain(face._asdict())


def unpack_face(value):
    from facefusion.types import Face
    value=dict(value)
    for key in ('bounding_box','embedding','embedding_norm'): value[key]=np.asarray(value[key],np.float32)
    value['landmark_set']={key:np.asarray(item,np.float32) for key,item in value['landmark_set'].items()}
    return Face(**value)


class Registry:
    def __init__(self, fps=30):
        self.people=[]
        self.max_gap=max(1,round(fps*.2))

    def reset_tracks(self):
        for person in self.people:
            person.pop('track_box',None);person.pop('track_embedding',None)

    def assign(self,faces,index,motion_links=None):
        # One identity may occur at most once per frame. Similar competing faces
        # are skipped instead of silently transferring another person's mapping.
        used=set(); result=[];motion_links=motion_links or {};positions={id(face):n for n,face in enumerate(faces)}
        for face in sorted(faces,key=lambda f:float(f.score_set['detector']),reverse=True):
            embedding=face.embedding_norm
            ranked=sorted([(max(float(np.dot(person['vector'],embedding)),
                               max(float(np.dot(sample,embedding)) for sample in person.get('gallery',[person['vector']]))),i)
                for i,person in enumerate(self.people)],reverse=True)
            best=ranked[0][0] if ranked else -1
            margin=best-ranked[1][0] if len(ranked)>1 else 1
            # Recognition alone is brittle when expression or pose changes.
            # Only continue a recent spatial track when its appearance agrees;
            # scene changes clear this state without forgetting identities.
            continuous=[]
            for i,person in enumerate(self.people):
                if i in used or 'track_box' not in person or not 0<index-person['last']<=self.max_gap: continue
                box=np.asarray(person['track_box']);current=face.bounding_box
                overlap=box_overlap(box,current)
                similarity=float(np.dot(person['track_embedding'],embedding))
                if overlap>=.35 and similarity>=.50:
                    continuous.append((similarity*.65+overlap*.35,i))
            continuous.sort(reverse=True)
            link=motion_links.get(positions[id(face)])
            linked=link['person_id']-1 if link else -1
            # A verified optical path between actual adjacent detections takes
            # priority over pose-sensitive recognition, except a strong rival.
            linked_score=next((score for score,i in ranked if i==linked),-1)
            if (link and 0<=linked<len(self.people) and linked not in used and
                not (ranked and ranked[0][1]!=linked and best>=.65 and best-linked_score>=.18)):
                identity=linked
            elif best>=.40 and margin>=.08:
                identity=ranked[0][1]
                if identity in used: continue
            elif continuous and (len(continuous)==1 or continuous[0][0]-continuous[1][0]>=.08):
                identity=continuous[0][1]
            elif best<.28 and float(face.score_set['detector'])>=.55 and float(face.score_set['landmarker'])>=.65:
                identity=len(self.people)
                self.people.append({'vector':embedding.copy(),'gallery':[embedding.tolist()],
                                    'count':0,'first':index,'last':index,'best':-1})
            else: continue
            used.add(identity)
            person=self.people[identity];person['count']+=1;person['last']=index
            person['track_box']=face.bounding_box.tolist()
            person['track_embedding']=embedding.tolist()
            gallery=person.setdefault('gallery',[person['vector'].tolist()])
            own=max(float(np.dot(sample,embedding)) for sample in gallery)
            reliable=own>=.50 or (link and linked==identity and link['score']>=.8)
            if reliable and float(face.score_set['landmarker'])>=.65 and own<.97:
                if len(gallery)<16:gallery.append(embedding.tolist())
                else:
                    matrix=np.asarray(gallery,np.float32);similarity=matrix@matrix.T
                    np.fill_diagonal(similarity,-1)
                    redundant=1+int(similarity[1:].max(axis=1).argmax())
                    gallery[redundant]=embedding.tolist()
            if own>.50:
                vector=person['vector']*.98+embedding*.02
                person['vector']=vector/max(np.linalg.norm(vector),1e-8)
            result.append((identity+1,face))
        return result


def box_overlap(a,b):
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersection=float(np.prod(size))
    union=float(np.prod(np.maximum(0,a[2:]-a[:2]))+np.prod(np.maximum(0,b[2:]-b[:2])))-intersection
    return intersection/max(union,1e-8)


def load_scan(path,video=None,start=None,end=None):
    path=Path(path)
    scan=json.loads(path.read_text(encoding='utf-8'))
    if video and fingerprint(video) not in {scan['fingerprint'],scan.get('input_fingerprint')}:
        raise RuntimeError('Видео изменилось после поиска людей. Выполните поиск заново.')
    if scan.get('frame_chunks'):
        frames={}
        if start is not None or end is not None:
            for chunk in scan['frame_chunks']:
                if (end is None or chunk['start']<end) and (start is None or chunk['end']>start):
                    stored=json.loads((path.parent/chunk['file']).read_text(encoding='utf-8'))
                    frames.update({k:v for k,v in stored['frames'].items() if (start is None or int(k)>=start) and (end is None or int(k)<end)})
        scan['frames']=frames
    for person in scan['characters']:
        person['thumbnail']=str(path.parent/Path(person['thumbnail']).name)
    return scan
