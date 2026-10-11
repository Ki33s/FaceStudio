"""Recover short recognition gaps in older projects without changing assignments."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from people import box_overlap,pack_face
from minute_queue import atomic_json


def recover_scan(job,scan):
    if scan.get('tracking_version',0)>=3: return scan
    from facefusion.face_creator import get_static_faces
    from facefusion.vision import read_static_video_frame
    from facefusion import face_store
    frames=scan['frames'];fps=scan['meta']['fps'];anchors={}
    for key,records in frames.items():
        for record in records: anchors.setdefault(record['person_id'],[]).append((int(key),record['face']))
    gaps={}
    first_frame=min(map(int,frames));last_frame=max(map(int,frames))
    # Require an accepted anchor on each side. Never extend onto a back of
    # a head, into an empty scene, or across a long absence.
    for identity,items in anchors.items():
        items.sort(key=lambda item:item[0])
        for (a,left),(b,right) in zip(items,items[1:]):
            if not 1<b-a<=round(fps)+1: continue
            lbox=np.asarray(left['bounding_box']);rbox=np.asarray(right['bounding_box'])
            resemblance=float(np.dot(left['embedding_norm'],right['embedding_norm']))
            if box_overlap(lbox,rbox)<.25 and resemblance<.60: continue
            if resemblance<.35: continue
            for index in range(a+1,b):
                gaps.setdefault(index,[]).append((identity,a,b,left,right))
        # A few frames at a track's edge can be recovered only with a much
        # stronger appearance match; this never supplies interpolated faces.
        for edge,anchor,direction in ((items[0][0],items[0][1],-1),(items[-1][0],items[-1][1],1)):
            for offset in range(1,min(3,max(1,round(fps*.1)))+1):
                index=edge+direction*offset
                if first_frame<=index<=last_frame:
                    gaps.setdefault(index,[]).append((identity,edge,edge,anchor,anchor))
    if not gaps: return scan
    target=Path(job['target']);analysis=Path(job['people_path'])
    stamp={'version':2,'target':[str(target),target.stat().st_size,target.stat().st_mtime_ns],
        'scan':[str(analysis),analysis.stat().st_size,analysis.stat().st_mtime_ns],
        'range':[min(map(int,frames)),max(map(int,frames))]}
    digest=hashlib.sha256(json.dumps(stamp,sort_keys=True).encode()).hexdigest()[:24]
    cache=Path(job.get('track_cache') or analysis.parent/'track-plan')/'coverage'
    cache.mkdir(parents=True,exist_ok=True);path=cache/(digest+'.json')
    if path.is_file():
        additions=json.loads(path.read_text(encoding='utf-8'))['frames']
        print('Восстановленная хронология загружена из кэша.',flush=True)
    else:
        print(f'Восстановление пропусков старого проекта: кадров {len(gaps)}.',flush=True)
        additions={};small={}
        def scene(index):
            if index not in small:
                frame=read_static_video_frame(job['target'],index)
                if frame is None: raise RuntimeError('Не удалось прочитать кадр '+str(index))
                small[index]=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,32))
                read_static_video_frame.cache_clear()
            return small[index]
        for number,index in enumerate(sorted(gaps)):
            frame=read_static_video_frame(job['target'],index)
            if frame is None: raise RuntimeError('Не удалось прочитать кадр '+str(index))
            current=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,32));small[index]=current
            faces=[f for f in get_static_faces([frame]) if float(f.score_set['detector'])>=.45 and float(f.score_set['landmarker'])>=.55]
            existing=frames.get(str(index),[]);candidates=[]
            for identity,a,b,left,right in gaps[index]:
                if any(r['person_id']==identity for r in existing): continue
                edge=a==b
                if any(np.abs(current.astype(float)-scene(side)).mean()>(15 if edge else 35) for side in (a,b)): continue
                ratio=(index-a)/(b-a) if not edge else 0
                expected=np.asarray(left['bounding_box'])*(1-ratio)+np.asarray(right['bounding_box'])*ratio
                for face_index,face in enumerate(faces):
                    if any(box_overlap(face.bounding_box,np.asarray(r['face']['bounding_box']))>.5 for r in existing): continue
                    overlap=box_overlap(expected,face.bounding_box)
                    scores=[float(np.dot(np.asarray(side['embedding_norm']),face.embedding_norm)) for side in (left,right)]
                    if overlap>=(.60 if edge else .35) and min(scores)>=(.65 if edge else .22) and max(scores)>=(.65 if edge else .35):
                        candidates.append((overlap*.35+max(scores)*.65,identity,face_index))
            used_faces=set();used_ids=set()
            for score,identity,face_index in sorted(candidates,reverse=True):
                if identity in used_ids or face_index in used_faces: continue
                competition=[s for s,i,f in candidates if (i!=identity and f==face_index) or (i==identity and f!=face_index)]
                if competition and score-max(competition)<.08: continue
                additions.setdefault(str(index),[]).append({'person_id':identity,'face':pack_face(faces[face_index])})
                used_ids.add(identity);used_faces.add(face_index)
            face_store.clear_faces();read_static_video_frame.cache_clear()
            if number%30==0: print(f'Восстановление хронологии: {number+1}/{len(gaps)}',flush=True)
        atomic_json(path,{'signature':stamp,'frames':additions})
    for key,records in additions.items(): frames.setdefault(key,[]).extend(records)
    scan['coverage_signature']=digest
    scan['recovered_detections']=sum(len(records) for key,records in additions.items()
        if job.get('start_frame',0)<=int(key)<(job.get('end_frame') or scan['meta']['frames']))
    print(f'Восстановлено пропущенных замен: {scan["recovered_detections"]}.',flush=True)
    return scan
