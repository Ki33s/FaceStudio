"""Bounded visual tracking through weak landmark detections, with identity guards."""
import hashlib,json
from bisect import bisect_left
from pathlib import Path
import cv2,numpy as np
from people import box_overlap,pack_face,unpack_face
from minute_queue import atomic_json
from joint_matching import assign
from difficult_detection import retry_faces,unique_faces
from face_motion import scene_changed


def boundary_geometry(face,width,height):
    """Do not use hallucinated interior eyes when detector puts them outside."""
    measured=face.landmark_set['5'];refined=face.landmark_set['5/68']
    eye=max(np.linalg.norm(measured[1]-measured[0]),1)
    outside=np.any((measured[:2]<0)|(measured[:2]>[width,height]),axis=1)
    if outside.any() and np.linalg.norm(refined[:2]-measured[:2],axis=1)[outside].max()>.15*eye:
        return face._replace(landmark_set={**face.landmark_set,'5/68':measured.copy()})
    return face


def plausible_five(points):
    axis=points[1]-points[0];eye=np.linalg.norm(axis)
    if eye<12 or not np.isfinite(points).all():return False
    down=np.array([-axis[1],axis[0]])/eye
    eye_center=(points[0]+points[1])*.5;mouth=(points[3]+points[4])*.5
    return .35<float(np.dot(mouth-eye_center,down)/eye)<1.65 and .05<float(np.dot(points[2]-eye_center,down)/eye)<1.3 and .3<np.linalg.norm(points[4]-points[3])/eye<1.4


def recover_continuity(job,scan):
    from facefusion.face_creator import get_static_faces,get_many_faces
    from facefusion.face_recognizer import calculate_face_embedding
    from facefusion.vision import read_static_video_frame
    from facefusion import face_store,state_manager
    from temporal import motion,warp
    frames=scan['frames']
    if not frames:return scan
    start=min(map(int,frames));end=max(map(int,frames))+1
    target=Path(job['target']);analysis=Path(job['people_path'])
    wanted=set(job.get('target_ids') or [item['person_id'] for item in job.get('assignments',[])])
    signature={'version':12,'video':[str(target),target.stat().st_size,target.stat().st_mtime_ns],
               'scan':[str(analysis),analysis.stat().st_size,analysis.stat().st_mtime_ns],
               'range':[start,end],'wanted':sorted(wanted)}
    if scan.get('identity_groups'):signature['identity_groups']=scan['identity_groups']
    base={key:value for key,value in signature.items() if key not in ('range','wanted')}
    digest=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()[:24]
    selection=hashlib.sha256(json.dumps(sorted(wanted)).encode()).hexdigest()[:8]
    folder=Path(job.get('track_cache') or analysis.parent/'track-plan')/'continuity'
    folder.mkdir(parents=True,exist_ok=True);path=folder/(digest+f'-{start:09d}-{end:09d}-{selection}.json')
    for candidate in sorted(folder.glob(digest+'-*.json'),key=lambda path:path.stat().st_mtime_ns,reverse=True):
        try:
            stored=json.loads(candidate.read_text(encoding='utf-8'));saved=stored['signature']
            if saved['range'][0]<=start and saved['range'][1]>=end and wanted<=set(saved.get('wanted',[])):
                path=candidate;break
        except (OSError,ValueError,KeyError):continue
    # Correct accepted boundary geometry too; this does not add new identities.
    for records in frames.values():
        for record in records:
            record['face']=pack_face(boundary_geometry(unpack_face(record['face']),scan['meta']['width'],scan['meta']['height']))
    if wanted and all(wanted<={record['person_id'] for record in records} for records in frames.values()) and not path.exists():
        atomic_json(path,{'signature':signature,'frames':{}})
    if path.exists():
        additions={key:[r for r in records if not wanted or r['person_id'] in wanted]
                   for key,records in json.loads(path.read_text(encoding='utf-8'))['frames'].items() if start<=int(key)<end}
        print('Удержание лица загружено из хронологии.',flush=True)
    else:
        print('Удержание лица при слабой разметке…',flush=True)
        # Pending ID 0 is an unresolved detector observation, not a person the
        # user chose to keep. Named unassigned people remain protected.
        original={i:{r['person_id']:unpack_face(r['face']) for r in frames.get(str(i),[]) if r['person_id']>0} for i in range(start,end)}
        all_ids=set(identity for records in original.values() for identity in records)
        identities=all_ids & wanted if wanted else all_ids
        anchors={identity:np.asarray([face.embedding_norm for records in original.values() for key,face in records.items() if key==identity]) for identity in all_ids}
        anchor_indices={identity:[i for i in original if identity in original[i]] for identity in identities}
        def similarity(identity,normal):return float(np.max(anchors[identity]@normal))
        def detect(frame,expected):
            faces=list(get_static_faces([frame]))
            if not any(float(f.score_set['detector'])>=.45 and float(f.score_set['landmarker'])>=.55 for f in faces):
                saved=state_manager.get_item('face_detector_margin')
                try:
                    state_manager.set_item('face_detector_margin',[100]*4)
                    faces+=get_many_faces([frame])
                finally:state_manager.set_item('face_detector_margin',saved)
            return retry_faces(frame,expected,unique_faces(faces))
        grays={};candidates={};additions={};active={};limit=max(3,round(scan['meta']['fps']))
        scale=min(1,480/scan['meta']['width']);smallsize=(round(scan['meta']['width']*scale),round(scan['meta']['height']*scale))
        def attempt(previous,gray,oldgray,raw,existing,age,identity,frame):
            if age>limit:return None
            h,w=frame.shape[:2]
            # For an edge-clipped face recognition and optical flow can fail
            # together. Keep a strong, nearby detector's own five points only
            # while its box, points and scene continue the accepted track.
            edge_candidates=[]
            for detected in raw:
                points=detected.landmark_set['5'];oldpoints=previous.landmark_set['5']
                if not np.any((points[:2]<0)|(points[:2]>[w,h])):continue
                eye=max(np.linalg.norm(oldpoints[1]-oldpoints[0]),30)
                if float(detected.score_set['detector'])<.60 or not plausible_five(points):continue
                if box_overlap(previous.bounding_box,detected.bounding_box)<.60 or np.linalg.norm(points-oldpoints,axis=1).mean()>.20*eye:continue
                if np.abs(cv2.resize(oldgray,(32,32)).astype(float)-cv2.resize(gray,(32,32))).mean()>15:continue
                if any(box_overlap(detected.bounding_box,value.bounding_box)>.35 for key,value in existing.items() if key!=identity):continue
                embedding,normal=calculate_face_embedding(frame,points)
                score=similarity(identity,normal)
                other=[similarity(key,normal) for key in anchors if key!=identity and key not in identities]
                if score<.20 or (other and score-max(other)<.08):continue
                edge_candidates.append(detected._replace(origin='edge-track',landmark_set={**detected.landmark_set,'5/68':points.copy()},embedding=embedding,embedding_norm=normal))
            if len(edge_candidates)==1:return edge_candidates[0]
            oldsmall=cv2.resize(oldgray,(32,32));newsmall=cv2.resize(gray,(32,32))
            oldhist=cv2.calcHist([oldsmall],[0],None,[32],[0,256]);newhist=cv2.calcHist([newsmall],[0],None,[32],[0,256])
            # A close camera pan or a hand crossing the lens can change many
            # pixels. Require a large histogram change as well before declaring
            # a cut; spatial, feature and identity checks below still apply.
            cut=np.abs(oldsmall.astype(float)-newsmall).mean()>45 and cv2.compareHist(oldhist,newhist,cv2.HISTCMP_BHATTACHARYYA)>.65
            matrix,valid=motion(None,None,previous.bounding_box*scale,(oldgray,gray),bool(cut))
            if not valid:return None
            matrix[:2,2]/=scale
            points=warp(previous.landmark_set['5/68'],matrix).astype(np.float32)
            corners=np.array([[previous.bounding_box[0],previous.bounding_box[1]],[previous.bounding_box[2],previous.bounding_box[3]]])
            corners=warp(corners,matrix);expected=np.r_[corners.min(axis=0),corners.max(axis=0)]
            possible=[f for f in raw if float(f.score_set['detector'])>=.35 and box_overlap(expected,f.bounding_box)>=.35]
            if len(possible)!=1:return None
            detected=possible[0]
            if any(box_overlap(expected,face.bounding_box)>.35 for key,face in existing.items() if key!=identity):return None
            # Visible detector points must agree with the tracked geometry.
            h,w=frame.shape[:2];measured=detected.landmark_set['5']
            visible=np.all((measured>0)&(measured<np.array([w,h])),axis=1)&np.all((points>0)&(points<np.array([w,h])),axis=1)
            eye=max(np.linalg.norm(points[1]-points[0]),30)
            if visible.sum()<2 or np.median(np.linalg.norm(measured[visible]-points[visible],axis=1))>.35*eye:return None
            # Strong detector landmarks can recover expression/scale changes
            # which a rigid optical-flow transform cannot represent.
            if float(detected.score_set['detector'])>=.60 and plausible_five(measured) and np.linalg.norm(measured-points,axis=1).mean()<.50*eye:
                direct_embedding,direct_normal=calculate_face_embedding(frame,measured)
                direct_score=similarity(identity,direct_normal)
                other=[similarity(key,direct_normal) for key in anchors if key!=identity and key not in identities]
                if direct_score>=.40 and (not other or direct_score-max(other)>=.08):
                    points=measured.astype(np.float32)
            embedding,normal=calculate_face_embedding(frame,points)
            score=similarity(identity,normal)
            competitors=[similarity(key,normal) for key in anchors if key!=identity and key not in identities]
            if score<.35 or (competitors and score-max(competitors)<.08):return None
            landmarks={key:warp(value,matrix).astype(np.float32) for key,value in previous.landmark_set.items()}
            landmarks['5']=landmarks['5/68']=points
            return previous._replace(origin='track',bounding_box=expected.astype(np.float32),landmark_set=landmarks,
                embedding=embedding,embedding_norm=normal,score_set={**previous.score_set,'detector':float(detected.score_set['detector'])})
        def direct_candidate(raw,existing,index,frame):
            proposals=[];measured={}
            for n,face in enumerate(raw):
                detector=float(face.score_set['detector']);reliable=float(face.score_set['landmarker'])>=.55
                points=boundary_geometry(face,frame.shape[1],frame.shape[0]).landmark_set['5/68'] if reliable else face.landmark_set['5']
                if detector<.35 or not plausible_five(points):continue
                if any(box_overlap(face.bounding_box,value.bounding_box)>.35 for value in existing.values()):continue
                embedding,normal=calculate_face_embedding(frame,points)
                scores={key:similarity(key,normal) for key in anchors}
                measured[n]=face._replace(origin='detect-5',landmark_set={**face.landmark_set,'5/68':points.copy()},embedding=embedding,embedding_norm=normal)
                for identity,score in scores.items():
                    if identity not in identities or identity in existing or score<(.55 if detector<.60 and not reliable else .45):continue
                    if any(value>=score-.08 for key,value in scores.items() if key not in identities and key!=identity):continue
                    indexes=anchor_indices[identity];position=bisect_left(indexes,index)
                    nearest=[(abs(i-index),i,original[i][identity]) for i in indexes[max(0,position-1):position+1] if abs(i-index)<=limit]
                    if not nearest:continue
                    overlaps=[box_overlap(anchor.bounding_box,face.bounding_box) for distance,side,anchor in nearest
                              if score>=.55 or np.abs(cv2.resize(grays[index],(32,32)).astype(float)-cv2.resize(grays[side],(32,32))).mean()<=35]
                    # A closer reference after a scene cut must not suppress a
                    # compatible reference before it (or the reverse).
                    overlap=max(overlaps,default=0)
                    if overlap<.25:continue
                    proposals.append((.8*score+.2*overlap,n,identity))
            return [(identity,measured[n]) for score,n,identity in assign(proposals,margin=.08)]

        def expectations(index,active):
            boxes=[]
            for identity in identities-original[index].keys():
                if identity in active and abs(index-active[identity][2])<=limit:
                    boxes.append(active[identity][0].bounding_box)
                indexes=anchor_indices[identity];position=bisect_left(indexes,index)
                boxes.extend(original[i][identity].bounding_box for i in indexes[max(0,position-1):position+1] if abs(i-index)<=limit)
            return boxes

        def joint_attempt(raw,existing,available,index,frame,old_index):
            proposals=[];values={}
            if not available or scene_changed(grays[index],grays[old_index]):return []
            for identity,previous,age in available:
                for n,candidate in enumerate(raw):
                    if any(box_overlap(candidate.bounding_box,f.bounding_box)>.35 for f in existing.values()):continue
                    recovered=attempt(previous,grays[index],grays[old_index],[candidate],existing,age,identity,frame)
                    if recovered is None:continue
                    score=.7*similarity(identity,recovered.embedding_norm)+.3*box_overlap(previous.bounding_box,recovered.bounding_box)
                    values[n,identity]=recovered;proposals.append((score,n,identity))
            return [(identity,values[n,identity]) for score,n,identity in assign(proposals,margin=.08)]
        # At most one minute is loaded by the render queue; only small grayscale
        # frames are kept for a reverse pass. No images or geometry cross cuts.
        for index in range(start,end):
            frame=read_static_video_frame(job['target'],index)
            if frame is None:raise RuntimeError('Не удалось прочитать кадр '+str(index))
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),smallsize);grays[index]=gray
            existing=original[index]
            missing=[identity for identity in identities if identity not in existing and identity in active and active[identity][1]==index-1 and index-active[identity][2]<=limit]
            if expectations(index,active):candidates[index]=detect(frame,expectations(index,active))
            available=[(identity,active[identity][0],index-active[identity][2]) for identity in missing]
            for identity,face in joint_attempt(candidates.get(index,[]),existing,available,index,frame,index-1):
                additions.setdefault(str(index),[]).append({'person_id':identity,'face':pack_face(face)})
                active[identity]=(face,index,active[identity][2])
            for identity in identities:
                if identity in existing:active[identity]=(existing[identity],index,index)
            face_store.clear_faces();read_static_video_frame.cache_clear()
            if index%30==0:print(f'Удержание лица: {index-start+1}/{end-start}',flush=True)
        active={}
        for index in range(end-1,start-1,-1):
            existing={**original[index],**{r['person_id']:unpack_face(r['face']) for r in additions.get(str(index),[])}}
            missing=[identity for identity in identities if identity not in existing and identity in active and active[identity][1]==index+1 and active[identity][2]-index<=limit]
            frame=None
            if missing:
                frame=read_static_video_frame(job['target'],index)
                if index not in candidates:candidates[index]=detect(frame,expectations(index,active))
            available=[(identity,active[identity][0],active[identity][2]-index) for identity in missing]
            for identity,face in joint_attempt(candidates.get(index,[]),existing,available,index,frame,index+1):
                additions.setdefault(str(index),[]).append({'person_id':identity,'face':pack_face(face)})
                active[identity]=(face,index,active[identity][2])
            for identity in identities:
                if identity in original[index]:active[identity]=(original[index][identity],index,index)
                elif identity in existing:active.pop(identity,None)
            face_store.clear_faces();read_static_video_frame.cache_clear()
        for index in sorted(candidates):
            existing={**original[index],**{r['person_id']:unpack_face(r['face']) for r in additions.get(str(index),[])}}
            if identities<=existing.keys():continue
            frame=read_static_video_frame(job['target'],index)
            for identity,face in direct_candidate(candidates[index],existing,index,frame):
                additions.setdefault(str(index),[]).append({'person_id':identity,'face':pack_face(face)})
                existing[identity]=face
            # A brief blur may destroy the current recognition embedding. Two
            # accepted anchors of the same identity and an actual detector face
            # must then agree on its location and visible feature geometry.
            bridge=[];bridge_faces={}
            for identity in identities-existing.keys():
                indexes=anchor_indices[identity];position=bisect_left(indexes,index)
                if position==0 or position==len(indexes):continue
                a=indexes[position-1];b=indexes[position]
                if b-a>limit+1:continue
                left=original[a][identity];right=original[b][identity]
                anchor_similarity=float(np.dot(left.embedding_norm,right.embedding_norm))
                scene_difference=max(np.abs(cv2.resize(grays[index],(32,32)).astype(float)-cv2.resize(grays[side],(32,32))).mean() for side in (a,b))
                t=(index-a)/(b-a);box=left.bounding_box*(1-t)+right.bounding_box*t
                points=left.landmark_set['5/68']*(1-t)+right.landmark_set['5/68']*t
                eye=max(np.linalg.norm(points[1]-points[0]),30)
                for n,face in enumerate(candidates[index]):
                    measured=face.landmark_set['5']
                    if float(face.score_set['detector'])<.35 or not plausible_five(measured) or box_overlap(box,face.bounding_box)<.45:continue
                    if np.linalg.norm(measured-points,axis=1).mean()>.35*eye:continue
                    if any(box_overlap(face.bounding_box,value.bounding_box)>.35 for value in existing.values()):continue
                    embedding,normal=calculate_face_embedding(frame,measured)
                    other=[similarity(key,normal) for key in anchors if key!=identity and key not in identities]
                    score=similarity(identity,normal)
                    if any(scene_changed(grays[index],grays[side]) for side in (a,b)):continue
                    if scene_difference>25 and score<.55:continue
                    if anchor_similarity<.45 and score<.55:continue
                    if score<.20 or (other and score-max(other)<.08):continue
                    bridge_faces[n,identity]=face._replace(origin='detect-5',landmark_set={**face.landmark_set,'5/68':measured.copy()},embedding=embedding,embedding_norm=normal)
                    bridge.append((.7*score+.3*box_overlap(box,face.bounding_box),n,identity))
            for score,n,identity in assign(bridge,margin=.08):
                face=bridge_faces[n,identity]
                additions.setdefault(str(index),[]).append({'person_id':identity,'face':pack_face(face)})
                existing[identity]=face
            read_static_video_frame.cache_clear()
        # A gap of at most three frames can be bridged by two independent
        # feature tracks. This needs agreement in the current image and an
        # identity match; it never copies a face onto an empty frame.
        complete={i:{**original[i],**{r['person_id']:unpack_face(r['face']) for r in additions.get(str(i),[])}} for i in original}
        for identity in identities:
            indexes=[i for i in complete if identity in complete[i]]
            for a,b in zip(indexes,indexes[1:]):
                if not 1<b-a<=4:continue
                left=complete[a][identity];right=complete[b][identity]
                if float(np.dot(left.embedding_norm,right.embedding_norm))<.35:continue
                for index in range(a+1,b):
                    if any(scene_changed(grays[index],grays[side]) for side in (a,b)):continue
                    ml,vl=motion(None,None,left.bounding_box*scale,(grays[a],grays[index]),False)
                    mr,vr=motion(None,None,right.bounding_box*scale,(grays[b],grays[index]),False)
                    if not vl or not vr:continue
                    ml[:2,2]/=scale;mr[:2,2]/=scale
                    pl=warp(left.landmark_set['5/68'],ml);pr=warp(right.landmark_set['5/68'],mr)
                    eye=max(np.linalg.norm(pl[1]-pl[0]),30)
                    if np.linalg.norm(pl-pr,axis=1).mean()>.15*eye:continue
                    points=(pl+pr)*.5
                    corners=warp(np.array([[left.bounding_box[0],left.bounding_box[1]],[left.bounding_box[2],left.bounding_box[3]]]),ml)
                    box=np.r_[corners.min(axis=0),corners.max(axis=0)]
                    if any(box_overlap(box,value.bounding_box)>.35 for key,value in complete[index].items() if key!=identity):continue
                    frame=read_static_video_frame(job['target'],index)
                    embedding,normal=calculate_face_embedding(frame,points)
                    score=similarity(identity,normal);other=[similarity(key,normal) for key in anchors if key!=identity]
                    if score<.35 or (other and score-max(other)<.08):continue
                    landmarks={key:warp(value,ml).astype(np.float32) for key,value in left.landmark_set.items()}
                    landmarks['5']=landmarks['5/68']=points.astype(np.float32)
                    face=left._replace(origin='flow-bridge',bounding_box=box.astype(np.float32),landmark_set=landmarks,embedding=embedding,embedding_norm=normal)
                    additions.setdefault(str(index),[]).append({'person_id':identity,'face':pack_face(face)})
                    read_static_video_frame.cache_clear()
        atomic_json(path,{'signature':signature,'frames':additions})
    applied={}
    for key,records in additions.items():
        current=frames.setdefault(key,[]);known={record['person_id'] for record in current}
        for record in records:
            if record['person_id'] in known:continue
            current[:]=[old for old in current if old['person_id']!=0 or
                        box_overlap(np.asarray(old['face']['bounding_box']),np.asarray(record['face']['bounding_box']))<=.35]
            current.append(record);known.add(record['person_id']);applied.setdefault(key,[]).append(record)
    scan['coverage_signature']=digest
    scan['continued_detections']=sum(len(records) for key,records in applied.items() if job.get('start_frame',0)<=int(key)<(job.get('end_frame') or scan['meta']['frames']))
    print(f'Удержано пропущенных лиц: {scan["continued_detections"]}.',flush=True)
    return scan
