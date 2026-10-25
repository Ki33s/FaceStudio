"""Automatic source selection and motion-compensated facial geometry."""
import json
import threading
from pathlib import Path
import cv2
import numpy as np
from protected_ranges import ProtectedFrames
from pose_stability import smooth_measured
from facefusion import state_manager, face_store
from facefusion.face_creator import get_static_faces, get_many_faces
from facefusion.face_recognizer import calculate_face_embedding
from facefusion.vision import read_static_image, read_static_video_frame, count_video_frame_total, write_image
from facefusion.processors.modules.face_swapper import core as swapper
from facefusion.workflows import to_video


def warp(points, matrix):
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def motion(a, b, box, grays=None, scene_cut=None):
    ga, gb = grays if grays is not None else (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in (a, b))
    identity = np.eye(3)
    # Do not propagate geometry across a scene cut.
    if scene_cut if scene_cut is not None else np.abs(cv2.resize(ga, (32, 32)).astype(float)-cv2.resize(gb, (32, 32))).mean() > 45:
        return identity, False
    mask = np.zeros(ga.shape, np.uint8)
    x1,y1,x2,y2 = np.rint(box).astype(int)
    h,w = mask.shape
    mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)] = 255
    pts = cv2.goodFeaturesToTrack(ga, 120, .015, 8, mask=mask)
    if pts is None or len(pts) < 8:
        return identity, False
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(25,25), maxLevel=3)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gb, ga, nxt, None, winSize=(25,25), maxLevel=3)
    good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    good &= np.linalg.norm(pts[:,0]-back[:,0],axis=1)<2.5
    if good.sum() < 8:
        return identity, False
    affine, inliers = cv2.estimateAffinePartial2D(pts[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=3)
    if affine is not None and inliers.sum()>=8 and .8<np.linalg.norm(affine[0,:2])<1.25:
        identity[:2] = affine
        return identity, True
    return identity, False


def source_faces(image):
    # A still photo needs reliable detections, not the permissive settings used
    # to keep tracking a large or partly clipped face in a video.
    keys = ('face_detector_margin', 'face_detector_score')
    saved = {key: state_manager.get_item(key) for key in keys}
    try:
        state_manager.set_item('face_detector_margin', [0, 0, 0, 0])
        state_manager.set_item('face_detector_score', .65)
        return [face for face in get_many_faces([image])
                if float(face.score_set['landmarker']) >= .5]
    finally:
        for key, value in saved.items():
            state_manager.set_item(key, value)


def prepare_sources(paths, folder):
    reference = None
    prepared = []
    notes = []
    for index, path in enumerate(paths):
        image = read_static_image(path)
        candidates = source_faces(image)
        if not candidates:
            if index == 0:
                raise RuntimeError('На основном фото не найдено достоверное лицо. Выберите фото с хорошо видимым лицом.')
            notes.append(f'Дополнительное фото {index}: лицо не найдено, пропущено.')
            continue
        if reference is None:
            face = max(candidates, key=lambda f: float(np.prod(f.bounding_box[2:]-f.bounding_box[:2])))
            reference = face.embedding_norm
        else:
            face = max(candidates, key=lambda f: float(np.dot(reference, f.embedding_norm)))
            if float(np.dot(reference,face.embedding_norm)) < .25:
                notes.append(f'Дополнительное фото {index}: сходство не подтверждено, пропущено.')
                continue
        x1,y1,x2,y2 = face.bounding_box
        margin = (x2-x1)*.35
        h,w = image.shape[:2]
        crop = image[max(0,int(y1-margin)):min(h,int(y2+margin)),max(0,int(x1-margin)):min(w,int(x2+margin))]
        dest = folder/f'source-{index}.png'
        if not write_image(str(dest),crop):
            raise RuntimeError('Не удалось сохранить фото лица.')
        # Validate again after cropping and pin the selected identity in the
        # FaceFusion cache so later permissive detection cannot add false faces.
        verified = source_faces(crop)
        if not verified:
            raise RuntimeError('Не удалось подтвердить лицо после подготовки фотографии.')
        verified_face = max(verified, key=lambda f: float(np.dot(face.embedding_norm, f.embedding_norm)))
        if float(np.dot(face.embedding_norm, verified_face.embedding_norm)) < .35:
            raise RuntimeError('Подготовленная фотография не соответствует выбранному лицу.')
        face_store.set_faces(crop, [verified_face])
        if index == 0:
            (folder/'source-selection.json').write_text(json.dumps({
                'bounding_box':face.bounding_box.tolist(),
                'detector_score':float(face.score_set['detector']),
                'landmarker_score':float(face.score_set['landmarker']),
                'rejected_weak_detections':True}, indent=2), encoding='utf-8')
            notes.append('Основное лицо подтверждено детектором и разметкой; слабые ложные срабатывания исключены.')
        prepared.extend([str(dest)]* (3 if index==0 else 1))
    return prepared, notes


def prepare_track(job, folder):
    path = state_manager.get_item('target_path')
    total = count_video_frame_total(path)
    radius = 8 if job.get('strong') else 5
    sigma = 3.2 if job.get('strong') else 2.2
    context=max(8,radius+1)
    start = max(0,job.get('start_frame',0)-context)
    end = min(total, (job.get('end_frame') or total)+context)
    protected = ProtectedFrames(job)
    faces, matrices, links = [], [], []
    previous_face, previous_frame = None, None
    if 'motion_track' in job:
        for index in range(start,end):
            chosen=None if protected.contains(index) else job['face_track'].get(index)
            faces.append(chosen)
            if index>start:
                link=job['motion_track'].get(str(index))
                valid=bool(link and link['valid'] and faces[-2] is not None and chosen is not None)
                matrices.append(np.asarray(link['matrix']) if valid else np.eye(3));links.append(valid)
        return smooth_track(job,folder,faces,matrices,links,start,total,radius,sigma,protected)
    for index in range(start,end):
        frame = read_static_video_frame(path,index)
        if frame is None:
            break
        candidates = [] if protected.contains(index) else (([job['face_track'][index]] if index in job.get('face_track', {}) else []) if 'face_track' in job else [face for face in get_static_faces([frame])
                      if float(face.score_set['landmarker']) >= .45])
        def score(face):
            value = float(face.score_set['detector'])
            if previous_face is not None:
                scale = max(np.linalg.norm(previous_face.landmark_set['5/68'][1]-previous_face.landmark_set['5/68'][0]),80)
                value -= .3*np.linalg.norm(face.landmark_set['5/68']-previous_face.landmark_set['5/68'],axis=1).mean()/scale
            return value
        chosen = max(candidates,key=score) if candidates else None
        if previous_frame is not None:
            if previous_face is not None and chosen is not None:
                matrix, valid = motion(previous_frame,frame,previous_face.bounding_box)
            else:
                matrix, valid = np.eye(3), False
            matrices.append(matrix)
            links.append(valid)
        faces.append(chosen)
        previous_frame = frame
        previous_face = chosen
        read_static_video_frame.cache_clear();face_store.clear_faces()
        if index%30==0:
            print(f'Проверка геометрии: {index-start}/{end-start}',flush=True)
    return smooth_track(job,folder,faces,matrices,links,start,total,radius,sigma,protected)


def smooth_track(job,folder,faces,matrices,links,start,total,radius,sigma,protected):
    valid = [i for i,f in enumerate(faces) if f is not None]
    if not valid and not protected.count(start,start+len(faces)) and not job.get('chunked'):
        raise RuntimeError('Лицо в видео не найдено. Попробуйте другой ролик.')

    def between(j,i):
        matrix=np.eye(3)
        if j<i:
            for k in range(j,i): matrix=matrices[k]@matrix
        else:
            for k in range(j-1,i-1,-1): matrix=np.linalg.inv(matrices[k])@matrix
        return matrix

    table, raw, stable = {}, [], []
    for i,anchor in enumerate(faces):
        if anchor is None:
            # Keep actual face-free frames unchanged; do not invent a face on title cards.
            raw.append(None); stable.append(None); table[start+i]=[]
            continue
        samples,weights,dense_samples=[],[],[]
        for j in range(max(0,i-radius),min(len(faces),i+radius+1)):
            if faces[j] is not None and (i==j or all(links[min(i,j):max(i,j)])):
                samples.append(warp(faces[j].landmark_set['5/68'],between(j,i)))
                dense_samples.append(warp(faces[j].landmark_set['68'],between(j,i)))
                weights.append(np.exp(-.5*((j-i)/sigma)**2)*max(float(faces[j].score_set['detector']),.15))
        points=smooth_measured(anchor.landmark_set['5/68'],samples,weights)
        landmarks=dict(anchor.landmark_set)
        landmarks['5/68']=landmarks['5']=points.astype(np.float32)
        from profile_detail import smooth_dense
        landmarks['68']=smooth_dense(anchor.landmark_set['68'],anchor.landmark_set['5/68'],dense_samples,samples,weights)
        table[start+i]=[anchor._replace(landmark_set=landmarks)]
        raw.append(anchor.landmark_set['5/68']);stable.append(points)
    from profile_detail import pose_weights
    profile_weights=pose_weights(stable,links)
    (folder/'profile-plan.json').write_text(json.dumps({'version':2,'frames':{str(start+i):v for i,v in enumerate(profile_weights)}}),encoding='utf-8')
    before,after,bad,pairs=[],[],[],[]
    fps=job['fps']
    for i in range(len(faces)-1):
        if raw[i] is not None and raw[i+1] is not None and links[i] and start+i>=job.get('start_frame',0) and start+i<(job.get('end_frame') or total)-1:
            a=float(np.linalg.norm(raw[i+1]-warp(raw[i],matrices[i]),axis=1).mean())
            b=float(np.linalg.norm(stable[i+1]-warp(stable[i],matrices[i]),axis=1).mean())
            before.append(a);after.append(b)
            pairs.append({'frame':start+i,'before':a,'after':b})
            scale=max(float(np.linalg.norm(stable[i][1]-stable[i][0])),60)
            if b>max(8,.08*scale): bad.append((start+i)/fps)
    def median(values): return float(np.median(values)) if values else None
    spans=[]
    for sec in bad:
        if spans and sec<=spans[-1][1]+.5: spans[-1][1]=sec+.2
        else: spans.append([max(0,sec-.2),sec+.2])
    report={'raw_median_px':median(before),'stable_median_px':median(after),
            'face_free_frames':len(faces)-len(valid)-protected.count(start,start+len(faces)),
            'excluded_frames':protected.count(job.get('start_frame',0),job.get('end_frame') or total),
            'excluded_ranges':protected.ranges,'checked_pairs':len(after),
            'suspicious_segments':spans,'start_frame':job.get('start_frame',0),'end_frame':job.get('end_frame') or total,
            'residuals':pairs,
            'measurement':'Motion-compensated landmark residual, not a visual quality guarantee.'}
    (folder/'geometry.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return table


def install(job,folder):
    if job.get('people_path'):
        from multi_temporal import install as install_multi
        return install_multi(job,folder)
    old_pre=swapper.pre_process
    old_select=swapper.select_faces
    old_process=to_video.process_memory_frame
    old_swap=swapper.swap_face
    old_finalize=to_video.finalize_video
    local=threading.local()
    table={}
    protected=ProtectedFrames(job)
    sample_frames=set()
    replacement_samples=[]
    def preprocess(mode):
        if not old_pre(mode): return False
        if mode=='output' and not table:
            sources,notes=prepare_sources(job['sources'],folder)
            state_manager.set_item('source_paths',sources)
            (folder/'source-notes.json').write_text(json.dumps(notes,ensure_ascii=False),encoding='utf-8')
            table.update(prepare_track(job,folder))
            eligible=[index for index,faces in table.items() if faces
                      and index >= job.get('start_frame',0)
                      and index < (job.get('end_frame') or max(table)+1)
                      and float(faces[0].score_set['landmarker']) >= .5]
            if eligible:
                sample_frames.update(eligible[int(i)] for i in np.linspace(0,len(eligible)-1,min(8,len(eligible))))
        return True
    def process(index,*args):
        local.index=index
        return old_process(index,*args)
    def select(reference,sources,targets):
        index=getattr(local,'index',None)
        if index is not None and protected.contains(index): return []
        return table[index] if index in table else old_select(reference,sources,targets)
    def checked_swap(source_face,target_face,source_image,frame):
        from occlusion import set_context
        index=getattr(local,'index',None)
        if index is not None:set_context(0,index)
        output=old_swap(source_face,target_face,source_image,frame)
        index=getattr(local,'index',None)
        if index in sample_frames:
            x1,y1,x2,y2=np.rint(target_face.bounding_box).astype(int)
            h,w=frame.shape[:2]
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            if x2>x1 and y2>y1:
                delta=float(np.abs(output[y1:y2,x1:x2].astype(float)-frame[y1:y2,x1:x2]).mean())
                _,before=calculate_face_embedding(frame,target_face.landmark_set['5/68'])
                _,after=calculate_face_embedding(output,target_face.landmark_set['5/68'])
                replacement_samples.append({'frame':index,'pixel_change':delta,
                    'source_similarity_before':float(np.dot(source_face.embedding_norm,before)),
                    'source_similarity_after':float(np.dot(source_face.embedding_norm,after))})
        return output
    def finalize(start_time):
        code=old_finalize(start_time)
        if code: return code
        changed=sum(sample['pixel_change']>.5 for sample in replacement_samples)
        check={'sampled_frames':len(replacement_samples),'changed_frames':changed,
               'samples':replacement_samples}
        gains=[s['source_similarity_after']-s['source_similarity_before'] for s in replacement_samples]
        check['median_identity_gain']=float(np.median(gains)) if gains else None
        path=folder/'geometry.json'
        geometry=json.loads(path.read_text(encoding='utf-8'))
        geometry['replacement_check']=check
        path.write_text(json.dumps(geometry,indent=2),encoding='utf-8')
        if sample_frames and not changed:
            print('Замена лица не подтверждена: проверенные кадры остались без изменения. Результат не принят.',flush=True)
            return 1
        return 0
    swapper.pre_process=preprocess
    swapper.select_faces=select
    to_video.process_memory_frame=process
    swapper.swap_face=checked_swap
    to_video.finalize_video=finalize
