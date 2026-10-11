"""Independent identities, source photos, smoothing and verification per person."""
import json
import threading
import cv2
import numpy as np
from facefusion import state_manager
from facefusion.common_helper import get_middle
from facefusion.face_creator import scale_face
from facefusion.face_recognizer import calculate_face_embedding
from facefusion.vision import read_static_image, read_static_video_frame
from facefusion.processors.modules.face_swapper import core as swapper
from facefusion.workflows import to_video
from people import load_scan, unpack_face
from temporal import prepare_sources, prepare_track
from track_plan import prepare_motion
from protected_ranges import ProtectedFrames


def install(job, folder):
    scan=load_scan(job['people_path'],None if job.get('verified_scan') else job['target'],max(0,job.get('start_frame',0)-10),(job.get('end_frame')+10) if job.get('end_frame') is not None else None)
    assignments={int(item['person_id']):item['sources'] for item in job['assignments'] if item['sources']}
    selected=set(job.get('target_ids') or assignments)
    if not selected or not selected<=assignments.keys():
        raise RuntimeError('Для выбранного человека не назначено фото.')
    known={item['id'] for item in scan['characters']}
    if not selected<=known: raise RuntimeError('Назначение относится к другому видео.')
    from person_groups import render_groups
    selected=render_groups(scan,selected,assignments,job.get('person_groups',[]))
    job={**job,'target_ids':sorted(selected)}
    tracks={identity:{} for identity in selected}
    for index, records in scan['frames'].items():
        for record in records:
            identity=record['person_id']
            if identity in tracks: tracks[identity][int(index)]=unpack_face(record['face'])
    old_pre,old_process,old_finalize=swapper.pre_process,to_video.process_memory_frame,to_video.finalize_video
    local=threading.local(); tables={}; sources={}; samples={}; eligible={}; protection=[]
    start=job.get('start_frame',0);end=job.get('end_frame') or scan['meta']['frames'] or len(scan['frames'])
    protected=ProtectedFrames(job)

    def preprocess(mode):
        if not old_pre(mode): return False
        if mode!='output' or tables: return True
        from coverage import recover_scan
        recover_scan(job,scan)
        from continuity import recover_continuity
        recover_continuity(job,scan)
        for index,records in scan['frames'].items():
            for record in records:
                identity=record['person_id']
                if identity in tracks: tracks[identity][int(index)]=unpack_face(record['face'])
        notes=[]; geometries={}
        if scan.get('recovered_detections'):
            notes.append(f'Восстановлено пропущенных лиц в хронологии: {scan["recovered_detections"]}.')
        if scan.get('continued_detections'):
            notes.append(f'Удержано лиц при слабой разметке: {scan["continued_detections"]}.')
        motion_plan=prepare_motion(job,tracks,scan,folder)
        for identity in sorted(selected):
            person_folder=folder/f'person-{identity}';person_folder.mkdir(exist_ok=True)
            if not any(start<=index<end for index in tracks[identity]):
                tables[identity]={};samples[identity]=[];eligible[identity]=set()
                notes.append(f'Человек {identity}: на указанном участке не найден, пропущен.')
                continue
            prepared,source_notes=prepare_sources(assignments[identity],person_folder)
            images=[read_static_image(path) for path in prepared]
            sources[identity]=(swapper.extract_source_face(images),images[0])
            notes.extend(f'Человек {identity}: '+note for note in source_notes)
            tables[identity]=prepare_track({**job,'face_track':tracks[identity],'motion_track':motion_plan[identity]},person_folder)
            geometries[identity]=json.loads((person_folder/'geometry.json').read_text(encoding='utf-8'))
            frames=[index for index,faces in tables[identity].items() if faces and start<=index<end]
            eligible[identity]={frames[int(i)] for i in np.linspace(0,len(frames)-1,min(8,len(frames)))}
            samples[identity]=[]
        # Minutes without selected people pass through unchanged.
        # Keep the worst person's residual for each frame, so automatic retry
        # also sees a problem affecting only one of several visible people.
        residuals={}
        for geometry in geometries.values():
            for item in geometry['residuals']:
                if item['frame'] not in residuals or item['after']>residuals[item['frame']]['after']:
                    residuals[item['frame']]=item
        pairs=list(residuals.values())
        spans=sorted(span for geom in geometries.values() for span in geom['suspicious_segments'])
        merged=[]
        for a,b in spans:
            if merged and a<=merged[-1][1]+.2: merged[-1][1]=max(merged[-1][1],b)
            else: merged.append([a,b])
        report={'start_frame':start,'end_frame':end,'residuals':pairs,'checked_pairs':len(pairs),
            'raw_median_px':float(np.median([item['before'] for item in pairs])) if pairs else None,
            'stable_median_px':float(np.median([item['after'] for item in pairs])) if pairs else None,
            'suspicious_segments':merged,
            'face_free_frames':sum(not protected.contains(index) and not any(table.get(index) for table in tables.values()) for index in range(start,end)),
            'excluded_ranges':protected.ranges,'excluded_frames':protected.count(start,end),
            'people_geometry':geometries,'uncertain_detections':scan.get('uncertain_detections',0)}
        report['tracking_version']=4
        report['continued_detections']=scan.get('continued_detections',0)
        missing=[index for index in range(start,end) if not protected.contains(index) and not any(table.get(index) for table in tables.values())]
        intervals=[]
        for index in missing:
            if intervals and index==intervals[-1][1]:intervals[-1][1]=index+1
            else:intervals.append([index,index+1])
        report['unswapped_segments']=[[a/job['fps'],b/job['fps']] for a,b in intervals[:120]]
        report['recovered_detections']=scan.get('recovered_detections',0)
        (folder/'geometry.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (folder/'source-notes.json').write_text(json.dumps(notes,ensure_ascii=False),encoding='utf-8')
        return True

    def process(index,*args):
        local.index=index
        return old_process(index,*args)

    def process_frame(inputs):
        index=local.index
        original=get_middle(inputs['target_vision_frames'])
        frame=inputs['temp_vision_frame']
        if protected.contains(index):
            return frame,inputs['temp_vision_mask']
        if job.get('base_target'):
            base=read_static_video_frame(job['base_target'],index-job.get('base_start_frame',0))
            read_static_video_frame.cache_clear()
            if base is None: raise RuntimeError('Не удалось прочитать предыдущий результат.')
            frame=cv2.resize(base,(frame.shape[1],frame.shape[0])) if base.shape[:2]!=frame.shape[:2] else base.copy()
        untouched=frame.copy()
        pending=[]
        for identity in sorted(selected):
            for face in tables[identity].get(index,[]):
                face=scale_face(face,original,frame)
                source,image=sources[identity]
                before=frame.copy() if index in eligible[identity] else None
                from occlusion import set_context
                set_context(identity,index)
                frame=swapper.swap_face(source,face,image,frame)
                if before is not None:
                    pending.append((identity,face,source,before))
        # A mask around an assigned face must not affect an unassigned person
        # standing close by, or another already swapped person during a repair.
        for record in scan['frames'].get(str(index),[]):
            if record['person_id'] in selected: continue
            face=scale_face(unpack_face(record['face']),original,frame)
            x1,y1,x2,y2=face.bounding_box; margin=max(x2-x1,y2-y1)*.12
            h,w=frame.shape[:2]
            x1,y1,x2,y2=max(0,int(x1-margin)),max(0,int(y1-margin)),min(w,int(x2+margin)),min(h,int(y2+margin))
            frame[y1:y2,x1:x2]=untouched[y1:y2,x1:x2]
            if index%10==0 and x2>x1 and y2>y1:
                protection.append({'frame':index,'person_id':record['person_id'],
                    'pixel_change':float(np.abs(frame[y1:y2,x1:x2].astype(float)-untouched[y1:y2,x1:x2]).mean())})
        for identity,face,source,before in pending:
            h,w=frame.shape[:2];x1,y1,x2,y2=np.rint(face.bounding_box).astype(int)
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            if x2>x1 and y2>y1:
                delta=float(np.abs(frame[y1:y2,x1:x2].astype(float)-before[y1:y2,x1:x2]).mean())
                _,a=calculate_face_embedding(before,face.landmark_set['5/68'])
                _,b=calculate_face_embedding(frame,face.landmark_set['5/68'])
                samples[identity].append({'frame':index,'pixel_change':delta,
                    'source_similarity_before':float(np.dot(source.embedding_norm,a)),
                    'source_similarity_after':float(np.dot(source.embedding_norm,b))})
        return frame,inputs['temp_vision_mask']

    def finalize(start_time):
        code=old_finalize(start_time)
        if code: return code
        all_samples=[];checks={};failed=[]
        for identity,items in samples.items():
            changed=sum(item['pixel_change']>.5 for item in items)
            gains=[item['source_similarity_after']-item['source_similarity_before'] for item in items]
            checks[identity]={'sampled_frames':len(items),'changed_frames':changed,'samples':items,
                'median_identity_gain':float(np.median(gains)) if gains else None}
            if eligible[identity] and not changed: failed.append(identity)
            all_samples.extend(items)
        gains=[item['source_similarity_after']-item['source_similarity_before'] for item in all_samples]
        path=folder/'geometry.json';geometry=json.loads(path.read_text(encoding='utf-8'))
        geometry['replacement_check']={'sampled_frames':len(all_samples),
            'changed_frames':sum(item['pixel_change']>.5 for item in all_samples),
            'median_identity_gain':float(np.median(gains)) if gains else None,'people':checks,
            'unassigned_protection':protection}
        path.write_text(json.dumps(geometry,indent=2),encoding='utf-8')
        if failed:
            print('Замена не подтверждена: люди '+', '.join(map(str,failed))+'. Результат не принят.',flush=True)
            return 1
        return 0

    swapper.pre_process=preprocess
    swapper.process_frame=process_frame
    to_video.process_memory_frame=process
    to_video.finalize_video=finalize
