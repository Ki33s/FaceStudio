"""Cached clear-face schedule. Unselected video remains on the original clock."""
import hashlib,json,subprocess
from bisect import bisect_left
from pathlib import Path
import cv2,numpy as np
from people import box_overlap
from protected_ranges import ProtectedFrames

VERSION=5
MIN_SIDE=48
MIN_SHARPNESS=8.


def selected_ids(project,target_ids=None):
    selected=set(target_ids or [a['person_id'] for a in project.get('assignments',[]) if a.get('sources')])
    for group in project.get('person_groups',[]):
        if selected.intersection(group):selected.update(group)
    return selected


def basic_reason(face,width,height,relaxed=False):
    try:
        box=np.asarray(face['bounding_box'],np.float32);score=face['score_set']
        points=np.asarray(face['landmark_set']['5/68'],np.float32)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):return 'geometry'
        confident=(score['detector']>=.45 and score['landmarker']>=.55) or (score['detector']>=.75 and score['landmarker']>=.50)
        if (not relaxed and not confident) or (relaxed and (score['detector']<.35 or score['landmarker']<.40)):return 'confidence'
        visible=np.minimum(box[2:],[width,height])-np.maximum(box[:2],0)
        if min(visible)<MIN_SIDE:return 'small'
        if points.shape!=(5,2) or not np.isfinite(points).all():return 'geometry'
        # A clear side view has compressed eye/mouth distances. Do not apply a
        # frontal-face ratio test to it; require finite, noncollapsed geometry.
        if np.linalg.norm(points[1]-points[0])<max(4.,min(visible)*.04):return 'geometry'
        if np.linalg.norm((points[3]+points[4]-points[0]-points[1])/2)<min(visible)*.15:return 'geometry'
        if np.all((points>=0)&(points<[width,height]),axis=1).sum()<4:return 'occluded'
    except (KeyError,ValueError,TypeError):return 'geometry'
    return None


def sharpness(gray,box,scale):
    x1,y1,x2,y2=np.asarray(box)*np.array([scale[0],scale[1],scale[0],scale[1]])
    h,w=gray.shape;x1,y1=max(0,int(x1)),max(0,int(y1));x2,y2=min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
    if x2-x1<12 or y2-y1<12:return 0.
    crop=cv2.resize(gray[y1:y2,x1:x2],(96,96),interpolation=cv2.INTER_LINEAR)
    if float(crop.std())<6:return 0.
    return float(cv2.Laplacian(crop,cv2.CV_32F).var())


def ranges(indices):
    result=[]
    for index in sorted(set(indices)):
        if result and index==result[-1][1]:result[-1][1]=index+1
        else:result.append([index,index+1])
    return result


def keep_short_dips(samples,fps):
    """Use existing measurements on both sides of a short dip; no new inference."""
    from face_motion import scene_changed
    from joint_matching import assign
    strong={};accepted={};promotions={};held=0
    for index,items in samples.items():
        for item in items:
            if item['strong'] and item['id']>0:
                accepted.setdefault(str(index),[]).append(item['id'])
                strong.setdefault(item['group'],[]).append((index,item))
    for items in strong.values():items.sort(key=lambda value:value[0])
    window=max(2,round(fps*.15))
    for index,items in samples.items():
        occupied={item['group'] for item in items if item['strong'] and item['id']>0}
        proposals=[];values={}
        for ordinal,item in enumerate(items):
            if item['strong'] or item['sharpness']<4:continue
            for group,anchors in strong.items():
                if group in occupied or (item['id']>0 and item['group']!=group):continue
                positions=[value[0] for value in anchors];at=bisect_left(positions,index)
                if at==0 or at==len(anchors):continue
                left,right=anchors[at-1],anchors[at]
                if index-left[0]>window or right[0]-index>window:continue
                if any(scene_changed(item['scene'],value['scene']) for _,value in (left,right)):continue
                overlaps=[box_overlap(item['box'],value['box']) for _,value in (left,right)]
                similarities=[float(item['vector']@value['vector']) for _,value in (left,right)]
                similarity_floor=.40 if item.get('landmarker',0)>=.70 and item['sharpness']>=MIN_SHARPNESS else .45
                if min(overlaps)<.30 or min(similarities)<.30 or max(similarities)<similarity_floor:continue
                score=.7*min(similarities)+.3*min(overlaps)
                proposals.append((score,ordinal,group));values[ordinal,group]=left[1]['id']
        for _,ordinal,group in assign(proposals,margin=.08):
            item=items[ordinal];identity=item['id'] or values[ordinal,group]
            accepted.setdefault(str(index),[]).append(identity);held+=1
            if item['id']==0:promotions.setdefault(str(index),[]).append({'ordinal':item['ordinal'],'person_id':identity})
    return accepted,promotions,held


def gray_frames(video,meta,start,end,color=False):
    from video_tools import FFMPEG,FLAGS
    scale=min(1.,1280/max(meta['width'],meta['height']))
    width=max(2,round(meta['width']*scale));height=max(2,round(meta['height']*scale));size=width*height*(3 if color else 1)
    args=[FFMPEG,'-v','error','-ss',format(start/meta['fps'],'.12f'),'-i',str(video),'-map','0:v:0',
          '-frames:v',str(end-start),'-vf',f'scale={width}:{height}','-pix_fmt','bgr24' if color else 'gray','-f','rawvideo','pipe:1']
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=FLAGS)
    try:
        for index in range(start,end):
            data=p.stdout.read(size)
            if len(data)!=size:raise RuntimeError('Не удалось прочитать кадр плана обработки: '+str(index))
            shape=(height,width,3) if color else (height,width)
            yield index,np.frombuffer(data,np.uint8).reshape(shape),(width/meta['width'],height/meta['height'])
        error=p.stderr.read();code=p.wait()
        if code:raise RuntimeError('Ошибка подготовки плана: '+error.decode(errors='replace')[-400:])
    finally:
        if p.poll() is None:p.kill()
        p.wait();p.stdout.close();p.stderr.close()


def seed_cached_measurements(samples,folder,stamp):
    """Carry confirmed clarity across minute boundaries, without carrying pixels."""
    if not samples:return 0
    first=min(samples);current_scan={p[0]:p[1:] for p in stamp['scan']}
    comparable=[k for k in stamp if k not in ('range','scan')]
    for path in sorted(folder.glob('*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:16]:
        try:
            plan=json.loads(path.read_text(encoding='utf-8'));old=plan['stamp']
            if not plan['start']<=first<plan['end']:continue
            if any(old.get(k)!=stamp[k] for k in comparable):continue
            saved={p[0]:p[1:] for p in old['scan']}
            if any(saved[k]!=current_scan[k] for k in saved.keys()&current_scan.keys()):continue
            if old['scan'][0]!=stamp['scan'][0]:continue
            known=set(plan['accepted'].get(str(first),[]));count=0
            for item in samples[first]:
                if item['id']>0 and item['id'] in known and not item['strong']:
                    item['strong']=True;count+=1
            if count:return count
        except (OSError,ValueError,KeyError,TypeError):continue
    return 0


def build_plan(project,start=0,end=None,target_ids=None,progress=None):
    """Read bounded scan chunks and measure clarity once, without any neural retry."""
    from minute_queue import atomic_json
    path=Path(project['people_path']);scan=json.loads(path.read_text(encoding='utf-8'))
    meta=scan['meta'];end=min(end or meta['frames'],meta['frames'])
    context_limit=max(2,min(30,round(meta['fps']*.75)));context=max(10,context_limit+2)
    lo=max(0,start-context);hi=min(meta['frames'],end+context)
    selected=selected_ids(project,target_ids);protected=ProtectedFrames({'fps':meta['fps'],'excluded_ranges':project.get('excluded_ranges',[])})
    stat=Path(project['target']).stat();parts=[p for p in scan.get('frame_chunks',[]) if p['start']<hi and p['end']>lo]
    stamps=[[str(path),path.stat().st_size,path.stat().st_mtime_ns]]
    for part in parts:
        item=path.parent/part['file'];value=item.stat();stamps.append([str(item),value.st_size,value.st_mtime_ns])
    stamp={'version':VERSION,'source':[str(project['target']),stat.st_size,stat.st_mtime_ns],'scan':stamps,
           'range':[start,end],'selected':sorted(selected),'groups':sorted(sorted(g) for g in project.get('person_groups',[])),
           'excluded':protected.ranges}
    digest=hashlib.sha256(json.dumps(stamp,sort_keys=True).encode()).hexdigest()[:24]
    folder=Path(project['folder'])/'processing-plan';folder.mkdir(exist_ok=True)
    ready=folder/(digest+'.json')
    if ready.exists():
        plan=json.loads(ready.read_text(encoding='utf-8'))
        if plan.get('stamp')==stamp:
            if progress:progress(None,desc=f'План загружен: обработка {plan["processed_frames"]} кадров, пропуск {plan["skipped_frames"]}.')
            return ready,plan
    candidates={};observed={};observations={};reasons={};rejected=0;mapping={i:i for i in selected}
    for group in project.get('person_groups',[]):
        if set(group)&selected:mapping.update({i:min(group) for i in group})
    def reject(reason):
        nonlocal rejected
        rejected+=1;reasons[reason]=reasons.get(reason,0)+1
    for part in parts or [None]:
        data=json.loads((path.parent/part['file']).read_text(encoding='utf-8')) if part else scan
        for key,records in data.get('frames',{}).items():
            index=int(key)
            if not lo<=index<hi or protected.contains(index):continue
            observed[index]=[np.asarray(r['face']['bounding_box']) for r in records]
            observations[index]=records
            for ordinal,record in enumerate(records):
                if record['person_id'] not in selected and record['person_id']!=0:continue
                reason=basic_reason(record['face'],meta['width'],meta['height'],relaxed=True)
                if reason:reject(reason)
                else:candidates.setdefault(index,[]).append((ordinal,record))
    from selection_tracking import measurement_patch,retain_measurements,bridge_faces
    samples={}
    if candidates:
        first=min(candidates);last=max(candidates)+1
        if progress:progress(None,desc='План обработки: отбираем чёткие лица, без повторного поиска…')
        for index,gray,scale in gray_frames(project['target'],meta,first,last):
            for ordinal,record in candidates.pop(index,[]):
                value=record['face'];quality=sharpness(gray,value['bounding_box'],scale)
                if quality<4:reject('blur');continue
                strong=record['person_id']>0 and quality>=MIN_SHARPNESS and basic_reason(value,meta['width'],meta['height']) is None
                samples.setdefault(index,[]).append({'id':record['person_id'],'group':mapping.get(record['person_id'],0),'ordinal':ordinal,
                    'box':np.asarray(value['bounding_box'],np.float32),'vector':np.asarray(value['embedding_norm'],np.float32),
                    'scene':cv2.resize(gray,(64,36)),'strong':strong,'sharpness':quality,'landmarker':value['score_set']['landmarker'],
                    'points':np.asarray(value['landmark_set']['5/68']), 'patch':measurement_patch(gray,value['bounding_box'],scale),'face':value})
            if progress and (index-first)%180==0:progress(None,desc=f'План обработки: кадр {index-first+1}/{last-first}…')
    accepted,promotions,held=keep_short_dips(samples,meta['fps'])
    seeded=seed_cached_measurements(samples,folder,stamp)
    if progress:progress(None,desc='Удержание выбранных лиц по движению и сходству…')
    continued,linked_promotions,retained,identified=retain_measurements(samples,meta['fps'])
    for key,ids in continued.items():accepted[key]=sorted(set(accepted.get(key,[]))|set(ids))
    for key,items in linked_promotions.items():
        existing={p['ordinal'] for p in promotions.get(key,[])}
        promotions.setdefault(key,[]).extend(p for p in items if p['ordinal'] not in existing)
    # Only short holes bracketed by the same selected person are inspected.
    # Long empty shots still skip all neural processing and tracking work.
    anchors={};bridges={};bridge_limit=max(2,min(6,round(meta['fps']*.20)))
    for index,items in sorted(samples.items()):
        for item in items:
            if item['id']>0 and item['id'] in accepted.get(str(index),[]):anchors.setdefault(item['group'],[]).append((index,item))
    pending=[]
    for values in anchors.values():
        for (a,left),(b,right) in zip(values,values[1:]):
            if not 1<b-a<=bridge_limit+1 or any(protected.contains(i) for i in range(a,b+1)):continue
            expected=lambda i:(left['box']*(b-i)+right['box']*(i-a))/(b-a)
            # Never replace a measured but rejected face or overlap another
            # detected person, even when its identity is still unresolved.
            if any(any(box_overlap(expected(i),box)>.25 for box in observed.get(i,[])) for i in range(a+1,b)):continue
            pending.append((a,b,left,right))
    # Decode overlapping repair windows once and keep at most three seconds
    # of small grayscale context, independent of the total video duration.
    windows=[]
    for a,b,_,_ in sorted(pending,key=lambda x:x[0]):
        if windows and a-windows[-1][1]<=meta['fps']*.5 and b-windows[-1][0]<=meta['fps']*3:
            windows[-1][1]=max(windows[-1][1],b)
        else:windows.append([a,b])
    for wa,wb in windows:
        if progress:progress(None,desc=f'Проверка коротких пропусков: {wa/meta["fps"]:.1f}–{wb/meta["fps"]:.1f} с…')
        context={};scale=None
        for i,gray,scale in gray_frames(project['target'],meta,wa,wb+1):context[i]=gray
        for a,b,left,right in [v for v in pending if wa<=v[0] and v[1]<=wb]:
            images=[context[i] for i in range(a,b+1)]
            recovered=bridge_faces(images,left['face'],right['face'],scale)
            if recovered is None:continue
            for i,face in enumerate(recovered,a+1):
                if basic_reason(face,meta['width'],meta['height'],relaxed=True) is not None:continue
                if sharpness(images[i-a],face['bounding_box'],scale)<4:continue
                if any(box_overlap(np.asarray(face['bounding_box']),box)>.25 for box in observed.get(i,[])):continue
                bridges.setdefault(str(i),[]).append({'person_id':left['id'],'face':face})
                observed.setdefault(i,[]).append(np.asarray(face['bounding_box']))
                accepted.setdefault(str(i),[]).append(left['id'])
    # Hands/objects require a separate visibility-aware path. Ordinary flow
    # bridges never become a reason to paint over a measured occluder.
    context_pending=[]
    for group,values in anchors.items():
        for (a,left),(b,right) in zip(values,values[1:]):
            if not 1<b-a<=context_limit+1 or b<=start or a>=end:continue
            if any(protected.contains(i) for i in range(a,b+1)):continue
            if all(any(mapping.get(identity,identity)==group for identity in accepted.get(str(i),[])) for i in range(a+1,b)):continue
            blocked=False
            for i in range(a+1,b):
                expected=(left['box']*(b-i)+right['box']*(i-a))/(b-a)
                for record in observations.get(i,[]):
                    f=record['face'];identity=record['person_id']
                    if box_overlap(expected,np.asarray(f['bounding_box']))<=.25:continue
                    if identity and mapping.get(identity,identity)!=group:blocked=True;break
                    if not identity and min(f['score_set'].values())>.75:
                        v=np.asarray(f['embedding_norm'])
                        if max(float(v@side['vector']) for side in (left,right))<.25:blocked=True;break
                if blocked:break
            if not blocked:context_pending.append((a,b,left,right))
    context_count=0
    if context_pending:
        from occlusion_context import Visibility,visible_bridge,scaled_face
        model=Visibility()
        context_windows=[]
        for a,b,_,_ in sorted(context_pending,key=lambda v:v[0]):
            if context_windows and a-context_windows[-1][1]<=meta['fps']*.5 and b-context_windows[-1][0]<=meta['fps']*3:
                context_windows[-1][1]=max(context_windows[-1][1],b)
            else:context_windows.append([a,b])
        try:
            for wa,wb in context_windows:
                if progress:progress(None,desc=f'Видимая часть лица при перекрытии: {wa/meta["fps"]:.1f}–{wb/meta["fps"]:.1f} с…')
                context_images={};scale=None
                for i,image,scale in gray_frames(project['target'],meta,wa,wb+1,color=True):context_images[i]=image
                for a,b,left,right in [v for v in context_pending if wa<=v[0] and v[1]<=wb]:
                    if all(any(mapping.get(identity,identity)==left['group'] for identity in accepted.get(str(i),[])) for i in range(a+1,b)):continue
                    images=[context_images[i] for i in range(a,b+1)]
                    recovered=visible_bridge(images,scaled_face(left['face'],scale),scaled_face(right['face'],scale),model.measure,context_limit)
                    if recovered is None:continue
                    for i,(face,visibility) in enumerate(recovered,a+1):
                        if any(mapping.get(identity,identity)==left['group'] for identity in accepted.get(str(i),[])):continue
                        face=scaled_face(face,(1/scale[0],1/scale[1]))
                        if basic_reason(face,meta['width'],meta['height'],relaxed=True) is not None:continue
                        if any(r['person_id'] and mapping.get(r['person_id'],r['person_id'])!=left['group'] and
                               box_overlap(np.asarray(face['bounding_box']),np.asarray(r['face']['bounding_box']))>.25
                               for r in observations.get(i,[])+bridges.get(str(i),[])):continue
                        replaced=[n for n,r in enumerate(observations.get(i,[])) if box_overlap(np.asarray(face['bounding_box']),np.asarray(r['face']['bounding_box']))>.50]
                        bridges.setdefault(str(i),[]).append({'person_id':left['id'],'face':face,'context_visibility':visibility,'replaces_ordinals':replaced})
                        accepted.setdefault(str(i),[]).append(left['id']);context_count+=1
        finally:model.session=None
    actual=[int(k) for k,v in accepted.items() if v and start<=int(k)<end]
    plan={'version':VERSION,'stamp':stamp,'digest':digest,'start':start,'end':end,'selected':sorted(selected),
          'accepted':accepted,'promotions':promotions,'held_measurements':held,'retained_measurements':retained,
          'linked_identities':identified,'seeded_measurements':seeded,'bridges':bridges,'bridged_faces':sum(map(len,bridges.values())),
          'occlusion_context_faces':context_count,
          'active_ranges':ranges(actual),'processed_frames':len(actual),
          'skipped_frames':end-start-len(actual),'rejected_faces':rejected,'reasons':reasons}
    atomic_json(ready,plan)
    if progress:progress(None,desc=f'По плану: обработка {len(actual)} кадров, пропуск {plan["skipped_frames"]}.')
    return ready,plan


def apply_plan(scan,plan):
    selected=set(plan['selected'])
    for key,items in plan.get('promotions',{}).items():
        records=scan['frames'].get(key,[])
        for item in items:
            if item['ordinal']<len(records) and records[item['ordinal']]['person_id']==0:
                records[item['ordinal']]={**records[item['ordinal']],'person_id':item['person_id']}
    for key,records in plan.get('bridges',{}).items():
        replaced={n for r in records for n in r.get('replaces_ordinals',[])}
        if replaced:scan['frames'][key]=[r for n,r in enumerate(scan['frames'].get(key,[])) if n not in replaced]
        scan['frames'].setdefault(key,[]).extend(records)
    scan['frames']={key:[r for r in records if r['person_id'] not in selected or r['person_id'] in plan['accepted'].get(key,[])]
                    for key,records in scan['frames'].items()}
    for key,records in scan['frames'].items():
        confirmed=[r for r in records if r['person_id'] in selected]
        # A duplicate unresolved measurement must not paint the original face
        # back over a confirmed replacement. Named unassigned people stay intact.
        def duplicate(record):
            if record['person_id']!=0:return False
            for known in confirmed:
                a,b=record['face'],known['face']
                overlap=box_overlap(np.asarray(a['bounding_box']),np.asarray(b['bounding_box']))
                if overlap>=.5 and float(np.asarray(a['embedding_norm'])@np.asarray(b['embedding_norm']))>=.6:return True
            return False
        scan['frames'][key]=[r for r in records if not duplicate(r)]
    scan['selection_signature']=plan['digest']
    return scan


def pass_through(project,folder,start,end,plan,base_target=None,base_start_frame=0):
    """No-face part: FFmpeg only, exact frame count and source audio."""
    from video_tools import FFMPEG,FLAGS,metadata,verify
    from minute_queue import atomic_json
    from quality import render_settings
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);output=folder/'render.mp4'
    meta=metadata(project['target']);settings=render_settings(project.get('processing_profile'))
    source=base_target or project['target'];offset=start-base_start_frame if base_target else start
    args=[FFMPEG,'-v','error','-y','-ss',format(offset/meta['fps'],'.12f'),'-i',str(source),'-map','0:v:0','-map','0:a:0?',
          '-t',format((end-start)/meta['fps'],'.12f'),'-fps_mode','cfr','-r',meta['rate'],'-frames:v',str(end-start),
          '-c:v','libx264','-crf','10','-preset','slow','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(output)]
    p=subprocess.run(args,capture_output=True,creationflags=FLAGS)
    if p.returncode:raise RuntimeError('Не удалось сохранить участок без замены: '+p.stderr.decode(errors='replace')[-700:])
    info=verify(output)
    if info['frames']!=end-start:raise RuntimeError('Изменилось число кадров участка без замены.')
    report={'start_frame':start,'end_frame':end,'residuals':[],'checked_pairs':0,'raw_median_px':None,'stable_median_px':None,
            'face_free_frames':end-start,'excluded_frames':0,'excluded_ranges':project.get('excluded_ranges',[]),
            'suspicious_segments':[],'people_geometry':{},'uncertain_detections':0,'tracking_version':4,'recovery_version':1,
            'selection_version':VERSION,'selection_signature':plan['digest'],'processed_frames':0,'skipped_frames':end-start,
            'rejected_faces':plan['rejected_faces'],'continued_detections':0,'recovered_detections':0,'unswapped_segments':[],
            'processing_profile':settings['id'],'render_settings':settings,'replacement_check':{'sampled_frames':0,'changed_frames':0,'people':{}},
            'neural_processing_skipped':True}
    atomic_json(folder/'geometry.json',report)
    atomic_json(folder/'source-notes.json',['Участок без подходящих лиц: сохранён без нейросетевой обработки.'])
    return output,report
