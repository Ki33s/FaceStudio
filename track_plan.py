"""Read each frame once for all selected identities; persist reusable motion maps."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from facefusion.vision import read_static_video_frame
from temporal import motion
from minute_queue import atomic_json


def prepare_motion(job,tracks,scan,folder):
    total=scan['meta']['frames']
    start=max(0,job.get('start_frame',0)-9)
    end=min(total,(job.get('end_frame') or total)+9)
    target=Path(job['target']);stat=target.stat();scan_stat=Path(job['people_path']).stat()
    signature={'version':2,'coverage':scan.get('coverage_signature'),'target_size':stat.st_size,'target_mtime':stat.st_mtime_ns,
        'scan_size':scan_stat.st_size,'scan_mtime':scan_stat.st_mtime_ns,'fps':job['fps']}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    cache=Path(job.get('track_cache') or folder/'track-plan');cache.mkdir(parents=True,exist_ok=True)
    result={identity:{} for identity in tracks}
    # A correction can reuse overlapping minute maps, including their context.
    for path in cache.glob(digest+'-*.json'):
        try:
            part_start,part_end=map(int,path.stem.rsplit('-',2)[-2:])
            if part_start>=end or part_end<=start: continue
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved['signature']!=signature or saved['start']>=end or saved['end']<=start: continue
            for identity in tracks:
                for index,link in saved['people'].get(str(identity),{}).items():
                    if start<=int(index)<end: result[identity][index]=link
        except (OSError,ValueError,KeyError): continue
    missing={identity for identity in tracks if any(str(i) not in result[identity] for i in range(start,end))}
    if not missing:
        print('Карта движения загружена: повторное чтение видео не требуется.',flush=True)
        return result
    print(f'Общий проход подготовки: людей {len(missing)}, кадров {end-start}.',flush=True)
    previous=None;previous_gray=None;read_count=0
    try:
        for index in range(max(0,start-1),end):
            frame=read_static_video_frame(job['target'],index)
            if frame is None: raise RuntimeError('Не удалось подготовить кадр '+str(index))
            read_count+=1;gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            scene_cut=previous_gray is not None and np.abs(cv2.resize(previous_gray,(32,32)).astype(float)-cv2.resize(gray,(32,32))).mean()>45
            if index>=start:
                for identity in missing:
                    if str(index) in result[identity]: continue
                    before=tracks[identity].get(index-1);after=tracks[identity].get(index)
                    matrix,valid=(motion(previous,frame,before.bounding_box,(previous_gray,gray),scene_cut)
                        if previous is not None and before is not None and after is not None else (np.eye(3),False))
                    result[identity][str(index)]={'matrix':matrix.tolist(),'valid':bool(valid)}
            previous=frame;previous_gray=gray;read_static_video_frame.cache_clear()
            if index%30==0: print(f'Общая подготовка траекторий: {index-start+1}/{end-start}',flush=True)
    finally: read_static_video_frame.cache_clear()
    atomic_json(cache/(digest+f'-{start:09d}-{end:09d}.json'),
        {'signature':signature,'start':start,'end':end,'decoded_frames':read_count,
         'people':{str(identity):links for identity,links in result.items()}})
    print(f'Карта сохранена: прочитано {read_count} кадров для {len(missing)} людей.',flush=True)
    return result
