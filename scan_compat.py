"""Rebase a saved scan only across a verified, frame-identical timeline re-encode."""
import json
from pathlib import Path
from people import fingerprint
from video_tools import metadata


def validate_processing(scan,original):
    processing=scan.get('processing_video',original)
    current=fingerprint(processing)
    if current==scan['fingerprint']:return scan
    message='Подготовленное видео изменилось. Назначения сохранены; нужна проверка временной шкалы.'
    original_hash=fingerprint(original)
    if original_hash!=scan.get('input_fingerprint'):raise RuntimeError(message)
    path=Path(processing);sidecar=path.with_suffix('.json')
    if not sidecar.is_file():raise RuntimeError(message)
    saved=json.loads(sidecar.read_text(encoding='utf-8'))
    if (saved.get('signature',{}).get('source')!=original_hash or
        saved.get('signature',{}).get('version') not in (1,2) or saved.get('size')!=path.stat().st_size):
        raise RuntimeError(message)
    old=scan['meta'];new=metadata(processing)
    if (any(old[key]!=new[key] for key in ('width','height','frames')) or
        not old['frames'] or abs(old['fps']-new['fps'])>1e-6 or
        abs(old['duration']-new['duration'])>1/old['fps']+.001):raise RuntimeError(message)
    # A matching filename/size is insufficient. Check the original content at
    # several timestamps before accepting old geometry on the new encoding.
    import cv2
    import numpy as np
    videos=[cv2.VideoCapture(str(p)) for p in (original,processing)];errors=[]
    try:
        for portion in (.13,.31,.53,.71,.89):
            videos[1].set(cv2.CAP_PROP_POS_MSEC,new['duration']*portion*1000)
            ok,frame=videos[1].read()
            if not ok:raise RuntimeError(message)
            timestamp=videos[1].get(cv2.CAP_PROP_POS_MSEC)
            target=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,36)).astype(float)
            # OpenCV seeks VFR recordings a few frames early. Match only the
            # nearby original frames that the CFR fps filter could repeat.
            videos[0].set(cv2.CAP_PROP_POS_MSEC,max(0,timestamp-120));near=[]
            for _ in range(12):
                ok,frame=videos[0].read()
                if not ok:break
                observed=videos[0].get(cv2.CAP_PROP_POS_MSEC)
                if observed>timestamp+120:break
                if abs(observed-timestamp)<=130:
                    image=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,36)).astype(float)
                    near.append(float(np.abs(image-target).mean()))
            if not near:raise RuntimeError(message)
            errors.append(min(near))
    finally:
        for video in videos:video.release()
    if max(errors)>12 or float(np.median(errors))>5:raise RuntimeError(message)
    return {**scan,'fingerprint':current,'meta':new,
            'timeline_rebase':{'previous_fingerprint':scan['fingerprint'],
                               'reason':'same source, clock, frame count and verified content',
                               'sample_errors':errors}}
