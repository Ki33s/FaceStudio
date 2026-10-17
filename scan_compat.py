"""Rebase a saved scan only across a verified, frame-identical timeline re-encode."""
import json
import subprocess
from pathlib import Path
from people import fingerprint
from video_tools import metadata,FFMPEG,FLAGS


def _samples(path,indices,rate=None):
    # Decode on the same CFR clock as discovery. OpenCV random seeks can jump
    # ahead on VFR recordings, even when the returned timestamp looks valid.
    select='+'.join(f'eq(n,{index})' for index in indices)
    filters=([f'fps={rate}:start_time=0'] if rate else [])
    filters += [f"select='{select}'",'scale=64:36','format=gray']
    result=subprocess.run([FFMPEG,'-v','error','-i',str(path),'-map','0:v:0',
        '-vf',','.join(filters),'-fps_mode','vfr','-frames:v',str(len(indices)),
        '-f','rawvideo','-pix_fmt','gray','pipe:1'],capture_output=True,creationflags=FLAGS)
    if result.returncode or len(result.stdout)!=len(indices)*64*36:
        raise RuntimeError('Не удалось прочитать контрольные кадры. Назначения фотографий сохранены.')
    import numpy as np
    return np.frombuffer(result.stdout,np.uint8).reshape(len(indices),36,64).astype(float)


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
    import numpy as np
    indices=sorted({min(new['frames']-1,round(new['frames']*p)) for p in (.13,.31,.53,.71,.89)})
    reference=_samples(original,indices,new['rate'])
    prepared=_samples(processing,indices)
    errors=np.abs(reference-prepared).mean(axis=(1,2)).tolist()
    if max(errors)>12 or float(np.median(errors))>5:raise RuntimeError(message)
    return {**scan,'fingerprint':current,'meta':new,
            'timeline_rebase':{'previous_fingerprint':scan['fingerprint'],
                               'reason':'same source, clock, frame count and verified content',
                               'method':'ffmpeg-cfr-frame-indices','sample_errors':errors}}
