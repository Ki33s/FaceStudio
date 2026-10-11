import json
import subprocess
import math
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
FFMPEG=str(ROOT/'runtime/bin/ffmpeg.exe')
FFPROBE=str(ROOT/'runtime/bin/ffprobe.exe')
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)


@lru_cache(maxsize=96)
def _metadata(path, size, modified):
    value=json.loads(subprocess.check_output([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(path)],creationflags=FLAGS))
    video=next((s for s in value['streams'] if s['codec_type']=='video'),None)
    if video is None: raise RuntimeError('В файле нет видеопотока.')
    rates=[video.get('avg_frame_rate'),video.get('r_frame_rate')]
    def valid_rate(value):
        try: return bool(value) and Fraction(value)>0
        except (ValueError,ZeroDivisionError): return False
    rate=next((r for r in rates if valid_rate(r)),'0/1')
    fps=float(Fraction(rate))
    if not math.isfinite(fps) or fps<=0: raise RuntimeError('Не удалось определить частоту кадров видео.')
    duration=float(video.get('duration') if video.get('duration') not in (None,'N/A') else value['format']['duration'])
    count=video.get('nb_frames')
    if not count or count=='N/A':
        counted=json.loads(subprocess.check_output([FFPROBE,'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(path)],creationflags=FLAGS))
        count=counted['streams'][0].get('nb_read_frames')
    if not count or count=='N/A': raise RuntimeError('Не удалось посчитать кадры видео.')
    return {'width':video['width'],'height':video['height'],'fps':fps,
        'rate':rate,'nominal_rate':video.get('r_frame_rate') if valid_rate(video.get('r_frame_rate')) else rate,'duration':duration,
        'container_duration':float(value['format']['duration']), 'frames':int(count),
        'audio':any(s['codec_type']=='audio' for s in value['streams'])}


def metadata(path):
    path=Path(path).resolve();stat=path.stat()
    return dict(_metadata(str(path),stat.st_size,stat.st_mtime_ns))


def prepare_timeline(path,folder,progress=None):
    """Keep native CFR files; normalize VFR before discovering frame identities."""
    from people import fingerprint
    from minute_queue import atomic_json
    meta=metadata(path)
    try: nominal=float(Fraction(meta['nominal_rate']))
    except (ValueError,ZeroDivisionError): nominal=meta['fps']
    # Different nominal/average clocks must never share one frame numbering map.
    variable=abs(nominal-meta['fps'])>max(.001,meta['fps']*.0001)
    if not variable:
        # A VFR stream can still report equal nominal and average rates.
        # Check packet timing at the beginning, middle and end as well.
        intervals=','.join(f'{max(0,sec):.6f}%+#120' for sec in (0,meta['duration']/2,meta['duration']-5))
        packet_data=json.loads(subprocess.check_output([FFPROBE,'-v','error','-select_streams','v:0',
            '-read_intervals',intervals,'-show_packets','-show_entries','packet=duration_time','-of','json',str(path)],creationflags=FLAGS))
        durations=[float(p['duration_time']) for p in packet_data.get('packets',[]) if p.get('duration_time') not in (None,'N/A')]
        if durations: variable=max(durations)-min(durations)>max(.001,.02/meta['fps'])
    if not variable: return str(path)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prepared=folder/'timeline.mp4';manifest=folder/'timeline.json'
    signature={'version':1,'source':fingerprint(path)}
    if prepared.exists() and manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if saved.get('signature')==signature and saved.get('size')==prepared.stat().st_size: return str(prepared)
    rate=meta['nominal_rate'] if 0<nominal<=120 else meta['rate']
    temporary=folder/'timeline-building.mp4'
    if progress: progress(None,desc='Подготовка временной шкалы видео с переменной частотой кадров…')
    result=subprocess.run([FFMPEG,'-v','error','-y','-i',str(path),'-map','0:v:0','-map','0:a:0?',
        '-vf',f'fps={rate}:start_time=0','-fps_mode','cfr','-c:v','libx264','-crf','18','-preset','fast',
        '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(temporary)],capture_output=True,creationflags=FLAGS)
    if result.returncode: raise RuntimeError('Не удалось подготовить видео: '+result.stderr.decode('utf-8',errors='replace')[-1000:])
    checked=verify(temporary,meta)
    if abs(checked['fps']-float(Fraction(rate)))>.001: raise RuntimeError('Не удалось выровнять частоту кадров.')
    temporary.replace(prepared)
    atomic_json(manifest,{'signature':signature,'size':prepared.stat().st_size})
    return str(prepared)


def verify(path, expected=None):
    meta=metadata(path)
    result=subprocess.run([FFMPEG,'-v','error','-i',str(path),'-f','null','-'],capture_output=True,creationflags=FLAGS)
    if result.returncode or result.stderr:
        raise RuntimeError('Проверка декодирования не прошла: '+result.stderr.decode('utf-8',errors='replace')[-1000:])
    if expected:
        if abs(meta['duration']-expected['duration'])>.35:
            raise RuntimeError('Длительность результата отличается от исходного видео.')
        if expected['audio'] and not meta['audio']:
            raise RuntimeError('В результате отсутствует исходный звук.')
    return meta


def join_patch(original, patch, output, start, end):
    meta=metadata(original)
    w,h=meta['width'],meta['height']; size=w*h*3
    def reader(path):
        return subprocess.Popen([FFMPEG,'-v','error','-i',str(path),'-map','0:v:0','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,creationflags=FLAGS)
    def read(proc):
        data=proc.stdout.read(size)
        return np.frombuffer(data,np.uint8).reshape(h,w,3) if len(data)==size else None
    a,b=reader(original),reader(patch)
    fpsrate=json.loads(subprocess.check_output([FFPROBE,'-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-of','json',str(original)],creationflags=FLAGS))['streams'][0]['avg_frame_rate']
    writer=subprocess.Popen([FFMPEG,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',fpsrate,'-i','pipe:0','-i',str(original),'-map','0:v:0','-map','1:a:0?',
        '-c:v','libx264','-crf','10','-preset','fast','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(output)],stdin=subprocess.PIPE,creationflags=FLAGS)
    index=0
    try:
        while True:
            frame=read(a)
            if frame is None: break
            if start<=index<end:
                changed=read(b)
                if changed is None: raise RuntimeError('Исправленный участок короче ожидаемого.')
                alpha=1.0
                if start>0: alpha=min(alpha,(index-start+1)/6)
                if end<meta['frames']: alpha=min(alpha,(end-index)/6)
                frame=np.rint(alpha*changed+(1-alpha)*frame).clip(0,255).astype(np.uint8)
            writer.stdin.write(frame.tobytes());index+=1
        writer.stdin.close()
        if writer.wait()!=0 or a.wait()!=0: raise RuntimeError('Ошибка сборки исправленного видео.')
    finally:
        for proc in (a,b,writer):
            if proc.poll() is None: proc.kill()
    return verify(output,meta)


def contact_sheet(path, destination):
    meta=metadata(path);tiles=[]
    for sec in np.linspace(0,max(0,meta['duration']-.5),6):
        data=subprocess.check_output([FFMPEG,'-v','error','-ss',str(sec),'-i',str(path),'-frames:v','1','-vf','scale=400:240:force_original_aspect_ratio=decrease,pad=400:240:(ow-iw)/2:(oh-ih)/2','-f','image2pipe','-vcodec','png','pipe:1'],creationflags=FLAGS)
        tile=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if tile is None: continue
        cv2.putText(tile,f'{sec:.1f}s',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
    if len(tiles)!=6: raise RuntimeError('Не удалось собрать кадры для проверки.')
    cv2.imwrite(str(destination),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
    return destination
