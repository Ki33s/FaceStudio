"""JSON-lines bridge for the native desktop UI; reuses the existing processor."""
import json
import re
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

OUT = sys.stdout
OUT.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# Keep library messages separate from the desktop protocol.
sys.stdout = sys.stderr
ROOT = Path(__file__).resolve().parent


def emit(kind, **values):
    OUT.write(json.dumps({'type': kind, **values}, ensure_ascii=False) + '\n')
    OUT.flush()


def progress(value, desc=''):
    if 'processing:' in desc:
        percent = re.search(r'(\d+)%', desc)
        desc = 'Замена лица' + (': ' + percent[1] + '%' if percent else '…')
        if percent:
            value = int(percent[1]) / 100
    elif 'analysing:' in desc:
        percent = re.search(r'(\d+)%', desc)
        desc = 'Проверка содержимого' + (': ' + percent[1] + '%' if percent else '…')
    elif '[FACEFUSION.' in desc or desc.startswith('{'):
        desc = 'Обработка видео…'
    emit('progress', value=value, text=desc)


def load_project(path):
    path = Path(path)
    project = json.loads(path.read_text(encoding='utf-8'))
    old_folder, new_folder = project['folder'], str(path.parent)
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(old_folder):
            return new_folder + value[len(old_folder):]
        return value
    return relocate(project)


def completed(project, report, produced_video=False):
    emit('completed', video=project['target'], primary=project['sources'][0],
         auxiliary=project['sources'][1:], result=project['result'], report=report,
         project=str(Path(project['folder']) / 'project.json'),
         analysis=project.get('people_path'),characters=project.get('characters',[]),
         assignments=project.get('assignments',[]),person_groups=project.get('person_groups',[]),excluded_ranges=project.get('excluded_ranges',[]),produced_video=produced_video)


def scan_video(video):
    from ollama_local import release_model
    from people import fingerprint
    from video_tools import metadata,prepare_timeline
    from minute_queue import atomic_json,frame_parts
    import msvcrt
    release_model(ROOT/'data')
    digest=fingerprint(video)
    timeline=prepare_timeline(video,ROOT/'data/timelines'/digest[:24],progress)
    meta=metadata(timeline)
    folder=ROOT/'data/scans'/('minutes-v3-'+digest[:24]);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'scan.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0');lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise RuntimeError('Поиск людей в этом видео уже запущен в другом окне.')
        ready=folder/'people.json'
        if ready.exists():
            scan=json.loads(ready.read_text(encoding='utf-8'))
            emit('characters',analysis=str(ready),characters=scan['characters'],video=video)
            return
        total=meta['frames'] or round(meta['duration']*meta['fps'])
        parts=frame_parts(0,total,meta['fps'])
        checkpoint=folder/'scan-checkpoint.json'
        saved=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'next_frame':0,'frame_chunks':[]}
        for number,(start,end) in enumerate(parts):
            if end<=saved['next_frame']: continue
            manifest=folder/'job.json'
            atomic_json(manifest,{'video':timeline,'folder':str(folder),'start_frame':start,'end_frame':end})
            progress(start/total,desc=f'Поиск людей: часть {number+1}/{len(parts)}…')
            with (folder/('scan-%04d.log' % (number+1))).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scan_worker.py'),str(manifest)],
                    stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                result=None
                for line in proc.stdout:
                    log.write(line);log.flush()
                    try: message=json.loads(line)
                    except ValueError: continue
                    if message.get('type')=='progress':
                        emit('progress',value=(start+message['value']*(end-start))/total,text=f"Поиск людей: часть {number+1}/{len(parts)} · "+message['text'])
                    if message.get('type')=='scan_part': result=message
                code=proc.wait()
            if code or result is None: raise RuntimeError('Поиск не завершён. Готовые минуты сохранены. Запустите поиск снова. Лог: '+str(log.name))
            saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        characters=saved.get('characters',[])
        if not characters: raise RuntimeError('Достоверные лица в видео не найдены.')
        scan={'tracking_version':3,'fingerprint':fingerprint(timeline),'input_fingerprint':digest,'processing_video':timeline,'meta':meta,'characters':characters,'frame_chunks':saved['frame_chunks'],
              'uncertain_detections':sum(part['uncertain_detections'] for part in saved['frame_chunks'])}
        atomic_json(ready,scan)
        emit('characters',analysis=str(ready),characters=characters,video=video)


def checkpoint(project,done,total):
    from app import save_project
    if project.get('pending'):
        project['pending'].update(completed_parts=done,total_parts=total)
        save_project(project)
    emit('checkpoint',project=str(Path(project['folder'])/'project.json'),completed_parts=done,total_parts=total)


def pending_project():
    files=sorted((ROOT/'data/projects').glob('*/project.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not files: return False
    project=load_project(files[0])
    if not project.get('pending'): return False
    emit('pending',project=str(files[0]),video=project['target'],result=project.get('result'),
         analysis=project.get('people_path'),characters=project.get('characters',[]),assignments=project.get('assignments',[]),person_groups=project.get('person_groups',[]),
         excluded_ranges=project.get('excluded_ranges',[]),completed_parts=project['pending'].get('completed_parts',0),total_parts=project['pending'].get('total_parts',0))
    return True


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    if job['action'] == 'startup':
        from ollama_local import ensure_running, choose_model
        status = ensure_running(ROOT / 'data')
        emit('ready', text=status + ' Локальная модель: ' + choose_model() + '.')
        return
    if job['action']=='scan':
        scan_video(job['video'])
        return
    from app import execute, load_last
    if job['action'] == 'last':
        if pending_project(): return
        try:
            _, _, _, _, report, project, _ = load_last()
        except Exception:
            emit('empty', text='Выберите видео, затем нажмите «Заменить лица».')
            return
        completed(project, report)
        return
    if job['action'] not in ('execute','swap','check','resume'):
        raise ValueError('Неизвестная операция.')
    automatic = job['action'] == 'swap'
    project = load_project(job['project']) if not automatic and job.get('project') else None
    _, report, project, _ = execute(job.get('video'), job.get('primary'),
        job.get('auxiliary', []), job.get('command', ''), project, progress=progress,automatic=automatic,
        analysis=job.get('analysis'),assignments=job.get('assignments'),technical_check=job['action']=='check',excluded_ranges=job.get('excluded_ranges'),resume=job['action']=='resume',checkpoint=checkpoint,person_groups=job.get('person_groups'))
    completed(project, report,produced_video=project['history'][-1]['plan']['action']!='check')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit('error', text=str(error))
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
