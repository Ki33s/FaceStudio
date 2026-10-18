"""Versioned, resumable discovery and whole-video identity analysis."""
import json
import subprocess
import sys
from pathlib import Path
from people import fingerprint,load_scan
from minute_queue import atomic_json,frame_parts
from video_tools import metadata,prepare_timeline


def scan_profiles(video,root,progress=None,provider=None,previous=None):
    import msvcrt
    root=Path(root);digest=fingerprint(video);folder=root/'data/scans'/('minutes-v5-'+digest[:24])
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'scan.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Анализ этого видео уже запущен в другом окне.')
        ready=folder/'people.json'
        if ready.exists():return ready,load_scan(ready,video)
        legacy=None;legacy_scan=None;draft=None
        candidates=[root/'data/scans'/f'minutes-v{version}-{digest[:24]}'/'people.json' for version in (4,3)]
        if previous and previous.get('analysis'):candidates.insert(0,Path(previous['analysis']))
        for path in candidates:
            if path.is_file():
                legacy=path;legacy_scan=load_scan(path,video)
                from assignment_drafts import restore_legacy_draft
                restore_legacy_draft(path,legacy_scan,root/'data/desktop-jobs')
                saved=path.parent/'assignments.json'
                if previous and Path(previous.get('analysis','')).resolve()==path.resolve() and 'assignments' in previous:
                    draft={'input_fingerprint':legacy_scan.get('input_fingerprint',legacy_scan['fingerprint']),
                        'person_ids':sorted(p['id'] for p in legacy_scan['characters']),
                        'assignments':previous['assignments'],'person_groups':previous.get('person_groups',[]),'grouping_reviewed':True}
                    atomic_json(folder/'previous-assignments.json',draft)
                elif saved.exists():
                    value=json.loads(saved.read_text(encoding='utf-8-sig'))
                    if (value.get('input_fingerprint')==legacy_scan.get('input_fingerprint',legacy_scan['fingerprint']) and
                        set(value.get('person_ids',[]))=={p['id'] for p in legacy_scan['characters']}):
                        draft=value;atomic_json(folder/'previous-assignments.json',value)
                break
        raw_folder=folder/'raw';raw_folder.mkdir(exist_ok=True);raw_ready=raw_folder/'people.json'
        if raw_ready.exists():raw=load_scan(raw_ready,video)
        else:
            from ollama_local import release_model
            release_model(root/'data')
            timeline=prepare_timeline(video,root/'data/timelines'/digest[:24],progress)
            meta=metadata(timeline);total=meta['frames'];parts=frame_parts(0,total,meta['fps'])
            checkpoint=raw_folder/'scan-checkpoint.json'
            saved=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'next_frame':0}
            for number,(start,end) in enumerate(parts):
                if end<=saved['next_frame']:continue
                job={'video':timeline,'folder':str(raw_folder),'start_frame':start,'end_frame':end,'retain_observations':True}
                if provider:job['provider']=provider
                manifest=raw_folder/'job.json';atomic_json(manifest,job)
                if progress:progress(None,desc=f'Тщательный поиск лиц: часть {number+1}/{len(parts)}…')
                with (raw_folder/f'scan-{number+1:04d}.log').open('w',encoding='utf-8') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(root/'scan_worker.py'),str(manifest)],
                        stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    result=None
                    for line in process.stdout:
                        log.write(line);log.flush()
                        try:message=json.loads(line)
                        except ValueError:continue
                        if message.get('type')=='progress' and progress:progress(None,desc=message['text'])
                        if message.get('type')=='scan_part':result=message
                    code=process.wait();process.stdout.close()
                if code or result is None:raise RuntimeError('Поиск прерван. Готовые части и назначения сохранены. Лог: '+str(log.name))
                saved=json.loads(checkpoint.read_text(encoding='utf-8'))
            raw={'tracking_version':5,'detection_schema':1,'fingerprint':fingerprint(timeline),'input_fingerprint':digest,
                'processing_video':timeline,'meta':meta,'characters':saved.get('characters',[]),'frame_chunks':saved['frame_chunks']}
            atomic_json(raw_ready,raw)
        from scan_compat import validate_processing
        compatible=validate_processing(raw,video)
        if compatible is not raw:atomic_json(raw_ready,compatible);raw=compatible
        if legacy_scan:
            old=legacy_scan['meta'];new=raw['meta']
            if (any(old[k]!=new[k] for k in ('width','height','frames')) or abs(old['fps']-new['fps'])>1e-6):
                # A different clock cannot migrate photo choices by frame index.
                # Preserve and expose the original selection instead of guessing.
                raise RuntimeError('Временная шкала прежнего поиска отличается. Старые карточки и фотографии сохранены; автоматический перенос назначений остановлен.')
        from identity_analysis import build_identity_scan
        scan=build_identity_scan(raw_ready,folder,progress,draft,legacy)
        return ready,scan
