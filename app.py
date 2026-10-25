"""Local text-controlled face swapping application."""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8',errors='replace')
DATA=ROOT/'data'
DATA.mkdir(exist_ok=True)
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
os.environ['GRADIO_TEMP_DIR']=str(DATA/'uploads')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PATH']=str(ROOT/'runtime/bin')+os.pathsep+os.environ.get('PATH','')
import gradio as gr
from ollama_local import ensure_running,interpret,choose_model,release_model
from video_tools import metadata,verify,join_patch
from protected_ranges import normalize_ranges
from quality import render_settings

LOCK=threading.Lock()
STATUS='Проверка Ollama…'


def startup_check():
    global STATUS
    try: STATUS=ensure_running(DATA)+' Локальная модель: '+choose_model()+'.'
    except Exception as error: STATUS=str(error)


def render_once(project,folder,start=0,end=None,strong=False,progress=None,target_ids=None,base_target=None,base_start_frame=0,chunked=False):
    settings=render_settings(project.get('processing_profile'))
    meta=metadata(project['target'])
    output=folder/'render.mp4'
    job={'sources':project['sources'],'target':project['target'],'output':str(output),
        'fps':meta['fps'],'start_frame':start,'end_frame':end,'strong':strong,'processing_profile':settings['id'],
        'excluded_ranges':project.get('excluded_ranges',[]),'base_start_frame':base_start_frame,'chunked':chunked,'verified_scan':chunked}
    if project.get('people_path'):
        job.update(people_path=project['people_path'],assignments=project['assignments'],
                   target_ids=target_ids,base_target=base_target,person_groups=project.get('person_groups',[]),track_cache=str(Path(project['folder'])/'track-plan'))
    if project.get('people_path'):
        from frame_selection import build_plan,pass_through
        selected_path,selected_plan=build_plan(project,start,end,target_ids,progress)
        if not selected_plan['processed_frames']:
            if progress:progress(None,desc='Участок без подходящих лиц: сохраняем без нейросетей…')
            return pass_through(project,folder,start,selected_plan['end'],selected_plan,base_target,base_start_frame)
        job['processing_plan']=str(selected_path)
    manifest=folder/'job.json'
    manifest.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'engine.log').open('wb') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'engine_worker.py'),str(manifest)],stdout=log,stderr=log,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while proc.poll() is None:
            if progress:
                text=(folder/'engine.log').read_text(encoding='utf-8',errors='replace')
                lines=[s.strip() for s in text.replace('\r','\n').splitlines() if s.strip()]
                progress(None,desc=lines[-1][-160:] if lines else 'Обработка видео…')
            time.sleep(1)
    if proc.returncode or not output.is_file():
        tail=(folder/'engine.log').read_text(encoding='utf-8',errors='replace')[-2000:]
        raise RuntimeError('Обработка не завершилась. Лог: '+str(folder/'engine.log')+'\n'+tail)
    verify(output)
    geometry=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
    geometry['profile_alignment_version']=3
    geometry['appearance_version']=1
    appearance_report=folder/'appearance.json'
    if appearance_report.exists():geometry['appearance']=json.loads(appearance_report.read_text(encoding='utf-8'))
    profile_report=folder/'profile-compositing.json'
    if profile_report.exists():geometry['profile_compositing']=json.loads(profile_report.read_text(encoding='utf-8'))
    geometry['processing_profile']=settings['id']
    geometry['render_settings']=settings
    (folder/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding='utf-8')
    return output,geometry


def save_project(project):
    from minute_queue import atomic_json
    atomic_json(Path(project['folder'])/'project.json',project)


def new_project(video,main,aux):
    folder=DATA/'projects'/time.strftime('%Y%m%d-%H%M%S')
    folder=folder.with_name(folder.name+'-'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True)
    def copy(path,label):
        source=Path(path);dest=folder/(label+source.suffix.lower());shutil.copy2(source,dest);return str(dest)
    project={'folder':str(folder),'target':copy(video,'original'),
             'sources':[copy(main,'primary')]+[copy(path,f'aux-{index}') for index,path in enumerate((aux or [])[:2])],
             'result':None,'geometry':None,'history':[]}
    save_project(project)
    return project


def new_multi_project(video,analysis,assignments,person_groups=None):
    from people import load_scan,fingerprint
    scan=load_scan(analysis,video)
    known={person['id'] for person in scan['characters']}
    if not assignments: raise RuntimeError('Назначьте хотя бы одну фотографию.')
    seen=set()
    for item in assignments:
        identity=int(item['person_id']);paths=item.get('sources',[])
        if identity not in known or identity in seen or not 1<=len(paths)<=3:
            raise RuntimeError('Неверное назначение фотографий людям.')
        if not all(Path(path).is_file() for path in paths): raise RuntimeError('Фотография не найдена.')
        seen.add(identity)
    from person_groups import normalize_groups
    groups=normalize_groups(person_groups,known,assignments)
    from scan_compat import validate_processing
    compatible=validate_processing(scan,video)
    if compatible is not scan:
        from minute_queue import atomic_json
        ready=Path(analysis);backup=ready.with_suffix('.before-rebase.json')
        if not backup.exists():shutil.copy2(ready,backup)
        atomic_json(ready,compatible)
        scan=compatible
    processing_video=scan.get('processing_video',video)
    if fingerprint(processing_video)!=scan['fingerprint']:
        raise RuntimeError('Подготовленное видео изменилось после поиска. Выполните поиск заново.')
    project=new_project(processing_video,assignments[0]['sources'][0],[])
    project['input_video']=str(video)
    folder=Path(project['folder']);copied=[]
    for item in assignments:
        paths=[]
        for index,path in enumerate(item['sources']):
            source=Path(path);dest=folder/f"person-{item['person_id']}-source-{index}{source.suffix.lower()}"
            shutil.copy2(source,dest);paths.append(str(dest))
        copied.append({'person_id':int(item['person_id']),'sources':paths})
    discovery=folder/'people';discovery.mkdir()
    for person in scan['characters']:
        source=Path(person['thumbnail']);dest=discovery/source.name
        shutil.copy2(source,dest);person['thumbnail']=str(dest)
        if person.get('identity_examples'):
            person['identity_examples']=[str(discovery/Path(path).name) for path in person['identity_examples']]
    for person in scan.get('persons',[]):
        person['examples']=[str(discovery/Path(path).name) for path in person.get('examples',[])]
    from person_groups import profiles_for_scan
    profiles=profiles_for_scan(scan,groups)
    for chunk in scan.get('frame_chunks',[]):
        shutil.copy2(Path(analysis).parent/chunk['file'],discovery/chunk['file'])
    people_path=discovery/'people.json'
    people_path.write_text(json.dumps(scan,ensure_ascii=False),encoding='utf-8')
    project.update(people_path=str(people_path),characters=scan['characters'],assignments=copied,
                   sources=[path for item in copied for path in item['sources']],person_groups=groups,
                   identity_schema=scan.get('identity_schema',0),persons=profiles,profile_numbering=1)
    save_project(project)
    return project


def report_text(meta,geometry,vision,notes):
    text=f"Видео проверено: {meta['width']}×{meta['height']}, {meta['duration']:.2f} с. Звук: {'есть' if meta['audio'] else 'нет в исходнике'}.\n\n"
    text+='Декодирование и длительность проверены. '
    if geometry:
        if geometry.get('selection_version'):
            text+=f'По сохранённому плану: обработано {geometry.get("processed_frames",0)} кадров, пропущено без замены {geometry.get("skipped_frames",0)}. Пустые и неясные участки сохранены вместе со звуком. '
            if geometry.get('retained_measurements') or geometry.get('bridged_faces'):
                text+=f'Удержано наблюдений при колебании резкости: {geometry.get("retained_measurements",0)}. Восстановлено коротких пропусков по движению с обеих сторон: {geometry.get("bridged_faces",0)}. '
            if geometry.get('occlusion_context_faces'):
                text+=f'Видимая часть лица восстановлена при перекрытии на {geometry["occlusion_context_faces"]} наблюдениях; закрывающие предметы сохранены. '
        appearance=geometry.get('appearance',{})
        if appearance.get('mustache_faces'):text+='Усы сохранены из исходных кадров. '
        if appearance.get('beard_faces'):text+='Борода сохранена из исходных кадров. '
        if geometry.get('processing_profile')=='quality-v1':
            text+='Режим качества: 512×512, новая маска перекрытий для каждого лица на каждом кадре. '
        text+=f"Разметка проверялась на участке {geometry['start_frame']/meta['fps']:.1f}–{geometry['end_frame']/meta['fps']:.1f} с. "
        a,b=geometry.get('raw_median_px'),geometry.get('stable_median_px')
        if a is not None and b is not None:
            label='Медианная нестабильность точек лица с учётом движения' + (' (выборка)' if geometry.get('median_is_sampled') else '')
            text+=f'{label}: {a:.1f} → {b:.1f} px. '
        spans=geometry.get('suspicious_segments',[])
        if spans: text+='Остались подозрительные участки: '+', '.join(f'{a:.1f}–{b:.1f} с' for a,b in spans[:12])+'. '
        else: text+='Порог нестабильности разметки не превышен. '
        text+=f"Кадры без разметки выбранных лиц: {geometry['face_free_frames']}."
        if geometry.get('continued_detections'):
            text+=f" Удержано лиц при слабой разметке: {geometry['continued_detections']}."
        if geometry.get('unswapped_segments'):
            text+='\nБез замены: '+', '.join(f'{a:.2f}–{b:.2f} с' for a,b in geometry['unswapped_segments'][:12])+'. Проверьте эти участки в плеере.'
        if geometry.get('excluded_ranges'):
            text+='\nБез замены лиц: '+', '.join(f'{a:.3f}–{b:.3f} с' for a,b in geometry['excluded_ranges'])+'.'
            text+=f" Сохранены исходные кадры: {geometry.get('excluded_frames',0)}."
        replacement=geometry.get('replacement_check')
        if replacement:
            text+=f"\nИзменение лица подтверждено на {replacement['changed_frames']} из {replacement['sampled_frames']} кадров выборки. Это не проверка каждого кадра."
            if replacement.get('median_identity_gain') is not None and replacement['median_identity_gain'] < .05:
                text+=' Сходство с новым лицом увеличилось слабо; результат требует визуальной проверки.'
            for identity,check in replacement.get('people',{}).items():
                text+=f"\nЧеловек {identity}: замена подтверждена на {check['changed_frames']} из {check['sampled_frames']} кадров."
                if check.get('median_identity_gain') is not None and check['median_identity_gain']<.05:
                    text+=' Сходство с фото увеличилось слабо — проверьте визуально.'
        if geometry.get('uncertain_detections'):
            text+=f"\nНеуверенные распознавания пропущены: {geometry['uncertain_detections']}."
    if vision:
        text+='\n\nОценка кадров локальной моделью:\n'+vision
    if notes: text+='\n\n'+ '\n'.join(notes)
    text+='\n\nАвтоматическая проверка может пропустить дефекты. Предпросмотр остаётся частью проверки.'
    return text


def execute(video,main,aux,command,project,progress=gr.Progress(),automatic=False,analysis=None,assignments=None,technical_check=False,excluded_ranges=None,resume=False,checkpoint=None,person_groups=None):
    if not automatic and not technical_check and not resume and (not command or not command.strip()): raise gr.Error('Напишите команду.')
    if not LOCK.acquire(blocking=False): raise gr.Error('Уже идёт обработка. Дождитесь результата.')
    try:
        progress(.01,desc='Техническая проверка видео…' if technical_check else 'Проверка Ollama…' if automatic else 'Проверка Ollama и разбор команды…')
        project=dict(project) if project else None
        if resume:
            if not project or not project.get('pending'): raise RuntimeError('Нет незавершённой обработки.')
            release_model(DATA)
            plan=project['pending']['plan'];command=project['pending']['command']
            model='сохранённый проект'
        elif automatic:
            release_model(DATA)
            project=None
            command='Автоматическая замена лица'
            plan={'action':'swap','start_seconds':0,'end_seconds':None,
                  'explanation':'Замена лица, проверка и автоматическое исправление нестабильных участков.'}
        elif technical_check:
            if not project or not project.get('result'): raise RuntimeError('Сначала выполните замену лица.')
            command='Техническая проверка результата'
            plan={'action':'check','start_seconds':0,'end_seconds':None,'explanation':command}
        else:
            from person_groups import command_people
            plan,model=interpret(command,bool(project and project.get('result')),DATA,
                known_ids=command_people(project) if project else None)
        target_ids=None
        if not automatic and not resume and project and project.get('assignments'):
            requested=re.findall(r'(?:человек\w*|персонаж\w*)\s*(?:№\s*)?(\d+)',command,re.IGNORECASE)
            if requested or plan.get('person_ids'):
                numbers=sorted(set(map(int,requested))) if requested else sorted(set(plan['person_ids']))
                from person_groups import resolve_people
                target_ids=resolve_people(project,numbers)
            # A text correction reuses mappings and the completed result.
            if plan['action']=='swap': plan['action']='repair'
        if not resume and (plan['action']=='swap' or project is None):
            if analysis and assignments:
                project=new_multi_project(video,analysis,assignments,person_groups)
            else:
                if not video or not main: raise RuntimeError('Выберите видео и основное фото.')
                project=new_project(video,main,aux)
            project['excluded_ranges']=normalize_ranges(excluded_ranges,metadata(project['target'])['duration'])
            save_project(project)
            plan['action']='swap'
        meta=metadata(project['target'])
        from minute_queue import render_minutes
        notes=['Замена запущена кнопкой. Проверка и автоматическое исправление нестабильных участков включены.'] if automatic else ['Выполнена техническая проверка без изменения видео.'] if technical_check else [f"Команда: {plan.get('explanation','')} Модель: {model} (локально)."]
        if resume:
            revision=Path(project['pending']['revision']);target_ids=project['pending'].get('target_ids')
        else:
            revision=Path(project['folder'])/('revision-'+uuid.uuid4().hex[:8]);revision.mkdir()
            if plan['action']!='check':
                project['pending']={'revision':str(revision),'plan':plan,'command':command,'target_ids':target_ids}
                save_project(project)
        if checkpoint and plan['action']!='check': checkpoint(project,0,0)
        if plan['action']=='swap':
            result,geometry=render_minutes(project,revision,render_once,progress=progress,checkpoint=checkpoint)
            notes.append(f"Видео обработано последовательно: частей {geometry['chunk_count']}.")
        elif plan['action']=='repair':
            if not project.get('result'): raise RuntimeError('Сначала выполните замену лица.')
            start=int(plan['start_seconds']*meta['fps'])
            end=math.ceil((plan['end_seconds'] or meta['duration'])*meta['fps'])
            end=min(meta['frames'] or end,end)
            if start>=end: raise RuntimeError('Указанный участок находится за пределами видео.')
            patch,geometry=render_minutes(project,revision,render_once,start,end,True,progress,target_ids=target_ids,checkpoint=checkpoint,
                base_target=project['result'] if project.get('people_path') else None)
            result=revision/'result.mp4';join_patch(project['result'],patch,result,start,end)
            notes.append(f'Повторно обработан участок {start/meta["fps"]:.1f}–{end/meta["fps"]:.1f} с. Остальные кадры взяты из предыдущего результата перед кодированием.')
        else:
            if not project.get('result'): raise RuntimeError('Сначала выполните замену лица.')
            result=Path(project['result']);geometry=project.get('geometry')
        progress(.9,desc='Техническая проверка видео…')
        checked=verify(result,meta) if plan['action']=='check' else metadata(result)
        source_notes=revision/'source-notes.json'
        if source_notes.is_file(): notes.extend(json.loads(source_notes.read_text(encoding='utf-8')))
        vision=None
        report=report_text(checked,geometry,vision,notes)
        (revision/'report.txt').write_text(report,encoding='utf-8')
        project['result']=str(result);project['geometry']=geometry
        project['history'].append({'command':command,'plan':plan,'result':str(result),'report':str(revision/'report.txt')})
        project.pop('pending',None)
        save_project(project)
        progress(1,desc='Готово')
        return str(result),report,project,str(result)
    except Exception as error:
        raise gr.Error(str(error)) from error
    finally: LOCK.release()


def load_last():
    projects=sorted((DATA/'projects').glob('*/project.json'),key=lambda path:path.stat().st_mtime,reverse=True) if (DATA/'projects').exists() else []
    for path in projects:
        project=json.loads(path.read_text(encoding='utf-8'))
        old_folder=project['folder']
        new_folder=str(path.parent)
        if old_folder!=new_folder:
            def relocate(value):
                if isinstance(value,dict): return {key:relocate(item) for key,item in value.items()}
                if isinstance(value,list): return [relocate(item) for item in value]
                if isinstance(value,str) and value.startswith(old_folder): return new_folder+value[len(old_folder):]
                return value
            project=relocate(project)
        if project.get('result') and Path(project['result']).is_file():
            last=project['history'][-1]
            report=Path(last['report']).read_text(encoding='utf-8')
            return project['target'],project['sources'][0],project['sources'][1:],project['result'],report,project,project['result']
    raise gr.Error('Сохранённых результатов пока нет.')


with gr.Blocks(title='FaceStudio — локально',theme=gr.themes.Soft(),css='.gradio-container {max-width: 1160px !important;}') as app:
    gr.Markdown('# FaceStudio\nЗамена лица и автоматическая проверка видео. Всё работает на этом компьютере.')
    status=gr.Markdown('Проверка Ollama…')
    state=gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            video=gr.File(label='Исходное видео',file_types=['.mp4','.mov','.mkv','.webm'],type='filepath')
            main=gr.File(label='Основное фото — лицо крупно',file_types=['image'],type='filepath')
            aux=gr.File(label='До двух дополнительных фото для глаз и ракурса',file_types=['image'],file_count='multiple',type='filepath')
            command=gr.Textbox(label='Что сделать',value='Сделай замену лица, проверь результат и исправь нестабильные участки.',lines=3)
            run=gr.Button('Выполнить команду',variant='primary')
            last=gr.Button('Открыть последний результат')
        with gr.Column(scale=2):
            result=gr.Video(label='Результат',interactive=False)
            download=gr.File(label='Сохранить видео',interactive=False)
            report=gr.Textbox(label='Проверка качества',lines=12,interactive=False)
    run.click(execute,[video,main,aux,command,state],[result,report,state,download],concurrency_limit=1)
    command.submit(execute,[video,main,aux,command,state],[result,report,state,download],concurrency_limit=1)
    last.click(load_last,outputs=[video,main,aux,result,report,state,download])
    app.load(lambda:STATUS,outputs=status)

if __name__=='__main__':
    startup_check()
    app.queue(max_size=4).launch(server_name='127.0.0.1',server_port=7865,share=False,inbrowser=False,
                                allowed_paths=[str(DATA)],show_error=True)
