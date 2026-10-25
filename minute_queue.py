"""Sequential, durable render checkpoints with frame-aligned minute boundaries."""
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from fractions import Fraction
from people import fingerprint
from video_tools import metadata, verify, join_patch, FFMPEG, FLAGS

CHUNK_SECONDS = 60


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def frame_parts(start, end, fps, seconds=CHUNK_SECONDS):
    step = max(1, int(round(seconds * fps)))
    return [(a, min(end, a + step)) for a in range(start, end, step)]


def combine_parts(paths, ranges, target, output, fps, start, end):
    output = Path(output)
    manifest = output.parent / 'concat.txt'
    def quoted(path):
        return str(Path(path).resolve()).replace('\\', '/').replace("'", "'\\''")
    temporary = output.with_name('.assembling-' + uuid.uuid4().hex + '.mp4')
    normalized = []
    rate = Fraction(str(fps)).limit_denominator(1000000)
    try:
        if not paths or len(paths) != len(ranges):
            raise RuntimeError('Не хватает сохранённых частей для сборки видео.')
        for path, (a, b) in zip(paths, ranges):
            part = metadata(path)
            if part['frames'] != b-a:
                raise RuntimeError('Количество кадров в сохранённой части отличается от ожидаемого.')
            source_rate = Fraction(part['rate'])
            timed = output.with_name('.timed-' + uuid.uuid4().hex + '.mp4')
            normalized.append(timed)
            # FaceFusion may use OpenCV's average FPS instead of the source rate.
            # Retimestamp each encoded frame to the project clock before concat.
            # Scale PTS and DTS together to retain B-frame presentation order.
            scale = source_rate / rate
            timing = (f'setts=pts=(PTS-STARTPTS)*{scale.numerator}/{scale.denominator}'
                      f':dts=(DTS-STARTPTS)*{scale.numerator}/{scale.denominator}'
                      f':duration=DURATION*{scale.numerator}/{scale.denominator}'
                      f':time_base=1/{rate.numerator}')
            subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(path),
                            '-map', '0:v:0', '-an', '-c:v', 'copy', '-bsf:v', timing,
                            '-video_track_timescale', str(rate.numerator), str(timed)],
                           check=True, capture_output=True, creationflags=FLAGS)
        manifest.write_text(''.join("file '" + quoted(path) + "'\nduration " + format((b-a)/fps, '.12f') + '\n'
                                    for path, (a,b) in zip(normalized,ranges)), encoding='utf-8')
        args = [FFMPEG, '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(manifest),
                '-ss', str(start/fps), '-i', str(target), '-map', '0:v:0', '-map', '1:a:0?',
                '-t', str((end-start)/fps), '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-movflags', '+faststart', str(temporary)]
        subprocess.run(args, check=True, capture_output=True, creationflags=FLAGS)
        expected = metadata(target)
        expected['duration'] = (end-start)/fps
        checked = verify(temporary, expected)
        if checked['frames'] and checked['frames'] != end-start:
            raise RuntimeError('Количество кадров при сборке частей изменилось.')
        os.replace(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()
        for path in normalized:
            if path.exists(): path.unlink()
    return output


def summarize(report_paths, start, end):
    import random
    import numpy as np
    sample=[];seen=0;rng=random.Random(0);checks={};spans=[]
    geometry={'start_frame':start,'end_frame':end,'checked_pairs':0,'face_free_frames':0,'excluded_frames':0,
              'excluded_ranges':[],'residuals':[],'chunk_count':len(report_paths),'median_is_sampled':True,'uncertain_detections':0,
              'recovered_detections':0,'continued_detections':0,'unswapped_segments':[],'tracking_version':4,'recovery_version':1,'profile_alignment_version':3,'appearance_version':1,'selection_version':5,'processed_frames':0,'skipped_frames':0,'rejected_faces':0,
              'retained_measurements':0,'linked_identities':0,'bridged_faces':0,'occlusion_context_faces':0}
    replacement={'sampled_frames':0,'changed_frames':0,'people':checks}
    for path in report_paths:
        report=json.loads(Path(path).read_text(encoding='utf-8'))
        profile=report.get('processing_profile')
        if 'processing_profile' not in geometry:
            geometry['processing_profile']=profile
            geometry['render_settings']=report.get('render_settings')
        elif geometry['processing_profile']!=profile:
            raise RuntimeError('Нельзя объединять части, сделанные с разным качеством.')
        for item in report.get('residuals',[]):
            seen+=1;pair=(item['before'],item['after'])
            if len(sample)<20000: sample.append(pair)
            else:
                position=rng.randrange(seen)
                if position<len(sample): sample[position]=pair
        geometry['uncertain_detections']=max(geometry['uncertain_detections'],report.get('uncertain_detections',0))
        geometry['checked_pairs']+=report.get('checked_pairs',0)
        geometry['face_free_frames']+=report.get('face_free_frames',0)
        geometry['excluded_frames']+=report.get('excluded_frames',0)
        geometry['recovered_detections']+=report.get('recovered_detections',0)
        geometry['continued_detections']+=report.get('continued_detections',0)
        geometry['tracking_version']=min(geometry['tracking_version'],report.get('tracking_version',0))
        geometry['recovery_version']=min(geometry['recovery_version'],report.get('recovery_version',0))
        geometry['selection_version']=min(geometry['selection_version'],report.get('selection_version',0))
        for key in ('processed_frames','skipped_frames','rejected_faces'):geometry[key]+=report.get(key,0)
        for key in ('retained_measurements','linked_identities','bridged_faces','occlusion_context_faces'):geometry[key]+=report.get(key,0)
        appearance=report.get('appearance',{})
        if appearance:
            total_appearance=geometry.setdefault('appearance',{'version':1,'mustache_faces':0,'beard_faces':0,'transported_masks':0})
            for key in ('mustache_faces','beard_faces','transported_masks'):total_appearance[key]+=appearance.get(key,0)
        geometry['unswapped_segments'].extend(report.get('unswapped_segments',[])[:max(0,120-len(geometry['unswapped_segments']))])
        geometry['excluded_ranges']=report.get('excluded_ranges',[])
        # The summary remains bounded; full problem intervals stay in part reports.
        spans.extend(report.get('suspicious_segments',[])[:max(0,120-len(spans))])
        check=report.get('replacement_check',{})
        replacement['sampled_frames']+=check.get('sampled_frames',0);replacement['changed_frames']+=check.get('changed_frames',0)
        for identity,values in check.get('people',{}).items():
            person=checks.setdefault(identity,{'sampled_frames':0,'changed_frames':0})
            person['sampled_frames']+=values['sampled_frames'];person['changed_frames']+=values['changed_frames']
    merged=[]
    for a,b in sorted(spans):
        if merged and a<=merged[-1][1]+.2: merged[-1][1]=max(merged[-1][1],b)
        else: merged.append([a,b])
    geometry.update(raw_median_px=float(np.median([a for a,b in sample])) if sample else None,
                    stable_median_px=float(np.median([b for a,b in sample])) if sample else None,
                    suspicious_segments=merged,replacement_check=replacement)
    return geometry


def render_minutes(project, folder, render_one, start=0, end=None, strong=False,
                   progress=None, target_ids=None, base_target=None, checkpoint=None):
    from quality import render_settings
    settings=render_settings(project.get('processing_profile'))
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True);meta = metadata(project['target']);fps = meta['fps']
    total = meta['frames'] or round(meta['duration']*fps)
    end = min(end or total,total)
    if not math.isfinite(fps) or fps <= 0 or start >= end:
        raise RuntimeError('Неверные границы видео.')
    parts = frame_parts(start,end,fps)
    signature_data = {'version':2,'target':fingerprint(project['target']),
                      'sources':[fingerprint(p) for p in project['sources']],
                      'assignments':project.get('assignments'),'excluded_ranges':project.get('excluded_ranges',[]),
                      'start':start,'end':end,'strong':strong,'target_ids':target_ids,
                      'base':fingerprint(base_target) if base_target else None,'parts':parts}
    if project.get('people_path'):
        scan=json.loads(Path(project['people_path']).read_text(encoding='utf-8'))
        if scan['fingerprint']!=signature_data['target']: raise RuntimeError('Видео изменилось после поиска людей. Выполните поиск заново.')
        del scan
    def checksum(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    signature = checksum(signature_data)
    state_path = folder/'chunks.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'signature':signature,'done':{}}
    if state['signature'] != signature:
        # Reuse a migrated checkpoint only when every input still matches.
        recorded=state.get('parts')
        candidate={**signature_data,'parts':recorded} if recorded else None
        if candidate and checksum(candidate)==state['signature']:
            parts=[tuple(pair) for pair in recorded];signature=state['signature']
        else:
            legacy_job=folder/'part-0001/job.json'
            old=json.loads(legacy_job.read_text(encoding='utf-8')) if legacy_job.exists() else None
            legacy_parts=frame_parts(start,end,old['fps']) if old else None
            legacy={**signature_data,'version':1,'parts':legacy_parts}
            if not old or checksum(legacy)!=state['signature']:
                raise RuntimeError('Исходник или назначения изменились. Запустите новую замену лиц.')
            old_output=folder/'part-0001/render.mp4';old_geometry=folder/'part-0001/geometry.json'
            # The Bandicam failure happened after a complete single-part render.
            # Recover it rather than repeat all neural face replacements.
            if len(legacy_parts)==1 and old_output.exists() and old_geometry.exists():
                ready=metadata(old_output)
                if ready['frames']==end-start and abs(ready['duration']-(end-start)/fps)<=max(.1,2/fps):
                    verify(old_output)
                    parts=legacy_parts;signature_data['parts']=parts;signature=checksum(signature_data)
                    state={'signature':signature,'parts':parts,'done':{'0':{'start':start,'end':end,'size':old_output.stat().st_size}}}
                    geometry=json.loads(old_geometry.read_text(encoding='utf-8'))
                    scale=old['fps']/fps
                    geometry['suspicious_segments']=[[a*scale,b*scale] for a,b in geometry.get('suspicious_segments',[])]
                    atomic_json(old_geometry,geometry)
                    if progress: progress(.85,desc='Готовая часть восстановлена после исправления частоты кадров…')
                else: state={'signature':signature,'done':{}}
            else:
                # Same inputs, but old minute boundaries used the wrong clock.
                state={'signature':signature,'done':{}}
    state['parts']=parts
    atomic_json(state_path,state)
    paths = []; reports = []; notes = []
    for index,(a,b) in enumerate(parts):
        part_folder = folder / ('part-%04d' % (index+1));part_folder.mkdir(exist_ok=True)
        saved = state['done'].get(str(index));output = part_folder/'render.mp4';report = None
        if saved and output.is_file() and output.stat().st_size == saved['size']:
            try:
                part_meta = metadata(output)
                if abs(part_meta['duration']-(b-a)/fps) <= max(.1,2/fps):
                    report = json.loads((part_folder/'geometry.json').read_text(encoding='utf-8'))
                    if project.get('people_path') and (report.get('tracking_version',0)<4 or report.get('recovery_version',0)<1 or report.get('selection_version',0)<5):
                        # Old completed parts contain the recognition gaps; a
                        # resume must render them with the corrected tracker.
                        report=None
                    if report is not None and report.get('profile_alignment_version',0)<3 and report.get('processed_frames',1):
                        report=None
                    if report is not None and report.get('appearance_version',0)<1 and report.get('processed_frames',1):report=None
                    if report is not None and report.get('processing_profile')!=settings['id']:
                        # Keep the timeline checkpoint and identity assignments,
                        # but never silently reuse lower-quality encoded parts.
                        report=None
            except (ValueError,OSError,subprocess.SubprocessError): pass
        def part_progress(value=None,desc=''):
            local = value if isinstance(value,(int,float)) else .05
            match = re.search(r'processing:\s*(\d+)%',desc)
            if match: local=int(match[1])/100;desc='Замена лиц: '+match[1]+'%'
            if progress: progress((index+max(0,min(1,local)))/len(parts),desc=f'Часть {index+1}/{len(parts)} · {a/fps:.0f}–{b/fps:.0f} с · {desc}')
        if report is None:
            output,report = render_one(project,part_folder,a,b,strong,part_progress,target_ids,base_target,chunked=True)
            if not project.get('people_path') and not strong and report.get('suspicious_segments'):
                retry_start=max(a,int(min(x for x,y in report['suspicious_segments'])*fps))
                retry_end=min(b,math.ceil(max(y for x,y in report['suspicious_segments'])*fps)+1)
                retry=part_folder/'automatic-repair';retry.mkdir(exist_ok=True)
                patch,rechecked=render_one(project,retry,retry_start,retry_end,True,part_progress,target_ids,
                                          str(output) if project.get('people_path') else None,
                                          base_start_frame=a,chunked=True)
                import numpy as np
                baseline={p['frame']:p['after'] for p in report['residuals']}
                common=[p for p in rechecked['residuals'] if p['frame'] in baseline]
                if common and np.percentile([p['after'] for p in common],95)<=np.percentile([baseline[p['frame']] for p in common],95):
                    joined=part_folder/'repaired.mp4';join_patch(output,patch,joined,retry_start-a,retry_end-a)
                    os.replace(joined,output)
                    updated={p['frame']:p for p in report['residuals']};updated.update({p['frame']:p for p in rechecked['residuals']})
                    report['residuals']=list(updated.values())
                    report['stable_median_px']=float(np.median([p['after'] for p in updated.values()]))
                    report['suspicious_segments']=[s for s in report['suspicious_segments'] if s[1]<retry_start/fps or s[0]>retry_end/fps]+rechecked['suspicious_segments']
                    notes.append(f'Часть {index+1}: нестабильный участок исправлен повторным проходом.')
            atomic_json(part_folder/'geometry.json',report)
            checked=verify(output)
            if checked['frames']!=b-a or abs(checked['duration']-(b-a)/fps)>max(.1,2/fps):
                raise RuntimeError(f"Часть {index+1}: ожидалось {b-a} кадров / {(b-a)/fps:.3f} с, получено {checked['frames']} / {checked['duration']:.3f} с. Частота кадров: {fps:.6f}.")
            state['done'][str(index)]={'start':a,'end':b,'size':output.stat().st_size}
            atomic_json(state_path,state)
            # Each render_one runs in a separate process, releasing model memory.
            time.sleep(1)
        else:
            if progress: progress((index+1)/len(parts),desc=f'Часть {index+1}/{len(parts)} уже сохранена.')
        paths.append(output);reports.append(part_folder/'geometry.json')
        if checkpoint: checkpoint(project,index+1,len(parts))
        source_notes=part_folder/'source-notes.json'
        if source_notes.exists(): notes.extend(json.loads(source_notes.read_text(encoding='utf-8')))
    if progress: progress(.92,desc='Собираем сохранённые части и исходный звук…')
    result=combine_parts(paths,parts,project['target'],folder/'render.mp4',fps,start,end)
    geometry=summarize(reports,start,end)
    atomic_json(folder/'geometry.json',geometry)
    atomic_json(folder/'source-notes.json',list(dict.fromkeys(notes)))
    return result,geometry
