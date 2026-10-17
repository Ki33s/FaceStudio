"""Preserve explicit photo choices; only group cards with compatible assignments."""
import json
from pathlib import Path
from people import fingerprint


def restore_legacy_draft(analysis,scan,job_folder):
    """Recover pre-autosave choices only for this exact scan and source file."""
    analysis=Path(analysis)
    if (analysis.parent/'assignments.json').exists():return
    for path in sorted(Path(job_folder).glob('*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        try:
            job=json.loads(path.read_text(encoding='utf-8-sig'))
            if (job.get('action')!='swap' or not job.get('analysis') or
                Path(job['analysis']).resolve()!=analysis.resolve() or not job.get('assignments')):continue
            if not all(Path(source).is_file() for a in job['assignments'] for source in a.get('sources',[])):continue
            capture_assignments(job)
            if (analysis.parent/'assignments.json').exists():return
        except (OSError,ValueError,KeyError):continue


def capture_assignments(job):
    if not job.get('analysis') or not job.get('assignments'):return
    analysis=Path(job['analysis']);scan=json.loads(analysis.read_text(encoding='utf-8'))
    if fingerprint(job['video']) not in {scan['fingerprint'],scan.get('input_fingerprint')}:return
    ids={p['id'] for p in scan['characters']}
    if not {a['person_id'] for a in job['assignments']}<=ids:return
    from person_groups import normalize_groups
    groups=normalize_groups(job.get('person_groups'),ids,job['assignments'])
    path=analysis.parent/'assignments.json';previous=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    from minute_queue import atomic_json
    atomic_json(path,{'assignments':job['assignments'],'person_groups':groups,'person_ids':sorted(ids),
        'input_fingerprint':scan.get('input_fingerprint',scan['fingerprint']),
        'grouping_reviewed':previous.get('grouping_reviewed',False)})


def choices_for_scan(analysis,scan,automatic_groups):
    path=Path(analysis).parent/'assignments.json'
    ids={p['id'] for p in scan['characters']};draft={}
    if path.exists():
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        if (set(value.get('person_ids',[]))==ids and
            value.get('input_fingerprint')==scan.get('input_fingerprint',scan['fingerprint'])):draft=value
    assignments=draft.get('assignments',[])
    if draft.get('grouping_reviewed'):return assignments,draft.get('person_groups',[])
    by_id={int(a['person_id']):a.get('sources',[]) for a in assignments}
    hashes={}
    def key(identity):
        result=[]
        for source in by_id.get(identity,[]):
            if source not in hashes:hashes[source]=fingerprint(source) if Path(source).is_file() else str(source)
            result.append(hashes[source])
        return tuple(result)
    groups=[];used=set()
    # Repeated explicit photo choices are authoritative about the requested
    # output, even when recognition cannot match a dark profile reliably.
    if assignments:
        photo_groups={}
        for identity in ids:
            signature=key(identity)
            if signature:photo_groups.setdefault(signature,[]).append(identity)
        for members in photo_groups.values():
            if len(members)>1:groups.append(sorted(members));used.update(members)
    for group in draft.get('person_groups',[]) or automatic_groups:
        partitions={}
        for identity in group:
            if identity in ids and identity not in used:partitions.setdefault(key(identity),[]).append(identity)
        for members in partitions.values():
            if len(members)>1:groups.append(sorted(members));used.update(members)
    return assignments,sorted(groups,key=lambda g:g[0])
