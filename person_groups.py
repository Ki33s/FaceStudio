"""User-confirmed groups; rendering retains the original track IDs."""
import hashlib
from people import fingerprint


def profiles_for_scan(scan,groups):
    """Number people in exactly the same order as the assignment window."""
    rows=scan['characters'];order={p['id']:i for i,p in enumerate(rows)}
    groups=[list(g) for g in groups];used={i for g in groups for i in g}
    groups += [[p['id']] for p in rows if p['id'] not in used]
    groups.sort(key=lambda g:min(order[i] for i in g));by_id={p['id']:p for p in rows}
    existing={tuple(sorted(p['members'])):p['uid'] for p in scan.get('persons',[])}
    result=[]
    for number,group in enumerate(groups,1):
        members=sorted(group,key=order.get);key=tuple(sorted(members))
        uid=existing.get(key) or hashlib.sha256((scan.get('input_fingerprint',scan['fingerprint'])+':'+','.join(map(str,key))).encode()).hexdigest()[:24]
        result.append({'uid':uid,'number':number,'label':f'Человек {number}','members':members,
            'examples':[by_id[i]['thumbnail'] for i in members[:4]],
            'ranges':[[by_id[i]['first_seconds'],by_id[i]['last_seconds']] for i in members]})
    return result


def command_people(project):
    selected={a['person_id'] for a in project.get('assignments',[])}
    if project.get('profile_numbering')==1:
        return [p['number'] for p in project['persons'] if selected&set(p['members'])]
    return sorted(selected)


def resolve_people(project,numbers):
    selected={a['person_id'] for a in project.get('assignments',[])}
    if project.get('profile_numbering')==1:
        people={p['number']:set(p['members'])&selected for p in project['persons']};result=set()
        for number in numbers:
            if not people.get(number):raise RuntimeError('Для указанного человека нет назначенного фото. Нажмите «Заменить лица» и назначьте фотографию.')
            result.update(people[number])
        return sorted(result)
    if not set(numbers)<=selected:raise RuntimeError('Для указанного человека нет назначенного фото. Нажмите «Заменить лица» и назначьте фотографию.')
    return sorted(set(numbers))


def normalize_groups(groups,known,assignments):
    if groups is None:return []
    result=[];seen=set();photos={int(a['person_id']):a['sources'] for a in assignments};hashes={}
    def signature(paths):
        if not paths:return ()
        for path in paths:
            if path not in hashes:hashes[path]=fingerprint(path)
        return tuple(hashes[path] for path in paths)
    if not isinstance(groups,(list,tuple)):raise ValueError('Неверный список объединённых карточек.')
    for group in groups:
        if not isinstance(group,(list,tuple)) or any(type(i) is not int for i in group):
            raise ValueError('Неверные номера объединённых карточек.')
        members=set(group)
        if len(members)<2 or len(members)!=len(group) or not members<=known or members & seen:
            raise ValueError('Карточка не может входить в несколько групп.')
        if len({signature(photos.get(i)) for i in members})!=1:
            raise ValueError('Для объединённых карточек выберите одинаковые фотографии.')
        seen.update(members);result.append(sorted(members))
    return sorted(result,key=lambda group:group[0])


def render_groups(scan,selected,assignments,groups):
    """Join confirmed aliases in memory; preserve simultaneous face tracks."""
    if not groups:return set(selected)
    known={person['id'] for person in scan['characters']}
    normalized=normalize_groups(groups,known,[{'person_id':identity,'sources':paths} for identity,paths in assignments.items()])
    selected=set(selected);mapping={};merged=[]
    for group in normalized:
        members=set(group)
        if not members & selected:continue
        selected.update(members)
        if any(sum(record['person_id'] in members for record in records)>1 for records in scan['frames'].values()):continue
        canonical=min(members);mapping.update({identity:canonical for identity in members});merged.append(group)
        selected.difference_update(members);selected.add(canonical)
    if mapping:
        scan['frames']={key:[{**record,'person_id':mapping.get(record['person_id'],record['person_id'])} for record in records] for key,records in scan['frames'].items()}
        scan['identity_groups']=merged
    return selected
