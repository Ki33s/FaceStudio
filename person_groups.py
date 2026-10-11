"""User-confirmed groups; rendering retains the original track IDs."""
from people import fingerprint


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
