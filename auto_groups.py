"""Suggest aliases using verified visual tracks and conservative recognition galleries."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from face_motion import FaceMotion

VERSION=2


def discover_groups(path,progress=None,cache_folder=None,cached_only=False):
    path=Path(path);scan=json.loads(path.read_text(encoding='utf-8'))
    folder=Path(cache_folder) if cache_folder else path.parent
    folder.mkdir(parents=True,exist_ok=True);cache=folder/'identity-groups.json'
    chunks=scan.get('frame_chunks',[])
    signature={'version':VERSION,'fingerprint':scan['fingerprint'],
               'scan':[path.stat().st_size,path.stat().st_mtime_ns],
               'chunks':[(c['file'],(path.parent/c['file']).stat().st_size,(path.parent/c['file']).stat().st_mtime_ns) for c in chunks]}
    # JSON round trip converts tuples to lists for stable cache comparison.
    signature=json.loads(json.dumps(signature))
    if cache.exists():
        saved=json.loads(cache.read_text(encoding='utf-8'))
        if saved.get('signature')==signature:return saved
    if cached_only:return {'groups':[]}
    cooccur=set();galleries={p['id']:[] for p in scan['characters']}
    for part in chunks or [None]:
        data=json.loads((path.parent/part['file']).read_text(encoding='utf-8')) if part else scan
        for key,records in data['frames'].items():
            ids=sorted({r['person_id'] for r in records})
            cooccur.update((a,b) for i,a in enumerate(ids) for b in ids[i+1:])
            for r in records:
                samples=galleries[r['person_id']];v=np.asarray(r['face']['embedding_norm'],np.float32)
                nearest=max([float(np.dot(s,v)) for s in samples]+[-1])
                if nearest<.97:
                    if len(samples)<24:samples.append(v)
                    else:
                        matrix=np.asarray(samples);redundancy=matrix@matrix.T;np.fill_diagonal(redundancy,-1)
                        best=1+int(redundancy[1:].max(axis=1).argmax())
                        if nearest<float(redundancy[best].max())-.01:samples[best]=v
    parents={p['id']:p['id'] for p in scan['characters']}
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    def members(i):return {n for n in parents if root(n)==root(i)}
    def forbidden(a,b):return any((min(x,y),max(x,y)) in cooccur for x in members(a) for y in members(b))
    def join(a,b):
        a,b=root(a),root(b)
        if a==b or forbidden(a,b):return False
        parents[max(a,b)]=min(a,b);return True
    evidence=Counter();strong=Counter();tracker=FaceMotion(scan['meta']['fps'])
    video=cv2.VideoCapture(str(scan.get('processing_video','')))
    if not video.isOpened():raise RuntimeError('Не удалось открыть видео для связывания повторных карточек.')
    total=scan['meta']['frames']
    current_part=0;frames={}
    try:
        for index in range(total):
            ok,frame=video.read()
            if not ok:raise RuntimeError('Видео не прочитано до конца; группы не сохранены.')
            if chunks:
                if not frames or index>=chunks[current_part]['end']:
                    while current_part+1<len(chunks) and index>=chunks[current_part]['end']:current_part+=1
                    frames=json.loads((path.parent/chunks[current_part]['file']).read_text(encoding='utf-8'))['frames']
                records=frames.get(str(index),[])
            else:records=scan['frames'].get(str(index),[])
            faces=[SimpleNamespace(bounding_box=np.asarray(r['face']['bounding_box']),
                    embedding_norm=np.asarray(r['face']['embedding_norm'])) for r in records]
            links=tracker.predict(frame,faces,index)
            for n,(record,face) in enumerate(zip(records,faces)):
                identity=record['person_id'];anchor=identity;link=links.get(n)
                if link and link['person_id']!=identity and not forbidden(identity,link['person_id']):
                    pair=tuple(sorted((identity,link['person_id'])))
                    evidence[pair]+=1
                    strong[pair]+=int(link['gap']<=2 and link['support']>=.9 and link['overlap']>=.7 and link['similarity']>=.22)
                    anchor=link['person_id']
                tracker.remember(anchor,face,index)
            if progress and (index%150==0 or index==total-1):progress((index+1)/total,desc='Связываем повторные лица по соседним кадрам…')
    finally:video.release()
    accepted=[]
    for (a,b),count in evidence.most_common():
        if (count>=2 or strong[a,b]) and join(a,b):accepted.append({'a':a,'b':b,'method':'motion','support':count})
    # Strong recognition can join reappearances across cuts; spatial continuity
    # never crosses cuts. Evaluate against the next competing cluster as well.
    def affinity(a,b):
        x=[v for i in members(a) for v in galleries[i]];y=[v for i in members(b) for v in galleries[i]]
        if not x or not y:return -1.
        similarity=np.asarray(x)@np.asarray(y).T
        left=np.sort(similarity.max(axis=1))[-min(3,len(x)):]
        right=np.sort(similarity.max(axis=0))[-min(3,len(y)):]
        return min(float(np.median(left)),float(np.median(right)))
    while True:
        ids=sorted({root(i) for i in parents});ranked=[]
        for n,a in enumerate(ids):
            for b in ids[n+1:]:
                if not forbidden(a,b):ranked.append((affinity(a,b),a,b))
        merged=False
        for score,a,b in sorted(ranked,reverse=True):
            if score<.44:break
            rival=max([s for s,x,y in ranked if (a in (x,y) or b in (x,y)) and {x,y}!={a,b}]+[-1])
            if score>=.62 or score-rival>=.10:
                if join(a,b):accepted.append({'a':a,'b':b,'method':'recognition','similarity':score});merged=True;break
        if not merged:break
    groups=sorted([sorted(members(i)) for i in parents if root(i)==i and len(members(i))>1])
    result={'signature':signature,'groups':groups,'original_cards':len(parents),
            'visible_cards':len(parents)-sum(len(g)-1 for g in groups),'evidence':accepted,
            'temporal_pairs':[{ 'a':a,'b':b,'count':count,'strong':strong[a,b]} for (a,b),count in evidence.items()]}
    from minute_queue import atomic_json
    atomic_json(cache,result)
    return result


if __name__=='__main__':
    import sys
    cv2.setNumThreads(2)
    result=discover_groups(sys.argv[1],cache_folder=sys.argv[2] if len(sys.argv)>2 else None)
    print(json.dumps(result,ensure_ascii=False),flush=True)
