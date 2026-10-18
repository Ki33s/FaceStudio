"""Whole-video people profiles built from reciprocal tracks of real detections.

Face geometry stays in chunk files. A disk index and bounded decode blocks keep
both passes independent of video length. No guessed/interpolated faces are added.
"""
import hashlib
import json
import sqlite3
import subprocess
import time
from bisect import bisect_right
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from face_motion import FaceMotion
from joint_matching import assign
from people import fingerprint,box_overlap
from minute_queue import atomic_json
from video_tools import FFMPEG,FLAGS
from continuity import plausible_five

VERSION=1
BLOCK=96


def iter_chunks(path,scan):
    for part in scan.get('frame_chunks') or [None]:
        yield part,(json.loads((Path(path).parent/part['file']).read_text(encoding='utf-8')) if part else scan)


def observations(data,index):
    values=data.get('observations',data.get('frames',{})).get(str(index),[])
    result=[]
    for record in values:
        face=record['face'];v=np.asarray(face['embedding_norm'],np.float32)
        box=np.asarray(face['bounding_box'],np.float32);scores=face['score_set']
        if (not np.isfinite(v).all() or not np.isfinite(box).all() or np.linalg.norm(v)<.9 or
            not np.isfinite([scores['detector'],scores['landmarker']]).all() or
            np.any(box[2:]<=box[:2]) or min(scores['detector'],scores['landmarker'])<.35):continue
        # Multiple detector models can report the same physical face.
        duplicate=next((n for n,old in enumerate(result) if box_overlap(box,np.asarray(old['face']['bounding_box']))>.65),None)
        if duplicate is None:result.append(record)
        elif sum(scores.values())>sum(result[duplicate]['face']['score_set'].values()):result[duplicate]=record
    return result


def good_anchor(face):
    box=np.asarray(face['bounding_box']);scores=face['score_set']
    measured=np.asarray(face['landmark_set']['5'],np.float32)
    return bool(scores['detector']>=.45 and scores['landmarker']>=.55 and
                measured.shape==(5,2) and np.isfinite(measured).all() and plausible_five(measured) and
                np.linalg.norm(measured[1]-measured[0])>=max(12.,min(box[2:]-box[:2])*.1))


def decode_blocks(path,meta,reverse=False,block=BLOCK):
    """Exact CFR seeking with FFmpeg; reverse holds at most `block` small frames."""
    scale=min(1.,640/max(meta['width'],meta['height']))
    width=max(2,int(meta['width']*scale)//2*2);height=max(2,int(meta['height']*scale)//2*2)
    size=width*height*3;total=meta['frames']
    starts=list(range(0,total,block))
    ranges=[(a,min(a+block,total)) for a in starts]
    if reverse:ranges.reverse()
    process=None
    try:
        if not reverse:
            process=subprocess.Popen([FFMPEG,'-v','error','-i',str(path),'-map','0:v:0',
                '-vf',f'scale={width}:{height}','-pix_fmt','bgr24','-f','rawvideo','pipe:1'],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=FLAGS)
        for start,end in ranges:
            if reverse:
                process=subprocess.Popen([FFMPEG,'-v','error','-ss',format(start/meta['fps'],'.12f'),
                    '-i',str(path),'-map','0:v:0','-frames:v',str(end-start),'-vf',f'scale={width}:{height}',
                    '-pix_fmt','bgr24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=FLAGS)
            content=process.stdout.read((end-start)*size)
            if len(content)!=(end-start)*size:raise RuntimeError('Не удалось прочитать кадры анализа людей. Сохранённые назначения не изменены.')
            frames=np.frombuffer(content,np.uint8).reshape(end-start,height,width,3)
            yield start,end,frames,np.array([width/meta['width'],height/meta['height']]*2,np.float32)
            if reverse:
                error=process.stderr.read();code=process.wait();process.stdout.close();process.stderr.close();process=None
                if code:raise RuntimeError('Ошибка обратного чтения видео: '+error.decode(errors='replace')[-500:])
        if process:
            process.stdout.close();error=process.stderr.read();code=process.wait();process.stderr.close();process=None
            if code:raise RuntimeError('Ошибка чтения видео: '+error.decode(errors='replace')[-500:])
    finally:
        if process:
            if process.poll() is None:process.kill()
            process.wait()
            if process.stdout:process.stdout.close()
            if process.stderr:process.stderr.close()


def _index(path,scan,db,progress,legacy_path=None):
    db.execute('CREATE TABLE nodes(n INTEGER PRIMARY KEY,frame INTEGER,ordinal INTEGER,old_id INTEGER,box TEXT,v BLOB,quality REAL,good INTEGER)')
    n=0;legacy_parts=[];legacy_data=None;legacy_part=None
    if legacy_path:
        legacy_scan=json.loads(Path(legacy_path).read_text(encoding='utf-8'))
        legacy_parts=iter(iter_chunks(legacy_path,legacy_scan))
        legacy_part,legacy_data=next(legacy_parts)
    for part,data in iter_chunks(path,scan):
        begin=part['start'] if part else 0;end=part['end'] if part else scan['meta']['frames']
        values=[]
        for index in range(begin,end):
            if legacy_part:
                while index>=legacy_part['end']:
                    legacy_part,legacy_data=next(legacy_parts,(None,None))
                    if legacy_part is None:break
            previous=legacy_data.get('frames',{}).get(str(index),[]) if legacy_data else []
            used=set()
            for ordinal,record in enumerate(observations(data,index)):
                face=record['face'];box=np.asarray(face['bounding_box']);scores=face['score_set']
                quality=float(scores['landmarker']*scores['detector']*np.sqrt(np.prod(box[2:]-box[:2])))
                good=good_anchor(face)
                old_id=int(record.get('person_id') or 0) if not legacy_path else 0
                if legacy_path:
                    vector=np.asarray(face['embedding_norm'],np.float32);possible=[]
                    for j,old in enumerate(previous):
                        overlap=box_overlap(box,np.asarray(old['face']['bounding_box']))
                        similarity=float(vector@np.asarray(old['face']['embedding_norm'],np.float32))
                        if j not in used and overlap>=.5 and similarity>=.4:possible.append((.5*overlap+.5*similarity,j,old['person_id']))
                    possible.sort(reverse=True)
                    if possible and (len(possible)==1 or possible[0][0]-possible[1][0]>=.08):
                        _,j,old_id=possible[0];used.add(j)
                values.append((n,index,ordinal,old_id,json.dumps(box.tolist()),
                    np.asarray(face['embedding_norm'],np.float32).tobytes(),quality,int(good)));n+=1
            if len(values)>=2048:
                db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)',values);values=[]
        db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)',values);db.commit()
        if progress:progress(None,desc=f'Собираем появления людей: кадр {end}/{scan["meta"]["frames"]}…')
    db.execute('CREATE INDEX node_frame ON nodes(frame)');db.commit()
    if not n:raise RuntimeError('Для анализа людей нет сохранённых обнаружений лиц.')
    return n


def _pass(db,scan,folder,signature,reverse,progress):
    path=folder/('backward.json' if reverse else 'forward.json')
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved.get('signature')==signature:return saved
    tracker=FaceMotion(scan['meta']['fps']);last={};edges=[];cuts=[];step=0;previous=[]
    for start,end,images,scale in decode_blocks(scan['processing_video'],scan['meta'],reverse):
        rows=defaultdict(list)
        for n,frame,box,v in db.execute('SELECT n,frame,box,v FROM nodes WHERE frame>=? AND frame<? ORDER BY frame,n',(start,end)):
            rows[frame].append((n,SimpleNamespace(bounding_box=np.asarray(json.loads(box),np.float32)*scale,
                                embedding_norm=np.frombuffer(v,np.float32))))
        indices=range(end-1,start-1,-1) if reverse else range(start,end)
        for index in indices:
            detections=rows[index];faces=[f for _,f in detections]
            links=tracker.predict(images[index-start],faces,step)
            if tracker.cut and step:cuts.append(index+1 if reverse else index)
            if tracker.cut:last.clear();previous=[]
            # Low-texture profiles can lose optical features while the detector
            # still measures a stable face. Use its actual adjacent box and
            # descriptor as separate evidence; reverse confirmation is required.
            proposals=[]
            for ordinal,(node,face) in enumerate(detections):
                for old_node,old_face,track in previous:
                    overlap=box_overlap(face.bounding_box,old_face.bounding_box)
                    resemblance=float(face.embedding_norm@old_face.embedding_norm)
                    size=face.bounding_box[2:]-face.bounding_box[:2];oldsize=old_face.bounding_box[2:]-old_face.bounding_box[:2]
                    shift=np.linalg.norm((face.bounding_box[:2]+face.bounding_box[2:]-old_face.bounding_box[:2]-old_face.bounding_box[2:])/2)
                    if overlap<.55 or resemblance<.3 or shift>max(size)*.25 or np.any(size/oldsize<.7) or np.any(size/oldsize>1.4):continue
                    competitor=max([float(face.embedding_norm@other.embedding_norm) for _,other,label in previous if label!=track]+[-1])
                    if competitor>resemblance-.08:continue
                    proposals.append((.6*overlap+.4*resemblance,ordinal,track))
            occupied={link['person_id'] for link in links.values()}
            for score,ordinal,track in assign(proposals,margin=.1,blocked_rows=links,blocked_columns=occupied):
                links[ordinal]={'person_id':track,'score':score,'gap':1}
            used=set();current=[]
            for ordinal,(n,face) in enumerate(detections):
                link=links.get(ordinal);track=link['person_id'] if link else n
                if track in used:track=n;link=None
                if link and track in last:
                    a,b=sorted((last[track],n));edges.append([a,b,round(link['score'],6),link['gap']])
                used.add(track);last[track]=n;tracker.remember(track,face,step)
                current.append((n,face,track))
            previous=current
            last={key:value for key,value in last.items() if key in tracker.tracks}
            step+=1
        if progress:progress(None,desc=('Проверяем дорожки назад' if reverse else 'Строим дорожки вперёд')+f': {step}/{scan["meta"]["frames"]} кадров…')
    result={'signature':signature,'edges':edges,'cuts':sorted(cuts)};atomic_json(path,result);return result


def _photo_labels(draft):
    photos={};labels={};hashes={}
    if not draft:return photos,labels
    for item in draft.get('assignments',[]):
        identity=int(item['person_id']);photos[identity]=item['sources'];tokens=[]
        for path in item['sources']:
            if path not in hashes:hashes[path]=fingerprint(path) if Path(path).is_file() else path
            tokens.append(hashes[path])
        labels[identity]=tuple(tokens)
    if draft.get('grouping_reviewed'):
        for identity in draft.get('person_ids',[]):labels.setdefault(identity,())
    return photos,labels


def _gallery_add(gallery,node,vector,quality,limit=16):
    if not gallery:gallery.append((node,vector,quality));return
    nearest=max(float(v@vector) for _,v,_ in gallery)
    if nearest>.97:
        closest=max(range(len(gallery)),key=lambda i:float(gallery[i][1]@vector))
        if quality>gallery[closest][2]:gallery[closest]=(node,vector,quality)
        return
    if len(gallery)<limit:gallery.append((node,vector,quality));return
    # Keep a high quality anchor and a varied collection of genuine poses.
    anchor=max(range(len(gallery)),key=lambda i:gallery[i][2]);matrix=np.asarray([v for _,v,_ in gallery])
    scores=matrix@matrix.T;np.fill_diagonal(scores,-1)
    replace=max((i for i in range(len(gallery)) if i!=anchor),key=lambda i:float(scores[i].max()))
    if nearest<float(scores[replace].max()) or quality>gallery[replace][2]*1.3:gallery[replace]=(node,vector,quality)


def affinity(x,y):
    if not x or not y:return -1.
    matrix=np.asarray([v for _,v,_ in x])@np.asarray([v for _,v,_ in y]).T
    left=np.sort(matrix.max(axis=1))[-min(3,len(x)):];right=np.sort(matrix.max(axis=0))[-min(3,len(y)):]
    return min(float(np.median(left)),float(np.median(right)))


def _components(db,count,forward,backward,labels):
    parent=list(range(count));incoming={};outgoing={};accepted=0
    frames=np.fromiter((r[0] for r in db.execute('SELECT frame FROM nodes ORDER BY n')),np.int32,count)
    old=np.fromiter((r[0] for r in db.execute('SELECT old_id FROM nodes ORDER BY n')),np.int32,count)
    pinned=[labels.get(int(identity)) for identity in old]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    opposite={(a,b) for a,b,*_ in backward['edges']};cuts=sorted(set(forward['cuts']+backward['cuts']))
    for a,b,score,gap in sorted(forward['edges']):
        if (a,b) not in opposite or a in outgoing or b in incoming:continue
        if bisect_right(cuts,int(frames[a]))!=bisect_right(cuts,int(frames[b])):continue
        left,right=root(a),root(b)
        if pinned[left] is not None and pinned[right] is not None and pinned[left]!=pinned[right]:continue
        parent[right]=left
        if pinned[left] is None:pinned[left]=pinned[right]
        outgoing[a]=b;incoming[b]=a;accepted+=1
    roots={};mapping=np.empty(count,np.int32)
    for n in range(count):
        key=root(n)
        if key not in roots:roots[key]=len(roots)+1
        mapping[n]=roots[key]
    tracks={}
    cooccur=set();previous=None;visible=set()
    for n,index,old_id,v,quality,good in db.execute('SELECT n,frame,old_id,v,quality,good FROM nodes ORDER BY n'):
        identity=int(mapping[n])
        item=tracks.setdefault(identity,{'id':identity,'first':index,'last':index,'count':0,'good':0,
            'gallery':[],'best':(-1,None),'old':Counter(),'labels':set()})
        item['last']=index;item['count']+=1;item['good']+=good
        if old_id:item['old'][old_id]+=1
        if old_id in labels:item['labels'].add(labels[old_id])
        if quality>item['best'][0]:item['best']=(quality,n)
        if good:_gallery_add(item['gallery'],n,np.frombuffer(v,np.float32).copy(),quality)
        if previous!=index:visible=set();previous=index
        cooccur.update(tuple(sorted((identity,other))) for other in visible if other!=identity);visible.add(identity)
    return mapping,tracks,cooccur,accepted,cuts


def _candidate_pairs(tracks):
    prototypes=[];owners=[]
    for identity,track in tracks.items():
        available=track['gallery'];best=[]
        if available:
            best.append(max(available,key=lambda x:x[2]))
            for _ in range(min(2,len(available)-1)):
                remaining=[p for p in available if p[0] not in {s[0] for s in best}]
                best.append(min(remaining,key=lambda p:max(float(p[1]@s[1]) for s in best)))
        for _,v,_ in best:owners.append(identity);prototypes.append(v)
    if not prototypes:return set()
    matrix=np.asarray(prototypes,np.float32);owners=np.asarray(owners);pairs=set()
    # Bounded matrix blocks; compare every scene, rather than only adjacent IDs.
    for start in range(0,len(matrix),128):
        similarity=matrix[start:start+128]@matrix.T
        for offset,row in enumerate(similarity):
            identity=int(owners[start+offset]);row[owners==identity]=-2
            for index in np.argpartition(row,-min(24,len(row)))[-min(24,len(row)):]:
                if row[index]>=.42:pairs.add(tuple(sorted((identity,int(owners[index])))))
    return pairs


def temporal_evidence(mapping,db,forward,backward,cuts):
    evidence={}
    for direction,data in enumerate((forward,backward)):
        for a,b,score,gap in data['edges']:
            left,right=int(mapping[a]),int(mapping[b])
            if left==right or gap>3 or score<.8:continue
            rows=db.execute('SELECT frame,v FROM nodes WHERE n IN (?,?) ORDER BY n',(a,b)).fetchall()
            if bisect_right(cuts,rows[0][0])!=bisect_right(cuts,rows[1][0]):continue
            resemblance=float(np.frombuffer(rows[0][1],np.float32)@np.frombuffer(rows[1][1],np.float32))
            if resemblance<.25:continue
            key=tuple(sorted((left,right)));item=evidence.setdefault(key,{'directions':set(),'count':0})
            item['directions'].add(direction);item['count']+=1
    return {pair for pair,value in evidence.items() if len(value['directions'])==2 and value['count']>=2}


def reconcile(tracks,cooccur,progress=None,temporal=None):
    parent={i:i for i in tracks};members={i:{i} for i in tracks};galleries={i:list(t['gallery']) for i,t in tracks.items()}
    labels={i:set(t['labels']) for i,t in tracks.items()};neighbors=defaultdict(set)
    for a,b in cooccur:neighbors[a].add(b);neighbors[b].add(a)
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def forbidden(a,b):
        return (any(neighbors[i]&members[b] for i in members[a]) or len(labels[a]|labels[b])>1)
    temporal=temporal or set()
    # A previously selected appearance is a useful human-supplied anchor, but
    # never a reason to merge simultaneous faces or contradictory photographs.
    prior_ids={i:set(t.get('old',{})) for i,t in tracks.items()}
    by_prior=defaultdict(list)
    for i,ids in prior_ids.items():
        for old_id in ids:by_prior[old_id].append(i)
    candidates=_candidate_pairs(tracks)|temporal
    for ids in by_prior.values():
        anchor=max(ids,key=lambda i:tracks[i]['good']);ordered_ids=sorted(ids,key=lambda i:tracks[i]['first'])
        candidates.update(tuple(sorted((anchor,i))) for i in ids if i!=anchor)
        candidates.update(tuple(sorted(pair)) for pair in zip(ordered_ids,ordered_ids[1:]))
    scores={(a,b):affinity(galleries[a],galleries[b]) for a,b in candidates}
    adjacent=defaultdict(list)
    for (a,b),score in scores.items():adjacent[a].append((b,score));adjacent[b].append((a,score))
    accepted=[];pending=[];ordered=sorted(scores.items(),key=lambda x:x[1],reverse=True)
    while True:
        changed=False;pending=[]
        for number,((left,right),initial) in enumerate(ordered):
            a,b=root(left),root(right)
            if a==b or forbidden(a,b):continue
            score=affinity(galleries[a],galleries[b]);motion=(left,right) in temporal
            prior=bool(prior_ids[left]&prior_ids[right])
            reviewed_prior=prior and bool(any(labels[a])) and labels[a]==labels[b]
            if not reviewed_prior and score<(.32 if motion else .36 if prior else .44):pending.append((left,right,score));continue
            rivals=[]
            for origin in members[a]|members[b]:
                for other_id,value in adjacent[origin]:
                    if value<score-.08:continue
                    other=root(other_id)
                    if other in (a,b):continue
                    if labels[a] and labels[b] and labels[other] and len(labels[a]|labels[b]|labels[other])>1:continue
                    # Competing appearances that can all agree as one family
                    # are not rival people. Co-occurrence always remains a veto.
                    if (forbidden(a,other) or forbidden(b,other) or
                        min(affinity(galleries[a],galleries[other]),affinity(galleries[b],galleries[other]))<.42):
                        rivals.append(value)
            contradiction=rivals and max(rivals)>=.65 and max(rivals)-score>=.18
            if not motion and rivals and ((prior and contradiction) or (not prior and score-max(rivals)<.08)):
                pending.append((left,right,score));continue
            canonical,alias=min(a,b),max(a,b);parent[alias]=canonical
            members[canonical]|=members.pop(alias);labels[canonical]|=labels.pop(alias)
            for n,v,q in galleries.pop(alias):_gallery_add(galleries[canonical],n,v,q,24)
            accepted.append({'a':left,'b':right,'similarity':round(score,5),
                'method':'preserved-user-profile' if reviewed_prior and score<.36 else 'bidirectional-path' if motion else 'whole-video-gallery'})
            changed=True
            if progress and number%100==0:progress(None,desc=f'Связываем людей между сценами: {number+1}/{len(scores)}…')
        if not changed:break
    groups=[sorted(value) for value in members.values()];groups.sort(key=lambda ids:min(tracks[i]['first'] for i in ids))
    return groups,accepted,pending


def build_identity_scan(path,folder,progress=None,draft=None,legacy_path=None):
    started=time.monotonic();path=Path(path);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    scan=json.loads(path.read_text(encoding='utf-8'));meta=scan['meta'];work=folder/'identity-work';work.mkdir(exist_ok=True)
    signature={'version':VERSION,'motion_policy':3,'geometry_policy':1,'fingerprint':scan['fingerprint'],'input':scan.get('input_fingerprint'),
        'chunks':[[p['file'],(path.parent/p['file']).stat().st_size,(path.parent/p['file']).stat().st_mtime_ns] for p in scan.get('frame_chunks',[])],
        'inline_hash':hashlib.sha256(path.read_bytes()).hexdigest() if not scan.get('frame_chunks') else None,
        'draft':hashlib.sha256(json.dumps(draft or {},sort_keys=True).encode()).hexdigest(),
        'legacy':hashlib.sha256(Path(legacy_path).read_bytes()).hexdigest() if legacy_path else None}
    status=work/'index.json';database=work/'observations.sqlite'
    valid=status.exists() and database.exists() and json.loads(status.read_text(encoding='utf-8')).get('signature')==signature
    if not valid and database.exists():database.unlink()
    photos,labels=_photo_labels(draft);db=sqlite3.connect(database)
    try:
        if valid:count=db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
        else:
            count=_index(path,scan,db,progress,legacy_path);atomic_json(status,{'signature':signature})
        forward=_pass(db,scan,work,signature,False,progress)
        backward=_pass(db,scan,work,signature,True,progress)
        mapping,tracks,cooccur,reciprocal,cuts=_components(db,count,forward,backward,labels)
        temporal=temporal_evidence(mapping,db,forward,backward,cuts)
        groups,merges,pending=reconcile(tracks,cooccur,progress,temporal)
        # A weak isolated detection remains pending, rather than inventing a new
        # person. It can participate if a reliable appearance anchors its group.
        groups=[g for g in groups if any(tracks[i]['good'] or any(tracks[i]['labels']) for i in g)]
        visible={i for g in groups for i in g};unresolved=sum(t['count'] for i,t in tracks.items() if i not in visible)
        groups_multi=[g for g in groups if len(g)>1];profiles=[];characters=[];assignments=[];photo_conflicts=[]
        best_nodes={t['best'][1] for i,t in tracks.items() if i in visible}
        requests=defaultdict(list);node_frames={n:frame for n,frame in db.execute('SELECT n,frame FROM nodes') if n in best_nodes}
        for identity in visible:
            node=tracks[identity]['best'][1];requests[node_frames[node]].append((identity,node))
        # Thumbnails are extracted in one sequential pass, not hundreds of seeks.
        for start,end,images,scale in decode_blocks(scan['processing_video'],meta):
            for frame in range(start,end):
                for identity,node in requests.get(frame,[]):
                    box=np.asarray(json.loads(db.execute('SELECT box FROM nodes WHERE n=?',(node,)).fetchone()[0]))*scale
                    x1,y1,x2,y2=box;margin=max(x2-x1,y2-y1)*.20;h,w=images[frame-start].shape[:2]
                    crop=images[frame-start][max(0,int(y1-margin)):min(h,int(y2+margin)),max(0,int(x1-margin)):min(w,int(x2+margin))]
                    if crop.size:
                        ok,encoded=cv2.imencode('.jpg',crop)
                        if ok:(folder/f'person-{identity}.jpg').write_bytes(encoded.tobytes())
            if progress:progress(None,desc=f'Готовим карточки людей: {end}/{meta["frames"]} кадров…')
        for number,group in enumerate(groups,1):
            token=scan.get('input_fingerprint',scan['fingerprint'])+':'+','.join(str(tracks[i]['best'][1]) for i in group)
            uid=hashlib.sha256(token.encode()).hexdigest()[:24]
            examples=[str(folder/f'person-{i}.jpg') for i in sorted(group,key=lambda i:tracks[i]['best'][0],reverse=True)[:4]]
            ranges=[[tracks[i]['first']/meta['fps'],tracks[i]['last']/meta['fps']] for i in group]
            profiles.append({'uid':uid,'label':f'Человек {number}','members':group,'examples':examples,'ranges':ranges})
            # Portable projects copy the same photo under different filenames.
            # Compare content signatures, not paths, when migrating a person.
            candidates={labels[old]:photos[old] for i in group for old in tracks[i]['old'] if old in photos}
            if len(candidates)>1:photo_conflicts.append({'members':group,'choices':list(candidates.values())})
            choice=list(next(iter(candidates.values()))) if len(candidates)==1 else None
            for identity in group:
                t=tracks[identity]
                characters.append({'id':identity,'label':f'Человек {number}','person_uid':uid,'identity_examples':examples,
                    'thumbnail':str(folder/f'person-{identity}.jpg'),'appearances':t['count'],
                    'first_seconds':t['first']/meta['fps'],'last_seconds':t['last']/meta['fps'],
                    'original_ids':dict(t['old']),'confidence':'confirmed' if t['good']>=2 else 'review'})
                if choice:assignments.append({'person_id':identity,'sources':choice})
        if not characters:raise RuntimeError('Уверенные появления людей не найдены. Прежний поиск и фотографии сохранены.')
        # Preserve all raw detections for future analysis; rendering sees only
        # confirmed tracks. ID 0 protects unresolved real faces from replacement.
        output_chunks=[];next_node=0
        for part,data in iter_chunks(path,scan):
            begin=part['start'] if part else 0;end=part['end'] if part else meta['frames'];frames={}
            for index in range(begin,end):
                records=[]
                for ordinal,record in enumerate(observations(data,index)):
                    identity=int(mapping[next_node]);next_node+=1
                    records.append({**record,'person_id':identity if identity in visible else 0})
                frames[str(index)]=records
            name=f'frames-{begin:09d}-{end:09d}.json'
            atomic_json(folder/name,{'start':begin,'end':end,'frames':frames,'uncertain_detections':unresolved})
            output_chunks.append({'start':begin,'end':end,'file':name,'uncertain_detections':0})
        result={k:v for k,v in scan.items() if k not in ('frames','observations','characters','frame_chunks')}
        result.update(tracking_version=5,identity_schema=VERSION,characters=sorted(characters,key=lambda p:p['first_seconds']),
            persons=profiles,person_groups=groups_multi,frame_chunks=output_chunks,uncertain_detections=unresolved,
            identity_analysis={'version':VERSION,'detections':count,'tracklets':len(tracks),'people':len(groups),
                'forward_links':len(forward['edges']),'backward_links':len(backward['edges']),'reciprocal_links':reciprocal,
                'cuts':cuts,'global_merges':merges,'pending_pairs':len(pending),'unresolved_detections':unresolved,
                'elapsed_seconds':round(time.monotonic()-started,2),'photo_conflicts':photo_conflicts})
        if draft:atomic_json(folder/'previous-assignments.json',draft)
        if assignments or draft:
            atomic_json(folder/'assignments.json',{'input_fingerprint':result.get('input_fingerprint',result['fingerprint']),
                'person_ids':sorted(p['id'] for p in characters),'assignments':assignments,'person_groups':groups_multi,
                'grouping_reviewed':bool(draft and draft.get('grouping_reviewed')),'person_profiles':profiles})
        atomic_json(folder/'people.json',result)
        atomic_json(work/'summary.json',result['identity_analysis'])
        return result
    finally:db.close()
