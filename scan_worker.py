"""GPU discovery in an isolated process, released before Ollama or rendering."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
os.environ['PATH']=str(ROOT/'runtime/bin')+os.pathsep+os.environ.get('PATH','')
sys.path[:0]=[str(ROOT/'runtime/directml'),str(ROOT/'runtime/facefusion')]
os.chdir(ROOT/'runtime/facefusion')
sys.argv=['scan','headless-run','--execution-providers',job.get('provider','directml'),
    '--face-detector-model','many','--face-detector-margin','60','60','60','60',
    '--face-detector-score','.35','--execution-thread-count','1']
from facefusion.program import create_program
from facefusion.args import apply_args
from facefusion import state_manager, face_store
from facefusion.face_creator import get_static_faces
from facefusion.vision import read_static_video_frame,write_image
from people import Registry,pack_face,fingerprint,unpack_face
from face_motion import FaceMotion
from minute_queue import atomic_json
import numpy as np
import cv2
if os.name=="nt":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
from video_tools import metadata
apply_args(vars(create_program().parse_args()),state_manager.init_item)
meta=metadata(job['video'])

total=meta['frames'] or round(meta['duration']*meta['fps'])
start=job.get('start_frame',0);end=min(total,job.get('end_frame') or total)
registry=Registry(meta['fps']);motion=FaceMotion(meta['fps']);frames={};skipped=0
previous_gray=None
if start:
    previous=read_static_video_frame(job['video'],start-1)
    if previous is not None: previous_gray=cv2.resize(cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),(32,32))
    read_static_video_frame.cache_clear()
checkpoint=folder/'scan-checkpoint.json';saved={}
if checkpoint.exists():
    saved=json.loads(checkpoint.read_text(encoding='utf-8'))
    if saved['next_frame']!=start: raise RuntimeError('Неверный порядок частей поиска людей.')
    registry.people=saved['people']
    for person in registry.people: person['vector']=np.asarray(person['vector'],np.float32)
    if start and previous is not None:
        from types import SimpleNamespace
        motion.seed(previous,[(i+1,SimpleNamespace(bounding_box=np.asarray(p['track_box']),
                    embedding_norm=np.asarray(p['track_embedding']))) for i,p in enumerate(registry.people)
                    if p.get('last')==start-1 and 'track_box' in p],start-1)
for index in range(start,end):
    frame=read_static_video_frame(job['video'],index)
    if frame is None: raise RuntimeError('Не удалось прочитать кадр '+str(index)+'. Готовые части поиска сохранены.')
    gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,32))
    previous_gray=gray
    detected=[face for face in get_static_faces([frame]) if
        float(face.score_set['detector'])>=.35 and float(face.score_set['landmarker'])>=.35]
    links=motion.predict(frame,detected,index)
    if motion.cut:registry.reset_tracks()
    accepted=[n for n,face in enumerate(detected) if
        (float(face.score_set['detector'])>=.45 and float(face.score_set['landmarker'])>=.55) or n in links]
    candidates=[detected[n] for n in accepted]
    matched=registry.assign(candidates,index,{i:links[n] for i,n in enumerate(accepted) if n in links})
    for identity,face in matched:motion.remember(identity,face,index)
    frames[str(index)]=[{'person_id':identity,'face':pack_face(face)} for identity,face in matched]
    skipped+=len(candidates)-len(matched)
    for identity,face in matched:
        person=registry.people[identity-1]
        size=face.bounding_box[2:]-face.bounding_box[:2]
        quality=float(face.score_set['landmarker'])*float(min(size))
        if quality>person['best']:
            person['best']=quality
            x1,y1,x2,y2=face.bounding_box;margin=max(size)*.25;h,w=frame.shape[:2]
            crop=frame[max(0,int(y1-margin)):min(h,int(y2+margin)),max(0,int(x1-margin)):min(w,int(x2+margin))]
            if crop.size: write_image(str(folder/f'person-{identity}.jpg'),crop)
    face_store.clear_faces();read_static_video_frame.cache_clear()
    if index%10==0 or index==end-1:
        print(json.dumps({'type':'progress','value':(index-start+1)/(end-start),
            'text':f'Поиск людей: кадр {index+1}/{total}, найдено {len(registry.people)}'},ensure_ascii=False),flush=True)
characters=[{'id':i+1,'label':f'Человек {i+1}','thumbnail':str(folder/f'person-{i+1}.jpg'),
    'appearances':person['count'],'first_seconds':person['first']/meta['fps'],
    'last_seconds':person['last']/meta['fps']} for i,person in enumerate(registry.people)]
chunk_file='frames-%09d-%09d.json' % (start,end)
atomic_json(folder/chunk_file,{'start':start,'end':end,'frames':frames,'uncertain_detections':skipped})
people=[{**person,'vector':person['vector'].tolist()} for person in registry.people]
chunks=saved.get('frame_chunks',[])+[{'start':start,'end':end,'file':chunk_file,'uncertain_detections':skipped}]
atomic_json(checkpoint,{'next_frame':end,'people':people,'characters':characters,'frame_chunks':chunks})
print(json.dumps({'type':'scan_part','start':start,'end':end,'file':chunk_file,'characters':characters,'uncertain_detections':skipped},ensure_ascii=False),flush=True)
