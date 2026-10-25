"""Keep profile crops upright instead of fitting a hidden eye as a frontal eye."""
import threading
import cv2
import numpy as np
from pose_stability import pose

VERSION=3
_LOCAL=threading.local()
_CACHE_READY=False

def commit_cache():
    global _CACHE_READY
    _CACHE_READY=True



def profile_matrix(five,landmarks,matrix,forced_strength=None):
    """Return a similarity transform; keep scale and nose position unchanged.

    Five-point alignment is ill-conditioned when the two eyes overlap in a
    profile. Use the wider, visible eye for roll, blending smoothly with the
    normal transform as the head turns. No facial features are deformed.
    """
    five=np.asarray(five,np.float32);points=np.asarray(landmarks,np.float32)
    original=np.asarray(matrix,np.float64)
    if (five.shape!=(5,2) or points.shape!=(68,2) or original.shape!=(2,3)
            or not all(np.isfinite(a).all() for a in (five,points,original))):
        return matrix,0.
    yaw,_,height=pose(five)
    t=float(np.clip((abs(yaw)-.30)/.22,0,1));strength=t*t*(3-2*t) if forced_strength is None else float(np.clip(forced_strength,0,1))
    if not strength:return matrix,0.
    axes=points[[39,45]]-points[[36,42]]
    lengths=np.linalg.norm(axes,axis=1)
    if max(lengths)<max(4.,.06*height):return matrix,0.
    # A fourth-power weight chooses the visible eye continuously, so equal
    # widths cannot switch the rotation abruptly between two different axes.
    units=axes/np.maximum(lengths[:,None],1e-6)
    weights=(lengths/max(lengths))**4
    eye_axis=(units*weights[:,None]).sum(axis=0)
    if np.linalg.norm(eye_axis)<.5 or eye_axis[0]<=0:return matrix,0.
    visible=np.arctan2(eye_axis[1],eye_axis[0])
    current=np.arctan2(original[0,1],original[0,0])
    delta=np.arctan2(np.sin(visible-current),np.cos(visible-current))
    # Reject implausible eye topology; it must not rotate a crop upside down.
    delta=float(np.clip(delta,-np.deg2rad(60),np.deg2rad(60)))
    angle=current+strength*delta
    scale=float(np.linalg.norm(original[0,:2]))
    if scale<=0 or np.linalg.det(original[:,:2])<=0:return matrix,0.
    result=cv2.getRotationMatrix2D((0,0),float(np.degrees(angle)),scale)
    anchor=five[2].astype(np.float64)
    result[:,2]=original[:,:2]@anchor+original[:,2]-result[:,:2]@anchor
    return result,strength


def install(folder=None,persistent=False):
    import atexit,json
    from facefusion.processors.modules.face_swapper import core as swapper
    from profile_detail import ProfileDetails
    from occlusion import LOCAL as CONTEXT
    details=ProfileDetails();roll_states={};plans={}
    cache=None
    if folder is not None and persistent:
        from pathlib import Path
        cache=Path(folder).parent/'profile-state-v3.npz'
        details.load(cache)
        roll_path=cache.with_suffix('.json')
        try:
            roll_states={int(k):{**v,'scene':np.asarray(v['scene'],np.uint8)} for k,v in json.loads(roll_path.read_text(encoding='utf-8')).items()}
        except (OSError,ValueError,KeyError):roll_states={}
    old_swap=swapper.swap_face;old_warp=swapper.warp_face_by_face_landmark_5;old_paste=swapper.paste_back
    def aligned(frame,five,template,size):
        target=getattr(_LOCAL,'target',None)
        if target is None or not np.array_equal(five,target.landmark_set.get('5/68')):
            return old_warp(frame,five,template,size)
        from facefusion.face_helper import estimate_matrix_by_face_landmark_5,WARP_TEMPLATE_SET
        matrix=estimate_matrix_by_face_landmark_5(five,template,size)
        landmarks=target.landmark_set.get('68')
        key,index=_LOCAL.key,_LOCAL.index
        if folder is None:plans[key]={}
        if key not in plans:
            from pathlib import Path
            path=Path(folder)/('person-'+str(key))/'profile-plan.json'
            if not path.exists():path=Path(folder)/'profile-plan.json'
            plans[key]=json.loads(path.read_text(encoding='utf-8'))['frames'] if path.exists() else {}
        if landmarks is not None:matrix,strength=profile_matrix(five,landmarks,matrix,plans[key].get(str(index)))
        else:strength=0.
        previous=roll_states.get(key)
        yaw=pose(five)[0]
        scene=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,32))
        angle=np.arctan2(matrix[0,1],matrix[0,0]);scale=float(np.linalg.norm(matrix[0,:2]))
        if (previous is not None and index==previous['index']+1
                and np.abs(scene.astype(float)-previous['scene']).mean()<35
                and .75<scale/previous['scale']<1.33
                and max(strength,previous['gain'])>.05):
            delta=np.arctan2(np.sin(angle-previous['angle']),np.cos(angle-previous['angle']))
            angle=previous['angle']+np.clip(delta,-np.deg2rad(4),np.deg2rad(4))*.6
            scale=previous['scale']*np.exp(np.clip(np.log(scale/previous['scale']),-.05,.05)*.6)
        anchor=five[2].astype(np.float64)
        mapped=matrix[:,:2]@anchor+matrix[:,2]
        mapped=(1-strength)*mapped+strength*(WARP_TEMPLATE_SET[template][2]*size)
        stable=cv2.getRotationMatrix2D((0,0),float(np.degrees(angle)),scale)
        stable[:,2]=mapped-stable[:,:2]@anchor
        roll_states[key]={'index':index,'angle':angle,'scale':scale,'gain':strength,'yaw':yaw,'scene':scene}
        _LOCAL.strength=strength
        return cv2.warpAffine(frame,stable,size,borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_AREA),stable
    def paste(frame,crop,mask,matrix):
        target=getattr(_LOCAL,'target',None);gain=getattr(_LOCAL,'strength',0)
        key,index=_LOCAL.key,_LOCAL.index
        previous=details.states.get(key)
        if target is not None and (gain>.05 or (previous is not None and previous.get('tail',0)>0)):
            from facefusion.face_landmarker import detect_with_2dfan4
            x1,y1,x2,y2=target.bounding_box
            corners=cv2.transform(np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]],np.float32),matrix)[0]
            box=np.r_[corners.min(0),corners.max(0)].astype(np.float32)
            predicted=None;confidence=0.
            try:predicted,confidence=detect_with_2dfan4(crop.clip(0,255).astype(np.uint8),box,0)
            except (cv2.error,ValueError,RuntimeError):pass
            expected=cv2.transform(target.landmark_set['68'][None].astype(np.float32),matrix)[0]
            original=cv2.warpAffine(frame,matrix,crop.shape[:2][::-1],borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_AREA)
            crop,mask=details.apply(key,index,original,crop,mask,expected,predicted,confidence,gain)
        return old_paste(frame,crop,mask,matrix)
    def swap(source,target,image,frame):
        previous=getattr(_LOCAL,'target',None);_LOCAL.target=target;_LOCAL.strength=0.
        _LOCAL.key=getattr(CONTEXT,'identity',0);_LOCAL.index=getattr(CONTEXT,'index',getattr(_LOCAL,'counter',0))
        _LOCAL.counter=_LOCAL.index+1
        try:return old_swap(source,target,image,frame)
        finally:_LOCAL.target=previous;_LOCAL.strength=0.
    if folder is not None:
        def save():
            from pathlib import Path
            (Path(folder)/'profile-compositing.json').write_text(json.dumps({'version':VERSION,**details.stats},indent=2),encoding='utf-8')
            if cache is not None and _CACHE_READY:
                details.save(cache)
                from minute_queue import atomic_json
                last=max((v['index'] for v in roll_states.values()),default=-1)
                atomic_json(cache.with_suffix('.json'),{str(k):{**v,'scene':v['scene'].tolist()} for k,v in roll_states.items() if v['index']>=last-2})
        atexit.register(save)
    swapper.warp_face_by_face_landmark_5=aligned
    swapper.paste_back=paste
    swapper.swap_face=swap
