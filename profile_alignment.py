"""Keep profile crops upright instead of fitting a hidden eye as a frontal eye."""
import threading
import cv2
import numpy as np
from pose_stability import pose

VERSION=1
_LOCAL=threading.local()


def profile_matrix(five,landmarks,matrix):
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
    t=float(np.clip((abs(yaw)-.30)/.22,0,1));strength=t*t*(3-2*t)
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
    if abs(delta)>np.deg2rad(60):return matrix,0.
    angle=current+strength*delta
    scale=float(np.linalg.norm(original[0,:2]))
    if scale<=0 or np.linalg.det(original[:,:2])<=0:return matrix,0.
    result=cv2.getRotationMatrix2D((0,0),float(np.degrees(angle)),scale)
    anchor=five[2].astype(np.float64)
    result[:,2]=original[:,:2]@anchor+original[:,2]-result[:,:2]@anchor
    return result,strength


def protect_features(mask,expected,predicted,confidence):
    """Keep original expression where a generated profile feature drifted.

    Reducing alpha is deliberate: a correctly positioned original lip is less
    distracting than a second mouth. The rest of the face can still be swapped.
    """
    expected=np.asarray(expected,np.float32);predicted=np.asarray(predicted,np.float32)
    if (expected.shape!=(68,2) or predicted.shape!=(68,2) or confidence<.60
            or not np.isfinite(expected).all() or not np.isfinite(predicted).all()):
        return mask,0
    from facefusion.face_helper import convert_to_face_landmark_5
    a=convert_to_face_landmark_5(expected);b=convert_to_face_landmark_5(predicted)
    height=pose(a)[2]
    result=mask.copy();protected=0
    for ids,error in [(list(range(48,68)),np.linalg.norm(b[3:].mean(0)-a[3:].mean(0))),
                      (list(range(27,36)),np.linalg.norm(b[2]-a[2]))]:
        strength=float(np.clip((error/height-.08)/.10,0,1))
        if not strength:continue
        holes=np.zeros(mask.shape,np.uint8)
        points=np.vstack([expected[ids],predicted[ids]])
        cv2.fillConvexPoly(holes,cv2.convexHull(np.rint(points).astype(np.int32)),1)
        radius=max(2,round(height*.08))
        holes=cv2.dilate(holes,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
        outside=cv2.distanceTransform(1-holes,cv2.DIST_L2,5)
        guard=np.clip(outside/max(3,height*.08),0,1)
        result*=1-strength*(1-guard);protected+=1
    return result,protected


def install(folder=None):
    import atexit,json
    from facefusion.processors.modules.face_swapper import core as swapper
    old_swap=swapper.swap_face;old_warp=swapper.warp_face_by_face_landmark_5;old_paste=swapper.paste_back
    stats={'version':VERSION,'profile_faces':0,'protected_features':0,'uncertain_geometry':0}
    def aligned(frame,five,template,size):
        target=getattr(_LOCAL,'target',None)
        if target is None or not np.array_equal(five,target.landmark_set.get('5/68')):
            return old_warp(frame,five,template,size)
        from facefusion.face_helper import estimate_matrix_by_face_landmark_5
        matrix=estimate_matrix_by_face_landmark_5(five,template,size)
        landmarks=target.landmark_set.get('68')
        if landmarks is not None:matrix,strength=profile_matrix(five,landmarks,matrix)
        else:strength=0.
        _LOCAL.strength=strength
        return cv2.warpAffine(frame,matrix,size,borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_AREA),matrix
    def paste(frame,crop,mask,matrix):
        target=getattr(_LOCAL,'target',None)
        if target is not None and getattr(_LOCAL,'strength',0)>.05:
            from facefusion.face_landmarker import detect_with_2dfan4
            stats['profile_faces']+=1
            x1,y1,x2,y2=target.bounding_box
            corners=cv2.transform(np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]],np.float32),matrix)[0]
            box=np.r_[corners.min(0),corners.max(0)].astype(np.float32)
            try:
                predicted,confidence=detect_with_2dfan4(crop.clip(0,255).astype(np.uint8),box,0)
                expected=cv2.transform(target.landmark_set['68'][None].astype(np.float32),matrix)[0]
                mask,count=protect_features(mask,expected,predicted,confidence)
                stats['protected_features']+=count
                if confidence<.60:stats['uncertain_geometry']+=1
            except (cv2.error,ValueError,RuntimeError):
                stats['uncertain_geometry']+=1
        return old_paste(frame,crop,mask,matrix)
    def swap(source,target,image,frame):
        previous=getattr(_LOCAL,'target',None);_LOCAL.target=target;_LOCAL.strength=0.
        try:return old_swap(source,target,image,frame)
        finally:_LOCAL.target=previous;_LOCAL.strength=0.
    if folder is not None:
        def save():
            from pathlib import Path
            (Path(folder)/'profile-compositing.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
        atexit.register(save)
    swapper.warp_face_by_face_landmark_5=aligned
    swapper.paste_back=paste
    swapper.swap_face=swap
