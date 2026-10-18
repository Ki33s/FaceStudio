"""Retry a lost face in its own region, retaining measured frame geometry.

Only detector/landmarker measurements are returned. Expected boxes guide the
search but are never painted onto a frame without a measured face.
"""
from contextlib import contextmanager
import cv2
import numpy as np
from people import box_overlap


def quality(face):
    return (float(face.score_set['landmarker'])>=.55,
            float(face.score_set['landmarker'])+float(face.score_set['detector']))


def unique_faces(faces):
    result=[]
    for face in sorted(faces,key=quality,reverse=True):
        if not any(box_overlap(face.bounding_box,old.bounding_box)>.45 for old in result):result.append(face)
    return result


@contextmanager
def detector_options(**options):
    from facefusion import state_manager
    saved={key:state_manager.get_item(key) for key in options}
    try:
        for key,value in options.items():state_manager.set_item(key,value)
        yield
    finally:
        for key,value in saved.items():state_manager.set_item(key,value)


def map_face(face,inverse):
    """Map every landmark set and the four box corners back to source pixels."""
    box=np.asarray(face.bounding_box,np.float32)
    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]],np.float32)
    points=cv2.transform(corners[None],inverse)[0]
    landmarks={key:cv2.transform(np.asarray(value,np.float32)[None],inverse)[0] for key,value in face.landmark_set.items()}
    return face._replace(origin='retry-detect',bounding_box=np.r_[points.min(axis=0),points.max(axis=0)].astype(np.float32),landmark_set=landmarks)


def retry_faces(frame, expected, faces=()):
    """Search uncovered boxes with a padded crop and, if needed, +/-20 degrees."""
    from facefusion.face_creator import get_many_faces
    result=list(faces);h,w=frame.shape[:2];regions=[]
    for value in expected:
        box=np.asarray(value,np.float32)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):continue
        if any(box_overlap(box,old)>.65 for old in regions):continue
        regions.append(box)
    for box in regions:
        # An uncertain face in the right area still needs a better measurement.
        if any(box_overlap(box,f.bounding_box)>.4 and min(f.score_set.values())>=.55 for f in result):continue
        size=box[2:]-box[:2];center=(box[:2]+box[2:])/2
        extent=np.maximum(size*1.9,96)
        left=np.floor(center-extent/2).astype(int);right=np.ceil(center+extent/2).astype(int)
        # Clamp oversized predictions to a finite region around the picture.
        left=np.maximum(left,[-w//2,-h//2]);right=np.minimum(right,[w+w//2,h+h//2])
        if np.any(right-left<32) or np.any(right<=0) or left[0]>=w or left[1]>=h:continue
        crop=frame[max(0,left[1]):min(h,right[1]),max(0,left[0]):min(w,right[0])]
        if crop.size==0:continue
        crop=cv2.copyMakeBorder(crop,max(0,-left[1]),max(0,right[1]-h),max(0,-left[0]),max(0,right[0]-w),cv2.BORDER_CONSTANT)
        scale=min(2.,640/max(crop.shape[:2]))
        cw=max(2,round(crop.shape[1]*scale));ch=max(2,round(crop.shape[0]*scale))
        crop=cv2.resize(crop,(cw,ch),interpolation=cv2.INTER_CUBIC)
        # Use actual rounded scale factors, not the nominal resize factor.
        to_source=np.array([[(right[0]-left[0])/cw,0,left[0]],[0,(right[1]-left[1])/ch,left[1]]],np.float32)
        found=[]
        with detector_options(face_detector_margin=[0]*4,face_detector_score=.20):
            for angle in (0,-20,20):
                rotation=cv2.getRotationMatrix2D((cw/2,ch/2),angle,1.)
                image=crop if angle==0 else cv2.warpAffine(crop,rotation,(cw,ch))
                inverse=cv2.invertAffineTransform(rotation)
                mapping=(np.vstack([to_source,[0,0,1]])@np.vstack([inverse,[0,0,1]]))[:2].astype(np.float32)
                for face in get_many_faces([image]):
                    measured=map_face(face,mapping)
                    if box_overlap(box,measured.bounding_box)>=.25 and float(face.score_set['detector'])>=.35:found.append(measured)
                if any(min(f.score_set.values())>=.55 for f in found):break
        # Preserve the ordinary measurement when an alternate is less reliable.
        result=unique_faces(result+found)
    return result
