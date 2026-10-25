"""Measured-face hysteresis and bounded, two-anchor optical-flow bridges.

All pixels come from the current source frame. No generated image is carried
forward, and a missing observation is only bridged inside a verified shot.
"""
import copy
import cv2
import numpy as np
from face_motion import scene_changed
from people import box_overlap
from joint_matching import assign


def transform(points, matrix):
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def transformed_box(box, matrix):
    x1,y1,x2,y2=box
    points=transform([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],matrix)
    return np.r_[points.min(0),points.max(0)]


def measurement_patch(gray, box, scale):
    box=np.asarray(box)*np.tile(scale,2)
    h,w=gray.shape;size=box[2:]-box[:2]
    low=np.maximum(0,np.floor(box[:2]-size*.10)).astype(int)
    high=np.minimum([w,h],np.ceil(box[2:]+size*.10)).astype(int)
    if min(high-low)<12:return None
    patch=cv2.resize(gray[low[1]:high[1],low[0]:high[0]],(96,96))
    matrix=np.eye(3);matrix[0,0]=(high[0]-low[0])/96/scale[0];matrix[1,1]=(high[1]-low[1])/96/scale[1]
    matrix[:2,2]=low/scale
    return patch,matrix


def flow_matrix(before, after, box=None,visibility_before=None,visibility_after=None):
    """Forward/backward checked feature motion, excluding background corners."""
    mask=np.zeros(before.shape,np.uint8)
    if box is None:box=np.array([8,8,before.shape[1]-8,before.shape[0]-8])
    x1,y1,x2,y2=box
    cv2.ellipse(mask,(round((x1+x2)/2),round((y1+y2)/2)),
                (max(1,round((x2-x1)*.40)),max(1,round((y2-y1)*.40))),0,0,360,255,-1)
    if visibility_before is not None:mask[visibility_before<.70]=0
    if visibility_after is not None:mask[visibility_after<.70]=0
    points=cv2.goodFeaturesToTrack(before,90,.01,3,mask=mask)
    if points is None or len(points)<8:return None
    window=(15,15) if visibility_after is not None else (21,21)
    levels=2 if visibility_after is not None else 3
    forward,status,_=cv2.calcOpticalFlowPyrLK(before,after,points,None,winSize=window,maxLevel=levels)
    if forward is None:return None
    backward,back_status,_=cv2.calcOpticalFlowPyrLK(after,before,forward,None,winSize=window,maxLevel=levels)
    if backward is None:return None
    good=status.ravel().astype(bool)&back_status.ravel().astype(bool)
    good&=np.linalg.norm(points[:,0]-backward[:,0],axis=1)<1.5
    if visibility_after is not None:
        h,w=after.shape;x,y=np.rint(forward[:,0]).astype(int).T
        inside=(x>=0)&(y>=0)&(x<w)&(y<h)
        visible=np.zeros(len(x),bool);visible[inside]=visibility_after[y[inside],x[inside]]>=.70
        good&=visible
    if good.sum()<8:return None
    matrix,inliers=cv2.estimateAffinePartial2D(points[good],forward[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
    if matrix is None or inliers is None or inliers.sum()<8 or inliers.mean()<.70:return None
    scale=float(np.linalg.norm(matrix[0,:2]));angle=abs(float(np.arctan2(matrix[1,0],matrix[0,0])))
    if not .85<scale<1.18 or angle>np.deg2rad(15):return None
    result=np.eye(3);result[:2]=matrix
    return result


def measured_link(left,right):
    """An adjacent real observation may keep a clear track below its entry threshold."""
    if left['group'] and right['group'] and left['group']!=right['group']:return None
    if scene_changed(left['scene'],right['scene']):return None
    similarity=float(left['vector']@right['vector'])
    if similarity<(.35 if not left['id'] or not right['id'] else .30):return None
    if box_overlap(left['box'],right['box'])<.45:return None
    a,b=left.get('points'),right.get('points')
    if a is None or b is None:return None
    height=max(left['box'][3]-left['box'][1],1.)
    if np.median(np.linalg.norm(a-b,axis=1))>.20*height:return None
    patches=left.get('patch'),right.get('patch')
    if any(p is None for p in patches):return None
    matrix=flow_matrix(patches[0][0],patches[1][0])
    if matrix is None:return None
    world=patches[1][1]@matrix@np.linalg.inv(patches[0][1])
    residual=np.linalg.norm(transform(a,world)-b,axis=1)
    if np.median(residual)>.10*height or max(residual)>.22*height:return None
    return .6*similarity+.4*box_overlap(transformed_box(left['box'],world),right['box'])


def retain_measurements(samples,fps):
    """Join only unambiguous adjacent observations; never merge named people.

    Clear entry requires the original sharpness threshold. Continuing a track
    uses the lower floor already applied by the selector, with checked optical
    flow. Unknown IDs need named anchors on both sides and a short bounded run.
    """
    nodes={(i,n):item for i,items in samples.items() for n,item in enumerate(items)}
    previous={};following={}
    for index,items in sorted(samples.items()):
        old=samples.get(index-1,[]);proposals=[]
        for n,right in enumerate(items):
            for m,left in enumerate(old):
                score=measured_link(left,right)
                if score is not None:proposals.append((score,n,m))
        for _,n,m in assign(proposals,margin=.08):
            previous[index,n]=(index-1,m);following[index-1,m]=(index,n)
    accepted={};promotions={};held=0;promoted=0
    for key,item in nodes.items():
        if item['strong'] and item['id']>0:accepted.setdefault(str(key[0]),set()).add(item['id'])
    for root in nodes.keys()-previous.keys():
        chain=[];key=root
        while key is not None:chain.append(key);key=following.get(key)
        groups={nodes[k]['group'] for k in chain if nodes[k]['id']>0}
        if len(groups)!=1 or not any(nodes[k]['strong'] and nodes[k]['id']>0 for k in chain):continue
        limit=max(2,round(fps*.20))
        for position,key in enumerate(chain):
            item=nodes[key]
            if item['id']>0:
                identity=item['id']
            else:
                left=next((nodes[k] for k in reversed(chain[:position]) if nodes[k]['id']>0),None)
                right=next((nodes[k] for k in chain[position+1:] if nodes[k]['id']>0),None)
                a=next((k[0] for k in reversed(chain[:position]) if nodes[k]['id']>0),-1000000)
                b=next((k[0] for k in chain[position+1:] if nodes[k]['id']>0),1000000)
                if left is None or right is None or b-a-1>limit:continue
                # Unknown observations cannot drift along a chain of lookalikes.
                if min(float(item['vector']@v['vector']) for v in (left,right))<.35:continue
                identity=left['id']
                promotions.setdefault(str(key[0]),[]).append({'ordinal':item['ordinal'],'person_id':identity});promoted+=1
            values=accepted.setdefault(str(key[0]),set())
            if identity not in values:held+=1;values.add(identity)
    return {k:sorted(v) for k,v in accepted.items()},promotions,held,promoted


def bridge_faces(grays,left,right,scale):
    """Return provisional geometry only if two independent motion paths agree."""
    if not 3<=len(grays)<=8:return None
    if float(np.asarray(left['embedding_norm'])@np.asarray(right['embedding_norm']))<.40:return None
    def path(face,images):
        box=np.asarray(face['bounding_box'])*np.tile(scale,2)
        matrix=np.eye(3);result=[matrix]
        for before,after in zip(images,images[1:]):
            if scene_changed(before,after):return None
            step=flow_matrix(before,after,transformed_box(box,matrix))
            if step is None:return None
            matrix=step@matrix;result.append(matrix)
        return result
    forward=path(left,grays);backward=path(right,list(reversed(grays)))
    if forward is None or backward is None:return None
    backward=backward[::-1]
    a=np.asarray(left['landmark_set']['5/68'])*scale;b=np.asarray(right['landmark_set']['5/68'])*scale
    height=max((left['bounding_box'][3]-left['bounding_box'][1])*scale[1],1.)
    if max(np.linalg.norm(transform(a,forward[-1])-b,axis=1))>.12*height:return None
    if max(np.linalg.norm(transform(b,backward[0])-a,axis=1))>.12*height:return None
    results=[]
    for i in range(1,len(grays)-1):
        pa,pb=transform(a,forward[i]),transform(b,backward[i])
        if max(np.linalg.norm(pa-pb,axis=1))>.10*height:return None
        t=i/(len(grays)-1);record=copy.deepcopy(left)
        inverse=np.diag([1/scale[0],1/scale[1],1.]);direct=np.diag([scale[0],scale[1],1.])
        ma=inverse@forward[i]@direct;mb=inverse@backward[i]@direct
        record['bounding_box']=((1-t)*transformed_box(left['bounding_box'],ma)+t*transformed_box(right['bounding_box'],mb)).tolist()
        record['landmark_set']={k:((1-t)*transform(v,ma)+t*transform(right['landmark_set'][k],mb)).tolist()
                                for k,v in left['landmark_set'].items() if k in right['landmark_set']}
        # A real occluder must not become a guessed face. Compare source texture
        # under the independently tracked eye/cheek region, in both directions.
        box=np.asarray(record['bounding_box'])*np.tile(scale,2)
        mask=np.zeros(grays[i].shape,np.uint8);x1,y1,x2,y2=box
        cv2.ellipse(mask,(round((x1+x2)/2),round((y1+y2)/2)),
                    (max(1,round((x2-x1)*.32)),max(1,round((y2-y1)*.32))),0,0,360,1,-1)
        visible=mask.astype(bool)
        if visible.sum()<80:return None
        for anchor,matrix in ((grays[0],forward[i]),(grays[-1],backward[i])):
            warped=cv2.warpAffine(anchor,matrix[:2],grays[i].shape[::-1])
            delta=np.abs(warped[visible].astype(float)-grays[i][visible])
            if np.median(delta)>12 or np.mean(delta>30)>.20:return None
        record['origin']='flow-bridge'
        results.append(record)
    return results
