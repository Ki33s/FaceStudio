"""Visible profile details, anchored to the source clock; never average video pixels."""
import cv2
import numpy as np
from pose_stability import pose


def five(points):
    return np.asarray([points[36:42].mean(0),points[42:48].mean(0),
                       points[30],points[48],points[54]],np.float32)


def profile_strength(points):
    yaw,spread,_=pose(points)
    t=float(np.clip((abs(yaw)-.20)/.25,0,1))
    compressed=float(np.clip((.65-spread)/.30,0,1))
    # Looking down compresses eye-to-mouth height and exaggerates the 2-D
    # nose offset. Two widely separated eyes are evidence against a profile;
    # an isolated nose estimate must not switch a frontal face to partial skin.
    broad=float(np.clip((spread-.95)/.35,0,1))
    return max(t*t*(3-2*t)*(1-broad*broad*(3-2*broad)),compressed*compressed*(3-2*compressed))


def visible_side(points):
    a=five(points);yaw,_,height=pose(a)
    axes=points[[39,45]]-points[[36,42]];lengths=np.linalg.norm(axes,axis=1)
    weights=(lengths/max(float(max(lengths)),1.))**4
    axis=(axes/np.maximum(lengths[:,None],1e-6)*weights[:,None]).sum(0)
    length=float(np.linalg.norm(axis))
    if length>.5:
        offset=float((a[2]-(a[0]+a[1])*.5)@(axis/length))
        if abs(offset)>.05*height:return 0 if offset>=0 else 1
    return 0 if yaw>=0 else 1


def smooth_dense(anchor,anchor_five,samples,fives,weights):
    """Bidirectional, motion-compensated smoothing of visible eye/nose geometry."""
    anchor=np.asarray(anchor,np.float32);yaw,spread,height=pose(anchor_five)
    numerator=anchor.astype(float).copy();denominator=np.ones((68,1))
    for sample,points,weight in zip(samples,fives,weights):
        other,other_spread,_=pose(points)
        if abs(other-yaw)>.20 or abs(other_spread-spread)>.35:continue
        delta=np.asarray(sample)-anchor
        distances=np.linalg.norm(delta,axis=1)
        trust=float(weight)*np.exp(-.5*(distances/max(1,height*.04))**2)
        numerator+=np.asarray(sample)*trust[:,None];denominator+=trust[:,None]
    result=numerator/denominator;delta=result-anchor
    lengths=np.linalg.norm(delta,axis=1)
    delta*=np.minimum(1,height*.03/np.maximum(lengths,1e-6))[:,None]
    return (anchor+delta).astype(np.float32)


def pose_weights(fives,links):
    """Look ahead within a shot so partial compositing is ready before the turn."""
    raw=[];yaws=[]
    for points in fives:
        if points is None:raw.append(0.);yaws.append(None);continue
        yaw=pose(points)[0]
        raw.append(profile_strength(points));yaws.append(yaw)
    maximum=[]
    def connected(i,j):
        return yaws[j] is not None and (i==j or all(links[min(i,j):max(i,j)]))
    for i,value in enumerate(raw):
        if yaws[i] is None:maximum.append(0.);continue
        maximum.append(max(raw[j] for j in range(max(0,i-4),min(len(raw),i+5))
                           if connected(i,j)))
    result=[]
    for i,value in enumerate(maximum):
        if yaws[i] is None:result.append(0.);continue
        neighbours=[j for j in range(max(0,i-3),min(len(raw),i+4)) if connected(i,j)]
        weights=np.exp(-.5*((np.asarray(neighbours)-i)/1.3)**2)
        result.append(float(np.average([maximum[j] for j in neighbours],weights=weights)))
    return result


def anatomy_mask(points,shape):
    """Preserve the measured profile nose and lips on every frame."""
    a=five(points);_,_,height=pose(a)
    nose=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(nose,cv2.convexHull(np.rint(points[27:36]).astype(np.int32)),1)
    radius=max(2,round(height*.18))
    nose=cv2.dilate(nose,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    outside=cv2.distanceTransform(1-nose,cv2.DIST_L2,5)
    keep=np.clip(outside/max(3,height*.08),0,1)
    eyes=(a[0]+a[1])*.5;mouth=(a[3]+a[4])*.5
    vertical=(mouth-eyes)/height
    y,x=np.mgrid[:shape[0],:shape[1]].astype(np.float32)
    projection=(x-mouth[0])*vertical[0]+(y-mouth[1])*vertical[1]
    keep*=np.clip((-height*.13-projection)/max(3,height*.10),0,1)
    return keep.astype(np.float32)


def detail_mask(points,shape,side):
    """Transfer the visible eye and cheek, preserving source nose geometry."""
    points=np.asarray(points,np.float32);a=five(points);_,_,height=pose(a)
    eye_ids=list(range(36,42)) if side==0 else list(range(42,48))
    brow_ids=list(range(17,22)) if side==0 else list(range(22,27))
    region=np.zeros(shape,np.uint8)
    polygon=points[eye_ids+brow_ids+list(range(27,36))]
    cv2.fillConvexPoly(region,cv2.convexHull(np.rint(polygon).astype(np.int32)),1)
    radius=max(2,round(height*.14))
    region=cv2.dilate(region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    alpha=cv2.GaussianBlur(region.astype(np.float32),(0,0),max(2,height*.045))
    # The same geometric lip boundary on every frame, regardless of the
    # generated landmarker confidence. No alternating original/generated mouth.
    alpha*=anatomy_mask(points,shape)
    return alpha.clip(0,1).astype(np.float32)


def anchor_parameters(expected,predicted,confidence,side):
    """Pin the visible eye without moving or stretching the profile nose."""
    if predicted is None or not np.isfinite(predicted).all():return None
    a=five(expected);b=five(predicted);height=pose(a)[2]
    shift=a[side]-b[side]
    if np.linalg.norm(shift)>.30*height:return None
    trust=float(np.clip((confidence-.40)/.30,0,1))
    if not trust:return None
    length=float(np.linalg.norm(shift));limit=.14*height
    if length>limit:shift*=limit/length
    return np.r_[shift/height,0.,0.]*trust


def correction_matrix(parameters,points):
    a=five(points);height=pose(a)[2];dx,dy,angle,log_scale=parameters
    c,s=np.cos(angle),np.sin(angle);linear=np.array([[c,-s],[s,c]])*np.exp(log_scale)
    return np.c_[linear,a[2]+np.array([dx,dy])*height-linear@a[2]]


class ProfileDetails:
    def __init__(self):
        self.states={}
        self.stats={'profile_faces':0,'anchored_faces':0,'transported_masks':0,'resets':0}

    def save(self,path):
        import json,os
        from pathlib import Path
        path=Path(path);values={};metadata={}
        last=max((s['index'] for s in self.states.values()),default=-1)
        for identity,state in self.states.items():
            if state['index']<last-2:continue
            key=str(identity)
            metadata[key]={k:v for k,v in state.items() if k not in ('gray','mask','parameters')}
            for name in ('gray','mask','parameters'):values[key+'_'+name]=state[name]
        values['metadata']=np.array(json.dumps(metadata))
        temporary=path.with_name(path.name+'.updating')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
        os.replace(temporary,path)

    def load(self,path):
        import json
        try:
            with np.load(path,allow_pickle=False) as values:
                metadata=json.loads(str(values['metadata']))
                states={int(key):{**state,**{name:values[key+'_'+name].copy()
                         for name in ('gray','mask','parameters')}} for key,state in metadata.items()}
            self.states=states
        except (OSError,ValueError,KeyError):self.states={}

    def apply(self,identity,index,original,crop,base_mask,points,predicted,confidence,gain):
        a=five(points);yaw,spread,height=pose(a);side=visible_side(points)
        gray=cv2.resize(cv2.cvtColor(original,cv2.COLOR_BGR2GRAY),(128,128))
        previous=self.states.get(identity)
        if previous is not None and index==previous['index']+1 and abs(yaw)<.25 and spread>.65:
            side=previous['side']
        continuous=bool(previous is not None and index==previous['index']+1
                        and previous['side']==side
                        and .75<height/previous['height']<1.33
                        and np.abs(gray.astype(float)-previous['gray']).mean()<35)
        if not continuous:
            previous=None;self.stats['resets']+=1
        measured=anchor_parameters(points,predicted,confidence,side)
        if previous is None:
            parameters=measured if measured is not None else np.zeros(4)
            stable_gain=float(gain)
        else:
            # A weak measurement holds the previous correction briefly and
            # then releases it smoothly. It never flips the mask to full-face.
            stale=previous['stale']+1 if measured is None else 0
            target=measured if measured is not None else (previous['parameters'] if stale<=3 else np.zeros(4))
            delta=np.asarray(target)-previous['parameters']
            delta[:2]=np.clip(delta[:2],-.012,.012)
            delta[2]=np.clip(delta[2],-np.deg2rad(1),np.deg2rad(1))
            delta[3]=np.clip(delta[3],-.012,.012)
            parameters=previous['parameters']+delta*.35
            # Gain is a bidirectional shot plan, not a delayed confidence gate.
            stable_gain=float(gain)
        stale=previous['stale']+1 if previous is not None and measured is None else 0
        semantic=(1-stable_gain)+stable_gain*detail_mask(points,base_mask.shape,side)
        small=cv2.resize(semantic,(128,128))
        if previous is not None:
            # Transport only opacity on the original moving image. Reusing
            # generated RGB would create a double nose/eye during motion.
            flow=cv2.calcOpticalFlowFarneback(gray,previous['gray'],None,.5,2,15,2,5,1.1,0)
            yy,xx=np.mgrid[:128,:128].astype(np.float32)
            warped=cv2.remap(previous['mask'],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            old_gray=cv2.remap(previous['gray'],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            valid=np.abs(old_gray.astype(float)-gray)<20
            small=np.where(valid,.72*warped+.28*small,small)
            self.stats['transported_masks']+=1
        semantic=cv2.resize(small,base_mask.shape[::-1]).clip(0,1)
        corrected=crop
        if np.linalg.norm(parameters)>1e-6:
            matrix=correction_matrix(parameters*stable_gain,points)
            corrected=cv2.warpAffine(crop,matrix,crop.shape[:2][::-1],flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            self.stats['anchored_faces']+=1
        # The current occlusion mask always wins. The temporal opacity must
        # not paint over a hand, glasses frame or other newly arrived occluder.
        anatomy=(1-stable_gain)+stable_gain*anatomy_mask(points,base_mask.shape)
        mask=np.minimum.reduce([base_mask,semantic,anatomy]).astype(np.float32)
        tail=12 if stable_gain>.05 else max(0,previous.get('tail',0)-1) if previous is not None else 0
        self.states[identity]={'index':index,'side':side,'yaw':yaw,'height':height,'gray':gray,'tail':tail,
                               'mask':small,'parameters':parameters,'gain':stable_gain,'stale':stale}
        self.stats['profile_faces']+=int(stable_gain>.05)
        return corrected,mask
