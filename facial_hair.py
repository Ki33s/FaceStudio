"""Preserve visible source facial hair by reducing replacement opacity.

No hair is generated or moved from another frame. A conservative contrast and
texture test is restricted to the measured upper-lip and beard regions.
"""
import cv2
import numpy as np


def polygon(points,shape):
    result=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(result,cv2.convexHull(np.rint(points).astype(np.int32)),1)
    return result


def hair_regions(points,shape):
    p=np.asarray(points,np.float32);mouth=(p[48]+p[54])*.5
    width=max(float(np.linalg.norm(p[54]-p[48])),8.)
    eyes=(p[36:42].mean(0)+p[42:48].mean(0))*.5
    down=mouth-eyes;down/=max(float(np.linalg.norm(down)),1.)
    y,x=np.mgrid[:shape[0],:shape[1]].astype(np.float32)
    vertical=(x-mouth[0])*down[0]+(y-mouth[1])*down[1]
    mustache=polygon(p[list(range(31,36))+list(range(48,55))],shape)
    # Include the hair beside the mouth corners, but keep lips and nostrils
    # as separate features. Their darkness alone is not facial-hair evidence.
    radius=max(1,round(width*.035))
    mustache=cv2.dilate(mustache,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    beard=polygon(p[list(range(1,16))+[48,54]],shape)
    beard&=(vertical>float((p[57]-mouth)@down)+width*.025).astype(np.uint8)
    inset=max(1,round(width*.04))
    beard=cv2.erode(beard,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*inset+1,2*inset+1)))
    lips=polygon(p[48:60],shape)
    lips=cv2.dilate(lips,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    nose=polygon(p[27:36],shape)
    nose=cv2.dilate(nose,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    # The thick moustache can overlap the measured upper-lip/nose boundaries,
    # especially when looking down. Excluding their full hull erases the very
    # hair being preserved. The contrast/texture entry test below prevents a
    # clean face from acquiring a broad preservation patch.
    # Keep the bounded upper-lip region intact: on pitched heads the inner
    # mouth landmarks can fall directly on a thick moustache. Removing that
    # guessed hole would erase its centre and make it flicker.
    # On a normally spaced mouth the opening is a real dark feature, not hair.
    # Exclude it only when the nose/mouth ordering is geometrically reliable.
    if float((mouth-p[33])@down)>.12*width:
        mustache*=1-polygon(p[60:68],shape)
    beard*=1-np.maximum(lips,nose)
    core=mustache.copy()
    # Landmarks locate a seed, not the edge of every moustache hair. Grow only
    # connected dark components in a bounded mouth neighbourhood, so a thick
    # or drooping moustache is not cut off at an uncertain lip landmark.
    reach=max(2,round(width*.18))
    mustache=cv2.dilate(mustache,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*reach+1,2*reach+1)))
    horizontal=(x-mouth[0])*down[1]-(y-mouth[1])*down[0]
    lower=.28 if float((mouth-p[33])@down)>.12*width else .45
    mustache*=((np.abs(horizontal)<width*.90)&(vertical>-width*.60)&(vertical<width*lower)).astype(np.uint8)
    if float((mouth-p[33])@down)>.12*width:mustache*=1-polygon(p[60:68],shape)
    cheeks=np.maximum(polygon(p[[1,2,3,31,39,36]],shape),polygon(p[[13,14,15,45,42,35]],shape))
    cheeks*=1-np.maximum(lips,nose)
    return mustache,beard,cheeks,width,core


def measure_hair(image,points,include_support=False):
    shape=image.shape[:2];zero=np.zeros(shape,np.float32)
    if np.asarray(points).shape!=(68,2) or not np.isfinite(points).all():return (zero,zero,{},zero) if include_support else (zero,zero,{})
    mustache,beard,cheeks,width,core=hair_regions(points,shape)
    lab=cv2.cvtColor(image.clip(0,255).astype(np.uint8),cv2.COLOR_BGR2LAB).astype(np.float32)
    light=lab[:,:,0];chroma=np.linalg.norm(lab[:,:,1:]-128,axis=2)
    reference=light[cheeks>0]
    if len(reference)<20:return (zero,zero,{},zero) if include_support else (zero,zero,{})
    skin=float(np.percentile(reference,60));difference=skin-light
    skin_chroma=float(np.median(chroma[cheeks>0]))
    probability=np.clip((difference-12)/18,0,1)
    detail=np.abs(light-cv2.GaussianBlur(light,(0,0),max(.7,width*.02)))
    masks=[];scores={}
    for name,region in (('mustache',mustache),('beard',beard)):
        count=int(region.sum())
        current=probability.copy()
        reference_hair=masks[0]>.5 if masks else np.zeros(shape,bool)
        if name=='beard' and scores.get('mustache',{}).get('confident') and reference_hair.any():
            # Match the actually observed hair darkness, rather than preserving
            # every dark jaw shadow as a beard on a clean-shaven chin.
            hair_light=float(np.median(light[reference_hair]))
            current*=np.clip((hair_light+22-light)/12,0,1)
        ink=((current>.35)&(region>0)).astype(np.uint8)
        amount,labels,stats,_=cv2.connectedComponentsWithStats(ink)
        supported=np.zeros(shape,np.uint8)
        for label in range(1,amount):
            _,_,w,h,area=stats[label]
            if area>=max(8,width*width*.004) and w>=width*.12:
                if name=='mustache' and not np.any((labels==label)&(core>0)):continue
                supported[labels==label]=1
        chosen=supported>0;scored=chosen&(core>0) if name=='mustache' else chosen
        score_count=int(core.sum()) if name=='mustache' else count
        coverage=float(scored.sum()/max(score_count,1))
        contrast=float(np.median(difference[scored])) if scored.any() else 0.
        texture=float(np.mean(detail[scored])) if scored.any() else 0.
        hair_chroma=float(np.median(chroma[scored])) if scored.any() else skin_chroma
        interior=cv2.distanceTransform(supported,cv2.DIST_L2,3)>max(1,width*.015)
        inner_texture=float(np.median(detail[interior])) if int(interior.sum())>=10 else 0.
        confident=count>=20 and coverage>=.25 and contrast>=20 and (texture>=1.1 or (texture>=.7 and contrast>=40))
        if name=='beard':
            confident=count>=20 and coverage>=.25 and contrast>=20 and (texture>=1.5 or (contrast>=65 and texture>=.45))
        if name=='beard' and not scores.get('mustache',{}).get('confident'):
            confident=confident and texture>=2.0 and contrast>=35
        # A smooth, skin-coloured shadow under the nose is not a moustache.
        # Dark hair generally loses skin chroma; richly coloured hair needs
        # stronger actual texture to start preservation.
        confident=confident and (hair_chroma<=skin_chroma*.90 or (inner_texture>=3.0 and contrast>=30))
        scores[name]={'coverage':coverage,'contrast':contrast,'texture':texture,'confident':bool(confident)}
        alpha=current*supported
        alpha=cv2.GaussianBlur(alpha,(0,0),max(.7,width*.012))*region
        masks.append(alpha.clip(0,1).astype(np.float32))
    if include_support:return *masks,scores,probability*polygon(np.asarray(points)[:27],shape)
    return *masks,scores


class FacialHair:
    def __init__(self):
        self.states={};self.stats={'mustache_faces':0,'beard_faces':0,'transported_masks':0}

    def apply(self,identity,index,original,points):
        mustache,beard,scores,support=measure_hair(original,points,True)
        gray=cv2.resize(cv2.cvtColor(original,cv2.COLOR_BGR2GRAY),(128,128))
        old=self.states.get(identity)
        continuous=old is not None and index==old['index']+1 and np.abs(gray.astype(float)-old['gray']).mean()<35
        flags={};parts=[]
        for name,mask in (('mustache',mustache),('beard',beard)):
            measured=scores.get(name,{})
            age=0 if measured.get('confident') else min(30,(old.get(name,30)+1) if continuous else 30)
            flags[name]=age
            # Low contrast alone cannot start preservation on a clean face.
            # A clearly measured hair region may survive a few weaker frames.
            enabled=age<12 and measured.get('coverage',0)>.05 and measured.get('contrast',0)>12
            parts.append(mask if enabled else np.zeros_like(mask))
            self.stats[name+'_faces']+=int(enabled and np.max(mask)>.2)
        result=np.maximum(*parts);small=cv2.resize(result,(128,128))
        if continuous:
            flow=cv2.calcOpticalFlowFarneback(gray,old['gray'],None,.5,2,15,2,5,1.1,0)
            yy,xx=np.mgrid[:128,:128].astype(np.float32)
            previous=cv2.remap(old['mask'],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            previous_gray=cv2.remap(old['gray'],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            # Only opacity is transported. The actual retained pixels always
            # belong to the current source frame, including a newly arrived hand.
            valid=(np.abs(previous_gray.astype(float)-gray)<20)&(cv2.resize(support,(128,128))>.35)
            small=np.maximum(small,previous*valid)
            self.stats['transported_masks']+=1
        self.states[identity]={'index':index,'gray':gray,'mask':small,**flags}
        return cv2.resize(small,original.shape[:2][::-1]).clip(0,1).astype(np.float32)

    def save(self,path):
        import json,os
        from pathlib import Path
        path=Path(path);values={};metadata={};last=max((s['index'] for s in self.states.values()),default=-1)
        for identity,state in self.states.items():
            if state['index']<last-2:continue
            key=str(identity);metadata[key]={k:v for k,v in state.items() if k not in ('gray','mask')}
            for name in ('gray','mask'):values[key+'_'+name]=state[name]
        values['metadata']=np.array(json.dumps(metadata));temporary=path.with_name(path.name+'.updating')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
        os.replace(temporary,path)

    def load(self,path):
        import json
        try:
            with np.load(path,allow_pickle=False) as values:
                metadata=json.loads(str(values['metadata']))
                self.states={int(k):{**state,'gray':values[k+'_gray'].copy(),'mask':values[k+'_mask'].copy()} for k,state in metadata.items()}
        except (OSError,ValueError,KeyError):self.states={}


def restore_source_detail(frame,source,keep,matrix):
    """Restore only measured source details when repairing an older result."""
    y,x=np.nonzero(keep>1e-4)
    if not len(x):return frame
    inverse=cv2.invertAffineTransform(matrix)
    corners=cv2.transform(np.array([[[x.min(),y.min()],[x.max()+1,y.min()],[x.max()+1,y.max()+1],[x.min(),y.max()+1]]],np.float32),inverse)[0]
    h,w=frame.shape[:2];low=np.maximum(0,np.floor(corners.min(0))).astype(int);high=np.minimum([w,h],np.ceil(corners.max(0))).astype(int)
    if min(high-low)<=0:return frame
    transform=inverse.copy();transform[:,2]-=low
    alpha=cv2.warpAffine(keep,transform,tuple(high-low)).clip(0,1)[:,:,None]
    if source.shape[:2]!=frame.shape[:2]:source=cv2.resize(source,(w,h))
    output=frame.copy();x1,y1=low;x2,y2=high
    output[y1:y2,x1:x2]=np.rint(frame[y1:y2,x1:x2]*(1-alpha)+source[y1:y2,x1:x2]*alpha).clip(0,255).astype(frame.dtype)
    return output


def install(folder,persistent=False,restore_source=False):
    import atexit,json,threading
    from pathlib import Path
    from facefusion.processors.modules.face_swapper import core as swapper
    from occlusion import LOCAL as CONTEXT
    local=threading.local();details=FacialHair();folder=Path(folder)
    cache=folder.parent/'facial-hair-state-v1.npz'
    if persistent:details.load(cache)
    old_swap,old_paste=swapper.swap_face,swapper.paste_back
    def swap(source,target,image,frame):
        local.target=target;local.identity=getattr(CONTEXT,'identity',0);local.index=getattr(CONTEXT,'index',0)
        local.visibility=getattr(CONTEXT,'visibility',None)
        local.original=getattr(CONTEXT,'original',None)
        try:return old_swap(source,target,image,frame)
        finally:local.target=None
    def paste(frame,crop,mask,matrix):
        target=getattr(local,'target',None)
        if target is not None and '68' in target.landmark_set:
            if local.visibility is not None:
                from occlusion_context import project_visibility
                mask=np.minimum(mask,project_visibility(local.visibility,matrix,crop.shape[:2],frame.shape[:2][::-1]))
            source=local.original if local.original is not None else frame
            if source.shape[:2]!=frame.shape[:2]:source=cv2.resize(source,frame.shape[:2][::-1])
            original=cv2.warpAffine(source,matrix,crop.shape[:2][::-1],borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_AREA)
            points=cv2.transform(target.landmark_set['68'][None].astype(np.float32),matrix)[0]
            keep=details.apply(local.identity,local.index,original,points)
            if restore_source:
                from occlusion import SOURCE_MASK
                face_area=polygon(points[:27],crop.shape[:2]).astype(np.float32)
                if SOURCE_MASK is not None:
                    source_mask=SOURCE_MASK(original)
                    mask=np.minimum(mask,source_mask);keep=np.maximum(keep,(1-source_mask)*face_area)
                from profile_alignment import _LOCAL as PROFILE
                from profile_detail import anatomy_mask
                gain=getattr(PROFILE,'strength',0.)
                if gain:keep=np.maximum(keep,gain*(1-anatomy_mask(points,crop.shape[:2]))*face_area)
                frame=restore_source_detail(frame,source,keep,matrix)
            mask=np.minimum(mask,1-keep)
        return old_paste(frame,crop,mask,matrix)
    def save():
        (folder/'appearance.json').write_text(json.dumps({'version':1,**details.stats},indent=2),encoding='utf8')
        from profile_alignment import _CACHE_READY
        if persistent and _CACHE_READY:details.save(cache)
    atexit.register(save);swapper.swap_face=swap;swapper.paste_back=paste
