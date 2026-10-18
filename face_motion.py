"""Conservative visual continuity between real face detections, without generating faces."""
import cv2
import numpy as np
from people import box_overlap
from joint_matching import assign


def small_gray(frame):
    scale=min(1.,640/frame.shape[1])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if scale<1:gray=cv2.resize(gray,None,fx=scale,fy=scale)
    return gray,scale


def scene_changed(a,b):
    if a is None or a.shape!=b.shape:return True
    x=cv2.resize(a,(64,36)).astype(np.float32);y=cv2.resize(b,(64,36)).astype(np.float32)
    difference=float(np.abs(x-y).mean())
    correlation=float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if x.std()>2 and y.std()>2 else 0
    # Dark cuts need a relative-change check; a fixed 35 gray threshold misses them.
    return difference>42 or (difference>12 and correlation<.35)


class FaceMotion:
    def __init__(self,fps=30):
        self.max_gap=max(2,round(fps*.75));self.gray=None;self.scale=1.;self.tracks={};self.cut=False

    def _points(self,box):
        mask=np.zeros(self.gray.shape,np.uint8);h,w=mask.shape
        x1,y1,x2,y2=np.rint(box).astype(int)
        center=(int((x1+x2)/2),int((y1+y2)/2))
        axes=(max(1,int((x2-x1)*.42)),max(1,int((y2-y1)*.43)))
        cv2.ellipse(mask,center,axes,0,0,360,255,-1)
        return cv2.goodFeaturesToTrack(self.gray,80,.015,5,mask=mask)

    def seed(self,frame,records,index):
        self.gray,self.scale=small_gray(frame)
        for identity,face in records:self.remember(identity,face,index)

    def predict(self,frame,faces,index):
        gray,scale=small_gray(frame);self.cut=scene_changed(self.gray,gray)
        if self.cut:self.tracks.clear()
        proposals=[];kept={}
        for identity,track in self.tracks.items():
            if index-track['seen']>self.max_gap:continue
            points=track['points']
            if points is None or len(points)<10:continue
            forward,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,winSize=(21,21),maxLevel=3)
            if forward is None:continue
            backward,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,forward,None,winSize=(21,21),maxLevel=3)
            if backward is None:continue
            good=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)
            good&=np.linalg.norm(points[:,0]-backward[:,0],axis=1)<1.5
            if good.sum()<10:continue
            affine,inliers=cv2.estimateAffinePartial2D(points[good],forward[good],method=cv2.RANSAC,ransacReprojThreshold=2)
            if affine is None or inliers is None:continue
            support=float(inliers.mean());size=float(np.linalg.norm(affine[0,:2]))
            if support<.7 or not .8<size<1.25:continue
            x1,y1,x2,y2=track['box'];corners=np.array([[[x1,y1],[x2,y2]]],np.float32)
            transformed=cv2.transform(corners,affine)[0];box=transformed.ravel()
            kept[identity]={**track,'box':box,'points':forward[good][inliers.ravel().astype(bool)]}
            for n,face in enumerate(faces):
                current=np.asarray(face.bounding_box)*scale;overlap=box_overlap(box,current)
                similarity=float(np.dot(track['embedding'],face.embedding_norm))
                if overlap<.5 or similarity<.12:continue
                center=np.linalg.norm((box[:2]+box[2:]-current[:2]-current[2:])/2)
                if center>max(current[2:]-current[:2])*.3:continue
                score=.55*overlap+.30*support+.15*max(0,similarity)
                proposals.append((score,n,identity,support,overlap,similarity,index-track['seen']))
        self.gray,self.scale,self.tracks=gray,scale,kept
        # Consider all tracks together; leave globally ambiguous faces unmatched.
        details={(p[1],p[2]):p for p in proposals}
        result={}
        for score,n,identity in assign((p[:3] for p in proposals),margin=.10):
            item=details[n,identity]
            result[n]={'person_id':identity,'score':score,'support':item[3],
                       'overlap':item[4],'similarity':item[5],'gap':item[6]}
        return result

    def remember(self,identity,face,index):
        box=np.asarray(face.bounding_box,np.float32)*self.scale
        self.tracks[identity]={'box':box,'points':self._points(box),
                              'embedding':np.asarray(face.embedding_norm),'seen':index}
