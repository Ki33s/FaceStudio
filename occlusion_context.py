"""Recover visible face geometry through a bounded, two-anchor occlusion.

Xseg masks are measured on current SOURCE pixels. Flow uses only visible face
features, never the moving hand. Hidden pixels are never synthesized here.
"""
import base64,copy
from pathlib import Path
import cv2,numpy as np
from selection_tracking import flow_matrix,transform,transformed_box
from face_motion import scene_changed

TEMPLATE=np.array([[.36167656,.40387734],[.63696719,.40235469],[.50019687,.56044219],
                   [.38710391,.72160547],[.61507734,.72034453]],np.float32)


class Visibility:
    def __init__(self):self.session=None
    def measure(self,image,face):
        if self.session is None:
            import onnxruntime as ort
            options=ort.SessionOptions();options.enable_mem_pattern=False
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.intra_op_num_threads=1
            providers=['DmlExecutionProvider','CPUExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.session=ort.InferenceSession(str(Path(__file__).parent/'runtime/facefusion/.assets/models/xseg_1.onnx'),sess_options=options,providers=providers)
            print('Контекст перекрытия: '+', '.join(self.session.get_providers()),flush=True)
        five=np.asarray(face['landmark_set']['5/68'],np.float32)
        matrix=cv2.estimateAffinePartial2D(five,TEMPLATE*256,method=cv2.RANSAC,ransacReprojThreshold=100)[0]
        if matrix is None or not np.isfinite(matrix).all():return None
        from profile_alignment import profile_matrix
        from profile_detail import profile_strength
        matrix,_=profile_matrix(five,face['landmark_set']['68'],matrix,profile_strength(five))
        crop=cv2.warpAffine(image,matrix,(256,256),borderMode=cv2.BORDER_REPLICATE)
        value=crop[None].astype(np.float32)/255
        try:probability=self.session.run(None,{'input':value})[0][0,:,:,0].clip(0,1)
        except Exception:
            if 'DmlExecutionProvider' not in self.session.get_providers():raise
            self.session.set_providers(['CPUExecutionProvider'])
            probability=self.session.run(None,{'input':value})[0][0,:,:,0].clip(0,1)
        full=cv2.warpAffine(probability,cv2.invertAffineTransform(matrix),image.shape[:2][::-1])
        points=np.asarray(face['landmark_set']['68'],np.float32)
        hull=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(np.rint(points[:27]).astype(np.int32)),1)
        return full*hull,matrix,probability


def interpolate_face(left,right,t):
    result=copy.deepcopy(left)
    result['bounding_box']=((1-t)*np.asarray(left['bounding_box'])+t*np.asarray(right['bounding_box'])).tolist()
    result['landmark_set']={k:((1-t)*np.asarray(v)+t*np.asarray(right['landmark_set'][k])).tolist()
                            for k,v in left['landmark_set'].items() if k in right['landmark_set']}
    return result


def scaled_face(face,scale):
    result=copy.deepcopy(face);result['bounding_box']=(np.asarray(face['bounding_box'])*np.tile(scale,2)).tolist()
    result['landmark_set']={k:(np.asarray(v)*scale).tolist() for k,v in face['landmark_set'].items()}
    return result


def visible_bridge(images,left,right,measure,maximum=24):
    """Require both identity anchors, visible-motion agreement and an occluder."""
    if not 3<=len(images)<=maximum+2:return None
    similarity=float(np.asarray(left['embedding_norm'])@np.asarray(right['embedding_norm']))
    if not np.isfinite(similarity) or similarity<.45:return None
    grays=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
    if any(scene_changed(a,b) for a,b in zip(grays,grays[1:])):return None
    measurements=[];masks=[];fractions=[]
    for i,image in enumerate(images):
        expected=interpolate_face(left,right,i/(len(images)-1))
        result=measure(image,expected)
        if result is None:return None
        mask,matrix,probability=result
        x1,y1,x2,y2=np.rint(expected['bounding_box']).astype(int);h,w=mask.shape
        region=mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        if not region.size:return None
        fractions.append(float((region>.70).mean()));masks.append(mask);measurements.append((matrix,probability))
    reference=min(fractions[0],fractions[-1])
    # Do not route a mere blurry/no-face interval through permissive tracking.
    if reference<.25 or min(fractions[1:-1])>reference*.85:return None
    if min(fractions[1:-1])<.12:return None  # Completely hidden: retain source.
    def path(face,reverse=False):
        order=list(range(len(images)));order=order[::-1] if reverse else order
        box=np.asarray(face['bounding_box']);current=np.eye(3);result=[current]
        for a,b in zip(order,order[1:]):
            step=flow_matrix(grays[a],grays[b],transformed_box(box,current),masks[a],masks[b])
            if step is None:return None
            current=step@current;result.append(current)
        return result[::-1] if reverse else result
    forward=path(left);backward=path(right,True)
    if forward is None or backward is None:return None
    a=np.asarray(left['landmark_set']['5/68']);b=np.asarray(right['landmark_set']['5/68'])
    height=max(left['bounding_box'][3]-left['bounding_box'][1],1.)
    if max(np.linalg.norm(transform(a,forward[-1])-b,axis=1))>.12*height:return None
    if max(np.linalg.norm(transform(b,backward[0])-a,axis=1))>.12*height:return None
    result=[]
    for i in range(1,len(images)-1):
        if max(np.linalg.norm(transform(a,forward[i])-transform(b,backward[i]),axis=1))>.10*height:return None
        visible=masks[i]>.70
        # The current visible face must still resemble both moving source
        # anchors. Pixels hidden by a hand/object take no part in this test.
        for source,matrix in ((grays[0],forward[i]),(grays[-1],backward[i])):
            warped=cv2.warpAffine(source,matrix[:2],grays[i].shape[::-1])
            delta=np.abs(warped[visible].astype(float)-grays[i][visible])
            if len(delta)<80 or np.median(delta)>15 or np.mean(delta>35)>.25:return None
        t=i/(len(images)-1);face=copy.deepcopy(left)
        face['bounding_box']=((1-t)*transformed_box(left['bounding_box'],forward[i])+t*transformed_box(right['bounding_box'],backward[i])).tolist()
        face['landmark_set']={k:((1-t)*transform(v,forward[i])+t*transform(right['landmark_set'][k],backward[i])).tolist() for k,v in left['landmark_set'].items() if k in right['landmark_set']}
        face['origin']='occlusion-context'
        matrix,probability=measurements[i]
        ok,png=cv2.imencode('.png',np.rint(probability*255).astype(np.uint8));assert ok
        result.append((face,{'matrix':matrix.tolist(),'png':base64.b64encode(png.tobytes()).decode('ascii'),'size':list(images[i].shape[:2][::-1])}))
    return result


def project_visibility(value,matrix,shape,frame_size):
    data=base64.b64decode(value['png'],validate=True)
    probability=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    if probability is None:raise ValueError('Повреждена сохранённая маска перекрытия.')
    stored=np.eye(3);stored[:2]=value['matrix']
    # Stored geometry belongs to the source image size, not the output crop.
    scale=np.diag([value['size'][0]/frame_size[0],value['size'][1]/frame_size[1],1.])
    current=np.eye(3);current[:2]=matrix
    mapping=current@np.linalg.inv(stored@scale)
    mask=cv2.warpAffine(probability.astype(np.float32)/255,mapping[:2],shape[::-1])
    return np.clip(mask-.5,0,1)*2
