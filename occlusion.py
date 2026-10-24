"""Preserve real occluders; quality mode measures the mask for every swapped face."""
import threading
from pathlib import Path
import cv2,numpy as np
LOCAL=threading.local()


def set_context(identity,index):
    LOCAL.identity=identity;LOCAL.index=index


def install(settings=None):
    from quality import render_settings
    settings = settings or render_settings()
    from facefusion.processors.modules.face_swapper import core as swapper
    old_swap=swapper.swap_face;local=LOCAL;session=None;cache={}
    grid_y,grid_x=np.mgrid[:256,:256].astype(np.float32)
    def mask(crop):
        nonlocal session
        if not getattr(local,'enabled',False):return np.ones(crop.shape[:2],np.float32)
        if session is None:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            path=Path(__file__).parent/'runtime/facefusion/.assets/models/xseg_1.onnx'
            options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                from facefusion import state_manager
                requested=state_manager.get_item('execution_providers') or []
                providers=['DmlExecutionProvider','CPUExecutionProvider'] if 'directml' in requested and 'DmlExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
                session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
            except Exception:
                session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            print('Маска xseg: '+', '.join(session.get_providers()),flush=True)
        resized=cv2.resize(crop,(256,256));gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        identity=getattr(local,'identity',0);index=getattr(local,'index',0);previous=cache.get(identity)
        reuse=not settings['mask_every_frame'] and previous is not None and previous['index']==index-1 and index-previous['measured']<3 and np.abs(gray.astype(float)-previous['gray']).mean()<25
        if reuse:
            flow=cv2.calcOpticalFlowFarneback(gray,previous['gray'],None,.5,2,15,2,5,1.1,0)
            result=cv2.remap(previous['mask'],grid_x+flow[:,:,0],grid_y+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            measured=previous['measured']
        else:
            value=resized[None].astype(np.float32)/255
            try:result=session.run(None,{'input':value})[0][0,:,:,0].clip(0,1)
            except Exception:
                if 'DmlExecutionProvider' not in session.get_providers():raise
                import onnxruntime as ort
                path=Path(__file__).parent/'runtime/facefusion/.assets/models/xseg_1.onnx'
                session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
                print('Маска xseg: запасной расчёт на CPU.',flush=True)
                result=session.run(None,{'input':value})[0][0,:,:,0].clip(0,1)
            if settings['mask_every_frame']:
                # Blur in the final crop scale, matching FaceFusion's mask.
                # Blurring a 256 mask before expanding to 512 doubles feathering.
                result=cv2.resize(result,crop.shape[:2][::-1])
                result=cv2.GaussianBlur(result,(0,0),5)
                return (result-.5).clip(0,1)*2
            result=cv2.GaussianBlur(result,(0,0),5)
            result=(result-.5).clip(0,1)*2;measured=index
        cache[identity]={'index':index,'measured':measured,'gray':gray,'mask':result}
        return cv2.resize(result,crop.shape[:2][::-1])
    def swap(source,target,image,frame):
        if not hasattr(local,'index'):local.index=getattr(local,'counter',0);local.counter=local.index+1
        local.enabled=settings['mask_every_frame'] or target.origin!='detect' or float(target.score_set['landmarker'])<.65
        try:return old_swap(source,target,image,frame)
        finally:
            local.enabled=False
            if hasattr(local,'index'):del local.index
    swapper.create_occlusion_mask=mask
    swapper.swap_face=swap
