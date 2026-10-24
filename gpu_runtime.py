"""Report actual ONNX Runtime providers used by the isolated worker."""
import json
from pathlib import Path


def install(folder):
    from facefusion import inference_manager
    original=inference_manager.create_inference_session;records=[]
    def create(model,providers):
        session=original(model,providers)
        actual=session.get_providers();records.append({'model':Path(model).name,'providers':actual})
        (Path(folder)/'runtime-providers.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print('ONNX '+Path(model).name+': '+', '.join(actual),flush=True)
        return session
    inference_manager.create_inference_session=create
