"""Isolated FaceFusion process. Built-in content checks stay active."""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1'
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8',errors='replace')
job_path = Path(sys.argv[1]).resolve()
job = json.loads(job_path.read_text(encoding='utf-8'))
from quality import render_settings
settings = render_settings(job.get('processing_profile'))
print('Режим качества: 512×512, маска каждого кадра, бережное кодирование.'
      if settings['mask_every_frame'] else 'Сравнение: прежний режим 256×256.', flush=True)
os.environ['PATH'] = str(ROOT / 'runtime/bin') + os.pathsep + os.environ.get('PATH', '')
os.environ['TEMP'] = os.environ['TMP'] = str(job_path.parent / 'temp')
Path(os.environ['TEMP']).mkdir(exist_ok=True)
os.chdir(ROOT / 'runtime/facefusion')
sys.path.insert(0, str(ROOT / 'runtime/directml'))
sys.path.insert(0, str(ROOT / 'runtime/facefusion'))

from occlusion import install as install_occlusion
install_occlusion(settings)
from temporal import install
install(job, job_path.parent)

args = ['headless-run', '-s', *job['sources'], '-t', job['target'], '-o', job['output'],
        '--jobs-path', str(job_path.parent / 'facefusion-jobs'),
        '--processors', 'face_swapper', '--face-swapper-model', 'hyperswap_1a_256',
        '--face-swapper-pixel-boost', settings['pixel_boost'],
        '--face-swapper-weight', '0.5', '--face-selector-mode', 'one',
        '--face-detector-model', 'many', '--face-detector-size', '640x640',
        '--face-detector-margin', '60', '60', '60', '60', '--face-detector-score', '0.15',
        '--face-tracker-score', '0.15', '--target-frame-amount', '5',
        '--face-mask-types', 'box', 'occlusion', '--face-occluder-model', 'xseg_1', '--face-mask-blur', '0.4',
        '--execution-providers', job.get('provider', 'directml'), '--execution-thread-count', '1',
        '--video-memory-strategy', 'tolerant', '--output-video-quality', settings['output_quality'],
        '--output-video-preset', settings['preset'], '--log-level', 'info']
if job.get('start_frame', 0) or job.get('end_frame') is not None:
    args += ['--trim-frame-start', str(job.get('start_frame', 0))]
    if job.get('end_frame') is not None:
        args += ['--trim-frame-end', str(job['end_frame'])]
sys.argv = ['facefusion.py', *args]
runpy.run_path('facefusion.py', run_name='__main__')
