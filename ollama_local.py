"""Local Ollama lifecycle and strictly bounded command interpretation."""
import json
import math
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

URL = 'http://127.0.0.1:11434'
SCHEMA = {'type': 'object', 'properties': {
    'action': {'type': 'string', 'enum': ['swap', 'repair', 'check']},
    'start_seconds': {'type': 'number', 'minimum': 0},
    'end_seconds': {'type': ['number', 'null']},
    'person_ids': {'type': 'array', 'items': {'type': 'integer', 'minimum': 1}},
    'explanation': {'type': 'string'}},
    'required': ['action', 'start_seconds', 'end_seconds', 'person_ids', 'explanation'],
    'additionalProperties': False}


def request(endpoint, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(URL + endpoint, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def find_executable():
    found = shutil.which('ollama')
    if found:
        return Path(found)
    local = Path(os.environ.get('LOCALAPPDATA', ''))
    for candidate in [local / 'Programs/Ollama/ollama.exe', local / 'Ollama/ollama.exe']:
        if candidate.is_file():
            return candidate
    raise RuntimeError('Ollama не найдена. Установите Ollama и загрузите модель qwen3.5:9b.')


def ensure_running(data_dir, timeout=45):
    try:
        request('/api/version')
        return 'Ollama уже работает.'
    except (OSError, ValueError):
        pass
    executable = find_executable()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / 'ollama.log').open('ab') as log:
        subprocess.Popen([str(executable), 'serve'], stdout=log, stderr=log,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request('/api/version', timeout=2)
            return 'Ollama запущена автоматически.'
        except (OSError, ValueError):
            time.sleep(.5)
    raise RuntimeError('Ollama не ответила после запуска. Подробности: data/ollama.log.')


def choose_model():
    names = [model['name'] for model in request('/api/tags')['models']
             if model.get('size', 0) > 1024*1024 and 'cloud' not in model['name'].lower()
             and not model.get('remote_host') and not model.get('remote_model')]
    if not names:
        raise RuntimeError('В Ollama нет локальных моделей. Загрузите qwen3.5:9b: ollama pull qwen3.5:9b')
    return next((name for name in names if name.startswith('qwen3.5:9b')), names[0])


def release_model(data_dir):
    ensure_running(data_dir)
    model=choose_model()
    loaded=[item['name'] for item in request('/api/ps').get('models',[])]
    if model in loaded:
        request('/api/generate',{'model':model,'keep_alive':0},timeout=30)


def interpret(command, has_result, data_dir, known_ids=None):
    ensure_running(data_dir)
    model = choose_model()
    system = ('Ты интерпретатор команд локального редактора замены лица. Верни JSON по схеме. '
              'swap — новая замена лица, repair — исправить нестабильную замену, check — только проверить. '
              'Работаем с целым видео, без цвета и обрезки. start_seconds/end_seconds — участок исправления; '
              'для всего видео 0/null. "Первые четыре секунды" значит 0/4. '
              'Не придумывай временные границы. explanation — коротко по-русски. '
              'person_ids — номера людей, прямо указанные пользователем, в том числе словами: второй значит 2. '
              'Если конкретный человек не указан или сказано все, верни пустой список. Не выбирай человека по описанию внешности. '
              f'Назначенные номера людей: {known_ids or []}. '
              f'Готовый результат уже есть: {has_result}.')
    response = request('/api/chat', {'model': model, 'stream': False, 'think': False,
        'keep_alive': 0, 'format': SCHEMA, 'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 256},
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': command}]}, timeout=240)
    plan = json.loads(response['message']['content'])
    if plan.get('action') not in ('swap', 'repair', 'check'):
        raise RuntimeError('Ollama вернула неизвестную команду. Попробуйте сформулировать иначе.')
    start = float(plan.get('start_seconds', 0))
    end = plan.get('end_seconds')
    if not math.isfinite(start) or start < 0 or (end is not None and
        (not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start)):
        raise RuntimeError('Не получилось определить участок. Напишите, например: «Исправь первые 4 секунды».')
    plan['start_seconds'] = start
    ids=plan.get('person_ids',[])
    if not isinstance(ids,list) or any(not isinstance(value,int) or isinstance(value,bool) or value<1 for value in ids):
        raise RuntimeError('Не получилось определить номер человека. Укажите его номер из окна назначения.')
    return plan, model


def inspect_images(images, data_dir):
    """Optional semantic check; never treated as proof of correctness."""
    import base64
    ensure_running(data_dir)
    model = choose_model()
    info = request('/api/show', {'model': model}, timeout=10)
    if 'vision' not in info.get('capabilities', []):
        return 'В модели нет просмотра изображений; выполнена техническая проверка.'
    encoded = [base64.b64encode(Path(path).read_bytes()).decode('ascii') for path in images]
    response = request('/api/chat', {'model': model, 'stream': False, 'think': False,
        'keep_alive': 0, 'options': {'temperature': 0, 'num_ctx': 4096, 'num_predict': 300},
        'messages': [{'role': 'user', 'content': 'Первая картинка — фото-источники для замены. Вторая — несколько кадров '
        'синтетической замены лица по времени. Оцени видимые артефакты лица: швы, двоение, странные глаза. '
        'Отвечай коротко по-русски. По отдельным кадрам нельзя доказать отсутствие дрожания. '
        'Если лицо не видно, так и скажи. Не идентифицируй людей.', 'images': encoded}]}, timeout=240)
    return response['message']['content']
