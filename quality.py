"""Versioned render settings; quality is the default for new and existing projects."""

DEFAULT_PROFILE = 'quality-v1'


def render_settings(profile=None):
    profile = profile or DEFAULT_PROFILE
    if profile == 'balanced-v1':
        # Retained for reproducible comparisons with FaceStudio 2.4.
        return {'id': profile, 'pixel_boost': '256x256', 'mask_every_frame': False,
                'output_quality': '90', 'preset': 'fast'}
    if profile != DEFAULT_PROFILE:
        raise ValueError('Неизвестный режим обработки. Запустите новую замену лиц.')
    return {'id': profile, 'pixel_boost': '512x512', 'mask_every_frame': True,
            'output_quality': '95', 'preset': 'slow'}
