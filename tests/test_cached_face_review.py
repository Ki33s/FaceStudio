"""Regression tests: reviewing saved cards must not prepare or validate video."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime/directml')]
# The bridge redirects library output for its JSON-lines protocol. Restore the
# test runner's streams after importing it; protocol events are captured below.
saved_out, saved_err = sys.stdout, sys.stderr
bridge = importlib.import_module('desktop_backend')
sys.stdout, sys.stderr = saved_out, saved_err


class CachedFaceReview(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / 'input.mp4'
        self.video.write_bytes(b'unchanged source video')
        self.digest = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.folder = self.root / 'data/scans' / ('minutes-v3-' + self.digest[:24])
        self.folder.mkdir(parents=True)
        self.analysis = self.folder / 'people.json'
        self.scan = {
            'fingerprint': 'obsolete-encoding-hash',
            'input_fingerprint': self.digest,
            'processing_video': str(self.root / 'missing-timeline.mp4'),
            'meta': {'fps': 30, 'frames': 60},
            'characters': [{'id': 1, 'thumbnail': str(self.folder / 'person-1.jpg')},
                           {'id': 2, 'thumbnail': str(self.folder / 'person-2.jpg')}],
        }
        self.analysis.write_text(json.dumps(self.scan), encoding='utf-8')
        self.photo = self.root / 'photo.png'
        self.photo.write_bytes(b'photo fixture')
        self.assignments = [{'person_id': 1, 'sources': [str(self.photo)]}]
        self.draft = self.folder / 'assignments.json'
        self.draft.write_text(json.dumps({
            'input_fingerprint': self.digest, 'person_ids': [1, 2],
            'assignments': self.assignments, 'person_groups': [],
            'grouping_reviewed': True,
        }), encoding='utf-8')
        self.messages = []
        for change in (
            patch.object(bridge, 'ROOT', self.root),
            patch.object(bridge, 'emit', side_effect=lambda kind, **v: self.messages.append({'type': kind, **v})),
            patch('ollama_local.release_model', side_effect=AssertionError('Review unloaded Ollama')),
            patch('video_tools.prepare_timeline', side_effect=AssertionError('Review prepared video')),
            patch('scan_compat.validate_processing', side_effect=AssertionError('Review validated timeline')),
        ):
            change.start()
            self.addCleanup(change.stop)

    def test_missing_processing_video_keeps_cards_and_assignments(self):
        before = self.draft.read_bytes()
        bridge.scan_video(str(self.video))
        self.assertEqual(self.messages[-1]['type'], 'characters')
        self.assertEqual(len(self.messages[-1]['characters']), 2)
        self.assertEqual(self.messages[-1]['assignments'], self.assignments)
        self.assertEqual(self.draft.read_bytes(), before)

    def test_old_choices_recovered_without_overwriting_existing_draft(self):
        jobs = self.root / 'data/desktop-jobs'
        jobs.mkdir(parents=True)
        (jobs / 'old.json').write_text(json.dumps({
            'action': 'swap', 'video': str(self.video), 'analysis': str(self.analysis),
            'assignments': self.assignments, 'person_groups': [],
        }), encoding='utf-8')
        self.draft.unlink()
        bridge.scan_video(str(self.video))
        self.assertEqual(self.messages[-1]['assignments'], self.assignments)
        self.assertTrue(self.draft.is_file())
        changed = json.loads(self.draft.read_text(encoding='utf-8'))
        changed['assignments'] = []
        changed['grouping_reviewed'] = True
        self.draft.write_text(json.dumps(changed), encoding='utf-8')
        bridge.scan_video(str(self.video))
        self.assertEqual(self.messages[-1]['assignments'], [])

    def test_another_source_is_rejected_without_losing_draft(self):
        before = self.draft.read_bytes()
        invalid = copy.deepcopy(self.scan)
        invalid['input_fingerprint'] = 'different-source'
        self.analysis.write_text(json.dumps(invalid), encoding='utf-8')
        with self.assertRaises(RuntimeError):
            bridge.scan_video(str(self.video))
        self.assertEqual(self.draft.read_bytes(), before)
        self.assertEqual(self.messages, [])

    def test_cached_group_lookup_does_not_open_processing_video(self):
        from auto_groups import discover_groups
        with patch('auto_groups.cv2.VideoCapture', side_effect=AssertionError('Review decoded video')):
            self.assertEqual(discover_groups(self.analysis, cached_only=True), {'groups': []})


if __name__ == '__main__':
    unittest.main()
