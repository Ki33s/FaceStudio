"""Intervals whose original frames must pass through without face replacement."""
import math
from bisect import bisect_right


def normalize_ranges(ranges, duration=None):
    result = []
    for interval in ranges or []:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError('Неверный промежуток без замены лиц.')
        a, b = map(float, interval)
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a:
            raise ValueError('Проверьте начало и конец промежутка без замены лиц.')
        if duration is not None:
            b = min(b, duration)
        if b > a:
            result.append([a, b])
    merged = []
    for a, b in sorted(result):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


class ProtectedFrames:
    def __init__(self, job):
        fps = float(job['fps'])
        self.ranges = normalize_ranges(job.get('excluded_ranges'))
        self.frames = [(math.ceil(a * fps - 1e-8), math.ceil(b * fps - 1e-8))
                       for a, b in self.ranges]
        self.starts = [a for a, b in self.frames]

    def contains(self, index):
        item = bisect_right(self.starts, index) - 1
        return item >= 0 and index < self.frames[item][1]

    def count(self, start, end):
        return sum(max(0, min(end, b) - max(start, a)) for a, b in self.frames)
