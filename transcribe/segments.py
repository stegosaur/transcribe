"""Cut a file into windows Whisper can transcribe in one pass (<= 30 s).

Windows are cut in pauses: between speech turns when diarization is
available, otherwise at the quietest point. Long stretches with no speech at
all are skipped, which also stops Whisper from hallucinating text in silence.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .audio import SAMPLE_RATE

MAX_WINDOW = 28.0   # seconds; leaves headroom inside Whisper's 30 s input
MIN_WINDOW = 15.0   # don't cut earlier than this unless there's a long silence
SILENCE_SKIP = 2.0  # gaps between speech longer than this end a window
PAD = 0.2           # context added around speech regions


@dataclass
class Window:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _quietest(audio: np.ndarray, lo: float, hi: float) -> float:
    """Middle of the quietest 200 ms in [lo, hi]."""
    win, hop = int(0.2 * SAMPLE_RATE), int(0.05 * SAMPLE_RATE)
    seg = audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
    if len(seg) < win:
        return (lo + hi) / 2
    frames = np.lib.stride_tricks.sliding_window_view(seg, win)[::hop]
    energy = (frames.astype(np.float64) ** 2).mean(axis=1)
    return lo + (int(np.argmin(energy)) * hop + win / 2) / SAMPLE_RATE


def _speech_regions(turns, total: float) -> list[tuple[float, float]]:
    """Union of all speakers' turns, with short gaps bridged."""
    regions: list[list[float]] = []
    for t in sorted(turns, key=lambda t: t.start):
        s, e = max(0.0, t.start - PAD), min(total, t.end + PAD)
        if regions and s - regions[-1][1] < SILENCE_SKIP:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])
    return [(s, e) for s, e in regions]


def make_windows(audio: np.ndarray, turns: Optional[list] = None) -> list[Window]:
    total = len(audio) / SAMPLE_RATE
    regions = _speech_regions(turns, total) if turns else [(0.0, total)]
    # Pause points we'd like to cut at: gaps between consecutive turns.
    gaps = []
    if turns:
        ordered = sorted(turns, key=lambda t: t.start)
        reach = ordered[0].end
        for t in ordered[1:]:
            if t.start > reach:
                gaps.append((reach, t.start))
            reach = max(reach, t.end)

    windows: list[Window] = []
    for rs, re_ in regions:
        start = rs
        while re_ - start > MAX_WINDOW:
            lo, hi = start + MIN_WINDOW, start + MAX_WINDOW
            inside = [g for g in gaps if lo <= (g[0] + g[1]) / 2 <= hi]
            if inside:
                g = max(inside, key=lambda g: (g[1] - g[0], g[0]))  # widest pause
                cut = (g[0] + g[1]) / 2
            else:
                cut = _quietest(audio, lo, hi)
            windows.append(Window(start, cut))
            start = cut
        windows.append(Window(start, re_))
    return [w for w in windows if w.duration >= 0.2]
