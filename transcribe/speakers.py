"""Match aligned words to diarization turns and split segments at speaker changes.

Each word simply takes the speaker whose turn overlaps it most. Smarter-looking
clean-ups were measured against the AMI meeting corpus (word-level speaker
accuracy on four test meetings, headset mix and single distant microphone):
snapping speaker changes onto sentence ends, majority voting per sentence, and
shifting turn boundaries all scored worse than leaving the labels alone. Short
interjections ("yeah", "right") in the middle of someone else's sentence are
common, and every smoothing rule ends up swallowing them.
"""

import bisect

from .asr import Segment


class _Index:
    """Overlap queries against a list of turns."""

    def __init__(self, turns):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]
        self.longest = max((t.end - t.start for t in self.turns), default=0.0)

    def overlaps(self, start: float, end: float) -> dict:
        """{speaker: seconds of overlap with [start, end]}"""
        out: dict = {}
        i = bisect.bisect_right(self.starts, end) - 1
        # Walk back while a turn could still reach `start`.
        while i >= 0 and self.turns[i].start >= start - self.longest:
            t = self.turns[i]
            ov = min(end, t.end) - max(start, t.start)
            if ov > 0:
                out[t.speaker] = out.get(t.speaker, 0.0) + ov
            i -= 1
        return out

    def nearest(self, start: float, end: float) -> str:
        if not self.turns:
            return ""
        mid = (start + end) / 2
        return min(self.turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


def _label(word, index: _Index) -> str:
    start, end = word["start"], max(word["end"], word["start"] + 0.01)
    ov = index.overlaps(start, end)
    return max(ov, key=ov.get) if ov else index.nearest(start, end)


def assign_speakers(segments: list[Segment], turns) -> list[Segment]:
    index = _Index(turns)
    out: list[Segment] = []
    for seg in segments:
        if not seg.words:
            seg.words = [{"word": seg.text, "start": seg.start, "end": seg.end}]
        labels = [_label(w, index) for w in seg.words]
        run = [seg.words[0]]
        for w, prev, label in zip(seg.words[1:], labels, labels[1:]):
            if label != prev:
                out.append(_make(run, prev))
                run = []
            run.append(w)
        out.append(_make(run, labels[-1]))
    return out


def _make(words, speaker) -> Segment:
    return Segment(start=words[0]["start"], end=words[-1]["end"],
                   text=" ".join(w["word"] for w in words), speaker=speaker, words=words)
