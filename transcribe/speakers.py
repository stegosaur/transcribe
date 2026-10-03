"""Match aligned words to diarization turns and split segments at speaker changes."""

import bisect

from .asr import Segment

SENTENCE_END = (".", "?", "!", "…")
SNAP = 2  # move a speaker change up to this many words to land on a sentence end


def _speaker_for(start: float, end: float, turns, starts) -> str:
    """Speaker with the most overlap with [start, end]; nearest turn if none overlap."""
    best, best_overlap = None, 0.0
    i = max(0, bisect.bisect_right(starts, end) - 1)
    # Turns are sorted by start; walk back over any that could still overlap.
    # Exclusive turns don't overlap each other, so once one ends well before
    # the word, every earlier one does too.
    while i >= 0:
        t = turns[i]
        if t.end < start - 30:
            break
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
        i -= 1
    if best is not None:
        return best
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker if turns else ""


def _snap_boundaries(words, labels):
    """Diarization edges are usually a word or two off. If a speaker change sits
    right next to the end of a sentence, move it onto the sentence end."""
    labels = list(labels)
    i = 1
    while i < len(labels):
        if labels[i] != labels[i - 1]:
            if words[i - 1]["word"].endswith(SENTENCE_END):
                i += 1
                continue
            for d in range(1, SNAP + 1):
                # sentence ends a little later: the new speaker started too early
                j = i + d
                if j <= len(labels) and words[j - 1]["word"].endswith(SENTENCE_END) \
                        and all(lbl == labels[i] for lbl in labels[i:j]):
                    for k in range(i, j):
                        labels[k] = labels[i - 1]
                    break
                # sentence ended a little earlier: the new speaker started too late
                j = i - d
                if j >= 1 and words[j - 1]["word"].endswith(SENTENCE_END) \
                        and all(lbl == labels[i - 1] for lbl in labels[j:i]):
                    for k in range(j, i):
                        labels[k] = labels[i]
                    break
        i += 1
    return labels


def assign_speakers(segments: list[Segment], turns) -> list[Segment]:
    turns = sorted(turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    # Label and snap over the whole word stream at once: speaker changes often
    # coincide with Whisper segment boundaries, and a stray "Oh," or "A" at the
    # end of one segment belongs with the words that follow it.
    words, seg_of = [], []
    for si, seg in enumerate(segments):
        if not seg.words:
            seg.words = [{"word": seg.text, "start": seg.start, "end": seg.end}]
        words.extend(seg.words)
        seg_of.extend([si] * len(seg.words))
    if not words:
        return []
    labels = [_speaker_for(w["start"], max(w["end"], w["start"] + 0.01), turns, starts) for w in words]
    labels = _snap_boundaries(words, labels)

    out: list[Segment] = []
    run = [words[0]]
    for i in range(1, len(words)):
        if labels[i] != labels[i - 1] or seg_of[i] != seg_of[i - 1]:
            out.append(_make(run, labels[i - 1]))
            run = []
        run.append(words[i])
    out.append(_make(run, labels[-1]))
    return out


def _make(words, speaker) -> Segment:
    return Segment(start=words[0]["start"], end=words[-1]["end"],
                   text=" ".join(w["word"] for w in words), speaker=speaker, words=words)
