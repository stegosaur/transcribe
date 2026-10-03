"""Transcript -> txt / md / srt / vtt / json."""

import json
from typing import Optional

FORMATS = ("txt", "md", "srt", "vtt", "json")


def _clock(t: float, sep: str = ",", hours: bool = True) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}" if hours else f"{h * 60 + m:02d}:{s:02d}"


def _stamp(t: float) -> str:
    s = int(t)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def _paragraphs(segments):
    """Consecutive segments from the same speaker joined into one paragraph
    (capped at ~30 s when there are no speaker labels to break on)."""
    paras = []
    for s in segments:
        if paras and paras[-1]["speaker"] == s.speaker and s.start - paras[-1]["end"] < 3.0 \
                and (s.speaker or s.end - paras[-1]["start"] < 30.0):
            paras[-1]["text"] += " " + s.text
            paras[-1]["end"] = s.end
        else:
            paras.append({"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text})
    return paras


def _named(transcript, names: Optional[dict]):
    if not names:
        return transcript.segments
    from dataclasses import replace
    return [replace(s, speaker=names.get(s.speaker) or s.speaker) for s in transcript.segments]


def render(transcript, fmt: str, names: Optional[dict] = None) -> str:
    segs = _named(transcript, names)
    if fmt == "txt":
        lines = []
        for p in _paragraphs(segs):
            who = f"{p['speaker']}: " if p["speaker"] else ""
            lines.append(f"[{_stamp(p['start'])}] {who}{p['text']}")
        return "\n\n".join(lines) + "\n"
    if fmt == "md":
        out = [f"# {transcript.file}", ""]
        for p in _paragraphs(segs):
            who = f"**{p['speaker']}** " if p["speaker"] else ""
            out += [f"{who}`{_stamp(p['start'])}`  ", p["text"], ""]
        return "\n".join(out)
    if fmt == "srt":
        cues = []
        for i, s in enumerate(segs, 1):
            who = f"[{s.speaker}] " if s.speaker else ""
            cues.append(f"{i}\n{_clock(s.start)} --> {_clock(s.end)}\n{who}{s.text}\n")
        return "\n".join(cues)
    if fmt == "vtt":
        cues = ["WEBVTT", ""]
        for s in segs:
            who = f"<v {s.speaker}>" if s.speaker else ""
            cues += [f"{_clock(s.start, '.')} --> {_clock(s.end, '.')}", f"{who}{s.text}", ""]
        return "\n".join(cues)
    if fmt == "json":
        d = transcript.to_dict()
        if names:
            d["segments"] = [dict(seg, speaker=names.get(seg["speaker"]) or seg["speaker"]) for seg in d["segments"]]
            d["speakers"] = [names.get(s) or s for s in d["speakers"]]
        return json.dumps(d, indent=2, ensure_ascii=False) + "\n"
    raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
