"""Loads the models once and runs audio files through
decode -> diarize -> transcribe -> align words -> assign speakers."""

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import gpu  # noqa: F401  (sets ROCm env vars before torch loads)
from .asr import Segment, Whisper
from .audio import SAMPLE_RATE, load_audio
from .segments import make_windows
from .speakers import assign_speakers

ProgressFn = Callable[[str, Optional[float]], None]


@dataclass
class Transcript:
    file: str
    duration: float
    language: str
    speakers: list[str]
    segments: list[Segment]
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [asdict(s) for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(d["file"], d["duration"], d["language"], d["speakers"],
                   [Segment(**s) for s in d["segments"]], d.get("timings", {}))


def hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return path.read_text().strip() if path.exists() else None


class Engine:
    def __init__(self, whisper_model: str = "turbo", diarization_model: str = "auto",
                 device: str = "auto", batch_size: int = 16, hf_token_value: Optional[str] = None,
                 diarization: bool = True, log: Callable[[str], None] = print):
        self.device = gpu.pick_device(device)
        self.log = log
        log(f"device: {gpu.describe(self.device)}")
        t = time.time()
        self.whisper = Whisper(self.device, whisper_model, batch_size)
        log(f"loaded {self.whisper.model_id} in {time.time() - t:.1f}s")
        self.aligner = None
        self.diarizer = None
        if diarization:
            from .align import Aligner
            t = time.time()
            self.aligner = Aligner(self.device)
            log(f"loaded MMS forced-alignment model in {time.time() - t:.1f}s")
            from .diarize import Diarizer
            t = time.time()
            self.diarizer = Diarizer(self.device, diarization_model, hf_token(hf_token_value), log=log)
            log(f"loaded {self.diarizer.model} in {time.time() - t:.1f}s")
        # One GPU, one job at a time; the web UI queues behind this.
        self.lock = threading.Lock()

    def run(self, path: str, num_speakers: Optional[int] = None, min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None, language: Optional[str] = None, task: str = "transcribe",
            diarize: bool = True, progress: Optional[ProgressFn] = None,
            display_name: Optional[str] = None) -> Transcript:
        progress = progress or (lambda stage, frac: None)
        timings = {}
        with self.lock:
            t = time.time()
            progress("decoding audio", None)
            audio = load_audio(path)
            duration = len(audio) / SAMPLE_RATE
            timings["decode"] = time.time() - t

            diarize = diarize and self.diarizer is not None
            turns = None
            if diarize:
                t = time.time()
                progress("finding speakers", 0.0)
                turns = self.diarizer(
                    audio, num_speakers, min_speakers, max_speakers,
                    progress=lambda step, frac: progress(f"finding speakers ({step.replace('_', ' ')})", frac))
                timings["diarize"] = time.time() - t

            t = time.time()
            progress("transcribing", 0.0)
            segments, lang = self.whisper.transcribe(
                audio, make_windows(audio, turns), language=language or None, task=task,
                progress=lambda done, total: progress("transcribing", done / total))
            timings["transcribe"] = time.time() - t

            names: dict[str, str] = {}
            if diarize and turns:
                t = time.time()
                progress("aligning words", 0.0)
                self.aligner.align(audio, segments,
                                   progress=lambda done, total: progress("aligning words", done / total))
                segments = assign_speakers(segments, turns)
                timings["align"] = time.time() - t
                # "Speaker 1", "Speaker 2", ... in order of first appearance.
                for s in segments:
                    s.speaker = names.setdefault(s.speaker, f"Speaker {len(names) + 1}")

        timings = {k: round(v, 2) for k, v in timings.items()}
        return Transcript(display_name or Path(path).name, round(duration, 2), lang,
                          list(names.values()), segments, timings)
