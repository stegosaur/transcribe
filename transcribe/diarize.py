"""Speaker diarization with pyannote.audio on the GPU."""

import contextlib
import io
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .audio import SAMPLE_RATE

COMMUNITY = "pyannote/speaker-diarization-community-1"
LEGACY = "pyannote/speaker-diarization-3.1"

# pyannote/speaker-diarization-3.1's published config, rebuilt by hand. Under
# pyannote.audio 4 its config is missing a PLDA entry, so loading it the normal
# way silently falls back to downloading PLDA weights from the (separately
# gated) community-1 repo. Agglomerative clustering doesn't use PLDA at all.
_LEGACY_PARAMS = {
    "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7045654963945799},
    "segmentation": {"min_duration_off": 0.0},
}


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def _load_legacy(token):
    import pyannote.audio.pipelines.speaker_diarization as sd

    real_get_plda = sd.get_plda
    sd.get_plda = lambda *a, **k: None
    try:
        pipeline = sd.SpeakerDiarization(
            segmentation="pyannote/segmentation-3.0",
            embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
            embedding_exclude_overlap=True,
            clustering="AgglomerativeClustering",
            token=token,
        )
    finally:
        sd.get_plda = real_get_plda
    pipeline.instantiate(_LEGACY_PARAMS)
    return pipeline


class Diarizer:
    def __init__(self, device: torch.device, model: str = "auto", token: Optional[str] = None,
                 batch_size: int = 32, log: Callable[[str], None] = print):
        from pyannote.audio import Pipeline

        self.model = None
        errors = []
        candidates = [COMMUNITY, LEGACY] if model == "auto" else [model]
        for name in candidates:
            try:
                if name == LEGACY:
                    pipeline = _load_legacy(token)
                else:
                    # pyannote prints (rather than raises) its "gated repo" help text.
                    with contextlib.redirect_stdout(io.StringIO()):
                        pipeline = Pipeline.from_pretrained(name, token=token)
                if pipeline is None:
                    raise RuntimeError("Pipeline.from_pretrained returned None")
                self.model = name
                break
            except Exception as e:  # gated repo not accepted, offline, ...
                errors.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        if self.model is None:
            raise RuntimeError(
                "Could not load a pyannote diarization model. Make sure you have a Hugging Face "
                "token (`hf auth login` or HF_TOKEN) and have accepted the model terms at "
                f"https://huggingface.co/{COMMUNITY} (or https://huggingface.co/{LEGACY} plus "
                "https://huggingface.co/pyannote/segmentation-3.0).\n  " + "\n  ".join(errors))
        if model == "auto" and self.model != COMMUNITY:
            log(f"note: {COMMUNITY} is not accessible, using {self.model} instead "
                f"(accept the terms at https://huggingface.co/{COMMUNITY} for better accuracy)")

        # Batch the segmentation and embedding models: the defaults (1) leave
        # the GPU mostly idle.
        if hasattr(pipeline, "embedding_batch_size"):
            pipeline.embedding_batch_size = batch_size
            pipeline.segmentation_batch_size = batch_size
        pipeline.to(device)
        self.pipeline = pipeline

    def __call__(self, audio: np.ndarray, num_speakers=None, min_speakers=None, max_speakers=None,
                 progress: Optional[Callable[[str, float], None]] = None) -> list[Turn]:
        waveform = torch.from_numpy(audio).unsqueeze(0)

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            if progress is not None:
                frac = (completed / total) if total and completed is not None else None
                progress(step_name, frac)

        kwargs = {k: v for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                        max_speakers=max_speakers).items() if v}
        with torch.inference_mode():
            out = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, hook=hook, **kwargs)

        # "exclusive" diarization assigns every instant to at most one speaker,
        # which is what we want for a transcript (pyannote resolves overlapping
        # speech to the most likely speaker).
        annotation = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
        turns = [Turn(float(seg.start), float(seg.end), str(spk))
                 for seg, _, spk in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: (t.start, t.end))
        return turns
