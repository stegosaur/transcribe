"""Batched Whisper inference with Hugging Face transformers on ROCm.

faster-whisper/CTranslate2 has no ROCm GPU backend, so on AMD it would run on
the CPU. Plain transformers + PyTorch-ROCm runs on the GPU in fp16 with SDPA
attention. We batch the model ourselves rather than going through
transformers' `pipeline()`: in testing on an RX 7900 XTX the pipeline was
~10x slower for the same model, and its word-timestamp mode forces eager
attention and runs out of VRAM at moderate batch sizes.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .audio import SAMPLE_RATE
from .segments import Window

MODELS = {
    "turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small",
    "distil-large-v3": "distil-whisper/distil-large-v3",  # English only
}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""
    words: list = field(default_factory=list)  # [{"word", "start", "end"}], filled in by alignment


class Whisper:
    def __init__(self, device: torch.device, model: str = "turbo", batch_size: int = 16):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.model_id = MODELS.get(model, model)
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.batch_size = batch_size
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self.dtype, attn_implementation="sdpa",
        ).to(device).eval()

    def _features(self, clips: list[np.ndarray]) -> torch.Tensor:
        feats = self.processor.feature_extractor(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return feats.input_features.to(self.device, self.dtype)

    @torch.inference_mode()
    def detect_language(self, clips: list[np.ndarray]) -> str:
        """Majority vote over a few of the longest clips. Detecting once per file
        and then fixing the language is more reliable than letting Whisper guess
        separately for every window."""
        ids = self.model.detect_language(self._features(clips)).tolist()
        token = Counter(self.processor.tokenizer.convert_ids_to_tokens(ids)).most_common(1)[0][0]
        return token.strip("<|>")

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, windows: list[Window], language: Optional[str] = None,
                   task: str = "transcribe",
                   progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[Segment], str]:
        clips = [audio[int(w.start * SAMPLE_RATE):int(w.end * SAMPLE_RATE)] for w in windows]
        if not clips:
            return [], language or ""

        if not language:
            longest = sorted(range(len(clips)), key=lambda i: -len(clips[i]))[:5]
            language = self.detect_language([clips[i] for i in longest])

        # Every clip is padded to 30 s for the encoder, but decoding time grows
        # with the number of tokens, so batching similar lengths together
        # wastes fewer decoder steps on finished sequences.
        order = sorted(range(len(windows)), key=lambda i: -windows[i].duration)
        segments: list[Segment] = []
        done = 0
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            feats = self._features([clips[i] for i in idx])
            out = self.model.generate(feats, language=language, task=task, return_timestamps=True)
            for i, seq in zip(idx, out):
                segments.extend(self._decode(seq, windows[i]))
            done += len(idx)
            if progress:
                progress(done, len(windows))

        segments.sort(key=lambda s: (s.start, s.end))
        return segments, language

    def _decode(self, seq: torch.Tensor, window: Window) -> list[Segment]:
        decoded = self.processor.tokenizer.decode(seq, output_offsets=True)
        pieces = decoded.get("offsets") or []
        if not pieces:
            text = self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
            return [Segment(window.start, window.end, text)] if text else []
        out = []
        for p in pieces:
            text = p["text"].strip()
            if not text:
                continue
            s, e = p["timestamp"]
            start = min(window.start + (s or 0.0), window.end)
            end = min(window.start + e, window.end) if e is not None else window.end
            out.append(Segment(round(start, 2), round(max(end, start), 2), text))
        return out
