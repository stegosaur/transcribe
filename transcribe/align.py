"""Word-level timestamps by CTC forced alignment (the WhisperX approach).

Whisper's own segment timestamps are only accurate to about a second, which is
not enough to tell where one speaker stops and the next starts. We run
torchaudio's multilingual MMS forced-alignment wav2vec2 model over the audio
on the GPU and align each Whisper segment's words to it. Each word then gets
its own start/end time and can be matched to a diarization turn.

MMS_FA works on romanised text (a-z and '). Accented Latin letters are reduced
to their base letter. Words with nothing alignable (numbers, scripts other than
Latin) get times interpolated from their neighbours.
"""

import re
import unicodedata
import warnings
from typing import Callable, Optional

import numpy as np
import torch

from .asr import Segment
from .audio import SAMPLE_RATE

warnings.filterwarnings("ignore", message=".*forced_align.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecat.*")

BLOCK = 30.0     # seconds of audio per wav2vec2 forward pass
MARGIN = 0.5     # slack around Whisper's segment boundaries, in seconds


def _normalise(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    w = w.replace("’", "'")
    return re.sub(r"[^a-z']", "", w)


class Aligner:
    def __init__(self, device: torch.device):
        from torchaudio.pipelines import MMS_FA

        self.device = device
        self.model = MMS_FA.get_model(with_star=False).to(device).eval()
        if device.type == "cuda":
            self.model = self.model.half()
        self.dictionary = MMS_FA.get_dict(star=None)
        self.stride = 320 / SAMPLE_RATE   # seconds per emission frame

    @torch.inference_mode()
    def _emissions(self, audio: np.ndarray, progress=None) -> torch.Tensor:
        """Frame-level log-probs for the whole file, computed in 30 s blocks."""
        block = int(BLOCK * SAMPLE_RATE)
        frames_per_block = block // 320
        out = []
        starts = list(range(0, len(audio), block))
        for n, i in enumerate(starts):
            wav = torch.from_numpy(audio[i:i + block]).to(self.device)
            if wav.numel() < 400:
                break
            wav = wav.unsqueeze(0).to(next(self.model.parameters()).dtype)
            em, _ = self.model(wav)
            em = torch.log_softmax(em.float(), dim=-1)[0].cpu()
            if n < len(starts) - 1:
                em = em[:frames_per_block]
                if em.shape[0] < frames_per_block:  # keep frame index == time / stride
                    em = torch.cat([em, em[-1:].expand(frames_per_block - em.shape[0], -1)])
            out.append(em)
            if progress:
                progress(n + 1, len(starts))
        return torch.cat(out) if out else torch.zeros(0, len(self.dictionary))

    def align(self, audio: np.ndarray, segments: list[Segment],
              progress: Optional[Callable[[int, int], None]] = None) -> list[Segment]:
        import torchaudio.functional as F

        emissions = self._emissions(audio, progress)
        n_frames = emissions.shape[0]

        for seg in segments:
            words = seg.text.split()
            if not words:
                continue
            f0 = max(0, int((seg.start - MARGIN) / self.stride))
            f1 = min(n_frames, int((seg.end + MARGIN) / self.stride) + 1)
            norm = [_normalise(w) for w in words]
            tokens = [self.dictionary[c] for w in norm for c in w if c in self.dictionary]
            times: list[Optional[tuple[float, float]]] = [None] * len(words)

            if tokens and f1 - f0 > len(tokens):
                try:
                    em = emissions[f0:f1].unsqueeze(0)
                    targets = torch.tensor([tokens], dtype=torch.int32)
                    ali, scores = F.forced_align(em, targets, blank=0)
                    spans = F.merge_tokens(ali[0], scores[0].exp())
                    k = 0
                    for wi, w in enumerate(norm):
                        n = sum(1 for c in w if c in self.dictionary)
                        if n:
                            ws = spans[k:k + n]
                            times[wi] = ((f0 + ws[0].start) * self.stride, (f0 + ws[-1].end) * self.stride)
                            k += n
                except Exception:
                    pass  # leave times as None -> interpolated below

            seg.words = [{"word": w, "start": t[0] if t else None, "end": t[1] if t else None}
                         for w, t in zip(words, times)]
            _interpolate(seg)
            seg.start = round(seg.words[0]["start"], 2)
            seg.end = round(max(seg.words[-1]["end"], seg.start), 2)
        return segments


def _interpolate(seg: Segment):
    """Give unaligned words times: share the gap between aligned neighbours
    by character count, or the whole segment if nothing aligned."""
    words = seg.words
    i = 0
    while i < len(words):
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j]["start"] is None:
            j += 1
        lo = words[i - 1]["end"] if i > 0 else seg.start
        hi = words[j]["start"] if j < len(words) else seg.end
        hi = max(hi, lo)
        lengths = [max(1, len(words[k]["word"])) for k in range(i, j)]
        total = sum(lengths)
        t = lo
        for k, n in zip(range(i, j), lengths):
            words[k]["start"] = t
            t += (hi - lo) * n / total
            words[k]["end"] = t
        i = j
    for w in words:
        w["start"], w["end"] = round(w["start"], 2), round(w["end"], 2)
