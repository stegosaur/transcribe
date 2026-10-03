"""Audio decoding via ffmpeg: any format ffmpeg reads (wav, mp3, m4a, flac,
ogg, even video files) -> 16 kHz mono float32, which both Whisper and
pyannote expect."""

import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (sudo apt install ffmpeg)")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='replace').strip()}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"{path} contains no audio")
    return audio.copy()
