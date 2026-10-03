# transcribe

Turn a WAV or MP3 (or anything ffmpeg can read) into a text transcript with
each speaker labelled. Runs entirely locally and is tuned for AMD Radeon GPUs
with ROCm (developed on an RX 7900 XTX).

```
[00:01:46] Speaker 4: Oh, hi, Joe. Hi, Ben. Did you get that message to call home?

[00:02:01] Speaker 3: Well, forget it, Ben. Think what a comfort he's gonna be in your old age.
```

There's a command-line tool and a local web UI. The web UI lets you upload
files, watch progress, play the audio with the current line highlighted, click
any line to jump to it, rename speakers, search, and download the transcript
as TXT, SRT, VTT, Markdown, or JSON.

## How it works

1. **Decode**: ffmpeg converts the input to 16 kHz mono.
2. **Diarize**: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
   works out who speaks when.
3. **Transcribe**: [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
   (large-v3-turbo by default) runs over 30-second windows. The windows are cut
   in pauses between speakers, and long silences are skipped.
4. **Align**: a wav2vec2 forced-alignment model (torchaudio's MMS_FA) gives
   every word its own start and end time.
5. **Assign**: each word goes to the speaker who was talking at that moment.
   Speaker changes that land within a word or two of a sentence end snap onto
   that sentence end.

All of the models run on the GPU.

### AMD / ROCm specifics

- **Whisper runs on PyTorch + transformers, not faster-whisper.** CTranslate2,
  the engine behind faster-whisper, has no ROCm backend, so on an AMD card it
  falls back to the CPU.
- **Batching is done by hand.** transformers' `pipeline()` was about 10x slower
  on the 7900 XTX for the same model. Its word-timestamp mode also forces eager
  attention, which runs out of VRAM.
- **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** turns on flash attention for
  RDNA3 (gfx1100). At batch size 32 it made Whisper about 7x faster.
- **`MIOPEN_FIND_MODE=FAST`** stops MIOpen from spending minutes searching for
  kernels on the first diarization run.
- **torchcodec comes from PyTorch's CPU index.** The PyPI wheel links against
  CUDA's `libnvrtc` and won't load on ROCm. Audio decoding goes through ffmpeg
  anyway.

All of this is set in `transcribe/gpu.py`, so you don't need to export
anything yourself.

Speed on an RX 7900 XTX: a 30-minute, 12-speaker radio episode took about 41 s
end to end, roughly 43x real-time. Diarization is the slowest stage.

## Install

You need Linux x86_64 with ROCm 7.x drivers, `ffmpeg`, and
[uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:stegosaur/transcribe.git
cd transcribe
uv sync
```

`uv sync` installs the ROCm build of PyTorch from download.pytorch.org (see
`[tool.uv.sources]` in `pyproject.toml`).

### Hugging Face access (one-time, needed for speaker separation)

The pyannote models are free but gated:

1. Log in to Hugging Face: `uv run hf auth login`, or set `HF_TOKEN`.
2. Accept the terms on
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
   This is the newest and most accurate pipeline, and it's the default.

If community-1 isn't accessible, the app falls back to
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
That fallback also needs its terms accepted, along with those of
[pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

The first run downloads about 3 GB of models: Whisper turbo, the MMS aligner,
and pyannote.

## Usage

### Web UI

```sh
uv run transcribe-web            # http://127.0.0.1:8765
uv run transcribe-web --host 0.0.0.0 --port 8765   # reachable from your LAN
```

Uploads and finished transcripts are stored in `~/.local/share/transcribe/jobs`
(change this with `--data-dir`) and stay in the sidebar history until you delete
them. Jobs run one at a time on the GPU, and the rest wait in a queue.

### Command line

```sh
uv run transcribe meeting.mp3                      # writes meeting.txt next to it
uv run transcribe interview.wav -f txt,srt,json    # several formats
uv run transcribe *.mp3 -o transcripts/ -f all
uv run transcribe call.wav -n 2                    # you know there are exactly 2 speakers
uv run transcribe lecture.mp3 --no-diarize         # one speaker, skip diarization
uv run transcribe entrevista.mp3 --translate       # translate to English
uv run transcribe podcast.mp3 --stdout
```

| Option | Default | |
|---|---|---|
| `-f, --format` | `txt` | `txt`, `md`, `srt`, `vtt`, `json`, or `all` |
| `-o, --output-dir` | next to input | |
| `-m, --model` | `turbo` | `turbo`, `large-v3` (slower, slightly more accurate), `medium`, `small`, `distil-large-v3` (English only), or any Hugging Face Whisper id |
| `-l, --language` | auto | e.g. `en`, `es`, `de`. Detected once per file |
| `-n, --num-speakers` | auto | exact speaker count, if known (improves diarization) |
| `--min-speakers / --max-speakers` | | bounds instead of an exact count |
| `--no-diarize` | | skip speaker separation |
| `--translate` | | output English, whatever language was spoken |
| `--batch-size` | `16` | Whisper batch size; lower it if you run out of VRAM |
| `--device` | `auto` | `cuda` (ROCm GPUs show up as cuda) or `cpu` |

The JSON output includes word-level timestamps.

## Running the web UI as a service

```ini
# ~/.config/systemd/user/transcribe.service
[Unit]
Description=transcribe web UI

[Service]
WorkingDirectory=%h/transcribe
ExecStart=%h/.local/bin/uv run transcribe-web
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload && systemctl --user enable --now transcribe
```

## Troubleshooting

- **Out of GPU memory**: something else may be holding VRAM, such as Ollama or
  ComfyUI (check `rocm-smi --showpids`). Stop it, or lower `--batch-size`. The
  models themselves need about 4 GB at the default batch size.
- **"Could not load a pyannote diarization model"**: see
  [Hugging Face access](#hugging-face-access-one-time-needed-for-speaker-separation).
- **Two people with similar voices are merged into one speaker**: pass the
  real count with `-n`.
- **Non-Latin scripts** (Chinese, Japanese, Russian, ...) still transcribe
  fine. Word timings are estimated within each segment instead of
  force-aligned, so speaker changes are a little less precise.
