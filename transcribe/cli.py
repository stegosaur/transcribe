"""Command line: transcribe FILE [FILE ...]"""

import argparse
import sys
import time
from pathlib import Path

from .formats import FORMATS, render


def _progress_printer():
    last = {"stage": None}

    def progress(stage, frac):
        line = f"  {stage}" + (f" {frac * 100:5.1f}%" if frac is not None else "")
        if stage != last["stage"] and last["stage"] is not None:
            sys.stderr.write("\n")
        sys.stderr.write("\r" + line.ljust(60))
        sys.stderr.flush()
        last["stage"] = stage

    return progress


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe WAV/MP3 (or anything ffmpeg can read) into a speaker-labelled transcript.")
    p.add_argument("files", nargs="+", help="audio files")
    p.add_argument("-o", "--output-dir", help="where to write transcripts (default: next to each input file)")
    p.add_argument("-f", "--format", default="txt",
                   help=f"comma-separated output formats: {','.join(FORMATS)} or 'all' (default: txt)")
    p.add_argument("-m", "--model", default="turbo",
                   help="Whisper model: turbo (default), large-v3, medium, small, distil-large-v3, "
                        "or any Hugging Face Whisper model id")
    p.add_argument("-l", "--language", help="language code, e.g. en, es, de (default: auto-detect)")
    p.add_argument("--translate", action="store_true", help="translate speech to English instead of transcribing")
    p.add_argument("-n", "--num-speakers", type=int, help="exact number of speakers, if known")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--no-diarize", action="store_true", help="skip speaker separation")
    p.add_argument("--diarization-model", default="auto",
                   help="pyannote pipeline (default: community-1, falling back to 3.1)")
    p.add_argument("--batch-size", type=int, default=16, help="Whisper batch size (default: 16)")
    p.add_argument("--device", default="auto", help="auto, cuda (ROCm GPUs show up as cuda), or cpu")
    p.add_argument("--hf-token", help="Hugging Face token (default: $HF_TOKEN or `hf auth login`)")
    p.add_argument("--stdout", action="store_true", help="print the transcript instead of writing files")
    args = p.parse_args(argv)

    formats = list(FORMATS) if args.format == "all" else [f.strip() for f in args.format.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        p.error(f"unknown format(s): {', '.join(bad)}")
    for f in args.files:
        if not Path(f).is_file():
            p.error(f"no such file: {f}")

    from .engine import Engine  # heavy imports after argument parsing

    log = lambda msg: print(msg, file=sys.stderr)
    engine = Engine(args.model, args.diarization_model, args.device, args.batch_size,
                    args.hf_token, diarization=not args.no_diarize, log=log)

    for path in args.files:
        log(f"\n{path}")
        t = time.time()
        result = engine.run(path, args.num_speakers, args.min_speakers, args.max_speakers,
                            args.language, "translate" if args.translate else "transcribe",
                            diarize=not args.no_diarize, progress=_progress_printer())
        elapsed = time.time() - t
        sys.stderr.write("\n")
        log(f"  {result.duration / 60:.1f} min of audio in {elapsed:.1f}s "
            f"({result.duration / max(elapsed, 1e-6):.0f}x real-time), language={result.language}, "
            f"speakers={len(result.speakers)}")

        if args.stdout:
            for fmt in formats:
                sys.stdout.write(render(result, fmt))
            continue
        out_dir = Path(args.output_dir) if args.output_dir else Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            dest = out_dir / f"{Path(path).stem}.{fmt}"
            dest.write_text(render(result, fmt), encoding="utf-8")
            log(f"  wrote {dest}")


if __name__ == "__main__":
    main()
