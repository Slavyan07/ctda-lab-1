"""Owned offline Piper subprocess with a single CPU inference thread."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def main() -> None:
    import onnxruntime as ort
    from piper import PiperVoice, SynthesisConfig
    from piper.config import PiperConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, sess_options=options, providers=["CPUExecutionProvider"])
    voice = PiperVoice(
        session=session,
        config=PiperConfig.from_dict(json.loads(Path(args.model + ".json").read_text(encoding="utf-8"))),
    )
    text = sys.stdin.buffer.read(16000).decode("utf-8")
    with wave.open(args.output_file, "wb") as output:
        voice.synthesize_wav(text, output, SynthesisConfig(length_scale=args.length_scale))


if __name__ == "__main__":
    main()
