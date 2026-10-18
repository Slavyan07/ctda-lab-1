"""Audio discovery with a headless Linux fallback; Windows uses the native backend."""

import os
import sys


def prepare_audio() -> None:
    # Qt 6.12 PipeWire discovery requires a desktop session and client.conf.
    # Cloud offscreen runners have neither; PulseAudio safely reports no devices.
    if sys.platform == "linux" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.setdefault("QT_AUDIO_BACKEND", "pulseaudio")
