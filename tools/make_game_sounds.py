#!/usr/bin/env python3
"""Build the game's sound effects in sounds/ from Kenney's free sound packs (CC0).

usage: make_game_sounds.py KENNEY_DIR [--force]

KENNEY_DIR holds the unpacked packs from kenney.nl/assets (Download, "Continue without
donating"): interface-sounds, digital-audio, music-jingles, impact-sounds. Each effect in
CHOICES is converted to sounds/<name>.wav (ffmpeg); 'thinking' is composed from real
samples: a clock tick that speeds up while the points drain, over a heartbeat. Voice lines
are not touched (record them, sounds/README.md). Existing files are kept unless --force.
To swap a sound: change CHOICES, run again with --force.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hri_legible_motion import sounds  # noqa: E402

# effect -> (pack, file name without .ogg)
CHOICES = {
    "tick": ("interface-sounds", "tick_004"),
    "go": ("digital-audio", "powerUp2"),
    "select": ("interface-sounds", "select_003"),
    "correct_fx": ("interface-sounds", "confirmation_002"),
    "wrong_fx": ("interface-sounds", "error_006"),
    "fanfare": ("music-jingles", "jingles_NES00"),
}
THINKING = {"tick": ("interface-sounds", "tick_002"), "beat": ("impact-sounds", "impactSoft_heavy_000")}


def find(kenney, pack, name):
    hits = glob.glob(os.path.join(kenney, pack, "**", name + ".ogg"), recursive=True)
    if not hits:
        sys.exit("Not found: %s/%s.ogg (unpacked pack '%s' in %s?)" % (pack, name, pack, kenney))
    return hits[0]


def load(path):
    """Mono float samples at sounds.RATE (ffmpeg decodes the ogg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1",
                          "-ar", str(sounds.RATE), "-"], check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, "<i2").astype(float) / 32768


def thinking(kenney, duration=6.0):
    """Clock ticks from 2/s to ~9/s (urgency grows as the points drain) over a heartbeat."""
    tick = load(find(kenney, *THINKING["tick"]))
    beat = load(find(kenney, *THINKING["beat"]))
    out = np.zeros(int(duration * sounds.RATE) + len(beat))
    def put(sample, t, gain):
        i = int(t * sounds.RATE)
        out[i:i + len(sample)] += gain * sample[:len(out) - i]
    t = 0.0
    while t < duration:
        put(tick, t, 0.8)
        t += 0.5 - 0.39 * t / duration
    t = 0.0
    while t < duration:
        put(beat, t, 0.5)
        put(beat, t + 0.18, 0.3)        # lub-dub
        t += 0.9 - 0.45 * t / duration
    return out[:int(duration * sounds.RATE)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("kenney_dir")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "sounds")
    made = []
    for name in list(CHOICES) + ["thinking"]:
        path = os.path.join(out_dir, name + ".wav")
        if os.path.exists(path) and not args.force:
            print("keep  %s (exists; --force replaces it)" % path)
            continue
        if name == "thinking":
            sounds.write_wav(path, thinking(args.kenney_dir))
        else:
            sounds.write_wav(path, load(find(args.kenney_dir, *CHOICES[name])))
        made.append(name)
        print("made  %s" % path)
    if made:
        print("Kenney sounds are CC0 (credit welcome, not required): sounds/CREDITS.md")


if __name__ == "__main__":
    main()
