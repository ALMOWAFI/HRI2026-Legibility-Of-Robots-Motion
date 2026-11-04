"""Game sounds (no ROS): the clip list, and fallbacks when a clip has not been recorded.

Every sound has a name. sound_node.py plays sounds/<name>.wav when it exists (the recorded
clip), otherwise the fallback: voice lines are spoken by espeak-ng, effects are synthesized
here. sounds/README.md lists what to record.
"""
import os
import shutil
import subprocess
import wave

import numpy as np

RATE = 44100

# Voice lines: name -> text (exact text to record, also the text-to-speech fallback).
VOICE = {
    "welcome": "Welcome! The robot will reach for one of the four cubes. Guess which one by pressing "
               "the key shown on that cube. Be quick: your points drain away while the robot moves, so "
               "the sooner you pick the right cube, the more you score. You can change your guess.",
    "get_ready": "Get ready.",
    "count_3": "Three.",
    "count_2": "Two.",
    "count_1": "One.",
    "which_cube": "Which cube? Quick!",
    "confidence": "How sure were you?",
    "correct": "Correct!",
    "wrong": "Not this time.",
    "no_answer": "You did not pick a cube.",
    "halfway": "Halfway there!",
    "last_round": "Last round!",
    "final_score": "That was the last round. Here is your score.",
    "high_score": "New high score!",
    "thank_you": "Thank you for playing!",
    "pause": "Short break. Please wait.",
    "practice_done": "Practice is over. Now the real game starts, and your score starts again from zero.",
    "questionnaire":"Halftime! Please fill in the questionnaire. The game continues afterwards.",
}

# Effects: name -> description (what to record or pick from a sound library).
EFFECTS = {
    "tick": "short tick under each countdown number, about 0.1 s",
    "go": "start signal when the robot starts moving, about 0.4 s",
    "select": "soft click when the player picks or changes a cube, under 0.05 s",
    "thinking": "suspense loop while the robot moves (the robot 'thinking'), loops seamlessly, 5-6 s",
    "correct_fx": "happy chime for a correct final guess, under 1 s",
    "wrong_fx": "low 'no' sound for a wrong final guess, under 1 s",
    "fanfare": "fanfare for the final score, 2-3 s",
}


def _tone(freq, dur, decay=0.0, shape="sine"):
    t = np.arange(int(RATE * dur)) / RATE
    phase = 2 * np.pi * np.cumsum(np.broadcast_to(freq, t.shape)) / RATE
    y = np.sin(phase) if shape == "sine" else np.sign(np.sin(phase)) * 0.4
    env = np.exp(-decay * t) if decay else np.ones_like(t)
    fade = min(len(t) // 10, int(0.005 * RATE))
    if fade:
        env[:fade] *= np.linspace(0, 1, fade)
        env[-fade:] *= np.linspace(1, 0, fade)
    return y * env


def _notes(freqs, dur, decay=6.0):
    return np.concatenate([_tone(f, dur, decay) for f in freqs])


def _thinking(dur=6.0):
    """Rising drone with ticks that speed up: suspense while the robot moves."""
    t = np.arange(int(RATE * dur)) / RATE
    drone = 0.25 * _tone(110 * 2 ** (t / dur), dur) + 0.1 * _tone(165 * 2 ** (t / dur), dur)
    ticks = np.zeros_like(t)
    tick = _tone(1500, 0.03, decay=80)
    s = 0.0
    while s < dur - 0.05:
        i = int(s * RATE)
        ticks[i:i + len(tick)] += 0.5 * tick[:len(ticks) - i]
        s += 0.5 - 0.38 * s / dur        # 2 ticks/s at the start, ~8/s at the end
    return drone + ticks


def synthesize(name):
    """Fallback samples (float, -1..1) for an effect."""
    if name == "tick":
        return _tone(1000, 0.1, decay=40)
    if name == "go":
        return _tone(1320, 0.4, decay=6) + 0.5 * _tone(1980, 0.4, decay=8)
    if name == "select":
        return 0.6 * _tone(2000, 0.03, decay=100)
    if name == "thinking":
        return _thinking()
    if name == "correct_fx":
        return _notes([523, 659, 784, 1047], 0.12)
    if name == "wrong_fx":
        return np.concatenate([_tone(300, 0.25, shape="square"), _tone(200, 0.45, decay=3, shape="square")])
    if name == "fanfare":
        return np.concatenate([_notes([523, 659, 784], 0.15), _notes([1047], 0.25, decay=1.5),
                               _notes([784, 1047], 0.15), _notes([1319], 0.9, decay=1.5)])
    raise KeyError(name)


def write_wav(path, samples):
    samples = np.asarray(samples, float)
    peak = np.abs(samples).max() or 1.0
    data = (samples / peak * 0.6 * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def speak_to_wav(text, path, voice="en-us", speed=150):
    """Text-to-speech fallback; False if espeak-ng is not installed."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return False
    subprocess.run([exe, "-v", voice, "-s", str(speed), "-w", path, text], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def resolve(name, recorded_dir, cache_dir, voice="en-us"):
    """Path of the wav to play for `name`: the recording, else a cached fallback, else None."""
    recorded = os.path.join(recorded_dir, name + ".wav")
    if os.path.exists(recorded):
        return recorded
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, name + ".wav")
    if name in EFFECTS:
        write_wav(cached, synthesize(name))
        return cached
    if name in VOICE and speak_to_wav(VOICE[name], cached, voice):
        return cached
    return None
