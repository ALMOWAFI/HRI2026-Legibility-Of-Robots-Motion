#!/usr/bin/env python3
"""Offline tests for sounds.py (no ROS, no audio device): python3 -m unittest discover -s test"""
import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import sounds  # noqa: E402


class TestSounds(unittest.TestCase):
    def test_every_effect_synthesizes(self):
        with tempfile.TemporaryDirectory() as d:
            for name in sounds.EFFECTS:
                path = sounds.resolve(name, os.path.join(d, "recorded"), d)
                with wave.open(path) as w:
                    self.assertGreater(w.getnframes(), 0, name)

    def test_recording_wins(self):
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, "rec")
            os.makedirs(rec)
            sounds.write_wav(os.path.join(rec, "tick.wav"), sounds.synthesize("go"))
            self.assertEqual(sounds.resolve("tick", rec, d), os.path.join(rec, "tick.wav"))

    def test_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(sounds.resolve("nothing", d, d))

    def test_sounds_used_by_the_game_exist(self):
        used = ["welcome", "get_ready", "halfway", "last_round", "count_3", "count_2", "count_1",
                "tick", "go", "select", "thinking", "which_cube", "confidence", "correct", "wrong",
                "no_answer", "correct_fx", "wrong_fx", "pause", "final_score", "fanfare", "high_score",
                "thank_you"]
        for name in used:
            self.assertTrue(name in sounds.VOICE or name in sounds.EFFECTS, name)


if __name__ == "__main__":
    unittest.main()
