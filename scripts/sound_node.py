#!/usr/bin/env python3
"""Sound node: plays the game's voice lines and effects (hri_legible_motion.sounds).

Sub:  /sound  (String)  "<name>"       play once (several sounds can overlap)
                        "loop:<name>"  repeat until "stop:<name>" or "stop"
                        "stop"         stop everything

sounds/<name>.wav (recorded clips, see sounds/README.md) is played when it exists,
otherwise a fallback: espeak-ng for voice lines, a synthesized sound for effects.
All sounds are prepared at startup, so playback starts without delay.
"""
import os
import shutil
import subprocess
import threading

import rospkg
import rospy
from std_msgs.msg import String

from hri_legible_motion import sounds


class SoundNode:
    def __init__(self):
        cfg = rospy.get_param("~sound", {})
        pkg = rospkg.RosPack().get_path("hri_legible_motion")
        recorded = os.path.join(pkg, cfg.get("dir", "sounds"))
        cache = os.path.expanduser(cfg.get("cache_dir", "~/.cache/hri_legible_motion/sounds"))
        self.enabled = bool(cfg.get("enabled", True))
        player = cfg.get("player", "aplay")
        self.player = [shutil.which(player) or player] + (["-q"] if player == "aplay" else [])
        self.paths, fallback, missing = {}, [], []
        for name in list(sounds.VOICE) + list(sounds.EFFECTS):
            path = sounds.resolve(name, recorded, cache, cfg.get("voice", "en-us"))
            if path is None:
                missing.append(name)
                continue
            self.paths[name] = path
            if not path.startswith(recorded):
                fallback.append(name)
        rospy.loginfo("Sounds: %d recorded in %s, fallback for: %s", len(self.paths) - len(fallback),
                      recorded, ", ".join(fallback) or "none")
        if missing:
            rospy.logwarn("No sound at all (install espeak-ng or record them): %s", ", ".join(missing))
        self.lock = threading.Lock()
        self.playing = []       # [(name, Popen)]
        self.loops = set()
        rospy.Subscriber("/sound", String, self.on_sound, queue_size=20)

    def play(self, name):
        if not self.enabled or name not in self.paths:
            return None
        proc = subprocess.Popen(self.player + [self.paths[name]], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        with self.lock:
            self.playing = [(n, p) for n, p in self.playing if p.poll() is None] + [(name, proc)]
        return proc

    def loop(self, name):
        while not rospy.is_shutdown():
            with self.lock:
                if name not in self.loops:
                    return
            proc = self.play(name)
            if proc is None:
                return
            proc.wait()

    def stop(self, name=None):
        with self.lock:
            if name is None:
                self.loops.clear()
            else:
                self.loops.discard(name)
            for n, p in self.playing:
                if (name is None or n == name) and p.poll() is None:
                    p.terminate()

    def on_sound(self, msg):
        cmd = msg.data.strip()
        if cmd == "stop":
            self.stop()
        elif cmd.startswith("stop:"):
            self.stop(cmd[5:])
        elif cmd.startswith("loop:"):
            name = cmd[5:]
            with self.lock:
                if name in self.loops:
                    return
                self.loops.add(name)
            threading.Thread(target=self.loop, args=(name,), daemon=True).start()
        elif cmd in self.paths:
            self.play(cmd)
        else:
            rospy.logwarn("Unknown sound '%s'", cmd)


if __name__ == "__main__":
    rospy.init_node("sound_node")
    node = SoundNode()
    rospy.on_shutdown(node.stop)
    rospy.spin()
