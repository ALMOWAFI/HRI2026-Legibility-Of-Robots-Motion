#!/usr/bin/env python3
"""Experiment node: the game session (state machine, keys, points, logging).

usage (own terminal, the experimenter at the keyboard; study.launch starts it with --auto):
  rosrun hri_legible_motion experiment_node.py P07 [--first legible] [--auto] [--start-trial N]

Sub:  /button_press  (std_msgs/Int32)  participant's cube key 0-3, left to right as they see them
      /motion_status (String)          executing / done / failed / home
      /rating        (std_msgs/Int32)  1-7 from the keypad (display_node)
Pub:  /trial_start (TrialInfo), /motion_home (Empty), /choice (ChoiceEvent),
      /display (String, JSON for display_node), /sound (String, sound_node.py)

Every robot motion is started by the experimenter at this terminal: 'go' to send the robot
home, Enter before each trial. --start-trial resumes a session after a crash (appends to the CSVs).

8 game rounds, every cube once per motion type (experiment.game_order), played as a game:
countdown, sounds, points, keyboard input. First the practice rounds (experiment.training):
scored and shown like the game, but logged apart (practice_trials.csv, practice_choices.csv,
practice.bag) and the score goes back to 0; then the player starts the game with a cube key.
Block A (rounds 1-4) is one motion type, block B (rounds 5-8) the other; --first chooses
block A's motion (else asked at startup). Between the blocks the session waits for the paper
questionnaire: Enter in this terminal, then a cube key from the player. Same files throughout.
--auto (study.launch): the experimenter types 'go' once and the trials run by themselves;
p + Enter pauses after the running trial, q + Enter stops after it. After a failure the
session waits for 'go' again. Only the e-stop stops the robot during a reach.
"""
import argparse
import csv
import datetime
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading

import rospkg
import rospy
import yaml
from std_msgs.msg import Empty, Int32, String

from hri_legible_motion import experiment as E
from hri_legible_motion import sounds
from hri_legible_motion import trajectories
from hri_legible_motion.msg import ChoiceEvent, TrialInfo
from hri_legible_motion.safety import require_operator

STATUS_TIMEOUT = {"executing": 10.0, "home": 40.0}  # s; the reach itself: duration + 10 s
ORDINAL = ["leftmost", "2nd from the left", "3rd from the left", "rightmost"]


class Aborted(Exception):
    pass


class ExperimentNode:
    def __init__(self, participant, start_trial, config=None, auto=False, first=None):
        self.pkg = rospkg.RosPack().get_path("hri_legible_motion")
        self.config_path = os.path.abspath(config or os.path.join(self.pkg, "config", "experiment.yaml"))
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)
        e = self.cfg["experiment"]
        self.e = e
        self.participant = participant
        self.duration = float(self.cfg["motion"]["duration"])
        self.blocks = self.cfg["blocks"]
        self.colors = [b["color"] for b in self.blocks]
        self.button_to_block = list(e["button_to_block"])
        self.out_dir = os.path.join(os.path.expanduser(e["output_dir"]), participant)
        self.start_trial = start_trial

        seed = e["assignment_seed"]
        self.first_motion = self.choose_first_motion(first, start_trial)
        ids = {b["name"]: b["id"] for b in self.blocks}
        training = e.get("training", [])
        if any(m not in E.MOTIONS or c not in ids for m, c in training):
            sys.exit("experiment.training: [motion, cube name] pairs, motion one of %s, cube one of %s."
                     % (", ".join(E.MOTIONS), ", ".join(ids)))
        self.trials = E.make_game_schedule(participant, [b["id"] for b in self.blocks], seed,
                                           self.first_motion, [(m, ids[c]) for m, c in training])
        self.auto = auto
        self.commands = queue.Queue()   # experimenter's lines while the session runs by itself
        self.paused = False
        self.total = 0
        self.points_max = float(e.get("points_max", 100))   # per round
        self.exponent = float(e.get("points_exponent", 1.0))  # > 1: points drain faster
        self.high_score = e.get("high_score")  # fixed number shown as the high score, or None

        traj_path = self.cfg["trajectories"]
        if not os.path.isabs(traj_path):
            traj_path = os.path.join(self.pkg, traj_path)
        self.traj_meta = trajectories.read_meta(traj_path)
        if abs(self.traj_meta["duration"] - self.duration) > 1e-6:
            sys.exit("Trajectory file was planned with T = %.2f s, config says %.2f s. Recompute."
                     % (self.traj_meta["duration"], self.duration))

        if start_trial == 0 and any(os.path.exists(os.path.join(self.out_dir, name))
                                    for name in ("trials.csv", "practice_trials.csv")):
            sys.exit("%s already has data. Use another participant id, or --start-trial N to resume."
                     % self.out_dir)
        self.log = E.SessionLog(self.out_dir)
        self.practice_log = E.SessionLog(self.out_dir, "practice_")  # practice rounds, kept apart
        self.write_session_yaml(seed, traj_path)

        self.lock = threading.Lock()
        self.record = None
        self.accepting = False
        self.waiting_start = None   # threading.Event while the welcome screen waits for a key
        self.statuses = queue.Queue()
        self.ratings = queue.Queue()
        self.display_pub = rospy.Publisher("/display", String, queue_size=10, latch=True)
        self.sound_pub = rospy.Publisher("/sound", String, queue_size=20)
        self.trial_pub = rospy.Publisher("/trial_start", TrialInfo, queue_size=1)
        self.home_pub = rospy.Publisher("/motion_home", Empty, queue_size=1)
        self.choice_pub = rospy.Publisher("/choice", ChoiceEvent, queue_size=10)
        rospy.Subscriber("/button_press", Int32, self.on_button)
        rospy.Subscriber("/motion_status", String, lambda m: self.statuses.put((m.data, rospy.get_time())))
        rospy.Subscriber("/rating", Int32, lambda m: self.ratings.put(m.data))
        self.bag = None

    # --- setup -----------------------------------------------------------
    def choose_first_motion(self, first, start_trial):
        """Motion of block A: --first, else the resumed session's, else asked here."""
        session = os.path.join(self.out_dir, "session.yaml")
        earlier = None
        if start_trial and os.path.exists(session):
            with open(session) as f:
                earlier = (yaml.safe_load(f) or {}).get("first_motion")
        if first and earlier and first != earlier:
            sys.exit("%s started with %s motions; resume with the same --first." % (self.participant, earlier))
        if first or earlier:
            return first or earlier
        default = E.default_first_motion(self.participant)
        while True:
            answer = input("Which motion does %s see first (rounds 1-4)? p = predictable, l = legible "
                           "(Enter = %s, the balanced choice; q = quit): " % (self.participant, default)).strip().lower()
            if answer == "q":
                sys.exit("Stopped before the session started.")
            if not answer:
                return default
            for motion in E.MOTIONS:
                if motion.startswith(answer):
                    return motion

    def write_session_yaml(self, seed, traj_path):
        path = os.path.join(self.out_dir, "session.yaml")
        info = {"participant_id": self.participant, "date": datetime.datetime.now().isoformat(timespec="seconds"),
                "colors_by_block": {b["name"]: c for b, c in zip(self.blocks, self.colors)},
                "schedule_seed": seed, "duration_T": self.duration,
                "trajectory_file": traj_path, "trajectory_hash": self.traj_meta["hash"],
                "button_to_block": self.button_to_block, "n_trials": len(self.trials), "notes": "",
                "first_motion": self.first_motion,
                "schedule": ["%s%d %s %s %s" % ("practice " if t["phase"] == "training" else "", t["trial_id"] + 1,
                                                t["motion"], self.blocks[t["target_id"]]["name"],
                                                self.colors[t["target_id"]])
                             for t in self.trials],
                "config_file": self.config_path}
        if self.start_trial:
            path = os.path.join(self.out_dir, "session_resumed_at_%d.yaml" % self.start_trial)
            info["resumed_at_trial"] = self.start_trial
        with open(path, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False)

    def start_bag(self, name=None):
        if not self.e.get("record_bag", True):
            return
        name = name or ("session.bag" if not self.start_trial else "session_from_%d.bag" % self.start_trial)
        self.bag = subprocess.Popen(["rosbag", "record", "-O", os.path.join(self.out_dir, name)]
                                    + sorted(set(self.e["bag_topics"])),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rospy.loginfo("Recording %s", name)

    def stop_bag(self):
        if self.bag and self.bag.poll() is None:
            self.bag.send_signal(signal.SIGINT)
            self.bag.wait(timeout=10)

    # --- display ---------------------------------------------------------
    def participant_cubes(self):
        """Cube colours left to right as the participant sees them (= button order)."""
        return [self.colors[b] for b in self.button_to_block]

    def show(self, screen, title="", text="", choice_block=-1, target_block=-1, **extra):
        """extra: display fields of the game screens (hud, number, points, focus)."""
        position = lambda block: self.button_to_block.index(block) if block >= 0 else -1
        msg = {"screen": screen, "title": title, "text": text, "cubes": self.participant_cubes(),
               "choice": position(choice_block), "target": position(target_block)}
        msg.update(extra)
        self.display_pub.publish(String(json.dumps(msg)))

    def sound(self, name):
        self.sound_pub.publish(String(name))

    def describe(self, block):
        pos = self.button_to_block.index(block)
        return "the %s cube (%s)" % (self.colors[block].upper(), ORDINAL[pos])

    # --- callbacks -------------------------------------------------------
    def on_button(self, msg):
        now = rospy.get_time()
        if not 0 <= msg.data < len(self.button_to_block):
            rospy.logwarn("Unknown button %d", msg.data)
            return
        block = self.button_to_block[msg.data]
        with self.lock:
            if self.waiting_start is not None:
                self.waiting_start.set()
                return
            if self.record is None or not self.accepting:
                rospy.loginfo("Button %d ignored (not during a reach)", msg.data)
                return
            if not self.record.add_choice(now, block):
                return
            trial_id = self.record.trial["trial_id"]
        ev = ChoiceEvent(trial_id=trial_id, choice_id=block)
        ev.header.stamp = rospy.Time.from_sec(now)
        self.choice_pub.publish(ev)
        self.display_pub.publish(String(json.dumps({"choice": self.button_to_block.index(block)})))
        self.sound("select")

    # --- waiting ---------------------------------------------------------
    def wait_status(self, wanted, timeout):
        """(status, stamp) of the first status in `wanted`, or (None, None) after timeout."""
        deadline = rospy.get_time() + timeout
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            try:
                status, stamp = self.statuses.get(timeout=0.2)
            except queue.Empty:
                continue
            if status in wanted:
                return status, stamp
        return None, None

    def wait_rating(self, timeout):
        while not self.ratings.empty():
            self.ratings.get_nowait()
        try:
            r = self.ratings.get(timeout=timeout)
            return r if 1 <= r <= 7 else None
        except queue.Empty:
            return None

    def ask(self, text):
        """A line from the experimenter. In --auto mode a reader thread owns stdin; lines typed
        before the question are dropped, so a stale 'go' never confirms a motion."""
        if not self.auto or not self.reader_started:
            return input(text).strip().lower()
        while not self.commands.empty():
            self.commands.get_nowait()
        print(text, end="", flush=True)
        while not rospy.is_shutdown():
            try:
                return self.commands.get(timeout=0.2)
            except queue.Empty:
                pass
        raise Aborted("ROS shutdown")

    def prompt(self, text):
        """Experimenter input; 'q' ends the session."""
        answer = self.ask(text)
        if answer == "q":
            raise Aborted()
        return answer

    def operator_go(self, text):
        """require_operator() through ask(): 'go' typed by a person at this terminal."""
        if not self.auto or not self.reader_started:
            return require_operator(text)
        return self.ask(text + " Hand on the e-stop. Type 'go' to move: ") == "go"

    reader_started = False

    def start_reader(self):
        """--auto: read the experimenter's lines in the background (p = pause, q = quit)."""
        def read():
            while not rospy.is_shutdown():
                try:
                    line = input()
                except EOFError:
                    self.commands.put("q")
                    return
                self.commands.put(line.strip().lower())
        threading.Thread(target=read, daemon=True).start()
        self.reader_started = True

    def go_home(self):
        while True:
            if not self.operator_go("Send the robot to the start pose?"):
                raise Aborted()
            self.drain_statuses()
            self.home_pub.publish(Empty())
            status, _ = self.wait_status({"home", "failed"}, STATUS_TIMEOUT["home"])
            if status == "home":
                return
            print("Robot did not reach the start pose (%s). Check the motion_node terminal." % (status or "timeout"))

    def drain_statuses(self):
        while not self.statuses.empty():
            self.statuses.get_nowait()

    # --- trial -----------------------------------------------------------
    def log_trial(self, record):
        log = self.practice_log if record.trial["phase"] == "training" else self.log
        row = log.write(record, self.participant, self.duration, self.blocks, self.colors)
        if row.get("f") is not None:
            extra = "f %.2f, points %s, final %s" % (row["f"], row["points"],
                                                     "ok" if row["final_correct"] else "wrong")
        else:
            extra = "no reach"
        print("  -> %s, %s, confidence %s" % (record.status, extra, row["confidence"]))

    def n_phase(self, phase):
        return sum(1 for t in self.trials if t["phase"] == phase)

    def hud(self, trial):
        label = "Practice" if trial["phase"] == "training" else "Round"
        hud = {"label": label, "round": trial["index_in_phase"] + 1, "rounds": self.n_phase(trial["phase"]),
               "total": self.total}
        if self.high_score is not None and trial["phase"] == "game":
            hud["best"] = int(self.high_score)
        return hud

    def round_voice(self, trial):
        n, rounds = trial["index_in_phase"], self.n_phase("game")
        if trial["phase"] == "training":
            return "get_ready"
        if n == rounds - 1:
            return "last_round"
        return "halfway" if n == rounds // 2 else "get_ready"

    def run_trial(self, trial):
        """Countdown, reach, (confidence), result. Robot is home afterwards unless failed.
        Training trials look the same (points and score included); their score is a separate
        practice score, reset when the game starts."""
        e, n, target = self.e, trial["index_in_phase"], trial["target_id"]
        hud = self.hud(trial)
        self.show("text", "%s %d of %d" % (hud["label"], n + 1, hud["rounds"]), "Get ready!", hud=hud, focus=True)
        self.sound(self.round_voice(trial))
        rospy.sleep(float(e["ready_time"]))
        for k in range(int(e.get("countdown", 3)), 0, -1):
            self.show("countdown", "", number=k, hud=hud)
            self.sound("tick")
            self.sound("count_%d" % k)
            rospy.sleep(1.0)

        self.drain_statuses()
        now = rospy.get_time()
        with self.lock:
            self.record = record = E.TrialRecord(trial, now, self.points_max, self.exponent)
            self.accepting = False
        self.show("reach", "Which cube?", "Pick fast: the points drain while the robot moves!", hud=hud, focus=True,
                  points_max=self.points_max, exponent=self.exponent)
        self.sound("go")
        msg = TrialInfo(trial_id=trial["trial_id"], target_id=target, condition=trial["motion"],
                        bend=float(e["bend"][trial["motion"]]))
        msg.header.stamp = rospy.Time.from_sec(now)
        self.trial_pub.publish(msg)

        # Guesses count from reach onset, so nobody scores by pressing during the countdown.
        status, t = self.wait_status({"executing", "failed"}, STATUS_TIMEOUT["executing"])
        if status == "executing":
            record.t_start = t
            with self.lock:
                self.accepting = True
            self.display_pub.publish(String(json.dumps({"drain": self.duration})))  # bar starts emptying
            self.sound("loop:thinking")
            self.sound("which_cube")
            status, t = self.wait_status({"done", "failed"}, self.duration + 10.0)
        with self.lock:
            self.accepting = False
        self.sound("stop:thinking")
        record.status = status or "timeout"
        record.t_done = t
        if record.status != "done":
            self.show("text", "Short pause", "Please wait a moment.", hud=hud)
            self.sound("pause")
            return record

        m = record.measures(self.duration)
        if e.get("confidence_rating", False):
            self.show("rating", "How sure were you?", "1 = just guessing   ...   7 = completely sure", hud=hud)
            self.sound("confidence")
            record.confidence = self.wait_rating(float(e["rating_timeout"]))
        self.total += m["points"]
        final = m["final_choice"]
        title = "Correct!" if m["final_correct"] else "No cube picked" if final < 0 else "Not this time"
        self.sound("correct_fx" if m["final_correct"] else "wrong_fx")
        self.sound("correct" if m["final_correct"] else "no_answer" if final < 0 else "wrong")
        self.show("result", title, "", choice_block=final, target_block=target,
                  points=m["points"], hud=self.hud(trial))
        rospy.sleep(float(e["score_time"]))

        status, _ = self.wait_status({"home", "failed"}, STATUS_TIMEOUT["home"])
        if status != "home":
            record.status = "return_" + (status or "timeout")
        return record

    # --- session ---------------------------------------------------------
    def check_commands(self):
        """--auto, between trials: p = pause, q = quit (typed in this terminal)."""
        while not self.commands.empty():
            cmd = self.commands.get_nowait()
            if cmd == "q":
                raise Aborted("stopped by the experimenter")
            if cmd == "p":
                self.paused = True
        if self.paused:
            self.show("text", "Short break", "Please wait.", hud=None)
            self.sound("pause")
            self.prompt("Paused. Enter = continue, q = quit: ")
            self.paused = False

    def wait_for_player(self, title="Guess the robot's cube!", text=None, sound="welcome"):
        """Welcome screen until the player presses any cube key (also checks the keys work)."""
        self.waiting_start = threading.Event()
        text = self.e.get("welcome_text", sounds.VOICE["welcome"]) if text is None else text
        self.show("text", title, text + "\n\nPress any cube key to start.", focus=True)
        if sound:
            self.sound(sound)
        print("Waiting for the player to press a cube key (q = quit) ...")
        while not self.waiting_start.wait(0.2):
            if rospy.is_shutdown():
                raise Aborted("ROS shutdown")
            if self.auto:
                self.check_commands()
        with self.lock:
            self.waiting_start = None
        # Started: release the keyboard, so the experimenter can type here until the next round.
        self.display_pub.publish(String(json.dumps({"focus": False})))

    def questionnaire_break(self, done, next_trial):
        """Between block A and B: the paper questionnaire, then the player starts block B."""
        self.show("text", "Break", self.e.get("questionnaire_text", sounds.VOICE["questionnaire"]), hud=None)
        self.sound("questionnaire")
        print("\n=== Block A done (%s, %d points so far). Questionnaire on paper. ===" % (done["motion"], self.total))
        rospy.loginfo("Questionnaire break after block A")
        self.prompt("Questionnaire done? Enter = continue with block B (%s) (q = quit): " % next_trial["motion"])
        rospy.loginfo("Block B starts")
        self.wait_for_player("Second half", "Ready for the next rounds?", sound=None)

    def training_done(self):
        """After the practice rounds: the score goes back to 0, the practice bag is closed and
        the study's bag starts, then the player starts the game with a cube key."""
        print("\n=== Practice done (%d practice points, not kept). The game starts. ===" % self.total)
        rospy.loginfo("Training done, block A starts")
        practice_points, self.total = self.total, 0
        self.stop_bag()
        self.start_bag("session.bag")
        self.wait_for_player("Practice done!", "Practice score: %d\n\n%s%s" % (
            practice_points, self.e.get("practice_done_text", sounds.VOICE["practice_done"]), self.challenge()),
            sound="practice_done")

    def challenge(self):
        """experiment.high_score_text with the fixed high score, before the scored game ('' if off)."""
        if self.high_score is None or not self.e.get("high_score_text"):
            return ""
        return "\n\n" + self.e["high_score_text"].format(high_score=int(self.high_score))

    def restore_total(self, log):
        """Points of the trials already done in `log` (resumed session)."""
        if not os.path.exists(log.trials_path):
            return 0
        with open(log.trials_path) as f:
            return sum(int(r["points"]) for r in csv.DictReader(f)
                       if r.get("status") == "done" and r.get("points") not in (None, ""))

    def finish_game(self):
        self.show("text", "That was the last round!", "", focus=True)
        self.sound("final_score")
        rospy.sleep(3.0)
        text = ""
        rank = None
        if self.e.get("leaderboard", True) and re.fullmatch(r"P\d+", self.participant):  # not dry runs
            board = E.Leaderboard(os.path.join(os.path.expanduser(self.e["output_dir"]), "leaderboard.csv"))
            rank, players, best = board.add(self.participant, datetime.date.today().isoformat(), self.total)
            text = "Rank %d of %d players" % (rank, players) if players > 1 else ""
        beaten = rank == 1 and text
        if self.high_score is not None:  # the fixed high score replaces the rank and never changes
            beaten = self.total > self.high_score
            text = ("You beat the high score of %d!" if beaten else "High score: %d") % self.high_score
        self.show("final", "Your score", text, points=self.total)
        self.sound("fanfare")
        if beaten:
            rospy.sleep(2.5)
            self.sound("high_score")
        rospy.sleep(3.0)
        self.sound("thank_you")
        self.show("final", "Thank you for playing!", text, points=self.total)

    def run(self):
        print("\nParticipant %s: %d trials, trajectories %s"
              % (self.participant, len(self.trials), self.traj_meta["hash"]))
        print("%d practice rounds (logged apart, score reset after), then block A (rounds 1-4): %s, block B (rounds 5-8): %s; "
              "questionnaire in between" % (self.n_phase("training"), self.first_motion, self.trials[-1]["motion"]))
        print("Cube colours (robot frame, place_blocks.py spots):")
        for b, c in zip(self.blocks, self.colors):
            print("  %s: %s" % (b["name"], c))
        print("Screen, left to right (keys %s): %s"
              % (" ".join("ASDF"[:len(self.button_to_block)]),
                 ", ".join("%s %s" % (self.blocks[b]["name"], self.colors[b]) for b in self.button_to_block)))
        print("Order: " + ", ".join("%s-%s" % (t["motion"][0].upper(), self.blocks[t["target_id"]]["name"])
                                   for t in self.trials))
        in_practice = self.start_trial < len(self.trials) and self.trials[self.start_trial]["phase"] == "training"
        self.total = self.restore_total(self.practice_log if in_practice else self.log) if self.start_trial else 0
        self.show("text", "Welcome", "Please wait for the experimenter.")
        self.prompt("Cubes in place, display on the participant's screen, sound on, slider at 100 %? "
                    "Enter (q = quit) ")
        if in_practice:
            self.start_bag("practice.bag" if not self.start_trial else "practice_from_%d.bag" % self.start_trial)
        else:
            self.start_bag()
        if self.auto:
            print("\nAutomatic session: after 'go' all %d rounds (practice included) run by themselves, with "
                  "a stop for the questionnaire after block A. While it runs:\n"
                  "  p + Enter  pause after the current round\n"
                  "  q + Enter  stop after the current round\n"
                  "  E-STOP     the only way to stop the robot during a reach\n"
                  "Typing here takes the keyboard focus from the game window; it takes it back at the "
                  "next round." % (len(self.trials) - self.start_trial))
        self.go_home()
        if self.auto:
            self.start_reader()
        if self.start_trial < len(self.trials) and self.trials[self.start_trial]["phase"] == "game":
            # No practice before this round (none configured, or resumed in the game): challenge here.
            self.wait_for_player(text=self.e.get("welcome_text", sounds.VOICE["welcome"]) + self.challenge())
        else:
            self.wait_for_player()

        i = self.start_trial
        block = self.trials[i]["block"] if i < len(self.trials) else None
        while i < len(self.trials):
            trial = self.trials[i]
            if trial["block"] != block:
                if block == "T":
                    self.training_done()
                else:
                    self.questionnaire_break(self.trials[i - 1], trial)
                block = trial["block"]
            if self.auto:
                self.check_commands()
                print("[%d/%d] %s, target %s" % (i + 1, len(self.trials), trial["motion"],
                                                 self.describe(trial["target_id"])))
            else:
                self.prompt("[%d/%d] %s, target %s. Enter = run (q = quit) "
                            % (i + 1, len(self.trials), trial["motion"], self.describe(trial["target_id"])))
            record = self.run_trial(trial)
            self.log_trial(record)
            if record.status != "done":
                print("Round not completed (%s). Check the robot and the motion_node terminal." % record.status)
                self.go_home()
                if self.prompt("Repeat this round? [Y/n] ") != "n":
                    continue
            i += 1
            if self.auto:
                rospy.sleep(float(self.e.get("inter_trial_time", 1.0)))

        self.finish_game()
        print("\nSession complete: %s (%d points). Questionnaire next." % (self.out_dir, self.total))


def main():
    rospy.init_node("experiment_node")
    ap = argparse.ArgumentParser()
    ap.add_argument("participant", help="e.g. P07")
    ap.add_argument("--start-trial", type=int, default=0, help="resume at this trial index")
    ap.add_argument("--config", help="same experiment YAML used by planner, optimizer and display")
    ap.add_argument("--first", choices=list(E.MOTIONS) + [""], default="",
                    help="motion of rounds 1-4 (the other one in rounds 5-8); asked if not given")
    ap.add_argument("--auto", action="store_true",
                    help="one 'go', then all trials run by themselves (study.launch)")
    args = ap.parse_args(rospy.myargv()[1:])
    if not sys.stdin.isatty():
        sys.exit("experiment_node needs the experimenter at an interactive terminal (use rosrun, not roslaunch).")
    try:
        node = ExperimentNode(args.participant, args.start_trial, args.config, args.auto, args.first or None)
    except SystemExit as error:
        if args.auto:  # own terminal window (study.launch): keep the message readable
            print(error)
            input("Enter closes this window and stops the launch. ")
        raise
    try:
        node.run()
    except (Aborted, RuntimeError, KeyboardInterrupt, EOFError) as error:
        print("\nSession stopped: %s. Inspect state before resuming with --start-trial." % error)
    finally:
        node.stop_bag()
    if args.auto:
        try:
            node.ask("Enter closes this window and stops the launch (robot, display, sound). ")
        except (Aborted, EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
