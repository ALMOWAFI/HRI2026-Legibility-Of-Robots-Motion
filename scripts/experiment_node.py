#!/usr/bin/env python3
"""Experiment node: session state machine, buttons, scoring, ratings, logging (plan §3.2-3.5).

usage (own terminal, the experimenter at the keyboard):
  rosrun hri_legible_motion experiment_node.py P07 [--start-trial N]

Sub:  /button_press  (std_msgs/Int32)  participant's button 0-3, left to right as they see them
      /motion_status (String)          executing / done / failed / home
      /rating        (std_msgs/Int32)  1-7 from the keypad (display_node)
Pub:  /trial_start (TrialInfo), /motion_home (Empty), /choice (ChoiceEvent),
      /display (String, JSON for display_node)

Every robot motion is started by the experimenter at this terminal: 'go' to send the robot
home, Enter before each trial. --start-trial resumes a session after a crash (appends to the CSVs).

experiment.design: game = 8 trials, every cube once per motion type (experiment.game_order),
played as a game: countdown, sounds (/sound, sound_node.py), points, keyboard input.
Block A (trials 1-4) is one motion type, block B (trials 5-8) the other; --first chooses
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
PHASE_INTRO = {
    "warmup": ("Practice", "A few practice rounds. Press the button of the cube you think the robot "
                           "is reaching for, as early as you can. You can change your mind."),
    "main": ("Next block", "Press the button of the cube you think the robot is reaching for, "
                           "as early as you can. You can change your mind."),
    "announced": ("Last block", "Now you are told the target in advance. Just watch the robot, "
                                "then rate how natural the motion looked."),
}


class Aborted(Exception):
    pass


class ExperimentNode:
    def __init__(self, participant, start_trial, config=None, pipeline=False, auto=False, first=None):
        self.pkg = rospkg.RosPack().get_path("hri_legible_motion")
        self.config_path = os.path.abspath(config or os.path.join(self.pkg, "config", "experiment.yaml"))
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.pipeline = None
        if pipeline:
            # Optional dependency: reuse the standalone package; no controller imports.
            from legibility_ros.experiment_client import ExperimentClient
            self.pipeline = ExperimentClient(rospy, self.cfg)
        e = self.cfg["experiment"]
        self.e = e
        self.participant = participant
        self.duration = float(self.cfg["motion"]["duration"])
        self.blocks = self.cfg["blocks"]
        self.button_to_block = list(e["button_to_block"])

        self.block_order, self.rotation = E.assignment(participant, e["assignment_seed"])
        if not e.get("rotate_colors", True):
            self.rotation = 0  # cubes keep the config colours, as on the table and in RViz
        self.colors = E.rotated_colors(e["colors"], self.rotation)
        seed = E.participant_number(participant)
        self.game = e.get("design", "classic") == "game"
        self.out_dir = os.path.join(os.path.expanduser(e["output_dir"]), participant)
        self.first_motion = None
        if self.game:
            seed = e["assignment_seed"]
            self.first_motion = self.choose_first_motion(first, start_trial)
            self.trials = E.make_game_schedule(participant, [b["id"] for b in self.blocks], seed,
                                               self.first_motion)
        else:
            if first:
                sys.exit("--first needs experiment.design: game.")
            self.trials = E.make_schedule(self.block_order, [b["id"] for b in self.blocks],
                                          e["reps_per_block"], seed, e["max_same_in_row"])
        if auto and (pipeline or not self.game):
            sys.exit("--auto needs experiment.design: game and the motion_node backend.")
        self.auto = auto
        self.commands = queue.Queue()   # experimenter's lines while the session runs by itself
        self.paused = False
        self.total = 0
        self.points_max = float(e.get("points_max", 100))   # per round (game)
        self.exponent = float(e.get("points_exponent", 1.0))  # > 1: points drain faster
        self.start_trial = start_trial

        traj_path = None
        self.traj_meta = {"hash": "live planner/optimizer", "duration": self.duration}
        if not self.pipeline:
            traj_path = self.cfg["trajectories"]
            if not os.path.isabs(traj_path):
                traj_path = os.path.join(self.pkg, traj_path)
            self.traj_meta = trajectories.read_meta(traj_path)
            if abs(self.traj_meta["duration"] - self.duration) > 1e-6:
                sys.exit("Trajectory file was planned with T = %.2f s, config says %.2f s. Recompute."
                         % (self.traj_meta["duration"], self.duration))

        if os.path.exists(os.path.join(self.out_dir, "trials.csv")) and start_trial == 0:
            sys.exit("%s already has data. Use another participant id, or --start-trial N to resume."
                     % self.out_dir)
        self.log = E.SessionLog(self.out_dir)
        self.write_session_yaml(seed, traj_path)

        self.lock = threading.Lock()
        self.record = None
        self.accepting = False
        self.waiting_start = None   # threading.Event while the welcome screen waits for a key
        self.statuses = queue.Queue()
        self.ratings = queue.Queue()
        self.display_pub = rospy.Publisher("/display", String, queue_size=10, latch=True)
        self.sound_pub = rospy.Publisher("/sound", String, queue_size=20)
        if not self.pipeline:
            self.trial_pub = rospy.Publisher("/trial_start", TrialInfo, queue_size=1)
            self.home_pub = rospy.Publisher("/motion_home", Empty, queue_size=1)
        self.choice_pub = rospy.Publisher("/choice", ChoiceEvent, queue_size=10)
        rospy.Subscriber("/button_press", Int32, self.on_button)
        if not self.pipeline:
            rospy.Subscriber("/motion_status", String, lambda m: self.statuses.put((m.data, rospy.get_time())))
        rospy.Subscriber("/rating", Int32, lambda m: self.ratings.put(m.data))
        self.bag = None

    # --- setup -----------------------------------------------------------
    def choose_first_motion(self, first, start_trial):
        """Motion of game block A: --first, else the resumed session's, else asked here."""
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
                "block_order": list(self.block_order), "color_rotation": self.rotation,
                "colors_by_block": {b["name"]: c for b, c in zip(self.blocks, self.colors)},
                "schedule_seed": seed, "duration_T": self.duration,
                "trajectory_file": traj_path, "trajectory_hash": self.traj_meta["hash"],
                "button_to_block": self.button_to_block, "n_trials": len(self.trials), "notes": ""}
        info["design"] = "game" if self.game else "classic"
        if self.game:
            info["first_motion"] = self.first_motion
            info["schedule"] = ["%d %s %s %s" % (t["trial_id"] + 1, t["motion"], self.blocks[t["target_id"]]["name"],
                                                 self.colors[t["target_id"]])
                                for t in self.trials]
        info["config_file"] = self.config_path
        if self.pipeline:
            info.update(backend="rrt_fk_optimizer", pipeline_session_id=self.pipeline.session_id,
                        config_sha256=self.pipeline.config_sha256,
                        timing="required duration validated downstream; measured onset/completion")
        if self.start_trial:
            path = os.path.join(self.out_dir, "session_resumed_at_%d.yaml" % self.start_trial)
            info["resumed_at_trial"] = self.start_trial
        with open(path, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False)

    def start_bag(self):
        if not self.e.get("record_bag", True):
            return
        name = "session.bag" if not self.start_trial else "session_from_%d.bag" % self.start_trial
        topics = list(self.e["bag_topics"])
        if self.pipeline:
            topics += ["/hri/block_layout", "/legibility/trial_request", "/legibility/baseline",
                       "/legibility/optimized_candidate", "/legibility/trajectory_proposal",
                       "/legibility/status", "/legibility/validated_trajectory", "/legibility/motion_feedback"]
        self.bag = subprocess.Popen(["rosbag", "record", "-O", os.path.join(self.out_dir, name)]
                                    + sorted(set(topics)),
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
        if self.game:
            self.display_pub.publish(String(json.dumps({"choice": self.button_to_block.index(block)})))
            self.sound("select")
        else:
            self.show("reach", choice_block=block)

    # --- waiting ---------------------------------------------------------
    def wait_status(self, wanted, timeout):
        """(status, stamp) of the first status in `wanted`, or (None, None) after timeout."""
        if self.pipeline:
            return self.pipeline.wait_status(wanted, timeout)
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
            if self.pipeline:
                try:
                    self.pipeline.request_motion("home", operator_confirmed=True)
                except RuntimeError as error:
                    raise Aborted(str(error))
            else:
                self.home_pub.publish(Empty())
            status, _ = self.wait_status({"home", "failed"}, STATUS_TIMEOUT["home"])
            if status == "home":
                return
            if self.pipeline:
                raise Aborted("Home not verified; inspect the downstream executor before any further command.")
            print("Robot did not reach the start pose (%s). Check the motion_node terminal." % (status or "timeout"))

    def drain_statuses(self):
        while not self.statuses.empty():
            self.statuses.get_nowait()

    # --- trial -----------------------------------------------------------
    def run_trial(self, trial):
        """Returns the TrialRecord; robot is home afterwards unless status is failed/timeout."""
        announced = trial["phase"] == "announced"
        target = trial["target_id"]
        pipeline_attempt = None
        if self.pipeline:
            self.show("text", "Please wait", "Preparing the next trial.")
            # Planning/optimization latency never starts the participant's reach clock.
            proposal = self.pipeline.prepare(trial)
            validation = self.pipeline.validate()
            pipeline_attempt = dict(trial_id=trial["trial_id"],
                                    pipeline_trial_id=proposal.source.trial_id,
                                    validation_id=validation.validation_id,
                                    config_sha256=proposal.source.config_sha256,
                                    layout_id=proposal.source.layout.layout_id)
            # Save the exact handoff even if rosbag recording is disabled.
            proposal_path = os.path.join(self.out_dir, proposal.source.trial_id + ".proposal")
            with open(proposal_path, "wb") as stream:
                proposal.serialize(stream)
            with open(os.path.join(self.out_dir, "pipeline_attempts.jsonl"), "a") as stream:
                stream.write(json.dumps(dict(pipeline_attempt, event="prepared")) + "\n")
            if self.prompt("Validated trial prepared. Type 'go' to request this reach (q = quit): ") != "go":
                raise Aborted("Trial not explicitly authorized.")
        if announced:
            self.show("text", "Watch", "The robot will reach for %s." % self.describe(target))
            rospy.sleep(3.0)
        else:
            self.show("reach", "Get ready")
            rospy.sleep(float(self.e["ready_time"]))

        self.drain_statuses()
        now = rospy.get_time()
        with self.lock:
            self.record = record = E.TrialRecord(trial, now)
            self.accepting = not announced
            record.pipeline_attempt = pipeline_attempt
        if not announced:
            self.show("reach")
        if self.pipeline:
            try:
                self.pipeline.request_motion("run", operator_confirmed=True)
            except RuntimeError as error:
                with self.lock:
                    self.accepting = False
                raise Aborted(str(error))
        else:
            msg = TrialInfo(trial_id=trial["trial_id"], target_id=target, condition=trial["motion"],
                            bend=float(self.e["bend"][trial["motion"]]))
            msg.header.stamp = rospy.Time.from_sec(now)
            self.trial_pub.publish(msg)

        status, t = self.wait_status({"executing", "failed"}, STATUS_TIMEOUT["executing"])
        if status == "executing":
            record.t_start = t
            status, t = self.wait_status({"done", "failed"}, self.duration + 10.0)
        with self.lock:
            self.accepting = False
        record.status = status or "timeout"
        record.t_done = t
        if record.status != "done":
            self.show("text", "Short pause", "Please wait a moment.")
            return record
        if self.pipeline:
            # Use measured arm onset-to-completion for scoring, not planner timestamps.
            record.measured_duration = record.t_done - record.t_start
            if record.measured_duration <= 0:
                record.status = "invalid_motion_timing"
                return record

        if announced:
            self.show("rating", "How natural did this motion look?",
                      "1 = not at all natural   ...   7 = very natural")
            record.naturalness = self.wait_rating(float(self.e["rating_timeout"]))
        else:
            m = record.measures(getattr(record, "measured_duration", self.duration))
            self.show("score", "Score: %d / 10" % m["score"], choice_block=m["final_choice"])
            rospy.sleep(float(self.e["score_time"]))
            self.show("rating", "How confident were you in your final choice?",
                      "1 = not at all confident   ...   7 = completely confident")
            record.confidence = self.wait_rating(float(self.e["rating_timeout"]))

        if self.pipeline:
            # The executor does not automatically return after a reach.
            # run() logs the completed reach before prompting for this separate move.
            return record
        status, _ = self.wait_status({"home", "failed"}, STATUS_TIMEOUT["home"])
        if status != "home":
            record.status = "return_" + (status or "timeout")
        return record

    def log_trial(self, record):
        row = self.log.write(record, self.participant, getattr(record, "measured_duration", self.duration),
                             self.blocks, self.colors)
        if self.pipeline:
            with open(os.path.join(self.out_dir, "pipeline_attempts.jsonl"), "a") as stream:
                stream.write(json.dumps(dict(record.pipeline_attempt, event="result",
                                             t_trial_start=record.t_trial_start, t_start=record.t_start,
                                             t_done=record.t_done, status=record.status,
                                             measured_duration=getattr(record, "measured_duration", None))) + "\n")
        if record.trial["phase"] == "announced":
            extra = "naturalness %s" % row["naturalness"]
        elif row.get("f") is not None:
            extra = "f %.2f, score %s, final %s" % (row["f"], row["score"],
                                                    "ok" if row["final_correct"] else "wrong")
        else:
            extra = "no reach"
        print("  -> %s, %s, confidence %s" % (record.status, extra, row["confidence"]))

    # --- game (experiment.design: game) ----------------------------------
    def hud(self, trial):
        return {"round": trial["index_in_phase"] + 1, "rounds": len(self.trials), "total": self.total}

    def round_voice(self, n):
        if n == len(self.trials) - 1:
            return "last_round"
        return "halfway" if n == len(self.trials) // 2 else "get_ready"

    def run_game_trial(self, trial):
        """Countdown, reach, (confidence), result. Robot is home afterwards unless failed."""
        e, n, target = self.e, trial["index_in_phase"], trial["target_id"]
        hud = self.hud(trial)
        self.show("text", "Round %d of %d" % (n + 1, len(self.trials)), "Get ready!", hud=hud, focus=True)
        self.sound(self.round_voice(n))
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

    def questionnaire_break(self, done, next_trial):
        """Between game block A and B: the paper questionnaire, then the player starts block B."""
        self.show("text", "Break", self.e.get("questionnaire_text", sounds.VOICE["questionnaire"]), hud=None)
        self.sound("questionnaire")
        print("\n=== Block A done (%s, %d points so far). Questionnaire on paper. ===" % (done["motion"], self.total))
        rospy.loginfo("Questionnaire break after block A")
        self.prompt("Questionnaire done? Enter = continue with block B (%s) (q = quit): " % next_trial["motion"])
        rospy.loginfo("Block B starts")
        self.wait_for_player("Second half", "Ready for the next rounds?", sound=None)

    def restore_total(self):
        """Points of the trials already done (resumed session)."""
        if not os.path.exists(self.log.trials_path):
            return 0
        with open(self.log.trials_path) as f:
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
        self.show("final", "Your score", text, points=self.total)
        self.sound("fanfare")
        if rank == 1 and text:
            rospy.sleep(2.5)
            self.sound("high_score")
        rospy.sleep(3.0)
        self.sound("thank_you")
        self.show("final", "Thank you for playing!", text, points=self.total)

    def run_game(self):
        print("\nParticipant %s: game, %d trials, colour rotation %d, trajectories %s"
              % (self.participant, len(self.trials), self.rotation, self.traj_meta["hash"]))
        print("Block A (rounds 1-4): %s, block B (rounds 5-8): %s; questionnaire in between"
              % (self.trials[0]["motion"], self.trials[-1]["motion"]))
        print("Cube colours for this participant (robot frame, place_blocks.py spots):")
        for b, c in zip(self.blocks, self.colors):
            print("  %s: %s" % (b["name"], c))
        print("Screen, left to right (keys %s): %s"
              % (" ".join("ASDF"[:len(self.button_to_block)]),
                 ", ".join("%s %s" % (self.blocks[b]["name"], self.colors[b]) for b in self.button_to_block)))
        print("Order: " + ", ".join("%s-%s" % (t["motion"][0].upper(), self.blocks[t["target_id"]]["name"])
                                   for t in self.trials))
        self.total = self.restore_total() if self.start_trial else 0
        self.show("text", "Welcome", "Please wait for the experimenter.")
        self.prompt("Cubes in place, display on the participant's screen, sound on, slider at 100 %? "
                    "Enter (q = quit) ")
        self.start_bag()
        if self.auto:
            print("\nAutomatic session: after 'go' all %d rounds run by themselves, with a stop for the "
                  "questionnaire after block A. While it runs:\n"
                  "  p + Enter  pause after the current round\n"
                  "  q + Enter  stop after the current round\n"
                  "  E-STOP     the only way to stop the robot during a reach\n"
                  "Typing here takes the keyboard focus from the game window; it takes it back at the "
                  "next round." % (len(self.trials) - self.start_trial))
        self.go_home()
        if self.auto:
            self.start_reader()
        self.wait_for_player()

        i = self.start_trial
        block = self.trials[i]["block"] if i < len(self.trials) else None
        while i < len(self.trials):
            trial = self.trials[i]
            if trial["block"] != block:
                self.questionnaire_break(self.trials[i - 1], trial)
                block = trial["block"]
            if self.auto:
                self.check_commands()
                print("[%d/%d] %s, target %s" % (i + 1, len(self.trials), trial["motion"],
                                                 self.describe(trial["target_id"])))
            else:
                self.prompt("[%d/%d] %s, target %s. Enter = run (q = quit) "
                            % (i + 1, len(self.trials), trial["motion"], self.describe(trial["target_id"])))
            record = self.run_game_trial(trial)
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

    # --- session ---------------------------------------------------------
    def run(self):
        print("\nParticipant %s: block order %s, colour rotation %d, %d trials, trajectories %s"
              % (self.participant, " -> ".join(self.block_order), self.rotation, len(self.trials),
                 self.traj_meta["hash"]))
        print("Cube colours for this participant (robot frame, place_blocks.py spots):")
        for b, c in zip(self.blocks, self.colors):
            print("  %s: %s" % (b["name"], c))
        self.show("text", "Welcome", "Please wait for the experimenter.")
        self.prompt("Cubes in place, buttons and keypad tested, slider at 100 %? Enter (q = quit) ")
        self.start_bag()
        self.go_home()

        phase = None
        i = self.start_trial
        while i < len(self.trials):
            trial = self.trials[i]
            if (trial["phase"], trial["block"]) != phase:
                if phase and phase[0] == "main":
                    self.show("text", "Break", "Please fill in the questionnaire.")
                    self.prompt("Questionnaire after block %s. Enter when done " % phase[1])
                phase = (trial["phase"], trial["block"])
                self.show("text", *PHASE_INTRO[trial["phase"]])
                print("\n=== %s block %s (%s) ===" % (trial["phase"], trial["block"],
                                                    trial["motion"] if trial["phase"] == "main" else "mixed"))
            self.prompt("[%d/%d] %s %s, target %s. Enter = run (q = quit) "
                        % (i + 1, len(self.trials), trial["phase"], trial["motion"],
                           self.describe(trial["target_id"])))
            record = self.run_trial(trial)
            self.log_trial(record)
            if self.pipeline:
                if record.status != "done":
                    # Do not issue a home command after an ambiguous timeout/failure.
                    raise Aborted("Trial failed; inspect executor/robot state before restarting.")
                self.go_home()
            if record.status != "done":
                print("Trial not completed (%s). Check the robot and the motion_node terminal." % record.status)
                self.go_home()
                if self.prompt("Repeat this trial? [Y/n] ") != "n":
                    continue
            i += 1

        self.show("text", "Thank you!", "The experiment is over.")
        print("\nSession complete: %s" % self.out_dir)


def main():
    rospy.init_node("experiment_node")
    ap = argparse.ArgumentParser()
    ap.add_argument("participant", help="e.g. P07")
    ap.add_argument("--start-trial", type=int, default=0, help="resume at this trial index")
    ap.add_argument("--config", help="same experiment YAML used by planner, optimizer and display")
    ap.add_argument("--first", choices=list(E.MOTIONS) + [""], default="",
                    help="game design: motion of rounds 1-4 (the other one in rounds 5-8); asked if not given")
    ap.add_argument("--auto", action="store_true",
                    help="game design: one 'go', then all trials run by themselves (study.launch)")
    ap.add_argument("--pipeline", action="store_true",
                    help="use RRT/FK pipeline and separate validator/executor; no legacy motion_node")
    args = ap.parse_args(rospy.myargv()[1:])
    if not sys.stdin.isatty():
        sys.exit("experiment_node needs the experimenter at an interactive terminal (use rosrun, not roslaunch).")
    try:
        node = ExperimentNode(args.participant, args.start_trial, args.config, args.pipeline, args.auto,
                              args.first or None)
    except SystemExit as error:
        if args.auto:  # own terminal window (study.launch): keep the message readable
            print(error)
            input("Enter closes this window and stops the launch. ")
        raise
    try:
        node.run_game() if node.game else node.run()
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
