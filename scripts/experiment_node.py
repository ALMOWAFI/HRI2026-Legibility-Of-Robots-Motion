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
"""
import argparse
import datetime
import json
import os
import queue
import signal
import subprocess
import sys
import threading

import rospkg
import rospy
import yaml
from std_msgs.msg import Empty, Int32, String

from hri_legible_motion import experiment as E
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
    def __init__(self, participant, start_trial):
        self.pkg = rospkg.RosPack().get_path("hri_legible_motion")
        with open(os.path.join(self.pkg, "config", "experiment.yaml")) as f:
            self.cfg = yaml.safe_load(f)
        e = self.cfg["experiment"]
        self.e = e
        self.participant = participant
        self.duration = float(self.cfg["motion"]["duration"])
        self.blocks = self.cfg["blocks"]
        self.button_to_block = list(e["button_to_block"])

        self.block_order, self.rotation = E.assignment(participant, e["assignment_seed"])
        self.colors = E.rotated_colors(e["colors"], self.rotation)
        seed = E.participant_number(participant)
        self.trials = E.make_schedule(self.block_order, [b["id"] for b in self.blocks],
                                      e["reps_per_block"], seed, e["max_same_in_row"])
        self.start_trial = start_trial

        traj_path = self.cfg["trajectories"]
        if not os.path.isabs(traj_path):
            traj_path = os.path.join(self.pkg, traj_path)
        self.traj_meta = trajectories.read_meta(traj_path)
        if abs(self.traj_meta["duration"] - self.duration) > 1e-6:
            sys.exit("Trajectory file was planned with T = %.2f s, config says %.2f s. Recompute."
                     % (self.traj_meta["duration"], self.duration))

        self.out_dir = os.path.join(os.path.expanduser(e["output_dir"]), participant)
        if os.path.exists(os.path.join(self.out_dir, "trials.csv")) and start_trial == 0:
            sys.exit("%s already has data. Use another participant id, or --start-trial N to resume."
                     % self.out_dir)
        self.log = E.SessionLog(self.out_dir)
        self.write_session_yaml(seed, traj_path)

        self.lock = threading.Lock()
        self.record = None
        self.accepting = False
        self.statuses = queue.Queue()
        self.ratings = queue.Queue()
        self.display_pub = rospy.Publisher("/display", String, queue_size=1, latch=True)
        self.trial_pub = rospy.Publisher("/trial_start", TrialInfo, queue_size=1)
        self.home_pub = rospy.Publisher("/motion_home", Empty, queue_size=1)
        self.choice_pub = rospy.Publisher("/choice", ChoiceEvent, queue_size=10)
        rospy.Subscriber("/button_press", Int32, self.on_button)
        rospy.Subscriber("/motion_status", String, lambda m: self.statuses.put((m.data, rospy.get_time())))
        rospy.Subscriber("/rating", Int32, lambda m: self.ratings.put(m.data))
        self.bag = None

    # --- setup -----------------------------------------------------------
    def write_session_yaml(self, seed, traj_path):
        path = os.path.join(self.out_dir, "session.yaml")
        info = {"participant_id": self.participant, "date": datetime.datetime.now().isoformat(timespec="seconds"),
                "block_order": list(self.block_order), "color_rotation": self.rotation,
                "colors_by_block": {b["name"]: c for b, c in zip(self.blocks, self.colors)},
                "schedule_seed": seed, "duration_T": self.duration,
                "trajectory_file": traj_path, "trajectory_hash": self.traj_meta["hash"],
                "button_to_block": self.button_to_block, "n_trials": len(self.trials), "notes": ""}
        if self.start_trial:
            path = os.path.join(self.out_dir, "session_resumed_at_%d.yaml" % self.start_trial)
            info["resumed_at_trial"] = self.start_trial
        with open(path, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False)

    def start_bag(self):
        if not self.e.get("record_bag", True):
            return
        name = "session.bag" if not self.start_trial else "session_from_%d.bag" % self.start_trial
        self.bag = subprocess.Popen(["rosbag", "record", "-O", os.path.join(self.out_dir, name)]
                                    + list(self.e["bag_topics"]),
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

    def show(self, screen, title="", text="", choice_block=-1):
        choice = self.button_to_block.index(choice_block) if choice_block >= 0 else -1
        self.display_pub.publish(String(json.dumps({
            "screen": screen, "title": title, "text": text,
            "cubes": self.participant_cubes(), "choice": choice})))

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
            if self.record is None or not self.accepting:
                rospy.loginfo("Button %d ignored (not during a reach)", msg.data)
                return
            if not self.record.add_choice(now, block):
                return
            trial_id = self.record.trial["trial_id"]
        ev = ChoiceEvent(trial_id=trial_id, choice_id=block)
        ev.header.stamp = rospy.Time.from_sec(now)
        self.choice_pub.publish(ev)
        self.show("reach", choice_block=block)

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

    def prompt(self, text):
        """Experimenter input; 'q' ends the session."""
        answer = input(text).strip().lower()
        if answer == "q":
            raise Aborted()
        return answer

    def go_home(self):
        while True:
            if not require_operator("Send the robot to the start pose?"):
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
    def run_trial(self, trial):
        """Returns the TrialRecord; robot is home afterwards unless status is failed/timeout."""
        announced = trial["phase"] == "announced"
        target = trial["target_id"]
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
        if not announced:
            self.show("reach")
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

        if announced:
            self.show("rating", "How natural did this motion look?",
                      "1 = not at all natural   ...   7 = very natural")
            record.naturalness = self.wait_rating(float(self.e["rating_timeout"]))
        else:
            m = record.measures(self.duration)
            self.show("score", "Score: %d / 10" % m["score"], choice_block=m["final_choice"])
            rospy.sleep(float(self.e["score_time"]))
            self.show("rating", "How confident were you in your final choice?",
                      "1 = not at all confident   ...   7 = completely confident")
            record.confidence = self.wait_rating(float(self.e["rating_timeout"]))

        status, _ = self.wait_status({"home", "failed"}, STATUS_TIMEOUT["home"])
        if status != "home":
            record.status = "return_" + (status or "timeout")
        return record

    def log_trial(self, record):
        row = self.log.write(record, self.participant, self.duration, self.blocks, self.colors)
        if record.trial["phase"] == "announced":
            extra = "naturalness %s" % row["naturalness"]
        elif row.get("f") is not None:
            extra = "f %.2f, score %s, final %s" % (row["f"], row["score"],
                                                    "ok" if row["final_correct"] else "wrong")
        else:
            extra = "no reach"
        print("  -> %s, %s, confidence %s" % (record.status, extra, row["confidence"]))

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
    args = ap.parse_args(rospy.myargv()[1:])
    if not sys.stdin.isatty():
        sys.exit("experiment_node needs the experimenter at an interactive terminal (use rosrun, not roslaunch).")
    node = ExperimentNode(args.participant, args.start_trial)
    try:
        node.run()
    except (Aborted, KeyboardInterrupt, EOFError):
        print("\nSession stopped. Resume with --start-trial <next trial index>.")
    finally:
        node.stop_bag()


if __name__ == "__main__":
    main()
