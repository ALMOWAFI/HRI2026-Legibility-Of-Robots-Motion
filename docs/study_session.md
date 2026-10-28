# Running a study session (game design, 8 rounds)

One command starts everything: motion node, participant display, sound and the experiment
node. The experiment node opens in its own terminal window, where the experimenter types `go`
once; after that the 8 rounds run by themselves.

```bash
# robot first, separate terminals (as before)
~/scripts/launch_hardware.sh
~/scripts/resend_robot_program.sh          # pendant in Remote Control
# the session
roslaunch hri_legible_motion study.launch participant:=P07
```

Dry run without the robot (fake motion node, nothing moves, no MoveIt or driver needed;
use a new id every time, ids that are not `P##` stay off the leaderboard):

```bash
roslaunch hri_legible_motion study.launch participant:=dry01 dry_run:=true
```

Resume after a crash: `start_trial:=N` (0-based index of the round to repeat; the score so far
is read back from `trials.csv`).

## Design

- 8 rounds per participant: every cube (B1-B4) once with the predictable and once with the
  legible reach. Participants are not told that there are two kinds of motion.
- Rounds 1-4: every cube once, 2 predictable + 2 legible. Rounds 5-8: every cube once
  with the other motion. Practice and fatigue therefore hit both motions equally.
- The same cube never comes twice in a row; the same motion at most twice in a row
  (`experiment.max_motion_in_row`).
- Counterbalancing over groups of 8 participants: the cube order of rounds 1-4 comes from a
  balanced Latin square (every cube in every position, every cube pair as neighbours, once per
  4 rows), and odd participant numbers start with the predictable motion, even ones with the
  legible motion. The rest comes from a fixed seed (`experiment.assignment_seed`), so the order
  depends only on the participant id and is written to `session.yaml`.
- Code: `experiment.game_order()`, tests in `test/test_experiment.py`.

Order for P01-P20 (P = predictable, L = legible):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| P01 | P B1 | L B2 | P B4 | L B3 | L B4 | P B2 | L B1 | P B3 |
| P02 | L B1 | L B2 | P B4 | P B3 | L B4 | P B2 | L B3 | P B1 |
| P03 | P B2 | L B3 | L B1 | P B4 | L B2 | P B1 | L B4 | P B3 |
| P04 | L B2 | P B3 | L B1 | P B4 | P B2 | L B4 | L B3 | P B1 |
| P05 | P B3 | L B4 | P B2 | L B1 | P B4 | L B3 | P B1 | L B2 |
| P06 | L B3 | L B4 | P B2 | P B1 | L B2 | P B3 | P B4 | L B1 |
| P07 | P B4 | L B1 | P B3 | L B2 | P B1 | L B4 | L B3 | P B2 |
| P08 | L B4 | L B1 | P B3 | P B2 | L B3 | L B2 | P B4 | P B1 |
| P09 | P B1 | P B2 | L B4 | L B3 | P B4 | L B1 | P B3 | L B2 |
| P10 | L B1 | P B2 | P B4 | L B3 | L B4 | P B3 | L B2 | P B1 |
| P11 | P B2 | L B3 | P B1 | L B4 | L B1 | P B3 | L B2 | P B4 |
| P12 | L B2 | P B3 | L B1 | P B4 | P B1 | L B4 | L B3 | P B2 |
| P13 | P B3 | L B4 | P B2 | L B1 | P B4 | P B1 | L B3 | L B2 |
| P14 | L B3 | P B4 | L B2 | P B1 | L B4 | P B2 | P B3 | L B1 |
| P15 | P B4 | L B1 | L B3 | P B2 | L B4 | P B1 | P B3 | L B2 |
| P16 | L B4 | P B1 | P B3 | L B2 | L B1 | P B2 | L B3 | P B4 |
| P17 | P B1 | L B2 | L B4 | P B3 | P B4 | L B3 | L B1 | P B2 |
| P18 | L B1 | P B2 | L B4 | P B3 | L B2 | L B3 | P B4 | P B1 |
| P19 | P B2 | L B3 | P B1 | L B4 | L B2 | P B3 | P B4 | L B1 |
| P20 | L B2 | P B3 | P B1 | L B4 | P B2 | P B4 | L B3 | L B1 |

## One round

| Step | Screen | Sound |
|---|---|---|
| Round screen, 2 s (`ready_time`) | "Round n of 8", score | `get_ready` (round 5: `halfway`, round 8: `last_round`) |
| Countdown 3-2-1, 1 s each | big number, cubes dimmed | `tick` + `count_3/2/1` |
| Reach starts (5 s) | "Which cube?", keys A S D F on the cubes, points bar 100 -> 0 from reach onset | `go`, at onset `which_cube` + `thinking` (ticks speed up) |
| Player presses keys | chosen cube outlined | `select` |
| Reach ends | (only with `confidence_rating: true`: "How sure were you?" 1-7) | (`confidence`) |
| Result, 3 s (`score_time`) | Correct! / Not this time, +points, robot's cube | `correct_fx` + `correct` / `wrong_fx` + `wrong` |
| Robot returns to the start pose | | |

Guesses count only from reach onset, not during the countdown. Points = 100 x the fraction of
the 5 s reach during which the selected cube was the robot's cube: a correct guess at 1 s gives
about 80 points, a correct guess only at the very end almost 0. A wrong guess earns nothing
while it is selected. Maximum 800 per game. The final screen shows the total and the rank among
all `P##` players (`<output_dir>/leaderboard.csv`).

Sounds: effects are Kenney CC0 samples (`tools/make_game_sounds.py`, `sounds/CREDITS.md`);
`sounds/README.md` lists the voice lines to record. Until a clip is recorded, a fallback
plays (espeak-ng voice or a synthesized effect).

## Checklist before each participant

1. Robot stack running, pendant in Remote Control, **speed slider 100 %**, e-stop in reach.
2. `tools/place_blocks.py --execute`: cubes placed at the spots in the config colours (B1 green,
   B2 red, B3 orange, B4 yellow; `rotate_colors: false`), as in RViz. Seen from the participant's
   side (facing the robot, like the RViz view) the screen shows them left to right as
   B4 yellow (A), B3 orange (S), B2 red (D), B1 green (F); the experiment node prints this
   order at startup.
3. Display on the participant's monitor (`display.geometry`), sound audible, volume set.
4. `roslaunch hri_legible_motion study.launch participant:=P##`.
5. In the experimenter window: Enter (setup checked), then `go` (robot to the start pose,
   then all rounds run by themselves).
6. The welcome screen waits until the player presses a cube key; this also tests the keys.
7. While it runs: `p` + Enter pauses after the current round, `q` + Enter stops after it.
   Typing in the experimenter window takes the keyboard from the game; it gets it back at
   the next round. Only the e-stop stops the robot during a reach.
8. After a failed round the session stops and asks for `go` again (robot home), then
   whether to repeat the round.
9. At the end: Enter in the experimenter window closes everything. Data: `~/hri_data/P##/`
   (`session.yaml`, `trials.csv` with `points`, `choices.csv`, `session.bag`). Questionnaire.
