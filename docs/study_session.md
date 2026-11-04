# Running a study session (game design, 8 rounds)

One command starts everything: motion node, participant display, sound and the experiment
node. The experiment node opens in its own terminal window, where the experimenter types `go`
once; after that the rounds run by themselves, with one stop after round 4 for the paper
questionnaire.

```bash
# robot first, separate terminals (as before)
roscore                                    # own terminal, keep it open
~/scripts/launch_hardware.sh
~/scripts/resend_robot_program.sh          # pendant in Remote Control
# the session; first:= is the motion of rounds 1-4 (leave it out to be asked)
roslaunch hri_legible_motion study.launch participant:=P07 first:=predictable
```

Dry run without the robot (fake motion node, nothing moves, no MoveIt or driver needed;
use a new id every time, ids that are not `P##` stay off the leaderboard):

```bash
roslaunch hri_legible_motion study.launch participant:=dry01 dry_run:=true
```

Resume after a crash: `start_trial:=N` (0-based index of the round to repeat, counting the
practice rounds: with 5 practice rounds, game round 1 is N = 5 and round 5 is N = 9; the
experimenter window prints `[N+1/13]` before every round). The score so far is read back from
`trials.csv`, the first motion from `session.yaml`. Resuming inside a block skips the screen
that starts it (practice done / questionnaire).

## Design

- 8 rounds per participant: every cube (B1-B4) once with the predictable and once with the
  legible reach. Participants are not told that there are two kinds of motion.
- Practice first: 5 rounds from `experiment.training` (default P B2, L B3, P B4, L B1, P B3).
  They look exactly like game rounds (points bar, +points, score in the corner, labelled
  "Practice n / 5"). Then "Practice done!" shows the practice score, the score goes back to 0
  (practice never reaches the leaderboard) and the player starts the game with a cube key.
- Practice data is recorded apart: `practice_trials.csv`, `practice_choices.csv` and
  `practice.bag`; the game is in `trials.csv`, `choices.csv` and `session.bag` (game
  trial_id 0-7 as without practice). The practice replays the study's reaches, so those
  (motion, cube) pairs are seen again in the game.
- Block A, rounds 1-4: every cube once with one motion. Block B, rounds 5-8: every cube once
  with the other motion. The experimenter chooses block A's motion (`first:=predictable` or
  `first:=legible`; without it the experimenter window asks, Enter = the balanced default:
  odd participant numbers predictable, even ones legible).
- Between the blocks the participant fills in the questionnaire on paper. The game screen
  says "Break", the experimenter window waits for Enter, then the player presses a cube key
  to start block B. Robot at the start pose throughout; the bag keeps recording.
- The same cube never comes twice in a row, and block B never repeats block A's cube order.
- Counterbalancing over groups of 8 participants (with the default first motion): the cube
  order of block A comes from a balanced Latin square (every cube in every position, every
  cube pair as neighbours, once per 4 rows), one row per pair of participants, so each motion
  sees every cube in every block-A position once. Block B's order comes from a fixed seed
  (`experiment.assignment_seed`). The order depends only on the participant id and the first
  motion, and is written to `session.yaml` (with `first_motion`).
- Code: `experiment.game_order()`, tests in `test/test_experiment.py`.

Order for P01-P20 with the default first motion (P = predictable, L = legible). Choosing the
other first motion swaps the motions only; the cube order stays.

| | first | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|---|
| P01 | P | B1 | B2 | B4 | B3 | B4 | B1 | B3 | B2 |
| P02 | L | B1 | B2 | B4 | B3 | B1 | B4 | B2 | B3 |
| P03 | P | B2 | B3 | B1 | B4 | B2 | B1 | B3 | B4 |
| P04 | L | B2 | B3 | B1 | B4 | B2 | B4 | B1 | B3 |
| P05 | P | B3 | B4 | B2 | B1 | B4 | B1 | B3 | B2 |
| P06 | L | B3 | B4 | B2 | B1 | B3 | B4 | B1 | B2 |
| P07 | P | B4 | B1 | B3 | B2 | B1 | B2 | B4 | B3 |
| P08 | L | B4 | B1 | B3 | B2 | B4 | B3 | B2 | B1 |
| P09 | P | B1 | B2 | B4 | B3 | B4 | B2 | B3 | B1 |
| P10 | L | B1 | B2 | B4 | B3 | B4 | B3 | B2 | B1 |
| P11 | P | B2 | B3 | B1 | B4 | B1 | B2 | B4 | B3 |
| P12 | L | B2 | B3 | B1 | B4 | B3 | B4 | B2 | B1 |
| P13 | P | B3 | B4 | B2 | B1 | B2 | B3 | B4 | B1 |
| P14 | L | B3 | B4 | B2 | B1 | B4 | B3 | B1 | B2 |
| P15 | P | B4 | B1 | B3 | B2 | B4 | B1 | B2 | B3 |
| P16 | L | B4 | B1 | B3 | B2 | B3 | B1 | B2 | B4 |
| P17 | P | B1 | B2 | B4 | B3 | B2 | B1 | B3 | B4 |
| P18 | L | B1 | B2 | B4 | B3 | B2 | B4 | B1 | B3 |
| P19 | P | B2 | B3 | B1 | B4 | B3 | B1 | B2 | B4 |
| P20 | L | B2 | B3 | B1 | B4 | B1 | B2 | B4 | B3 |

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

Guesses count only from reach onset, not during the countdown. Points drain while the robot
moves: points left = 1000 x (1 - t/5 s)^2 (`experiment.points_max`, `points_exponent`), so a
right cube picked at onset and kept earns 1000, at 1 s 640, at 2 s 360, at 3 s 160. A wrong pick
earns nothing while it is selected; switching to the right cube later earns what is left then.
The bar on the screen shows this number. Maximum 8000 per game. The final screen shows the
total and the rank among all `P##` players (`<output_dir>/leaderboard.csv`).

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
4. `roslaunch hri_legible_motion study.launch participant:=P## first:=predictable|legible`.
5. In the experimenter window: Enter (setup checked), then `go` (robot to the start pose,
   then all rounds run by themselves).
6. The welcome screen waits until the player presses a cube key; this also tests the keys.
7. While it runs: `p` + Enter pauses after the current round, `q` + Enter stops after it.
   Typing in the experimenter window takes the keyboard from the game; it gets it back at
   the next round. Only the e-stop stops the robot during a reach.
8. After a failed round the session stops and asks for `go` again (robot home), then
   whether to repeat the round.
9. 5 practice rounds run first (scored on screen, score reset after); then "Practice done!"
   waits for a cube key.
   After round 4: the screen says "Break". Hand out the first questionnaire; when it is done,
   Enter in the experimenter window, then the player presses a cube key and block B starts.
10. At the end: Enter in the experimenter window closes everything. Data for both blocks in
   `~/hri_data/P##/` (`session.yaml`, `trials.csv` with `block` A/B and `points`,
   `choices.csv`, `session.bag`). Second questionnaire.

## Analysing the data

```bash
sudo apt install python3-statsmodels  # once (Debian); without it the mixed models are skipped
python3 tools/analyze.py --questionnaires questionnaires.csv
```

Reads every `~/hri_data/P##/` folder (ids that are not `P##` are dry runs and skipped),
keeps the last attempt of each game round, drops rounds that did not finish and
participants with fewer than 6 finished rounds (`--min-trials`, `--exclude P05`), and
writes to `build/analysis/`:

- `trials.csv`: the analysed trials, with `legible`, `edge` (B1/B4), `second` (block B),
  `first_motion`, `seen_in_practice` and `never_correct` (then `t_first_correct` = 1.0).
- `descriptives.csv`, `participant_means.csv`; `tests.csv`: per participant legible -
  predictable on all, edge and middle cubes and the edge - middle difference (bootstrap
  95 % CI, d_z, paired t, Wilcoxon).
- `models.txt`: `f`, `t_first_correct`, `t_commit` ~ legible * edge + second with a
  participant random intercept (+ legible slope when it fits), the order check
  (legible:second), and a logistic GEE for `final_correct`.
- `manipulation.csv`: lateral deviation and model legibility of every reach.
- `questionnaire_tests.csv`: paired Wilcoxon per scale plus the order check.
  `questionnaires.csv` has one row per participant and block: `participant_id`, `motion`
  (predictable | legible), then one numeric column per scale or item.
- `f_by_cube.png`, `recognition_curve.png` (share with the right cube over the reach).
