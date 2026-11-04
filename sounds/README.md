# Game sounds: what to record

`sound_node.py` plays `sounds/<name>.wav` when the file exists. The effects are already here
(Kenney, CC0, built by `tools/make_game_sounds.py`, see `CREDITS.md`); the voice lines are yours to record. Every missing clip falls back to
espeak-ng (voice lines) or a synthesized sound (effects), so the game works before anything is
recorded. Record one clip at a time and drop it in here; restart the launch to pick it up.

**Format:** WAV (aplay plays WAV only), mono or stereo, 44.1 or 48 kHz, 16 bit. Trim the silence
at the start: the countdown and the `go` sound must play on the beat. Keep all voice clips at
the same loudness (normalize to about -3 dB) and recorded by the same speaker.

Check what is still a fallback: the sound node prints `fallback for: ...` at startup.

## Voice lines (record exactly this text)

| File | Text | When |
|---|---|---|
| `welcome.wav` | Welcome! The robot will reach for one of the four cubes. Guess which one by pressing the key shown on that cube. Be quick: your points drain away while the robot moves, so the sooner you pick the right cube, the more you score. You can change your guess. | welcome screen, before round 1 |
| `get_ready.wav` | Get ready. | start of each round |
| `count_3.wav` | Three. | countdown |
| `count_2.wav` | Two. | countdown |
| `count_1.wav` | One. | countdown |
| `which_cube.wav` | Which cube? Quick! | when the robot starts moving (over the thinking loop) |
| `confidence.wav` | How sure were you? | confidence question after the reach |
| `correct.wav` | Correct! | final guess right |
| `wrong.wav` | Not this time. | final guess wrong |
| `no_answer.wav` | You did not pick a cube. | no key pressed during the reach |
| `halfway.wav` | Halfway there! | start of round 5 (instead of get_ready) |
| `last_round.wav` | Last round! | start of round 8 (instead of get_ready) |
| `final_score.wav` | That was the last round. Here is your score. | after round 8, before the score |
| `high_score.wav` | New high score! | best score so far (leaderboard rank 1) |
| `thank_you.wav` | Thank you for playing! | end of the game |
| `pause.wav` | Short break. Please wait. | experimenter paused, or a round failed |
| `questionnaire.wav` | Halftime! Please fill in the questionnaire. The game continues afterwards. | after round 4, before the second block |

Keep `count_3/2/1` under 0.8 s each (one number per second) and `get_ready`, `halfway` and
`last_round` under 2 s (they play during the 2 s round screen, `experiment.ready_time`). `welcome` can be
long: the welcome screen waits for the player to press a key.

## Effects (record or take from a free sound library, e.g. freesound.org, CC0)

| File | What |
|---|---|
| `tick.wav` | short tick under each countdown number, about 0.1 s |
| `go.wav` | start signal when the robot starts moving, about 0.4 s |
| `select.wav` | soft click when the player picks or changes a cube, under 0.05 s |
| `thinking.wav` | suspense loop while the robot moves (the robot 'thinking'), loops seamlessly, 5-6 s |
| `correct_fx.wav` | happy chime for a correct final guess, under 1 s |
| `wrong_fx.wav` | low 'no' sound for a wrong final guess, under 1 s |
| `fanfare.wav` | fanfare for the final score, 2-3 s |

`thinking` plays in a loop from reach onset until the reach ends (5 s, `motion.duration`), so
a 5-6 s clip that builds up suits it best. To hear the current fallbacks:

```bash
ls ~/.cache/hri_legible_motion/sounds/      # written by sound_node.py at startup
aplay ~/.cache/hri_legible_motion/sounds/thinking.wav
```
