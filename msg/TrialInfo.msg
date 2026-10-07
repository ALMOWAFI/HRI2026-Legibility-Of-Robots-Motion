# Experiment node -> motion node: start of a trial.
std_msgs/Header header
int32 trial_id
int32 target_id      # Block.id the robot reaches for
string condition     # e.g. "baseline", "legible"
float32 bend         # legibility/bend level for this condition (0 = baseline)
