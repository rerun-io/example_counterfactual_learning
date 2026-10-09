# Reference results

Our numbers from the run behind the blog post, for comparison. Generated files with the same names land in `results/`.

- `eval.csv`: closed-loop success per nominal seed and condition (100 rollouts each), with the strict success rate
  (cube still held at the end) next to ACT's original metric.
- `demo_qa.csv`: per-demo QA summary (reward stages, stable stages, `lift_gap`, timing, jerk, final cube height).
- `selection/`: the three seed-0 repair selections (K=20): top drift, random, drift-weighted coverage.
- `real/qa_real.csv`: per-episode teleop QA for `lerobot/aloha_static_cups_open`.
- `real/offline_eval.csv`: held-out behavior-cloning error on clean and edited frames for the original and repaired
  real-data policies, with paired bootstrap 95% CIs (computed on CPU).
