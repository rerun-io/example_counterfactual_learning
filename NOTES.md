# Notes

Things we found while building this example, roughly in workflow order. Most of them are problems that would have
quietly skewed the results; the rest are practical tips for working with the data and with Rerun.

## Recording

- **Save the full simulator state with every demo.** ACT stores actions as float32, but the scripted policy steps the
  simulator in float64, so replaying the saved actions does not reproduce the episode (joint differences up to 0.025).
  Every later step that re-renders or rebuilds a moment (counterfactuals, 3D scenes, ghost arms) reads the recorded
  state instead. `act.patch` adds a `sim_state` group to the demo HDF5 files.
- **The demos are 17× smaller in Rerun than in HDF5**, with no loss (pixel-identical frames).
- **numpy 2 changes seeded random draws by about one ulp.** Cube start poses differ from numpy 1 in the last digit. It
  doesn't change any outcome here, but don't expect bit-identical poses across numpy versions.

## Reviewing the demos

- **ACT's reward stages flicker.** In 7 of 50 demos the "lifted" stage fires at first touch while the cube is still on
  the table: the cube's table contact drops out of MuJoCo's contact list for one step when a finger presses it, and
  ACT records the maximum stage over the episode. Success is unaffected, but partial-progress numbers are wrong (of 14
  failed rollouts, the original stage says 5 lifted the cube, only 1 did). `qa` adds a stable stage (held 5 steps, cube
  at least 5 mm up) next to the original one, and `qa:lift_gap` makes the flicker demos a one-click sort.
- **The scripted demos are near-identical in timing.** The handoff happens at step 291 or 292 in all 50. The only real
  variation is the cube's start position, which is part of why the policy is brittle.
- **Write the comparison you want to sort by as its own property.** The viewer's segment table sorts columns but can't
  compute them.
- **Give each layer's properties a unique name.** If several layers on a segment send a property group with the same
  name, the segment table shows only one of them, with no hint that the others exist.

## Counterfactuals

- **Nuisance objects must not touch the physics.** The distractors and support patch have no collision geometry, and
  physics with the nuisance scene is checked to be bit-identical to the original.
- **Check task preservation geometrically, not visually.** At our original strongest distractor setting, the lifted
  cube passed through a distractor in 34 of 50 demos (up to 7.1 mm). From the policy's top camera that looks normal.
  `task_preservation` measures signed distances from the cube and gripper meshes to every distractor per step, and
  `scene3d` rebuilds the scene in 3D so a flagged step can be checked from any angle. Moving the distractors from
  4.5 to 5.5 cm fixed it.
- **Verify rendered colors, not just parameters.** In mujoco 2.3.7 / dm_control 1.0.14, the numpy views of the
  headlight's ambient/diffuse/specular have stride 0, so writing all three channels only sets red and gives the scene
  a teal cast, while reading the values back looks correct. Lighting conditions switch the headlight off via its
  `active` flag and set ambient through `light_ambient` instead.

## Measuring drift

- **Drift predicts closed-loop failure.** Across 11 conditions, mean open-loop drift ranks the conditions by
  closed-loop failure rate with Spearman 0.86 (the paper reports 0.88).
- **Distractor severity is inverted for this policy.** Far distractors hurt more than near ones, in drift and in
  closed-loop success. Worth knowing before assuming "closer means harder".
- **Look at high-drift cases before trusting them.** A large drift can be a harmless alternative action. Since the
  clean and counterfactual frames show the same recorded pose, the drift isn't visible in the cameras; the ghost arms
  (`ghost`) pose the robot at the predicted actions under each frame so you can see what the policy intended.
- **Drift is a single catalog query.** With both predictions stored as layers, `avg(array_distance(a_cond, a_clean))`
  grouped by segment reproduces the numpy pipeline to about 1e-7 (`drift_query`).

## Selecting and training

- **Top-drift selection spends the whole budget on one failure mode** (all 20 picks on the hardest combination).
  Random picks include edits the policy already ignores. Drift-weighted coverage keeps drift high and covers more
  distinct response modes.
- **Pick the Rerun dataset type by access pattern.** The clean demos are dense (every step of every episode is a
  sample), so `RerunIterableDataset` streams them with internal shuffling; it matched ACT's own loader sample for
  sample at about 280 samples/s on one process. The repair set is a short list of exact steps on sparse layers (frames
  every 10th step), so `RerunMapDataset` indexes straight to them and the set is cached for all epochs.
- **Express training-time padding as a layer.** ACT pads the action chunk with zeros past the episode end. Windowed
  fields only return samples whose window exists, so the `train_pad` layer supplies zero actions and an `is_pad` mask
  past step 399.
- **On macOS, use `ACT_NUM_WORKERS=0`.** DataLoader worker processes were 5× slower than loading in the main process
  for ACT's HDF5 loader, and worker processes need semaphores that some sandboxes don't allow.
- **ACT's best checkpoint is chosen by validation loss, which late in training is mostly the KL term.** We evaluate the
  final checkpoint.

## Evaluating

- **ACT's success metric counts dropped handoffs.** A rollout succeeds if the receiving gripper ever touches the lifted
  cube, even if the cube then falls. Under nuisances that's hundreds of rollouts; with the nearest distractors, success
  drops from 52% to 28% when the cube has to still be held at the end. `qa --rollouts` adds `qa:final_cube_z` and
  `qa:success_strict` to every rollout.
- **Clean failures cluster at the edge of the cube pose range** (all 14 at y ≈ 0.40–0.42 and/or x ≈ 0–0.07). Compare
  nuisance failures against that set.
- **Closed-loop evaluation is compute-bound.** With temporal ensembling ACT runs at every step (about 40 GFLOP per
  query). `fast_eval` steps several environments in lockstep with one batched forward pass (1.5× faster on an M4 Pro,
  identical outcomes on all 100 clean rollouts). Rendering only the policy's camera saves another ~25%.
- **Give every rollout its 3D scene.** If the layout shows a 3D view, every segment needs the data behind it.
  Building it from recorded state takes about a second per rollout.

## Real data

- **78% of the episodes are still moving when the recording stops** (fixed 8 s). Nothing in the dataset says whether
  the task finished.
- **The follower arm lags the operator by 80–100 ms.** ACT's real-data loader pairs observation t with action t−1 for
  this reason; with the Rerun loader that's an action window of `range(-1, 99)`.
- **Operators are idle 19–41% of the time, and gripper-command reversals range from 4 to 102 per episode.**
- **Joint readings are quantized** (about 0.0015 rad), so "the arm is still" needs a threshold above sensor jitter.
- **Camera brightness varies by only ±3% across all 50 episodes**: one session, almost no natural visual variation.
- **Counterfactuals on real frames are edits, and the masks carry the safety argument.** The overhead camera is
  static, so each episode's temporal median is the empty table. Everything that isn't table gets a keep-out mask with
  a margin, distractors are placed clear of everywhere the arms go, and every edit is checked to change no keep-out
  pixel. Pixels that change during the episode (the arms' shadows) are left alone, which is conservative.
- **Encode clean and edited frames the same way.** Both go through the same JPEG round trip, so drift measures the
  edit, not the compression.
- **Offline evaluation needs confidence intervals.** With 190 held-out samples, gaps of 1–2% are often noise. The
  drift-weighted repaired policy does about 2% better with the support patch than without (the patch is in half of
  its training picks); the other small negative gaps are within noise.

## Rerun practicalities

- **On the open-source server, give tables an explicit location.** Without one, `create_table` puts the table in a
  temporary directory. `hub.ensure_table` stores them under `rerun/tables/`, where `serve` finds them on restart.
- **Never rewrite a registered file.** The catalog references files in place; overwriting one breaks its entry until
  it's re-registered. `hub sync` re-registers changed files, and blueprints are registered as content-addressed copies.
- **Keep a client alive while you use a lazy DataFrame.** Query results are streamed through the client that created
  them.
- **Create recordings in batches of processes.** One process ran out of threads after about 3,000 `RecordingStream`s;
  `scene3d --all_rollouts` runs in subprocess batches of 800.
- **Raise the open-file limit for thousands of layer files.** `serve` raises it to the hard limit on macOS.
- **Viewer layouts are remembered per application.** If a dataset's default blueprint changes and the viewer keeps
  showing the old layout, `rerun reset` clears it.
- **mujoco 2.3.7 can't import on recent macOS** because `mujoco/cgl` loads a path that no longer resolves.
  `setup.sh` patches the path.
