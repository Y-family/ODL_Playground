# Open Duck Mini ZP10D dynamics model

## Run the current robot

From the repository root, run the current 16-actuator model in MuJoCo:

```bash
LP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/open-duck-uv-cache uv run --no-sync \
  scripts/cad/run_memory_limited.py --wait playground/open_duck_mini_zp10d/simulate.py
```

The default holds the published home pose using the model's position actuators
and floating-base physics. Closing the window or pressing Ctrl-C exits.
If another memory-limited task is active, `--wait` queues this command until
that task exits. Ctrl-C cancels the wait without interrupting the active task.
The saved low-clearance liftoff commands predate the R11 leg shortening.
`--replay left` and `--replay right` currently reject those commands because their
model hashes differ. New trajectories must be validated before replay is available
for the shortened robot. Replay integrates position commands through physics and
does not overwrite qpos with saved animation frames.

These are command replays, not live feedback, the current unaccepted 10 mm
controller candidate, or a learned walking policy. The original
`BEST_WALK_ONNX_2.onnx` is not used. A trained walking policy for this ZP10D
model is not yet available. See `cad/local_fix/UPPER_REFIT_ISSUES.md` for current
control development and the accepted test's scope.

For a finite run without a window, add `--headless --duration 5`. With
`--headless --replay left` (or `right`) and no duration, the full saved test is
rerun and its integrated qpos is checked against the saved trajectory. All runs
use the shared serial lock and 4 GB address-space cap.

Synchronization source: `cad/local_fix/open_duck_mini_upper_refit.step`.
The formal XML and meshes were synchronized and validated on 2026-10-17 after shortening each thigh and shank by 10 mm (78.65 to 68.65 mm).
The source STEP is read without modifying geometry. Its SHA256 is
`bb9c851dfcae554653981919bf69a950febc2730c4362e562f58a0a63f7ce740`.

The generator updates all 115 physical solids, their visual meshes, moving-link
assignments, centres of mass and inertia tensors. Construction frames are omitted.
It uses the measured current CAD axes for the 14 ZP10D joints and the two existing
SG90 antenna hinges. S13 is ±100°, S11 is −20…39°; antenna motion remains forward
and backward. SG90 ±90° is a diagnostic range pending hardware calibration.

The estimated total mass is 2.080124393 kg. ZP10D assemblies use 56 g each,
including S01/S06 whose actual mass after removing the upper output is unmeasured.
Printed parts retain the original effective material density. Purchased components
retain their legacy mass, with current geometry for the COM and uniform-density
inertia estimate. Two legacy zero-mass flash parts use provisional 1250 kg/m³.
Actual part weights, infill, wiring and actuator dynamics still need calibration.

Printed collision geometry uses convex decomposition; per-part target settings
and actual piece counts are recorded in the mass report. Hull caps can exceed the
target tolerance. Purchased components use conservative envelopes that can fill
holes. MuJoCo excludes same-body and immediate parent-child contact by default.
These proxies serve rigid-body dynamics, not a 0.2 mm CAD clearance certificate.
Existing STEP self-intersection warnings remain recorded in the CAD review.

The first 14 actuator indices retain their previous order; the two antenna
actuators are appended. Joint traversal inserts the antenna coordinates inside
the head subtree: use joint names to locate qpos/qvel entries. The complete scene
has 23 qpos entries, 22 velocity entries and 16 actuators. The existing 14-action
policy and inference code require named mappings and observation/action adaptation.

## Regenerate and validate

Run from the repository root, with the project's CAD and MuJoCo environment and
CoACD 1.0.14. Commands are serial and enforce the 4 GB address-space cap before
native imports. Keep other geometry jobs idle; the workspace total limit is 10 GB.
Candidates use the fixed `/tmp/open-duck-dynamics-sync` directory.

```bash
.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/build_zp10d_mjcf_from_final_step.py prepare
for index in {0..114}; do
  .venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/build_zp10d_mjcf_from_final_step.py collision "$index" || break
done
.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/build_zp10d_mjcf_from_final_step.py assemble
.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/validate_zp10d_dynamics_sync.py
.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/build_zp10d_mjcf_from_final_step.py publish
MUJOCO_GL=egl LP_NUM_THREADS=1 .venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/validate_zp10d_dynamics_sync.py preview
```

`open_duck_mini_zp10d_draft.xml` supplies the legacy tree, frame orientation and
actuator defaults. The generator replaces its geometry, poses, inertia and limits
to produce the formal `xmls/open_duck_mini_zp10d.xml` and `scene_flat_terrain.xml`.

Validation checks SI units against an analytic solid, all axes and limits,
all current CAD bounds, explicit link inertials and 575 part poses. It probes
head/ring/body cavities, detects the four demonstrated collision endpoints, and
runs three 4-second numerical checks: fixed hold, fixed sweep and free home hold.
Fixed tests constrain the free base with a weld equality, preserving the normal
parent-contact filtering. Removing its free joint would cause artificial contact
between servo housings and their own horns. The home base height is recomputed
from the current contact meshes: 0.157739322 m, with a 0.2 mm initial floor gap.
The gravity-only torque diagnostic assumes the base is externally supported; it
is not the joint loading of a standing robot with ground reactions.

Publication binds validation to the candidate manifest, meshes and XML hashes,
then recompiles the formal scene and compares its model arrays with the validated
candidate. Full evidence is in
`cad/local_fix/open_duck_mini_zp10d_mass_properties.json`.

## Published validation

R14 (2026-10-18) adopts **60-degree knee flexion** on both shortened legs.
Geometric home is exactly 60 degrees; estimated position-actuator load
compensation uses commands of 58.209 / 58.179 degrees. Actual loaded standing
angles are 60.023 / 60.038 degrees (left S09 / right S04). Hip pitch, ankle and
small hip-roll compensation commands are synchronized to retain flat soles and
front-view vertical legs. These are simulation coordinates, not calibrated
hardware commands.

Matched 13-second holds reduce mean absolute knee torque from 0.517 / 0.510 to
0.423 / 0.434 N.m (18.11% / 14.88%). This does not establish the earlier 32%
estimate. COM rises 12.15 mm and the contact support margin changes from 37.34
to 37.67 mm. Under the same 4 N backward push for 0.2 s, maximum body tilt
increases from 2.066 to 3.066 degrees; both recover. The tested change reduces
standing load while increasing the backward disturbance excursion.

Nine cases total 78.2 seconds: before/after holds, a 9-second stance transition,
four 2 N / 0.2 s directional pushes at the new stance, and the matched 4 N
backward comparison. All finish without numerical warnings, torque saturation,
nonfoot-ground stops, active self contact or joint-limit violations. Transition
peak motor torque is 0.668446 N.m. Local checks cover 25 poses and 74 leg/adjacent
parts, with 200 close-pair BRep measurements retaining at least 0.500 mm and zero
intersection. This is sampled local clearance, not full-range certification.

The model retains 115 physical parts, 16 actuators and 3180 convex collision
meshes. CAD shapes, masses, inertias, joint axes, gains and physical limits are
unchanged; compiled structural arrays were compared. Formal scene home, mass
report metadata and generation cache are synchronized. Current evidence is
`cad/local_fix/CURRENT_STANCE_ASSESSMENT.md`, `.json` and `.png`; previous R09
files under those names are archived in
`/tmp/open-duck-knee-stance/previous_CURRENT_STANCE_ASSESSMENT.*`.

`CURRENT_STANCE_ASSESSMENT.png` compares the actual loaded side views.
`open_duck_mini_zp10d_dynamics.glb` and `.png` show the published 60-degree home.
Earlier R12/R13 tests use the former ~81-degree knee pose. Their liftoff commands
remain incompatible; no earlier failed or accepted trajectory is current gait
evidence. Walking, single-leg support, actual actuator calibration, PETG strength
and continuous rated torque have not been validated by these standing checks.

## Current training workflow

Follow the original project: **Placo reference → polynomial fit → MJX/Brax PPO →
full MuJoCo motion validation → ONNX**. The previous hand-written liftoff
controller is no longer a prerequisite. CAD, the published 60-degree home,
inertias, actuator gains, damping, torque caps and physical limits are unchanged.

The current URDF has 21 links, 20 joints and 16 movable joints, with all 115
visual parts and the source inertias. Strict URDF validation and Placo/MuJoCo
comparison of every rigid-body frame and COM at 33 poses passed (maximum frame
error 3.02e-13). External `check_urdf` is unavailable; a browser joint sweep is
still pending. The URDF is for kinematic planning and deliberately omits collision
geometry; native full-model MuJoCo supplies the contact validation.

The reference covers **standing and 0.03333333 m/s forward motion only**. Its
period is 0.6 s. Placo now compensates each foot origin using the actual oriented
collision hull, so the lowest corner follows the planned 15 mm swing height.
The earlier degree-15 Chebyshev fit had 0.040491 rad (2.32 degrees) joint error
and about 3.132 mm floor penetration. The current fit uses 120 periodic cubic
pieces, sampled from 200 Hz IK output; its joint velocities are the analytic
derivatives of those position polynomials. At 1 ms replay intervals maximum
penetration is 0.095565 mm and height reconstruction error 0.033550 mm.
Planning speed is limited to 4 rad/s for interpolation headroom; all polynomial
speed extrema remain below 4.752354 rad/s, within the unchanged 5.24 rad/s cap.
This verifies reference kinematics with the planned base, not physical tracking.
The subsequent demonstration dynamics audit **failed**: among 120 samples in
one cycle, 58 have no unilateral frictional support-force solution, even with
the entire projected sole allowed to contact the floor. The other 62 require
motor torques above the current provisional caps; the smallest optimized peak
ratio is 1.556. This uses the fitted cubic's analytic joint derivatives and the
unchanged full MuJoCo inertias, armatures and damping, with optimistic joint dry
friction. The same allocation accepts static home as a sanity check. This is
a necessary-condition diagnostic with finite-difference base acceleration,
not a free-base tracking trial or proof that the robot cannot lift a foot.
The current Placo demonstration therefore needs support/COM/acceleration
correction and dynamic validation before it can be called a usable example.
Reproduce with `.venv/bin/python scripts/cad/run_memory_limited.py
playground/open_duck_mini_zp10d/assess_reference.py --projected-contacts` after generating the current
scratch candidate. Results are recorded under `demonstration_dynamics_audit`
in `training/reference_assessment.json`; formal models and reference data are
unchanged by this read-only audit.
The 2026-10-18 follow-up restricts support to collision-hull vertices within
0.2 mm of the floor (the default; `--actual-contacts` remains an explicit alias).
The entire projected sole requires the diagnostic `--projected-contacts` flag.
Published home's
startup air gap is removed only for the static sanity check. Of the same 120
demonstration samples, 76 have no contact-force solution and the remaining 44
exceed the provisional motor caps. These are contact geometry and inverse-dynamics
checks, not physical replay. A small `--check` case verifies that a raised sole
corner cannot support the robot and that an airborne foot supplies no floor force.

Repair trials retained the 15 mm swing goal and physical limits. Slower timing,
flat soles, inward swing arcs, free COM height, implicit height compensation,
IK regularization and post-filtering have not produced an accepted replacement.
Temporary landing-spacing/COM-height trials also failed; they do not change home
or the published reference. Stronger regularization reduced acceleration in some
trials while increasing geometric tracking error. Failed candidates were not
promoted. `training/reference_assessment.json` records the configurations, script
sources and results under `demonstration_repair_study`.

ZP10D gain, damping, friction and armature values remain inherited and unmeasured;
the audit now records the separate damping/armature terms for diagnosis. They
are not measured servo torque. Supply voltage, torque/speed data and response
measurements are needed for hardware conclusions. Reference repair is still open.
`reference.py --fit` now defaults to `/tmp/zp10d-reference/data`; use that same
`--data-dir` for candidate audits. Reward metadata preparation binds new fits to
the candidate SHA and checks legacy joint samples against the fitted reference.
The positive scratch fit and negative mismatched-candidate checks passed, with
both published reference files unchanged.
The shared readers still accept original monomial and Chebyshev files. Reference reader, joint/reward
mapping, both support phases, contact filtering and 113-feature actor observation
parity have runnable checks in `test_adaptation.py`.

`training_model.py` prepares `/tmp/zp10d-training/scene.xml`, an explicitly
approximate 18-box, floor-only training scene without self-contact. Joint frames,
inertias, armature, friction, damping, actuator gains/force/position limits and
home are compared to the full scene. The unchanged full model has 3296 geoms;
its MJX collision allocation exceeded the current GPU allocator allowance when
requesting 559.54 MiB. That does not mean all 2 GB of VRAM was exhausted. The
training approximation must always be followed by full-model evaluation.

The workspace RAM budget is 10,000,000,000 bytes. All JAX commands, including CPU
JAX, use `scripts/cad/run_memory_limited.py --gpu`: before native imports a
systemd cgroup limits the task and every descendant to 7,000,000,000 bytes actual
RAM and zero swap. WSL currently exposes about 8.2 GB RAM, so this task cap leaves
about 1 GB for the system and tools. CAD retains its 4 GB address-space
limit. Both modes use the same serial lock; native computation uses one thread
and training is pinned to one CPU. JAX GPU preallocation is disabled, with default
allocator fraction 1.0 (user selected the full VRAM budget); allocation remains
on demand. VRAM is separate from the RAM limit.

The MX350 (2 GB, compute capability 6.1, driver 582.78) now runs GPU PPO with
JAX/JAXlib/CUDA plugin 0.8.2 and Flax 0.12.2, pinned by the CUDA extra.
The earlier `CUDA_ERROR_NOT_SUPPORTED` was traced to MJX's dense mass-matrix
boolean scatter: its compiled kernel contained `atom.sys.global.cas`, which
Windows/WSL Pascal drivers reject. The GPU adapter installs
`playground/common/mjx_compat.py`, constructing the same topology mask with
NumPy before GPU compilation. Mass-matrix arithmetic is unchanged; batched
dense/sparse results, with and without a diagonal, match the original CPU
implementation. The failed PPO PTX contained one system-scope atomic; the fixed
PTX contains none. Real GPU parameter updates and checkpoints pass; this does
not establish walking performance. CPU remains available via `JAX_PLATFORMS=cpu`.

The GPU regression run completed 2,048 environment transitions with 16 environments
and `--no-eval`, saving checkpoints at 1,024 and 2,048 transitions. Policy weights
changed by up to 0.00480229; the run took 152.154 seconds including compilation
and peaked at 2.142 GB process RAM. Evidence is in `training_gpu/gpu_validation.json`
and `training_gpu/train.log`. A separate checkpoint reload/inference check was
not run because another training task held the shared serial lock.

### Selected training configuration: 512 environments

On 2026-10-18 the user selected **512 parallel environments on the MX350** for
subsequent ZP10D training. The runner and GPU launcher now default to this mode:
CUDA, no VRAM preallocation, memory fraction 1.0, optimized rewards, unroll 16,
batch size 512, and two PPO updates per batch. The default short run requests
49,152 transitions with checkpoints every 8,192 transitions; longer training
sets `--steps` and `--checkpoint-every` explicitly. Normal training retains
evaluation; `--no-eval` is for throughput measurements.

The 512-environment test completed six batches / 49,152 transitions and saved
six checkpoints. After discarding the first two epochs, which still included
JAX compilation, throughput was 849.43–859.51 transitions/s (median 852.47).
Observed GPU memory peaked at 600 MiB total, GPU utilization at 98%, and process
RAM at 2.066 GB. At that rate 300 million transitions take about 4.07 days of
training computation, excluding evaluations, checkpoints and interruptions.
The earlier 20.5 transitions/s estimate included compilation and should not be
used as steady training speed. This is a short throughput check, not a walking
acceptance or a convergence guarantee. All completed trials used identical
model/reference/reward inputs.

Evidence and comparison: `training_gpu/parallel_benchmark/RESULTS.md` and
`results.json`; per-run logs, resource samples and checkpoints remain in that
directory. 1,024 environments also passed, with similar throughput. The user
stopped the capacity search during the 2,048-environment trial and selected
512; the hardware capacity limit was not determined.

The subsequent GPU profile kept these training and physics settings. Warm
throughput was 853.66 transitions/s; physics accounted for approximately 95.59%
of measured kernel time, including 53.57% in solver line search. Batched solver
iterations were 6–9 (mean 7.06), while each line search still ran ten scans.
That baseline identified line-search reductions and conditional execution as
the first optimization target. Measured attribution, limitations and raw traces are documented in
[`training_gpu/profile/RESULTS.md`](training_gpu/profile/RESULTS.md).

GPU training now automatically limits line-search friction calculations to the
16 friction rows among 304 constraint rows. The original scan, tolerances, FP32,
2 ms physics step and ten substeps remain. A sequential 512-environment PPO
comparison completed 49,152 transitions and six updates per version: warm
throughput increased from **843.75 to 933.16 transitions/s (+10.60%)**, and
attributed line-search kernel time fell from 4.976 to 4.076 s (18.09%). Initial
compilation took longer; complete six-epoch diagnostics took 213.83 versus
277.20 s, so this improvement applies to sustained training after warm-up.

Point costs/derivatives match in standalone checks; sampled in-simulation
checks and 512-environment short physics trajectories passed the documented
bounds. Full trajectories are not bitwise identical: contact solves amplify
FP32 rounding. This does not establish long-run convergence or walking quality.
The runner records the actual implementation and adapter SHA256 in metadata.
See [`training_gpu/linesearch/RESULTS.md`](training_gpu/linesearch/RESULTS.md)
for numerical limits, PPO samples and raw traces. Normal training keeps evaluation
enabled; the comparison used `--no-eval` only to measure throughput.

A second accepted change replaces coefficient concatenation with a slice update.
Another sequential 512-environment comparison, with 49,152 transitions and six
updates per version, increased warm throughput from **929.60 to 989.48
transitions/s (+6.44%)** relative to the preceding optimization. Independent
FP64 point equations, autodiff, boundary cases and two-second physics comparisons
passed fixed bounds. The 128-environment physics check was 5.47% slower, so this
gain applies to the selected 512-environment configuration. Long-run convergence
and walking remain unverified. See
[`training_gpu/linesearch/reduction/RESULTS.md`](training_gpu/linesearch/reduction/RESULTS.md)
for the adopted one-line change, validation limits and raw results.

Example next training stage (choose a fresh output directory):

```bash
JAX_PLATFORMS=cuda .venv/bin/python scripts/cad/run_memory_limited.py --gpu --wait \
  playground/open_duck_mini_zp10d/runner.py \
  --steps 1048576 --checkpoint-every 262144 \
  --output playground/open_duck_mini_zp10d/training_gpu/walk_512
```

This command uses the default 512 environments and keeps evaluation enabled.
It starts a new policy; warm starts explicitly supply `--restore` and reset the
optimizer/run step count, as documented by the runner metadata. The command is
recorded for future use and was not launched by the configuration-selection task.

The `reference` extra installs Placo 0.9.16 for Python 3.12, with urdfdom 4 and
TinyXML2 10 for its Pinocchio binary dependency. Upstream Placo 0.6.3 has no wheel
for this Python version. Use the existing environment with `--no-sync`; to recreate
it, install the project with the `cuda` and `reference` extras.

Run these commands serially, from the repository root:

```bash
# Numeric URDF/MuJoCo comparison; generate a scratch reference, then fit it:
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/reference.py --check
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/reference.py
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/reference.py --fit --data-dir /tmp/zp10d-reference/data
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/assess_reference.py --data-dir /tmp/zp10d-reference/data --actual-contacts --output /tmp/zp10d-reference/assessment.json

# Prepare and check the training adaptation:
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/training_model.py
.venv/bin/python scripts/cad/run_memory_limited.py --gpu playground/open_duck_mini_zp10d/test_adaptation.py

# Small CPU PPO run, including periodic evaluation/checkpointing:
JAX_PLATFORMS=cpu .venv/bin/python scripts/cad/run_memory_limited.py --gpu \
  playground/open_duck_mini_zp10d/runner.py --envs 16 --steps 16384 --checkpoint-every 4096

# GPU mass-matrix regression check:
JAX_PLATFORMS=cuda,cpu .venv/bin/python scripts/cad/run_memory_limited.py --gpu --wait \
  playground/common/mjx_compat.py

# GPU PPO update/checkpoint check; choose an unused output directory:
JAX_PLATFORMS=cuda .venv/bin/python scripts/cad/run_memory_limited.py --gpu --wait \
  playground/open_duck_mini_zp10d/runner.py --envs 16 --steps 2048 \
  --checkpoint-every 1024 --no-eval --output /tmp/zp10d-gpu-smoke

# Inspect the actual step directory before evaluating a checkpoint:
.venv/bin/python scripts/cad/run_memory_limited.py --gpu \
  playground/open_duck_mini_zp10d/evaluate.py \
  playground/open_duck_mini_zp10d/training/checkpoints/000000016384 --seconds 5
```

Training outputs stay in `training/`: `metadata.json`, append-only
`metrics.jsonl`, and ignored `checkpoints/`. `--restore PATH` warm-starts policy,
value and observation-normalizer parameters; Brax does not restore optimizer
state or the run step counter. Reusing an already existing checkpoint step name
fails rather than overwriting it. `--output PATH` can select a separate experiment
when required. `--no-eval` is for update/checkpoint connectivity checks only.
The runner explicitly names the same default `lecun_uniform` mean initializer
to avoid Brax 0.14.2's null-initializer checkpoint reload bug.

The first verified CPU run completed 256 environment steps with four environments;
policy weights changed by up to 0.00239718 and checkpoint `000000000256` was saved
and reloaded. The epoch rate was 4.833 transitions/s including compilation;
training took 69.649 s and peaked at 2.239 GB RAM. This is not steady throughput.
In a full native-model 5-second forward-command diagnostic, this early policy
did not fall (maximum tilt 3.675 degrees), produced no numerical warnings or
active self-contact, but neither whole sole lifted 10 mm. Both knees briefly
reached the provisional stall cap. This was an early diagnostic.

The next 16-environment run completed 16,384 additional transitions and saved
four checkpoints. Training took 433.489 s, peaked at 2.647 GB RAM and reached
217–223 transitions/s in the last two epochs. Evaluation reward increased from
263.955 to 268.747, with mean episode length 500; reward is not walking acceptance.
Full native-model evaluation of checkpoint `000000016384` ran 5 s without a fall,
numerical warning or active self-contact (maximum tilt 4.711 degrees). Knee peak
torques were 0.950 / 0.824 N.m, without knee saturation; right hip roll saturated
for 0.008 s. Neither whole sole lifted 10 mm. This policy has not learned to walk.

A further **262,144 additional-transition run completed**, warm-starting from
`000000016384`, with 16 training environments, four evaluation environments and a
checkpoint/evaluation every 32,768 transitions. The launching shell automatically
runs a full-model 5-second diagnostic after successful training completion.
Training took 1432.849 s. Its full-model 5-second diagnostic did not fall
(maximum tilt 4.376 degrees), but neither foot lifted: maximum whole-foot
clearance was 0.00974 / 0.00834 mm. Left/right knees saturated for 1.154 / 1.416 s.
This failed walking baseline is preserved. Warm-start run counters reset as above.

The default task now uses `--reward-mode optimized`; `--reward-mode original`
retains the previous objective for matched comparisons. Only the ZP10D adapter
changes. With control interval 0.02 s, the optimized objective before clipping is:

```
20 + 2.5 * velocity_tracking + 6 * yaw_tracking + original_imitation
- .001 * sum(torque**2) - .5 * sum(action_delta**2) - .2 * zero_command_pose_cost
- 2 * whole_foot_height_error - 2 * support_load_error
+ 1 * support_contact_match - .25 * loaded_foot_site_speed - .2 * torque_margin_cost
```

Forward tracking uses `exp(-((vx-command_x)/.02)**2)`; its lateral exponent is
`-((vy-command_y-reference_vy)/.1)**2`, with `reference_vy=0` for zero commands.
Following phase-dependent lateral motion is necessary for COM transfer; the
current Placo reference reaches about ±0.40 m/s laterally. Punishing it against
zero with the forward tolerance would oppose the reference gait. Stationary
forward tracking at 0.03333333 m/s falls from 0.89484 to 0.06218 before the lateral
factor. The unchanged survival constant does not distinguish actions of equal
duration; compare physical outcomes rather than total scores across objectives.

Height cost is the mean squared whole-foot minimum-height error divided by
0.015², capped at 9 per foot. Exact Placo contacts and smooth cubic load transfer
through double support replace inferred polynomial contact labels for these new
terms. Load cost compares each foot's fraction of actual upward floor reaction
with that target. Contact matching is a fraction between zero and one. Slip cost
is the sum of contacting foot-site planar speeds divided by 0.03333333 m/s;
unloaded geometric contact can still incur this conservative training cost.
Height/load/contact gait terms are disabled at zero movement command. The
remaining original joint/velocity/contact imitation formula is retained.
Torque-margin cost sums `max((abs(torque)/limit-.8)/.2, 0)**2` over all actuators;
it discourages approaching the provisional stall cap without changing that cap.

`reference.py --reward-reference` enriches the same `data/reference.json` using
the existing settled scratch Placo cycle; `--fit` also runs this preparation.
No polynomial coefficients or physical model parameters are changed by the
standalone preparation. Full collision-foot convex-hull vertices preserve the
exact minimum height at every rigid orientation. The MuJoCo foot site is 2.601 mm
above the planned sole origin. Before the whole-foot reference correction,
raw reference feet rolled up to 3.150 mm below the floor; targets were expressed relative to the lowest supporting sole,
with zero height demanded for stance feet. This is a reward target correction,
not proof that the joint reference can be dynamically tracked. The raw peaks are
12.188 / 12.134 mm; ground-aligned target peaks are 15.168 / 15.157 mm. The fitted
joint reference then had up to 3.132 mm kinematic floor penetration before that
vertical alignment. This is the historical reward-test reference; the current
whole-foot correction and periodic cubic fit replace that training data. Current
control-tick stance height error is at most 0.093530 mm; swing peaks are
15.003055 / 15.003197 mm. `training/reference_assessment.json` records the current
geometry check and its independent PPO/full-scene follow-up.


The corrected-reference CPU trial completed 32,768 new transitions from the same
16,384-step warm start, seed 0 and 16/4 train/eval environments, in 340.074 s.
Maximum policy parameter change was .0351305. Checkpoints, reference input
snapshots and logs are in `training/reference_test/`. Full-model replays completed after the shared lock became available.
In the 8 s forward test, maximum whole-foot clearance was approximately
0 / 0.315 mm (left/right), with zero time at least 10 mm clear and unloaded.
Loaded slip was 56.993 / 80.283 mm; knee saturation lasted 2.478 / 1.168 s,
versus .314 / .036 s for the prior optimized-reward policy. Maximum tilt was
10.240 degrees; mean forward speed was .001327 m/s for a .03333333 m/s command.
The 12 s stand/walk-command/stop profile also produced no 10 mm unloaded
liftoff. Both completed without a fall or numerical warning, but the new
policy regressed in slip, knee saturation and tilt and is **not accepted**.
Reference geometry correction is verified; physical liftoff/walking is not.
The current reference has only .1 s per foot above 10 mm per cycle, so the
separate .2 s airborne acceptance requirement also remains unfulfilled.

Reward counterexamples, native/MJX force agreement, and JIT physics/reward checks
are in `test_adaptation.py`. An initial 32,768-transition trial relaxed smoothing
from -.5 to -.05 and yaw tracking from 6 to 1. Its full-scene 8-second test
regressed: loaded contact-point slip rose from 22.269/38.467 to 88.553/60.921 mm,
left/right knee saturation rose from .044/.062 to 3.812/1.172 s, and neither foot
lifted 10 mm. That candidate is rejected and preserved in
`/tmp/zp10d-reward-first-candidate`. The current formula restores both original
weights and adds torque-margin cost. Its matched-budget retest completed in
327.507 s (policy maximum parameter change .029664; final training epoch
160.63 transitions/s). Against the old 32,768-step checkpoint, full-model 8 s
results are:

| Metric | Original reward | Current candidate |
| --- | --- | --- |
| Left/right loaded contact slip | 22.269 / 38.467 mm | 16.434 / 12.187 mm |
| Left/right knee time at torque cap | .044 / .062 s | .314 / .036 s |
| Maximum torso tilt | 6.041 degrees | 8.693 degrees |
| Mean forward speed, command .03333333 m/s | .000365 m/s | .000388 m/s |
| Either foot at least 10 mm clear and unloaded | No | No |

Slip decreased 26.2% / 68.3%, but left-knee saturation and tilt regressed.
These single-seed short-budget trials verify integration and partial slip
improvement; effective liftoff and walking remain **not accepted**. The result
does not establish eventual convergence or mechanical infeasibility.

A separate full-model 12 s profile (1 s standing, 6 s forward command, 5 s stop)
finished without falling, self/nonfoot contact or numerical warnings. Final tilt
was 1.802 degrees and maximum joint deviation from geometric home was .04679 rad
(2.681 degrees); neither foot lifted 10 mm. This tests command transitions, not
successful landing/recovery from a walking step.

`training/reward_assessment.json` retains all comparisons. The earlier reward-test checkpoint,
metadata, logs and three native evaluation traces are saved in
`training/reward_test/`; copied checkpoint bytes were checked against the tested
scratch run. This remains an experimental candidate, without a walking/ONNX
handoff. To reproduce the current comparison, explicitly choose the scratch
output directory of a fresh experiment (checkpoint names cannot be overwritten):

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/cad/run_memory_limited.py --gpu \
  playground/open_duck_mini_zp10d/runner.py --envs 16 --steps 32768 \
  --checkpoint-every 32768 --reward-mode optimized \
  --restore playground/open_duck_mini_zp10d/training/checkpoints/000000016384 \
  --output /tmp/zp10d-reward-reproduction
```

`evaluate.py` runs the unchanged full scene from home and checks actuator order,
source hashes and command domain. It records whole-foot minimum heights, vertical
foot loads, tilt, joint-limit margin, torque saturation and active contacts.
It also integrates load-weighted contact-point slip, records support matching,
and requires the other foot to carry over half the robot weight for the 10 mm
unloaded-foot metric. `--warmup S --walk-seconds S` adds stand/walk/stop commands;
`--output PATH` keeps experiment diagnostics separate from the preserved baseline.
Its report is **diagnostic**, not a complete motion certificate: sampled positive
self-clearance, loaded slip, both-side landing/recovery and continuous-walking
acceptance remain to be completed. Reports overwrite `training/evaluation.json`
and `evaluation.npz` with the checkpoint path recorded in each report.

Final acceptance must cover home → weight transfer → each whole sole at least
10 mm above ground with less than 0.5 N load and the other foot supporting → at
least 0.2 s airborne → landing/recovery → continuous walking. Loaded slip must
stay within 5 mm per foot, sampled self-clearance at least 0.2 mm, and physical
joint limits must be respected. No current policy has passed these gates and no
new accepted ONNX has been exported. Printed mass, contact, actuator response,
continuous torque and thermal margin still require calibration.

## Historical R13 motion tests at the previous ~81-degree home

The following results predate R14. Their scene hashes differ from the current
home; the load and structural-load files remain historical until motion is
reassessed for the new stance.

The 2026-10-17 assessment uses the accepted R11 shortened legs and R12 front-view
vertical standing pose. Estimated mass is 2.080124393 kg. Seven independent
free-base scenarios ran for 73 seconds: standing, shallow crouching, left/right
lift attempts, a slow-step attempt, head motion and both antennas. All completed
without a fall/nonfoot-ground safety stop, numerical warning, nonfloor contact
or actuator saturation. Standing ground force was 20.405362 N, within 0.00323%
of estimated weight. This is a limited motion test, not a full-range collision
or walking certificate.

| Servo | Mean absolute standing torque N.m | Seven-scenario peak N.m | Maximum scenario RMS N.m |
| --- | ---: | ---: | ---: |
| S09 left knee | 0.515 | 0.737 | 0.542 |
| S04 right knee | 0.508 | 0.759 | 0.541 |
| S07 left hip roll | 0.406 | 0.672 | 0.476 |
| S02 right hip roll | 0.445 | 0.683 | 0.481 |

The lift and step attempts did **not** unload or clear the floor. Swing feet
still carried at least 7.264/7.289 N. Both support knees reached the 90-degree
soft limit (maximum soft violation 0.013864 degrees), which can share load.
Consequently these peaks do not establish true single-leg torque capacity.
All IK targets passed 0.5 mm / 1 degree tolerance (maximum 0.429 mm / 0.624
degrees), but joint-target feasibility alone did not achieve weight transfer.
Actual sole-mesh clearance and foot loads are checked together.

The 10 mm feedback-controller assessment is recorded separately in
`cad/local_fix/CURRENT_LIFTOFF_CONTROL.md` and `.json`; no old-leg trajectory is
accepted as current evidence. The R13 left trial entered transfer at 2.28 s
and failed the internal unloading timeout at 14.28 s, without entering lift.
Swing load was still 3.385 N; S03 reached 69.530 degrees against its 70-degree
limit. Peak body tilt was 31.378 degrees. Subsequent recovery diagnostics were
interrupted at 14.56 s and are unassessed. The right feedback trial was not run;
both baseline directions above were tested and failed liftoff.

Unlike the seven baseline scenarios, the left feedback trial briefly reached
the provisional 0.980665 N.m cap: S09/S04 accumulated 58/44 ms at the 99.9%
threshold, with longest individual runs of 2 ms. Whole-trial RMS was
0.452/0.326 N.m. Other brief cap events are in the 16-axis report. These are
failed-transfer loads, not validated single-leg or landing loads. Do not infer
insufficient stall torque, impossible mechanics or successful walking from this
one failed transfer path.

Next, adapt the transfer path to the shortened-leg kinematics while preserving
formal home and physical limits; verify unloading, whole-sole liftoff, landing
and recovery before using results as single-leg/landing load cases. This round
assesses the existing controls; it does not modify CAD or tune the controller.

Print material is now specified as **PETG**, with **10% shell infill** and
**30% for the other printed parts**. Walls, solid top/bottom layers, layer height,
infill pattern and build orientation remain unspecified. The model still uses
legacy estimated masses: neither infill percentage nor a generic PETG strength
value has been substituted for actual slicing, weighing or anisotropic material
properties. Historical TPU/PLA foot filenames are geometry labels; calibrate
ground friction/compliance for the actual chosen material.

`cad/local_fix/CURRENT_LOAD_ASSESSMENT.md` and `.json` contain results and complete
trajectories; `.csv` summarizes each actuator and `.png` plots loads and ground
support. `CURRENT_STRUCTURAL_LOADS.json` contains 336 simultaneous wrench/pose
states for these seven baseline scenarios in current STEP coordinates
(N, N.mm, mm), the 50 printed-part material
assignments and existing fastener-axis references. Peaks of force and moment
are separate real states, not concatenated component maxima. Local bolt seats,
contacts and load sharing are still pending; no local FEA stress, displacement
or safety factor is certified.

```bash
.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/assess_zp10d_loads.py
# Regenerate reports from the saved trajectory without repeating the simulation:
.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/assess_zp10d_loads.py report
```

## Historical assessments before shortening the legs

The following posture and control experiments use earlier geometry and home;
their numeric results do not validate the current model. Current standing evidence is R14; historical motion/load
evidence is R13 in `cad/local_fix/UPPER_REFIT_ISSUES.md` and the reports above.

A standing-only posture comparison (`assess_zp10d_loads.py stance`)
tested six 15-second transitions/holds. A candidate pelvis target 25 mm higher
and 5 mm backward reduced mean knee holding torque from 0.548/0.580 to
0.336/0.368 N.m (left/right); actual knee bend was about 44 degrees. Effective
ground-force lever arms fell from 62.8/65.4 to 39.3/42.2 mm, with no substantial
hip/ankle load increase. Transition peak torque still reached 0.705 N.m. All six
trials stayed upright without warnings or saturation. This historical static-only
comparison did not establish disturbance recovery or walking performance.
The original R09 files are archived as
`/tmp/open-duck-knee-stance/previous_CURRENT_STANCE_ASSESSMENT.*`; the current
files with those names now document R14.

The user then withdrew approval to change stance and requested an impact
assessment. The matched follow-up ran 20 cases: a 20-second hold, eight independent
2/4 N horizontal pushes lasting 0.2 seconds, and a 19-second slow-step diagnostic
for each stance. Under the same 4 N backward push the current stance peaked at
2.824 degrees and completed 8 seconds; the candidate reached 35.012 degrees and
stopped at 2.998 seconds. Both results were independently replayed from the same
initial states. The candidate raises COM from 197.58 to 218.87 mm and reduces the
rear support margin from 35.77 to 30.54 mm. It also leaves less geometric leg
extension reserve. Neither slow-step diagnostic achieved liftoff, despite valid
IK targets, so walking improvement remains unproven. No numerical warnings or
actuator saturation occurred, including the failed push trial.

**Keep the published home pose; the recommendation to adopt the candidate is
withdrawn.** CAD, formal XML/home, runtime configuration and hardware were not
changed by these assessments. See `cad/local_fix/CURRENT_STANCE_IMPACT.md`, `.json`
and `.png` for results, trajectories, replay confirmation and model limitations.
`assess_zp10d_loads.py stance-impact-report` regenerates that report from saved
results; `stance-impact` runs the comparison serially under the same hard limit.

Next, calibrate print mass and actuator response, establish realistic
torque/speed/thermal limits, extend isolated liftoff to steps and collect single-leg
and landing loads. Apply verified loads to local structural analysis with the
specified print settings, measured PETG data and actual mounting conditions.
Then adapt PPO observations/actions and randomization to this topology and mass
model, train and validate before hardware calibration. The published ZP10D limit
of 0.980665 N.m is stall torque; continuous torque is not established.

Historical liftoff diagnosis identified a planning/control gap in the diagnostic
trajectory: even its ideal 16-degree roll and 8 mm lift target leaves COM about
43 mm outside the support sole projection. During the actual left lift hold,
the free base drops 4.49 mm and rolls back 1.89 degrees from the assumed pose;
the measured base pose consumes 7.66 mm of swing-foot height in an FK comparison.
Fixed-time transitions do not wait for unloading. A replay matches saved states;
an in-memory fourfold leg-kp comparison improves tracking but still does not lift
the foot and introduces brief torque saturation. Neither comparison has nonfloor
contact force above 0.05 N. The baseline support knee reaches its 90-degree soft
limit; the stiffer trial avoids that limit and still fails liftoff, so this is a
secondary constraint. No formal gains, limits, home, CAD or hardware were changed.
See `cad/local_fix/CURRENT_LIFTOFF_DIAGNOSIS.md`, `.json` and `.png`.
The 10 mm whole-sole lift and torque optimization is in progress; not accepted.
Latest stopped probe c27737fa reached 16.228 mm,
with support S09 knee RMS 0.844 N.m and 0.340 s at the cap
during 0.760 s of lift. Stopped with prediction_solve_failed: primal infeasible; landing/recovery not accepted.
Current candidate c27737fa shifts temporary lift COM 10 mm rearward/upward,
extends the support leg, retains the entry support bank and prioritizes lift balance/height.
Airborne-pivot feedforward and fresh/replay weights are synchronized; balance preview/priority persists until touchdown. The completed replay diagnostic still failed at10.74s; S09 unloading and landing/recovery are not accepted. The clipping removal is included in that diagnostic. Formal home, CAD, physical actuator limits and the 5 mm slip gate stay unchanged.
The results below belong to the earlier low-clearance validation only.
The earlier feedback-controller baseline passed
independent left and right liftoff, landing and original-home recovery trials.
Both start from the formal home, use the same source hash and replay no prefix.
Independent review checks actual whole-sole mesh height and foot loads:

- left: airborne sample span 0.440 s, maximum whole-sole minimum height 4.552 mm, final home error 2.224 degrees, final tilt 0.151 degrees.
- right: airborne sample span 0.220 s, maximum whole-sole minimum height 2.757 mm, final home error 1.429 degrees, final tilt 1.092 degrees.

Each trial completes transfer/lift/lower/recover/done and holds the original
home stable for at least 0.5 s. Loaded slip stays <=5 mm per foot, sampled
self separation stays >=0.494 mm, and there are no self contacts, nonfoot
ground contacts, numerical warnings or physical joint-limit violations.

Repairs synchronize transfer position/velocity retiming, hold measured liftoff
COM height, and solve five-axis whole-sole/landing-point references. Recovery
uses one measured full-pose route, then fits the original home to the actual
two-foot planar frame. Recovery balance previews 0.24 s, strengthens torso yaw
and sole-heading tracking after flat-foot alignment, and slows final head/height
restoration twofold. The landing pivot recenters 5 mm inward during measured
flight. Final joint weight is 100, with COM weight fading back to ordinary
double support. Low-load transfer also brakes the lowest swing-sole point
through brief contact loss. Stable hold uses elapsed sample span. Sensitivity probes cover
existing static-friction interval. Native position targets run at 50 Hz;
actual states remain native floating-base dynamics with nonlinear constraints.

See `cad/local_fix/CURRENT_LIFTOFF_CONTROL.md`, `.json` and `.png`.
Formal CAD, home, actuator gains, friction, torque and physical limits are
unchanged. These isolated simulation trials do not validate continuous walking,
a trained policy, FEA or hardware execution. PETG mass/contact/servo parameters
still require calibration. Some actuators reach the provisional stall-torque
cap; continuous torque and thermal margin remain unvalidated. Sampled collision
proxy separation does not replace exact CAD clearance checks.

The historical policy comparison used older geometry: in 10 seconds the original
model travelled about 0.93 m, the draft about 0.16 m and fell, with approximately
28.38% of actuator samples at stall torque. It is not an evaluation of the
synchronized model. No current policy training, FEA or hardware validation is
claimed by the synchronization checks. Physical servo limits remain pending the
previously requested hardware connection and zero calibration.
