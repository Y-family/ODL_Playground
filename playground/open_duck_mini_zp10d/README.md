## 2026-10-31 Drift feedback candidate and continuation

`sampled_control/drift_feedback` completed 262,144 new steps from the retained
v6 actor, reaching 21,387,264 cumulative spent transitions. It adds only the
filtered planar velocity error in the actor's two previously unused inputs;
the reward, Placo reference, controller, contact adapter, servo parameters and
CAD remain unchanged. KL stayed below 0.001 after the first update.

The full native 8 s checks all survived: deterministic walk 9/8 completed
lifts, transition 5/4, stochastic seeds 4/3 and 2/1, and initial yaw 6/5.
The 20 s checks survived for deterministic walk 19/18, transition 5/4,
stochastic seed 1 5/4 and initial yaw 9/8; seed 0 fell at 47.8 degrees after
4/3 lifts. The deterministic walk moved backward 123 mm in 20 s and 405 mm in
40 s, with 23/22 completed lifts and maximum tilt 17.1 degrees. This is a
repeatable short lift/landing candidate with substantial drift, not walking
acceptance. It is selected for deterministic review because it improves the
retained actor's 18/17 20 s lifts and 157 mm drift while remaining within the
same native collision checks.

The next 131,072-step `drift_feedback_continuation` reduced its 8 s GPU result
to 5/4 lifts and its initial-yaw 20 s test fell after 3/2; it is rejected and
kept only as evidence. The selected checkpoint remains
`drift_feedback/checkpoints/000000262144`. The single local viewer is pointed
at the selected saved walk trajectory; no training process is active.

A reward-weight ablation `drift_cost4` doubled the active planar-drift cost
from 2 to 4 for 131,072 steps, resetting only the value estimate. Its 8 s
profiles survived, but 20 s ordinary walk completed 16/15 lifts with 126 mm
backward drift and the initial-yaw profile fell after 10/10. It is rejected;
the higher drift weight did not improve the selected deterministic candidate.

## 2026-10-31 Conservative updates and observable drift

`sampled_control/conservative_updates` completed 262,144 new environment steps,
reaching 21,125,120 cumulative spent transitions. Initial/adaptive maximum
learning rate were both 1e-5; measured mean KL fell from the previous ~0.09 to
0.000639/0.000249 in the two checkpoint intervals. The first GPU evaluation
improved completion credit 6.75 -> 7.75, then the final dropped to 6.25.
All seven 8 s native profiles survived at both checkpoints. Final commanded
completion counts were 9/8 deterministic, 6/5 transition, 5/4 stochastic seed 0,
2/1 seed 1 and 6/5 initial yaw.

The final actor fails longer acceptance: 20 s counts were 13/12 deterministic,
6/5 transition, 8/7 stochastic seed 0 and 2/1 seed 1; the initial-yaw profile
fell at 8.78 s after 6/5 completions. Deterministic backward drift was 187 mm;
stochastic drift 227/285 mm. All arrays in each long test's first 8 s match the
saved short evaluation exactly, checking that the new optional input path
does not alter historical checkpoint replay. The retained older v6 actor is
still selected. Smaller updates help control KL but do not solve drift or
long-term stability.

The actor has gyro, acceleration and joint inputs but no direct planar velocity
input, while the objective charges a filtered planar velocity error. Optional
`--drift-observation` exposes that existing 1 s filtered error divided by
0.010 m/s in unused head-command slots 11:13 of actor and critic. Planned Placo
transfer and initial heading are handled exactly as in the existing cost.
The error is rotated into the robot's current yaw frame for the actor, so
correction directions remain meaningful after turning. Input dimensions,
reward, reference, controller and physics are unchanged.
Warm start zeros only those input weights and gives their normalizer unit
scale, preserving old actor/value outputs before learning. Native replay uses
the same post-step reference phase/filter; old checkpoints keep zero inputs.
This observation is simulator state; physical deployment would require a
base-velocity estimate. Migration preserves actor/value outputs and checkpoint
round trips for synthetic and actual checkpoints. The complete sampled-control
and randomized batched MJX regression passed (3.094 GB peak RSS). A further
four-step MJX comparison matches an independent NumPy filter/current-yaw
conversion, and nine initial/relative heading cases preserve corrective
directions (1.436 GB peak RSS).

`sampled_control/drift_feedback` starts from the retained older v6 actor,
requesting 262,144 new steps from 21,125,120 cumulative spent transitions.
It retains 512 CUDA environments and the 1e-5 initial/maximum learning rate,
with the same corrected contacts and v6 reward. The old actor is preserved by
input migration; the value function resets for its changed contact model.
Code/reference inputs are archived in the run. Full native liftoff, landing,
recovery and longer stability remain pending; the live window follows actual
samples from this continuation.

## 2026-10-31 Corrected-contact training and long validation

`sampled_control/native_contacts` completed 131,072 new environment steps in
723.93 s, reaching 20,862,976 cumulative spent transitions. All seven native
8 s profiles survived, and all five commanded profiles completed bilateral
10 mm lifts. Five 20 s tests also survived without self/nonfoot floor contacts,
but completion counts were only 6/6 deterministic, 3/2 stochastic seed 0,
6/5 seed 1, 8/7 initial yaw and 4/3 transition. Deterministic backward drift
was 196 mm and stochastic drift 276/289 mm. The actor is not promoted over
the retained v6 actor: contact selection is corrected, persistent stepping is
still unaccepted. GPU completion credit also decreased 8.455 -> 3.25.

Three successive trials had first-interval mean PPO KL about 0.089-0.091,
well above the 0.01 target, with adaptive mean learning rates around 1.5e-4.
The scheduler reacts after the loss evaluation and can increase toward 3e-4;
it does not impose a hard trust region. `runner.py` now permits an explicit
initial rate and adaptive ceiling, preserving the existing defaults.
The next isolated comparison uses 1e-5 for both, restoring the retained v6
actor with corrected contacts and the unchanged reward/reference/controller.
This tests update damage before changing observations or reward weights.

## 2026-10-31 Native sole contact selection adopted for training

The same native poses produce different plane/mesh manifolds: native MuJoCo
selects the deepest hull vertex plus at most two graph neighbours sufficiently
far from it, while upstream MJX selects four spread points in a 1 mm skin.
For example, a saved pose has one native contact per foot but four MJX
contacts per foot, with constraint-force differences up to 0.32 Nm/N in the
measured generalized-force components. This is a concrete engine-path cause;
CAD, inertias and identified servos were already identical.

`contacts.py` ports MuJoCo 3.12.0's `mjc_PlaneConvex`/`addplanemesh` selection
for only this named training model's two original sole meshes. It uses the
unchanged native hull graph, keeps the native neighbour order, and pads unused
contacts to retain four static MJX slots. Other scene names/mesh pairs use the
upstream collider; plane/box collision is unchanged. No CAD geometry, contact
friction, solver tolerances, timesteps, reference or servo parameters change.
The adapter is explicitly tied to MuJoCo 3.12.0 and zero-margin sole contacts.

GPU checks matched C MuJoCo contact counts, order and positions for 1,312 foot
states (256 independent random poses plus 400 trajectory poses, both feet),
with max position error 1.46e-8 m. The original CAD sole meshes also match
in 2,048 support directions per foot. The old support test incorrectly
compared the entire foot assembly against just the sole; it now compares the
same named CAD sole. Actual randomized batched MJX reset/step and command
observation regression passed, peak RSS 2.257 GB.

A fixed-home eight-second closed-loop comparison reduces the maximum hinge
position error at 20 ms from 4.77e-4 to 8.52e-8 rad and at 1 s from 3.69e-3
to 4.99e-7 rad. Long trajectories still diverge: at 8 s the maximum base
translation error is 38 mm and the maximum hinge error 0.041 rad; the candidate
has three completion credits versus one native credit. Contact sensitivity
and FP32 therefore remain practical limits; this is not exact long-trajectory
equivalence or walking acceptance. Full native evaluation remains mandatory.

`sampled_control/native_contacts` requests 131,072 new steps from 20,731,904
cumulative spent transitions, restoring the retained v6 actor with the same
512 CUDA environment settings and v6 completion weight 2. The contact model
signature now includes adapter source bytes, so restoring the changed contact
model resets the value function and records the exact adapter hash/method.
Physical liftoff/stationary stepping acceptance remains pending.

## 2026-10-30 Completion-priority trial rejected; engine comparison

The v7 completion-weight trial finished 131,072 updates in 772.88 s, reaching
20,731,904 cumulative spent transitions. Randomized GPU evaluation completed
full 500-tick episodes and improved return (-12.726 to -5.796), but completion
credit decreased (8.238 to 6.25). Full native deterministic clearance fell to
11.04/7.71 mm with 1/0 completed landings; transition had 0/0. All seven 8 s
profiles survived, which does not satisfy the bilateral lift objective.
V7 is rejected and the default sampled recipe is restored to v6/weight 2;
the completion-event regression passed again. Checkpoints and trial evidence
remain available. No further reward reweighting is adopted from this trial.

The reduced native scene reproduces the full native walking trajectory,
including the same 11.04/7.71 mm peaks. Omitted CAD collisions are not the cause
in this test. With the same checkpoint, nominal model and exact stationary
home, MJX instead reaches 11.60/11.54 mm and five completion credits in 8 s.
The first issued actions agree within 2.1e-9, but qpos differs by up to
4.77e-4 after the first 20 ms, growing to 0.101 after 8 s. This establishes an
engine-path difference before domain randomization; contact/solver causes
are still being isolated. It does not establish a solved simulator mismatch.
The retained v6 131,072 checkpoint remains the native candidate below.
Saved-evaluation playback honors its selection even across continuation runs.

## 2026-10-30 Longer validation and completion priority

The v6 131,072 checkpoint is retained as the current native candidate. Five
20 s full-model tests (deterministic, stochastic seeds 0/1, initial yaw 0.7 rad,
and stand 1 s / step 5 s / recover 14 s) all reached duration without a fall,
self-contact or nonfoot ground contact. Supported >=10 mm lifts followed by
stable landings were respectively 18/17, 5/4, 5/5, 10/9 and 4/3 (left/right).
The transition ended at 4.86 degrees tilt. Only the original two sole collision
meshes contacted the floor, so omitted upper-foot contacts did not explain
these trajectories. Continuous deterministic backward drift was 157 mm;
stochastic seed 1 drifted backward 293 mm. This is repeated short liftoff,
not accepted stationary stepping or walking. Later v6 checkpoints fell in some
8 s profiles and are not promoted merely because they are newer.

Sampled recipe v7 changes only completion weight from 2 to 10; the qualifying
height, upper-height band, reference, controller and physical model stay fixed.
In the existing randomized evaluation, the initial/262,144 policies earned
8.237/6.25 completion credits but returns -99.438/-84.757: fewer completions
could improve return. Reweighting those same recorded credits to 10 gives
-33.544/-34.757, reversing this particular undesirable ordering. This is an
objective diagnostic, not a prediction of learning or a general guarantee.
The `sampled_control/completion_priority` trial requests 131,072 new steps
from 20,600,832 cumulative transitions with 512 CUDA environments. It restores
the retained v6 131,072 actor and resets the value function for the changed
objective. The full sampled-control/randomized-MJX regression passed (peak
RSS 3.120 GB). Native evaluation also uses the same
active-command/stop-fade gate as training for its completion counts.

## 2026-10-30 Completion reward after final-policy regression

The requested budget finished at 20,076,544 cumulative transitions
(5,767,168 in `sampled_control/goal_10mm`; PPO batches rounded past the request).
The final native walking policy ran 8 s but reached only 5.909/5.740 mm,
with no supported 10 mm samples. It is not an accepted liftoff policy.
The early 262,144 checkpoint is retained: its native trajectory completed
three left and two right supported 10 mm lifts followed by landings in 8 s.

The duration objective retained almost the same positive credit after losing
the target motion. Initial/final GPU evaluation air-time payments were
6.039/5.724 reward units, and dense-height payments 6.279/6.005, although final
eligible 10 mm time was zero on both sides. A low partial lift could therefore
retain substantial payment while avoiding more demanding motions.

Sampled-reference recipe v6 disables the air-time payment and adds two reward
units per supported landing after a qualified alternating 10 mm lift. A single
qualifying sample suffices; hovering longer cannot increase the event payment.
Landing requires opposite-foot contact above 0.5 N, tilt at most 20 degrees
and a nonterminal frame with a moving command or remaining stop fade. Standing
without a task/fade earns no completion payment. The whole-foot peak is retained until landing:
payment is full through 15 mm and decays above it. Existing continuous height
guidance, reference, controller, physics and drift costs remain. Native
evaluation now reports completion counts as well as peak heights. The complete
sampled-control/randomized-physics regression passed (peak RSS 3.099 GB), along
with the watchdog's completion-count and changed-heading-failure checks.
`sampled_control/completion_reward/active_command` requests 524,288 new steps, starting from
the retained `goal_10mm/checkpoints/000000262144` actor with a reset value
function. Cumulative spent transitions start at 20,076,544, independently of
the restored actor's age. It uses 512 CUDA environments and native evaluations
every 131,072 steps; physical acceptance is still pending.

## 2026-10-30 Short 10 mm lifts and continued stability training

The user accepts a brief supported 10 mm lift; hovering is not required.
Sampled-reference recipe v5 qualifies alternation at 10 mm instead of 2 mm:
one supported sample followed by landing unlocks the opposite side. A 9 mm
lift cannot rearm the successful side. The finite-horizon fall compensation
uses a 30-unit baseline after the stronger drift costs made early termination
cheaper in the observed course. This is checked for constant losses up to 30
units/s, not a bound on every possible trajectory. Lift credit still saturates
at 10 mm and decreases above the existing 15 mm upper band.

`test_lift_curriculum.py --sampled-only` passed the actual delayed MJX control,
native/JAX formula and observation checks, 10 mm alternation counterexamples,
and randomized batched reset/step checks (peak RSS 3.140 GB).

`training_gpu/placo_lift_12mm/course/sampled_control/goal_10mm` continues from
the synchronized 786,432-step checkpoint with 512 CUDA environments and full
native checks every approximately 262,144 new steps. It requests 5,693,440 new
steps from 14,309,376 observed cumulative transitions, for 20,002,816 total.
The restored weights are younger than the spent-step counter. Identified servo
dynamics and the existing experimental 2x force-cap assumption are unchanged.
Services are `zp10d-placo-goal-training` and `zp10d-placo-goal-supervisor`;
the latter checks health each minute and prints summaries every 20 minutes.
The single existing window follows actual training samples. Its live mode has
no playback multiplier: about 755 aggregate steps/s across 512 environments
advances one environment at only 0.0295x real time. Saved evaluation playback
is a separate mode and defaults to 0.5x. Stable landing, drift, heading and
recovery still require native verification; the requested step budget is not
walking acceptance.

The first new checkpoint (262,144 steps; 14,571,520 cumulative spent steps)
completed all seven 8 s native profiles. Deterministic stepping reached
12.40/10.94 mm with 0.04/0.04 s supported 10 mm runs, but moved backward
79 mm. Stochastic seed 0 reached only 9.71 mm on the right; transition reached
9.98 mm. Seed 1 and the 0.7 rad initial-yaw case achieved bilateral short
10 mm lifts. Longer stability, consistent bilateral clearance and drift remain
unaccepted; the training and read-only supervision services continue.

## 2026-10-29 12 mm Placo reference and inward standing pose

The experimental training reference now targets 12 mm whole-sole clearance,
with 1 degree inward hip roll per side and a 0.6 s cycle. The fitted minimum-foot
height peaks at 11.99997 / 11.99998 mm, and its analytic peak joint speed is
4.67522 rad/s. Full-body unilateral contact/command allocation passes 1192/1192
raw and 601/601 fitted samples; only 928/1192 and 431/601 respectively also meet
the existing torque prior. These are necessary-condition checks, not stable
physical liftoff. Experimental-reference mode remains required.

The previous 2.5 mm reference and polynomial are archived under
`training_gpu/placo_imitation_unlimited_v2/inputs`; its checkpoints are retained.
Current candidate poses are published to the existing
`training/reference_demo/candidate.npz`. CAD, model inertias, joint limits and
identified servo parameters are unchanged. The inward home is applied in memory
consistently by the training and native evaluation paths.

Direct free-base feedforward replay fails at 2.234 s before the first requested
lift, both with and without motor force caps. Startup blending, joint feedback,
orientation/COM feedback and an experimental dynamics tracker do not establish
bilateral supported 10 mm liftoff. These diagnostic controllers are not the PPO
policy. Slower 0.9/1.2 s candidates failed contact or joint-range checks and were
not adopted. Training/evaluation results below remain historical unless their
reference hashes match the new files; a planned 12 mm pose is not actual lift.

The first new-reference baseline completed 262,144 real PPO steps in 797.85 s
(`training_gpu/placo_lift_12mm`). At 131,072 steps all five native profiles ran
8 s but neither foot achieved supported 10 mm clearance. At 262,144 steps the
three deterministic profiles again ran 8 s with about 1.6 degrees maximum tilt
and no liftoff; stochastic standing/walking fell at 1.7/2.5 s. The baseline's
20-unit survival reward and positive yaw-stability reward still favor idle poses.

`--lift-curriculum` can now retain the Placo imitation/contact/COM targets while
adding supported 10 mm lift and air-time credit for the planned swing foot only.
It removes the survival payment, uses nonpositive upright/yaw regularizers,
allows negative rewards and charges 2 units per fall. Both lift and air-time
height credit saturate at 10 mm in this reference course; forward progress is
not required. Phase observations remain the Placo clock. The direct course's
event observations and rewards are preserved. The new course uses adaptive-KL
PPO and resets the value network when restoring across reward recipes.

The first Placo lift course completed 524,288 additional steps in 1145.90 s
(`training_gpu/placo_lift_12mm/course`). All four checkpoints' five native
profiles ran 8 s, but none demonstrated supported 10 mm liftoff. The learned
normal exploration standard deviation remained about 0.0126; deterministic
hip-roll actions barely varied. This is a failed liftoff result, not acceptance.
Continuation `course/exploration` tests a 0.05 exploration standard deviation
for 1,048,576 additional steps, preserving the learned mean, normalizer, value,
reference, rewards and physics. `--policy-std` now supports this explicit
warm-start override. The policy regression verifies unchanged deterministic
actions, the requested stochastic scale and checkpoint persistence; invalid
standard deviations are rejected. The exploration comparison was deliberately
stopped after 524,288 steps / 1167.06 s and its two checkpoints' native profiles;
neither checkpoint lifted either foot by 2 mm in deterministic replay. At the
second checkpoint stochastic walking ran 8 s and stochastic standing fell at
5.28 s; neither achieved supported 10 mm liftoff. Rare training-rollout clearance
events did not transfer to the fixed-seed native evaluations. Larger exploration
alone has not solved liftoff; its checkpoint and interrupt trace are retained.

The fitted 12 mm audit requires more than the provisional 0.980665 Nm cap in
170/601 samples, split equally between the two knees, with a peak minimum ratio
of 1.87150. Disabling the torque reward never disabled that physical cap.
`course/torque_sensitivity` therefore isolates a runtime 2x ZP10D force-cap
override for 524,288 additional steps, restoring the exploration checkpoint
without changing its standard deviation or action scaling. This is explicitly
a simulation sensitivity test, not identified hardware capability. Antennas,
joint/command limits, motor gain, damping, friction, armature and delay remain
unchanged; source XML/CAD is untouched. Native evaluation inherits the scale
and reports the actual force caps. Default scale remains 1. The adaptation
regression checks native/MJX parity and unchanged controller scaling/response.
This comparison completed 524,288 steps in 1189.10 s. Both deterministic
checkpoints remained grounded for 8 s. All profiles at 262,144 steps ran 8 s;
at 524,288 steps stochastic standing/walking fell at 1.88/1.26 s. Neither
checkpoint achieved native supported 2 mm or 10 mm liftoff. The 2x cap alone
has not established stable stepping.

An optional `--reference-reset-fraction` now lets the Placo lift course sample
moving reference poses, velocities and phases at environment initialization.
The other environments retain the home start. Their cached reset/controller
histories match the selected state. An initially airborne foot is excluded
from lift credit until it lands and the other foot performs a real lift. This
is reference-state initialization for learning, not demonstration acceptance:
native evaluations always start from home, and training heights can include
the supplied initial poses. The default fraction is zero. Geometry, phase,
velocities, queued commands, initial-foot reward exclusion, standing reset and
an actual JIT physics step have regression coverage.
`course/reference_initialization` tests a 50% reference/home mix for 524,288
additional steps, retaining the 2x sensitivity cap. It restores the more stable
262,144-step sensitivity checkpoint; cumulative spent steps and the restored checkpoint's reported counter
are recorded separately. Native home-start liftoff remains unverified.
The 524,288-step initialization comparison completed in 1361.83 s. Deterministic
native profiles still had no 2 mm clearance. Stochastic walking reached left
13.19 mm / 0.06 s of supported 10 mm clearance at 6.46--6.50 s, but body tilt
was already 35.8--42.7 degrees and it fell at 6.52 s; the right foot never
achieved 10 mm. This is a near-fall event, not normal stepping. Hip-roll cycle
ranges at 262,144 steps were 1.41/1.48 degrees with reference correlations
0.942/0.939; at 524,288 steps ranges grew to 6.99/7.15 degrees, versus roughly
34 degrees in the reference. The selected policy still lacks enough transfer.
`reference_initialization/continued` preserves this setup for 2,097,152 more
steps, with native checks every 262,144 steps. The 2x cap remains a simulation
sensitivity assumption; stable bilateral home-start liftoff has not passed.

That continuation was deliberately interrupted after 851,968 logged steps
(1971.93 s); the last saved/evaluated checkpoint is 786,432. All three
deterministic walking evaluations remained grounded. At the last checkpoint
the hip-roll ranges reached 18.15/18.89 degrees versus 34.25/34.31 degrees in
the reference, but minimum foot loads were still 5.29/5.75 N. Stochastic
clearance immediately preceding a fall is not accepted stepping.

The optimized task now gates `stand_still` on a zero command (norm <=1e-6).
The upstream <0.01 m/s deadband incorrectly included the current 0.00333333 m/s
stepping command: following the reference incurred an average 2.7681 reward
units/s penalty for joint pose/velocity. This is a confirmed reward conflict,
not proof that it was the sole cause of failed liftoff. The Placo course recipe
is now v2, so warm starts reset the value function for the changed objective.
Slow forward/yaw and zero-command counterexamples plus an actual JIT course
step pass; physical training and bilateral supported 10 mm liftoff remain pending.
The complete liftoff regression also passes (peak RSS 3.480 GB). Continuation
`course/zero_command_fix` restores the saved 786,432-step policy, accounts for
the 851,968 steps already spent in the interrupted phase, resets its value
function, and requests 1,310,720 more steps with native checks every 131,072.
The reference, exploration, 50% reference resets and 2x sensitivity cap are
unchanged; the corrected standing-command gate is the only objective change.

The corrected course ultimately completed its 1,310,720 steps across a host
restart (524,288 saved before restart, 786,432 afterward in
`zero_command_fix/continued`; the latter took 1679.01 s). None of the ten
deterministic walking checkpoints demonstrated supported 10 mm liftoff.
The post-restart 393,216-step checkpoint reached right 1.744 mm clearance;
the left foot still carried at least 2.53 N. The earlier stochastic bilateral
event failed its extended GPU replay at 14.88 s, and four other GPU seeds did
not reproduce bilateral 10 mm liftoff. CPU inference produced different
contact trajectories and is recorded separately, not treated as a like-for-like
GPU replay. Six reference-posture-only diagnostics (0/40/80 ms anticipation,
0.6/1.2 s startup blends) all fell within 1.04--1.58 s and are not adopted.

`course/low_noise` restores the post-restart 393,216-step policy and retains
its mean/value/normalizer, with exploration std explicitly reduced from about
0.05 to 0.0125. It requests 524,288 steps and full native checks every 131,072;
reference, reward, controller and force-cap assumptions remain unchanged.
The cumulative spent-step counter is 4,521,984 before this comparison; restored
weights are younger. Stable bilateral stepping remains unaccepted.

The low-noise comparison completed 524,288 steps in 1237.37 s. None of its four
deterministic native walking evaluations achieved bilateral 10 mm clearance.
The third checkpoint reached left/right 0.074/5.814 mm; the final checkpoint
reached 0.624/4.292 mm. Random walking at the final checkpoint fell at 7.06 s.
The user correctly identified body rocking without useful stepping; this is a
failed liftoff result. Cumulative spent steps are now 5,046,272.

The Placo lift course v3 removes grounded imitation/contact baseline payments:
`imitation_v3 = imitation - 5*(1-supported_lift_credit)` for nonzero commands,
where supported lift credit saturates at 10 mm and retains phase/alternation
eligibility. The upstream imitation has at most five positive velocity/contact
units; all joint tracking costs remain. Contact matching becomes a nonpositive
mismatch cost. Dense lift/air-time weights remain 6/6; bounded unloading and
stability costs are unchanged. This makes physical clearance necessary for
positive imitation credit, not a guarantee that PPO will learn it.

Placo now observes last-landing/qualified-lift memory in the two unused head
command slots (actor indices 9:11, also copied to critic), preserving its clock
and 113-input layout. Previously identical observations could conceal opposite
reward eligibility. Old Placo weights require clearing those two input rows in
actor/critic and setting their normalization statistics to unit scale, so the
new memory cannot cause amplified untrained actions. Native evaluation keeps
legacy layouts and uses the checkpoint's recorded layout. Regression and
physical training results for v3 are pending; CAD/physics/reference are unchanged.

The observation/reward regression and actual-checkpoint migration pass. A Placo
course early fall also accounts for the discounted remainder of the six removed
baseline units/s (gamma .99), preventing the baseline shift itself from favoring
early termination. Its finite-horizon identity and real JIT course step pass.

The user requested separate support transfer and swing-leg shortening. Native
counterfactuals under `training_gpu/placo_lift_12mm/swing_coordination_probe`
hold the swing roll near standing and add coordinated hip/knee/ankle flexion.
Holding roll alone and adding only 5 degrees at the knee produce no 10 mm lift.
Coordinated 15-degree knee flexion (half at hip/ankle, mirrored hip sign) produces
left/right 31.54/38.14 mm peaks and 0.88/0.62 s supported 10 mm samples over 8 s,
but slides 0.398/0.542 m and exceeds the right knee limit by 3.73 degrees. Hard
10-degree and smooth 10-degree variants fall at 6.50/7.04 s, despite bilateral
liftoff. These are untrained controller counterfactuals, not accepted PPO results.
Read-only FK also shows that freezing swing roll while retaining the remaining
reference pose moves that foot about 40 mm outward; do not equate it with pure
vertical lift.

`--coordinated-lift` is an experimental residual training mode based on that
diagnosis. It smoothly blends swing roll toward standing and adds
`10deg*sin(phase)^2` knee shortening with half-angle hip/ankle compensation,
before the unchanged feedback, delay, speed/angle/force bounds. Zero commands
receive no guidance. It requires standing initialization, preserves raw PPO
adjustments for leg flexion and keeps Placo phase/COM/imitation guidance. Slip
weight becomes -1 (was -.25); squared joint-margin cost starts 3 degrees from
each existing limit. Native evaluation applies exactly the same action function
and history ordering. This mode still requires regression, training and stable
home-start liftoff validation; no CAD or servo identification is changed.

The coordinated mode regression passes (peak RSS 2.334 GB): mirrored physical
hip/knee/ankle increments, zero-command identity, NumPy/JAX parity, transformed
action history, observed alternation state and a real JIT physics step. Run
`course/coordinated_lift` requests 1,048,576 new steps from low-noise checkpoint
393,216, with 512 environments, low exploration, standing-only initialization
and the existing 2x research force cap. The raw actor is migrated, its value is
reset for the new recipe, and cumulative spent steps start at 5,046,272. This is
training research, not an accepted controller.

The previous local window was looping saved native evaluations, not streaming
the active training rollouts. It was closed after the user pointed out the
mismatch. `runner.py --live-training` now atomically publishes the actual first
of the 512 MJX training environments after each batched physics control step;
evaluation environments do not overwrite it. `training_window.py RUN
--live-training` displays that current state, phase/total steps, simulation time,
episode resets and sample age. It does not loop, extrapolate or run a second
simulation. Compilation/evaluation can leave the last sample stationary; waiting
and age are visible. This is the approximate training physics rendered with the
current CAD, not full-native acceptance. The 512-environment JIT stream regression
passes, including atomic pose/step/reset correspondence and exclusion of four
evaluation environments. The initial coordinated startup was interrupted before
any logged PPO update to add this stream and restarted in the same run directory;
the original zero-step startup and interrupt remain in the log.

Coordinated v1 was stopped for a confirmed excess-height incentive after
321,024 observed environment transitions (294,912 logged by PPO, last saved
checkpoint 262,144; 1200.14 s). At 131,072 the deterministic full model ran 8 s,
left/right peaks 38.10/39.92 mm and supported 10 mm samples 1.38/1.08 s. Its
limit margin was +0.372 degrees and there was no self/nonfoot contact, but slip
was 0.287/0.300 m and maximum tilt 25.75 degrees. Stochastic walking fell at
7.78 s; transition exceeded a joint limit. At 262,144 deterministic walking
regressed to a 5.78 s fall. No result passed stable stepping.

Even with per-frame credit capped at 10 mm, longer high lifts still earn more
integrated reward. Holding the other reward inputs fixed, scaling the current
12 mm height waveform to 24 mm raises its height-dependent return from 2.8498
to 3.5642/cycle, including the existing clearance-error cost. Coordinated recipe
v2 multiplies dense lift, supported air time and imitation lift credit by
`exp(-sum((max(height-.015,0)/.005)^2))`. The full band is 10--15 mm; excessive
height loses credit smoothly. The counterexample and actual JIT coordinated
physics regression pass (peak RSS 2.354 GB).

Continuation `coordinated_lift/height_band` requests 786,432 steps, with full
native checks every 262,144. It restores the more stable 131,072 checkpoint,
retains actor/normalizer and resets value for recipe v2. Cumulative observed
steps start at 5,367,296, distinct from saved-checkpoint age. Guidance, slip and
joint-margin costs, reference, force-cap assumption and zero-command standing
behavior remain unchanged. The single actual-training window follows this
continuation, showing explicit waiting during compilation. Physical improvement
from the new height band remained pending at startup.

Height-band v2 was stopped after its first 262,144-step checkpoint and all five
native profiles (996.71 s; cumulative observed steps 5,629,440). Deterministic
walking and transition ran 8 s, but walking peaks remained 34.17/32.70 mm,
supported 10 mm samples totaled 1.10/0.90 s and slip was 0.293/0.377 m. Body
heading spanned 116.8 degrees and ended 96.2 degrees from its start. Transition
exceeded a joint limit by 0.42 degrees; stochastic walking fell at 5.54 s.
No result passed normal stable stepping; all checkpoints remain available.

Coordinated recipe v3 retains the height band and changes only its saturated
yaw-rate cost: for `x=abs(gyro_z)/.5`, the cost is `x*x` through `x=1`, then
`2*x-1`, still weighted .2. The previous cap charged the same for .5 and 5 rad/s;
the linear tail preserves the small-rate objective and a bounded, nonzero
restoring gradient at high rates. Other courses keep their prior cost. Formula
counterexamples and a real coordinated JIT step are checked before continuation.
The live window explicitly labels a sample older than 5 s as waiting; it keeps
the last actual training frame during evaluation rather than playing a loop.
The coordinated regression passes (peak RSS 2.362 GB), including mirrored
control, height counterexample, yaw symmetry/restoring gradients and the actual
JIT reward. `height_band/yaw_tail` continues from v2 checkpoint 262,144 for
393,216 requested steps, checking every 131,072; actor/normalizer are retained
and value resets for recipe v3. Its observed-step baseline is 5,629,440. Native
physical improvement from this change remains unverified.

At v3 checkpoint 131,072, deterministic, transition and stochastic walking all
ran 8 s. Deterministic peaks were 23.4/19.7 mm, supported 10 mm samples totaled
.54/.42 s and yaw span/end were 53.2/-5.8 degrees. Slip remained .246/.417 m;
transition still exceeded a limit by .52 degrees. At 262,144 deterministic
walking stayed up but stochastic walking fell at 1.48 s. These results preserve
the first candidate for comparison and do not pass stable stepping.

V3 completed 393,216 new steps in 1428.97 s (6,022,656 cumulatively observed
steps). Final deterministic peaks were 21.0/24.8 mm, with .345/.388 m loaded
slip and 82.9 degrees of heading span. Stochastic walking ran 8 s but had
37.4/30.6 mm peaks and a slight joint-limit overrun. Lowering the same first
candidate's knee guide to 0/2.5/5 degrees removed bilateral 10 mm liftoff;
7.5 degrees and diagnostic yaw damping did not resolve slipping. None is adopted.

The T0000 identification uses delayed step targets (`identify_servo.py` rejects
firmware interpolation), while the inherited training controller additionally
ramped targets at 5.24 rad/s. New ZP10D environments remove this extra ramp;
the original V2 default stays intact. Identified gains/friction/inertia, the
two-tick delay and force/angle caps stay in use. Metadata records
`motor_target_slew_rad_s=null`; native replay defaults missing historical
metadata to the old 5.24 ramp. Warm starting across this controller change
resets value and records the change. Real JIT regression observes a .232828 rad
target step after the same delay versus the old .104800 rad increment, with
force/angle bounds retained (peak RSS 2.3595 GB). The old walking policy falls
at 1.78 s when run without its learned ramp; this is a diagnostic, not a fixed
gait. Legacy 8 s native replay remains array-identical. The zero actor with
unchanged gain 3 stands for 8 s (maximum tilt .706 degrees; loaded slip
.285/.263 mm). Gain 0 falls at 2.84 s. Keep gain 3 for policy adaptation.
Four full-startup reference tests with/without audited feedforward and 0/40 ms
reference lead all fall at 3.00–3.34 s; feedforward alone is not accepted.
Stable bilateral stepping remains open.

T0000 v3 adaptation completed 262,144 steps in 1010.23 s (6,284,800
cumulatively observed). Native walking still falls at 2.40/2.82 s; final
standing runs 8 s but slips 60.0/39.4 mm. Not accepted. A separate delayed
feedback step test now invalidates retaining gain 3 for moving joints based
only on static standing: a 10-degree step produces sustained 47–52-degree
peak-to-peak oscillation at four load inertias, hitting the unchanged 2x
research force cap. Gain 1 settles near 10 degrees. Full reference tracking
with gain 1 still falls, so software stabilization alone is not gait acceptance.
Explicit `--adapt-feedback` checks the (1+new)/(1+old) leg-scale conversion,
retains actor/normalizer and unloaded posture semantics, and resets value.
The next reward recipe also rotates world translation into the initial
episode heading before imitation comparison; randomized yaw must not change
the score for the same physical motion. Dynamic/heading/gain-1 JIT checks pass
(peak RSS 2.538 GB). The `t0000/gain1` continuation requests 393,216 steps
with full native checks every 131,072, starting from 6,284,800 observed steps.
Metadata confirms 3-to-1 migration, v4 reward, both value-reset reasons and
all 16 unchanged unloaded posture scales. Walking acceptance is still pending.

The first two gain-1 checkpoints run all five native profiles for 8 s,
but neither lifts. First walking slips 18.4/18.3 mm (maximum tilt 7.19 degrees).
A further controller conflict is now isolated: at double-support phase 6/30,
Placo needs hip rolls near -12.1/-13.6 degrees; the old sin-squared quiet-roll
guide reduces the second to -2.2 degrees before unloading. Guidance v2 leaves
both rolls under policy control during transfer, then captures/holds a roll
only after actual supported 2 mm liftoff until landing. Historical metadata
without a version retains v1 playback. Real JIT controller checks pass
(peak RSS 2.551 GB), and all legacy 8 s NPZ arrays remain exactly identical.
The unchanged actor with v2 stays up for 8 s but does not lift (slip 19.1/17.9 mm).
Gain-1 v1 finished 393,216 steps in 1140.43 s (6,678,016 observed total);
all three checkpoints run five native profiles for 8 s without liftoff.
The `roll_release` continuation requests another 393,216 steps, same gain 1
and v4 reward, with value reset only for guidance v1-to-v2. Initial GPU
evaluation stays up for 10 s but has no valid liftoff; adaptation is ongoing.

Guidance v2 completed 393,216 steps in 1226.37 s (7,071,232 observed);
all three checkpoints ran every native profile for 8 s but never lifted.
Yaw span is now 1.230 degrees versus 53.195 for the old gain-3 candidate.
The hand-coded knee fold also disagrees with Placo knee timing. Fixed-base
CAD kinematics confirms its 10-degree increment clears only 6.420 mm; this is
not a dynamic-height bound. Removing the fold stays up for 8 s but does not
lift; four additive pose/torque-prior counterfactuals fall at 1.62–2.30 s.
Guidance v3 removes that fold and keeps physical liftoff roll holding.
Regression/JIT passes (RSS 2.547 GB); legacy v2 arrays remain bit-identical.
`placo_only` now requests 1,048,576 more steps, native checks every 262,144,
same gain 1/512 GPU/2x simulation force cap, std .025 (gain-1 default),
actor/normalizer retained and value reset for controller v2-to-v3.
Stable bilateral 10 mm liftoff remains unaccepted.

The `placo_only` v3-guidance comparison completed 1,048,576 new steps
(8,119,808 cumulatively observed). Every deterministic walking checkpoint
remained grounded; the final stochastic standing evaluation fell at 4.12 s.
The actual-training window is a streamed MJX sample, not a replay.
Native controller counterfactuals reproduce the saved baseline array-for-array.
Larger unilateral transfer can unload each foot, but the body drops roughly
10 mm and the lowest sole corner still barely clears; extra swing flexion or
support extension alone causes excessive slip and is not adopted.

Coordinated reward v5 adds the previously absent Placo base-height target:
`-2*min(((base_z-reference_base_z)/.010)^2,9)` during nonzero commands.
The current reference raises its base to about 154.4/154.6 mm at peak lift,
whereas a diagnostic trajectory remains near 140--148 mm. This is a confirmed
missing target, not proof it was the only cause. Formula symmetry, restoring
gradients, same-phase target/cost/metric checks and zero-command exclusion pass
an actual JIT step (peak RSS 2.535 GB). `placo_only/base_height` requests
1,048,576 more steps from the earlier 262,144 checkpoint, preserving all
controller/reference/servo settings and learned exploration; only the value
network resets for the new reward. Spent steps include the discarded later
updates. V5 was interrupted after 10,240 observed transitions and no saved
checkpoint when the support-force conflict below was identified.

The current Placo peak has a planned normal force of about 0.4355 times body
weight (from its hard COM target acceleration). The old >0.5-weight gate rejects
10 of 14 late-cycle >=10mm reference samples per side, including the height
maximum. Coordinated v6 therefore uses actual opposite-foot floor contact
>0.5 N for lift, air-time and physical-liftoff event qualification; unloaded
swing force remains <0.5 N. Actual supported 10mm clearance earns full credit
without a static weight-fraction multiplier. No-contact flight and grounded
rocking still earn no height credit; unloading remains bounded per attempt.
Conservative half-weight metrics remain unchanged, and native replay/watchdog
also report the contact-supported durations separately. Historical checkpoints
retain their old qualification rule. Guidance version 4 records this changed
hold-activation predicate; the action transformation itself is unchanged.
Counterexamples, default compatibility, NumPy/JAX memory parity and actual JIT
physics pass (peak RSS 2.553 GB); screen-only monitoring regression also passes.
`base_height/contact_support` retains the Placo-height cost and requests
1,048,576 steps from the earlier 262,144 policy, with prior observed steps
8,130,048 and actor/normalizer/exploration retained. Stable bilateral lift and
landing/recovery remain unaccepted.

The contact-support comparison was stopped after 548,352 observed transitions
(524,288 saved/evaluated, 1354.82 s; 8,678,400 cumulatively observed). Both
deterministic checkpoints stayed grounded for 8 s. Hip-roll ranges regressed
from 8.47/9.76 to 2.51/4.57 degrees versus about 34 degrees in Placo, and the
second stochastic standing profile fell at 3.66 s. The force-gate and body-height
corrections did not establish liftoff. Checkpoints and all native profiles remain.

`--reference-action` tests Placo joint postures as the nominal command rather
than only an imitation-reward target. PPO adds residual corrections, with the
existing geometric-home feedback scale compensated. The nominal leg offset
anticipates the two-tick transport delay and that tick's end pose; a quintic
1.2 s startup starts at home. Zero commands and head/antenna commands receive
no nominal offset. Guidance version 5 records this explicit controller change;
old checkpoints retain their old behavior. The reward, geometry, actuator fit,
delay and physical limits remain unchanged. All-phase NumPy/JAX reference and
controller parity, unloaded feedback scaling, startup/standing/head exclusions
and a real three-step JIT motor queue/bounds regression pass (RSS 2.363 GB).
Historical native playback remains array-identical. The unadapted nominal-action
counterfactual reaches 7.00/6.90 mm peaks but falls at 1.74 s (23.7/47.7 mm slip);
it is not a usable demonstration. `course/reference_residual` requests 524,288
adaptation steps, checks every 131,072, restores the earlier 262,144-step
contact-support actor/normalizer/exploration, and resets value for guidance v5.
Cumulative observed steps start at 8,678,400; the last 24,064 interrupted
transitions from the previous run have no saved checkpoint. The current single
window follows actual training samples. Stable bilateral 10 mm clearance,
limited slipping and landing/recovery remain pending; reference postures alone
are not evidence of physical liftoff.

The nominal-posture comparison was stopped after 317,952 observed transitions
(262,144 saved/evaluated; 8,996,352 cumulatively observed). Its first native
walking checkpoint reaches 13.85/14.40 mm, but falls at 2.04 s; left >=10 mm
samples all have >25-degree tilt. The second falls at 2.50 s with 6.38/0.04 mm
peaks. All moving profiles fail. At the first checkpoint, support knees reach
90--92 degrees while the corresponding Placo targets are roughly 67--77;
ankle errors and backward pitch accumulate. Nominal joint angles retain a
static two-foot home preload, which does not supply the phase-dependent
support torques. That is a confirmed missing feedforward term, not proof it
is the sole cause of the falls.

`--reference-feedforward` adds the existing fitted reference's audited motor
torques to the nominal leg commands. It replaces the static preload and
compensates the separate software velocity feedback, so zero residual at the
reference state requests `q_reference + tau_reference/kp`. It uses the same
delay/startup/limits, and only leg commands. Guidance v6 records the change;
the exact torque file hash is checkpointed and replay verifies it. This is
an optional controller comparison. Final-source regression/JIT passes
(RSS 2.391 GB), and guidance-v5 native playback remains array-identical.
The unchanged actor with feedforward falls at 2.28 s (7.15/7.71 mm peaks);
zero residual falls at 1.76 s (1.04/0.86 mm). Zero residual improves knee
tracking but falls forward; the old actor still over-bends support knees and
falls backward. Three intermediate residual scales also fall and are not
adopted. None is an accepted demonstration.

`reference_residual/feedforward` starts a fresh zero-mean residual policy,
value and normalizer with nominal Placo posture/torque guidance, rather than
carrying the old knee-posture bias. It requests 1,048,576 steps with checks
every 262,144, 512 GPU environments and default std .025. The separate spent
step counter starts at 8,996,352; restored checkpoint is null and the weight
age starts at zero. Exact source, trajectory and torque-audit bytes are archived
in its inputs. Reward v6, CAD, mass, identified dynamics and the existing 2x
simulation force-cap assumption remain unchanged. Stable physical liftoff,
landing and recovery remain pending.

The fresh feedforward run was interrupted after 544,256 observed transitions
(524,288 saved/evaluated; 9,540,608 cumulatively spent). Both standing profiles
run 8 s. Deterministic moving checkpoints fall at 1.92/2.22 s, with final
left/right peaks 4.13/.37 mm and no supported 10 mm clearance. Forward pitch
grows despite hip-roll motion; native tests report no self-contact.

Guidance v7 separates the two reference times: nominal motor posture/torque
anticipates the transport delay, while measured position/velocity feedback uses
the current phase. Previously, perfect current-state tracking still injected
`kp*(gain*(q_future-q_now)+damping*(qd_future-qd_now))`, peaking at .622 Nm
for hip roll. The all-phase moving-state regression now checks zero such bias,
with actual JIT queue and bounds coverage (peak RSS 2.391 GB). Versions 5/6
retain their historical replay paths. An exact-reference-state diagnostic
extends time to falling from .94 to 2.42 s with current-phase feedback, but
still falls; supplied reference poses/velocities are not learned liftoff.
Home-start native improvement and stable bilateral stepping remain pending.

The unchanged 524,288-step actor with guidance v7 runs the full native home-start
walking test for 8 s (maximum tilt 13.28 degrees), versus a fall at 2.22 s with
v6. Clearance is still only 3.41/2.53 mm and supported 10 mm time is zero.
Zero residual with v7 still falls at 2.70 s. All v6 native replay arrays remain
identical. These are controller counterfactuals, not v7 training acceptance.

`feedforward/current_phase` continues from the v6 524,288 checkpoint for
1,048,576 requested steps, native checks every 262,144, the same 512 GPU
environments, v6 reward and physical parameters. Actor/normalizer/exploration
are retained; value resets for guidance 6-to-7. The spent-step baseline is
9,540,608, including 19,968 interrupted transitions beyond the saved weights.
The single actual-training window follows this continuation. The no-torque
nominal-posture v7 JIT regression also passes (RSS 2.378 GB). Stable bilateral
10 mm clearance, slip, landing and recovery remain unaccepted.

Current-phase training was stopped after 576,512 observed transitions
(524,288 saved/evaluated; 10,117,120 cumulatively spent). Both deterministic
native walking checkpoints run 8 s, but peaks regress from 1.46/1.82 to
.08/.07 mm. All final native profiles have zero supported 10 mm time; final
stochastic walking also runs 8 s. The timing correction improves balance but
has not overcome the grounded policy solution. These weights are retained.

The coordinated course can now use reference-state initialization only with
both nominal posture and audited torque feedforward. A supplied moving state
gets its current periodic motor preload and the correctly phased three-command
transport queue; its nominal startup blend is already complete. Standing
initialization keeps the 1.2 s ramp. Initial airborne feet remain excluded from
lift rewards, and new `eligible_contact_10mm_s` metrics apply that same phase/
alternation exclusion. Historical raw clearance metrics may include supplied
initial poses. The live stream marks the source of each episode's initial
state; native evaluations always start standing. This changes training data,
not CAD, masses, servo limits or the physical-liftoff acceptance requirement.

Final-source guided reset and home-start JIT checks pass (peak RSS 1.931/2.391
GB); the stream regression verifies the initialization flag, real sampled
states and evaluation exclusion. `current_phase/reference_initialization`
requests 1,048,576 steps with a 50% reference/home mix and checks every 262,144.
It restores the earlier v6 feedforward checkpoint 524,288: its unchanged v7
counterfactual had the largest balanced clearance (3.41/2.53 mm), whereas both
v7 training checkpoints regressed. Actor/normalizer are retained, value resets
for guidance 6-to-7, and all 10,117,120 previously spent steps remain accounted
for. The live window labels supplied initial states. Home-start bilateral
10 mm liftoff and recovery still require verification.

## 2026-10-28 Alternating liftoff training

v6 objective correction (`lift_objective` continuation): controlled replay and
training logs show that yaw/upright idle payments outweigh the observed stepping
bonus. In the liftoff course only, upright becomes `2 * (up_z - 1)` and yaw becomes
`-0.2 * (yaw_rate / 0.5)^2`; neither pays for standing. Supported air time and
10 mm saturated lift each have weight 6. A fall costs 2 reward units per event
(previously 0.1 after control-step scaling), exceeding the maximum 0.24 terminal
frame lift bonus. Unloading retains v5 potential shaping. Physics, limits, delay,
reference-imitation status and observations are unchanged.
After 10,944,512 cumulative observed steps, the remaining request is 9,055,488
steps, warm-starting the original 3,014,656-step actor/normalizer; value and
optimizer restart for the changed objective. This is a reward correction, not
proof that stable liftoff has been learned.

Unloading-reward correction: v4 pays a static biased stance indefinitely. v5
replaces that state payment with discounted potential shaping:
`8 * (0.99 * Phi(next) - Phi(previous))` per control step. `Phi` is the existing
bounded eligible-foot unloading score. The implementation divides by `dt` before
the task's common `dt` multiplication. Terminal/final-episode-step potential is
zero; negative corrections are retained. Discounted episode totals telescope to
zero, so standing, rocking and reset cycles cannot farm this shaping term.
The separate 10 mm saturated lift reward, supported air time, actuator physics,
limits and observation layout are unchanged. Passing these checks does not
establish learned liftoff.

The 8,257,536-step checkpoint is preserved. Continuation output is
`training_gpu/lift_10mm_current/unload_progress`, with the remaining 11,742,464-step
request. `--prior-steps` verifies the restored checkpoint and reports cumulative
steps; on-disk checkpoint names within a phase remain local. Policy and observation
normalizer warm-start; value is reset for the new recipe and optimizer restarts.
The original run metadata points the watchdog to its continuation once started.

Continuation selection correction: the latest weights had already collapsed to
standing. After 9,011,200 cumulative observed steps, continuation `lift_recovery`
restores the original 3,014,656-step weights with v5 rewards and a remaining
10,988,800-step request. Original full-model replay shows commanded left-right-left
supported 10 mm events with tilt below 20 degrees and no recorded self contact;
it still falls later and is NOT accepted gait. Cumulative budget and restored
weight age are recorded separately; prior steps may exceed checkpoint age.
Historical 4,456,448/6,029,312 transition-profile bilateral events occurred AFTER
the moving command stopped. They are not commanded-liftoff acceptance evidence.
New native evaluations separately record active/inactive-command supported 10 mm
sample runs; the watchdog displays the active-command statistic when available.

User-requested budget: 20,000,000 environment steps, 512 MX350 CUDA environments,
optimized direct liftoff curriculum, checkpoint/evaluation interval approximately
131,072 steps. Output: `training_gpu/lift_10mm_current`; supervised process:
`systemctl --user status zp10d-lift-20m`. `train.log`, `metrics.jsonl`, and
`metadata.json` report actual progress; launch does not certify liftoff.

Independent read-only watchdog: `systemctl --user status zp10d-lift-supervisor`.
It checks process health once per minute and prints a readable training summary
once every 600 seconds (first summary immediately). No monitoring report files
are generated. View the screen output with:
`journalctl --user -fu zp10d-lift-supervisor -o cat --no-pager`.
The output includes steps, increment, current checkpoint per profile, physical
clearance, supported 10 mm duration and failures. Changed process/read failures
are printed during the minute health checks. It does not restart PPO, change
rewards, send chat notifications, or certify landing/recovery from peak height.
It exits when the requested PPO budget completes. Check its tests with
`.venv/bin/python playground/open_duck_mini_zp10d/test_supervise.py`.

`lift_dense` saturates at 10 mm whole-foot clearance with opposite-foot support.
Direct training disables historical trajectory imitation and support/COM targets;
the historical polynomial remains an interface placeholder, not a validated gait.
Foot-height geometry is loaded from the current source model independently.
Imitation mode still rejects stale references. Measured bench response parameters,
physical torque limits, transport delay and joint limits remain enforced.
Reward gates, alternation/landing memory, native/JIT parity, controller resets and
an actual JIT physics step pass `test_lift_curriculum.py`; this is not walking acceptance.
First checkpoint: 131,072 real environment steps; policy parameter delta 0.0074448846.
All five full native profiles completed their requested duration, but both supported
2 mm and 10 mm liftoff durations were zero: liftoff is NOT accepted. Training continues.

## 2026-10-28 Latest CAD / Placo synchronization and local reference window

Current STEP `528d82d2a2bda0a7757f28600dc02b289c73220f0cf98c8b0d39ebacd92bf65b` is now fully synchronized through prepare, per-part collision export, assemble, independent validation and publish. The previous preparation-only run had not published the model. The canonical URDF now matches the current physical link frames, mirrored/swapped knee/ankle axes, inertials and current corresponding foot-site XY. Sole Z is projected from each current collision foot. CAD geometry and joint limits are unchanged.

Placo/MuJoCo comparison passes all physical frames and COM at home plus both limits of all 16 joints (33 poses; maximum frame error 2.73e-13). The unchanged 50 x 20 mm planning rectangles fit the actual low sole hulls with left/right margins 1.164/0.600 mm. Strict URDF validation passes 21 links, 20 joints, 113 visual mesh references. Full-model numerical smoke checks have no warnings; free-home tilt is 0.081 degrees over 4 s. These are synchronization checks, not physical walking acceptance.

A new 6 s Placo kinematic demonstration is retained at `training/reference_demo/candidate.npz`. All current limits pass; peak planned whole-sole clearance is about 2.500 mm on both sides. This reference is NOT adopted as a validated imitation target or a trained walking policy. The existing fitted training reference/checkpoints remain historical and require revalidation with this model.

Open the native local window (no browser):

```bash
MUJOCO_GL=glfw LP_NUM_THREADS=1 .venv/bin/python scripts/cad/run_memory_limited.py \
  playground/open_duck_mini_zp10d/simulate.py \
  --reference playground/open_duck_mini_zp10d/training/reference_demo/candidate.npz
```

It loops the planned poses, uses actual latest model visuals, and does not integrate contact physics. Space pauses/resumes; mouse rotates/zooms; closing the window stops playback and releases the shared serial lock. Opaque appearance is display-only. Evidence: `cad/local_fix/PLACO_MODEL_SYNC_REVIEW.json`. A native-window 6 s test and a full headless pose replay passed. Source/model and joint-limit mismatches are rejected. `reference.py` now rejects unpublished CAD/model combinations and checks URDF/MuJoCo agreement before generation; the CAD builder requires an explicit stage argument.

# Open Duck Mini ZP10D dynamics model

## Current diagnostic stage — 2026-10-22

**Walking remains unresolved; no accepted bilateral supported liftoff.**
MicroDuck issue [#46](https://github.com/pollen-robotics/microduck_rl/issues/46)
is still open. Achieved-motion gating and lateral-error weighting come from a
community report; neither is an official proven fix for this robot.

Current training uses exact convex foot envelopes instead of the previous thin
foot boxes. Native/MJX plane-support checks agree with full CAD to 5e-9 m.
The repaired model completed 49,152 real PPO transitions in 440.399 s, and all
five full native stand/walk/transition/stochastic checks lasted 8 s without
joint overshoot, self contact, nonfoot ground contact or warnings. Both feet
still have zero supported 2 mm liftoff. This validates the repaired training
pipeline, **not walking**. See `training_gpu/convex_contact_check/stage_status.json`.

The subsequent `training_gpu/convex_contact_lift` trial requested 1,048,576
transitions but was stopped after the 262,144 saved checkpoint. All five native
profiles now fell (1.62–3.28 s), with zero supported liftoff. Additional unsaved
updates were discarded. This is an incomplete, rejected trial.

Source inspection found that the upstream cached automatic reset restores only
physics/observations, retaining previous motor targets, delayed commands and
other task memory. `runner.wrap_training` now clears task/controller memory for
each reset environment while preserving RNG streams and PPO terminal metrics /
timeout truncation. A separate v3 fix exposes alternating-foot landing memory
in the actor's two formerly unused clock inputs, with unchanged observation
size. Actual reset/physics/JAX checks passed, including a runnable reproduction of the
old leak and mixed termination/timeout bookkeeping. The fresh v3 short PPO
check completed 49,152 transitions in 437.751 s (4.092 GB peak RSS). GPU evaluation
lasted 10 s and all five native profiles lasted 8 s (max tilt 1.107–3.822 degrees),
but both feet still have zero supported 2 mm liftoff.

`training_gpu/episode_reset_lift` stopped after the saved/evaluated 524,288
checkpoint (1,048,576 requested, not completed). Deterministic profiles lasted
8 s without supported liftoff; sampled stand/walk fell at 1.88/3.70 s. Logging
exposed a large transient critic-loss outlier, not a persistently large loss:
the second evaluation interval's average value loss was back to 0.153.

A checkpoint counterexample confirms a numerical hazard: at the 49,152 warm
start, first changes in previously constant landing features normalize to
1–2 million, and critic values can jump to 107,134. Bounding normalized inputs
at 5 removes this counterexample. The shared `policy.py` factory checkpoints
this bound for both actor and critic; native reload uses the same factory and
historical configurations keep their original unbounded preprocessing.
`test_policy.py` passed the rare-event and actual save/reload checks (0.940 GB
peak RSS). This identifies a real failure mode, not the proven sole cause of
nonwalking or an exact reconstruction of the failed training minibatch.

The paired correction `training_gpu/bounded_observation_lift` was interrupted
for the user's T0000 test: last saved checkpoint 262,144, last logged transition
360,448, elapsed 1049.533 s; the requested 524,288 budget was not completed.
Reward v3, seed, limits, feedback and PPO were unchanged except the input bound.
No numeric outlier recurred (rolling value loss about 0.187 at 262,144), but
supported 2 mm liftoff remained zero. Deterministic profiles lasted 8 s;
sampled stand/walk fell at 2.58/3.56 s. No training is currently running.

The requested T0000 native simulation comparison completed 12 eight-second cases.
T0000 specifies position-command duration, not torque; identification already
used T0000. Removing only the extra 5.24 rad/s target ramp brought lift-onset
knee peaks to the unchanged 0.980665 N m model cap. With support-hip-only transfer,
adding a step target and an initial 100 ms maximum-error request produced only
1.307/0.330 mm instantaneous left/right clearance, with zero swing-foot load at
the peaks, followed by renewed loading. All 12 cases failed sustained supported
2 mm liftoff/recovery. The six support-hip-only cases had no self contact or joint
overshoot; the two coordinated-hip maximum-effort cases did. This is simulation
only, not an adopted controller change or hardware test. See
[results](training_gpu/t000_liftoff/RESULTS.md) and
[curves](training_gpu/t000_liftoff/torque_and_clearance.png).
The local banked-foot search saved two cases before being stopped to inspect
a candidate. One passed static necessary conditions, but both native holds
fell at 0.78 s with self contact and joint overshoot; it is not a usable demo.

Earlier v5 completed 1,048,576 transitions but did not lift either foot. Both
box-contact liftoff courses stopped at 393,216 transitions after deterioration;
they are rejected policies. Retained Placo trajectories with widened motor
position targets also failed all eight full-model cases, with joint overshoot /
self contact in some cases; expanded motor target limits were not adopted.

CAD, leg spacing, 60-degree home, identified servo physics, real joint / motor
position limits, torque caps and 40 ms command delay remain unchanged. Other
body contacts are still boxes and self-contact is omitted during training, so
full native validation remains required. This dynamics snapshot does not include
the other session's CAD plate thickening. Details and historical evidence are in
[MICRODUCK_REWARD_COMPARISON.md](MICRODUCK_REWARD_COMPARISON.md).

## Historical diagnostic stages (not current running status)

## 2026-10-22 Support-feedback locomotion trial (unaccepted)

The full-range `feedback_gait` run completed 524,288 transitions in 1023.190 s.
Its native stand/direct-walk/start-and-recover tests all run for 8 s, but direct
walk advances only 1.198 mm with no supported bilateral 2 mm liftoff. Standing
passed; walking failed. The retained raw Placo startup also still falls at about
3 s with multiple torque saturations; short airborne events during that failure
are not a passed demonstration.

The earlier `support-locomotion-v3` experiment used no joint imitation, no
old dynamic COM trajectory, and zero privileged reference features. It retains
only the alternating support schedule, starts with equal loads, stretches the
period to 2 s and scales peak clearance to about 5 mm. The COM goal follows
actual foot centers weighted by desired support load. Air-time credit requires
real clearance/unloading, an opposite supporting foot, and signed progress;
standing, backward/sideways motion and unsupported jumps get no air-time reward.
Forward/lateral velocity filters are 0.2/2 s, preserving periodic transfer while
penalizing sustained drift. The unconditional alive 20 is removed, upright/yaw
rewards use weight 2, negative costs remain negative and PPO uses its built-in
adaptive KL schedule (target 0.01, learning rate 1e-5 to 3e-4).

Actual native/JAX actuator parity, JIT physics, the 40 ms feedback queue, 2 s
phase seam and air-time counterexamples pass under the same memory limits
(peak RSS 3.175 GB for v2). `training_gpu/support_locomotion` requested
1,048,576 steps but was intentionally stopped after 655,360 saved transitions
(1330.603 s); bilateral 2 mm liftoff remained zero in GPU evaluation.
The v1/v2 idle subtraction also erased forward credit above twice the command.
V3 keeps achieved-progress credit above target with a separate smooth overspeed
cost, preserving zero-command behavior. Backward credit stays negative so
zero-net forward/backward rocking cannot earn progress. Full adaptation and
focused JIT regression pass (3.177/0.818 GB peak RSS). The serial native callback ran all three full-model profiles at the first
131,072-step checkpoint: each lasts 8 s without warnings, self/nonfoot contacts
or joint overruns; walking advances 3.627 mm but bilateral 2 mm liftoff stays zero.
This is not accepted walking. `training_gpu/support_progress` requests
2,097,152 new transitions, starting from the v2 actor at 655,360 steps and
resetting the critic for the changed recipe. No new walking acceptance is claimed.
The earlier v2 run warm-started `feedback_gait/000000524288`. Both trials
retain the default PPO rollout/batch/update settings. It is an engineering trial inspired by official code and community
diagnosis, not an official proven fix. Bilateral liftoff, forward walking and
landing/recovery have not passed. See the comparison report for exact weights
and evidence limits.

## 2026-10-22 Feedback-based balance repair and gait restart

The reward-only diagnosis trial failed, including zero-command balance. A
393,216-step stand-only curriculum also regressed and was stopped. Native
stochastic checks failed too, so there is no evidence here that action averaging
alone caused the failure. Full results and public-source limits are in
[MICRODUCK_REWARD_COMPARISON.md](MICRODUCK_REWARD_COMPARISON.md).

A software joint-position feedback probe, with unchanged identified actuator
physics and force/range/speed limits, stands for 8 seconds at gain 3 with maximum
0.977-degree tilt; the fixed-command baseline falls at 2.18 seconds. Both knees
remain near their 60-degree geometric home. This changes motor position requests,
not the simulated physical actuator gain. Feedback is queued before the existing
40 ms transport delay.

The stand-only prototype `training_gpu/feedback_balance` completed 49,152 real
PPO transitions / 12 gradient updates. All 4 evaluation initial states stand for
10 seconds before and after training. Saved-policy reload into the full native
model passes 8 seconds with deterministic and stochastic actions (maximum tilt
1.020 and 3.371 degrees), with no warnings, joint overruns or nonfoot contact.
This prototype retains the older motor residual range; it verifies standing
only, and is not a walking policy.

The current full-range variant scales leg motor residuals by `(1+feedback_gain)`
to preserve the policy's reachable geometric motion, and reduces initial policy
noise accordingly. The learner and native evaluator share the feedback formula.
Actual JIT physics, native/learner actuator parity and the 40 ms feedback-delay
regression pass. `runner.py` blocks silent restore across feedback gains or
action scales. Old checkpoints retain their recorded zero-feedback behavior.

`training_gpu/feedback_gait` now trains mixed stand/forward commands from a fresh
zero-mean, low-noise normal policy with `--position-feedback 3`. It requests
524,288 transitions and evaluates/saves every 131,072, with the default MX350,
512 environments, original PPO batch/update settings and optimized rewards.
The action-scale change means it does not silently restore the narrower
standing prototype. Walking, bilateral liftoff and landing/recovery remain
unaccepted; the reference is still an explicitly unvalidated research target.
Native evaluation must use this run's stored scales and feedback gain.

## 2026-10-22 MicroDuck diagnosis trial

The public-source comparison, fixed commits and evidence limits are recorded in
[MICRODUCK_REWARD_COMPARISON.md](MICRODUCK_REWARD_COMPARISON.md). Official issue
46 is still open; the achieved-progress gate is a community proposal, not an
official validated fix. The old 77.88% idle score describes one velocity
component only and does not by itself explain falling or total reward.

The `microduck-diagnosis-v1` trial subtracts the idle tracking baseline only for
nonzero linear commands, increases lateral residual weighting to 3 (preserving
reference COM-transfer velocity), raises linear tracking weight to 4, and gates
the positive contact-match term on signed progress of at least 30% of command.
There is no independent air-time reward in this adapter. CAD, standing pose,
reference, physical limits, actuator parameters and remaining rewards are held
fixed. Full adaptation regression passed, including actual JIT physics and
learner/native actuator parity, within the existing memory limit.

`training_gpu/microduck_diagnosis` adds 262,144 transitions, using the same
original 131,072-step warm start, seed and 512 environments as `transfer_walk`.
It does not restore the later regressed checkpoint. Input hashes and the actual
regression log are archived in its `inputs` directory. The trial completed in 798.992 seconds and failed native acceptance: stand-only
0.92 s, direct walk 0.76 s, transition 0.92 s (before walking begins), with zero
bilateral 2 mm liftoff. The starting checkpoint also fails: stand-only 4.14 s,
direct walk 1.78 s. See its `comparison.json`. No accepted policy or ONNX.

A separate `runner.py --stand-only` balance curriculum was stopped under
`training_gpu/standing_curriculum`: 1,048,576 requested steps with evaluations and
checkpoints every 131,072. It starts from the old 131,072-step checkpoint, keeps
all physical parameters unchanged and samples only zero commands. Metadata
records the restricted command domain; its checkpoints must not be evaluated or
presented as a trained walking policy. The old controller failed balance; the later feedback repair is described above.

## 2026-10-22 Current-width gait training resumed

The user paused the narrow-hip CAD redesign and requested gait training with the
existing synchronized wide-leg dynamics model. No CAD geometry was changed by that redesign. A new simulation
research run uses `training_gpu/transfer_walk`, 512 MX350 CUDA environments,
optimized COM-transfer/pitch rewards, and the identified actuator response.
It warm-starts policy/value/normalizer from the 131,072-step checkpoint; optimizer
and run step count reset. The requested stage adds 262,144 transitions with
evaluation/checkpoints every 131,072. Actual progress is in that directory's
`metadata.json` and `metrics.jsonl`; starting a run is not walking acceptance.

The retained imitation target lacks a passing corrected moving-friction audit.
`runner.py --experimental-reference` explicitly permits this target for simulation
research and records failed/unverified stages in checkpoint metadata. The default
still rejects it. This option changes no actuator limits, physics, CAD or reference
data, and does not declare the old optimistic audits valid. Full native-model
liftoff, support, landing/recovery and walking checks remain required.

Input provenance is saved under `training_gpu/transfer_walk/inputs`. The formal
STEP was separately updated with two `roll_motor_top` plate thickening changes
after the last dynamics synchronization. Those changes are not included in this
run's dynamics; its native evaluation applies to the recorded XML hashes. The
wide-leg spacing is retained. `stage_status.json` records both STEP hashes, so
this run must not be presented as validation of the later thickened geometry.

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --gpu \
  playground/open_duck_mini_zp10d/runner.py \
  --steps 262144 --checkpoint-every 131072 --envs 512 \
  --reward-mode optimized --experimental-reference \
  --restore playground/open_duck_mini_zp10d/training_gpu/identified_walk/checkpoints/000000131072 \
  --output playground/open_duck_mini_zp10d/training_gpu/transfer_walk
```

Do not rerun into an existing output directory. The memory-limited wrapper keeps
the shared serial lock, 7 GB RAM cgroup and zero swap; JAX preallocation is off.

Actual result: 262,144 new transitions completed in 934.384 seconds. Both new
checkpoints failed full native evaluation (1 s standing, then 5 s walk request,
then 2 s recovery request). At 131,072 new steps, the model stops at 1.38 s and
46.437 degrees tilt, with 34.598 mm forward base displacement and zero supported
2 mm liftoff on either foot. At 262,144 new steps it stops at 0.86 s / 47.012
degrees, before the walk request, with zero bilateral liftoff. These displacements
are loss of balance, not accepted walking distance. This run regressed from the
old checkpoint's 2.54 s stop and is not an accepted policy or ONNX. Reports:
`training_gpu/transfer_walk/native_131072/evaluation.json`,
`training_gpu/transfer_walk/native/evaluation.json`, and `stage_status.json`.

## 2026-10-21 Actual liftoff still fails — task unfinished

2026-10-22: corrected the requested transfer direction before **left** lift:
S02 inward, actual COM and body toward the robot's **right**, right foot taking
more load. Run `diagnose_identified_liftoff.py --left-lift-transfer
--live-directory DIRECTORY` through the memory-limited wrapper. This uses Placo
COM IK and bounded motor-position feedforward, never imposed physical base poses.
In the full model, 1.4 s gives 26.896 mm COM and 5.857 mm body movement right,
S02 24.006 degrees, left/right load 6.980/13.216 N, and no joint overrun.
Direction passes; the left foot remains loaded and **liftoff still fails**.
This short diagnostic does not establish sustained balance or change the formal
reference/policy. Evidence: `training/reference_repair/directed_transfer.json`.

Native live viewing and rendering/physics-parity validation are implemented.
`evaluate.py CHECKPOINT --live
--output DIRECTORY` publishes `live.html` and atomic image/metric snapshots to
that directory. The Placo diagnostic accepts `--planning-only
--live-directory DIRECTORY`; it shows physical tracking, not imposed base poses.
Serve the directory with the existing HTTP server and open `live.html`.
The current LAN page is <http://172.20.179.51:8765/live.html>; it reports waiting,
running or finished explicitly. It now shows the corrected transfer and labels
actual COM/body displacement positive toward robot-right. `replay.html` remains
the old failure recording.
Neither viewer is evidence of accepted liftoff. The startup preparation segment
is still excluded by the settled-cycle fit and has not entered PPO execution.

The reference checks below passed necessary conditions only. They did not
complete the requested Placo/liftoff repair. Full-model free-base unloading,
bilateral liftoff, landing and recovery remain the acceptance gate.

Further review found that the former LP allowed moving-joint dry friction to
assist motion. With friction opposing velocity, the unchanged adopted reference
fails 3 of 298 sampled points (right knee, 1.36–1.40 s), peaking at 1.149434 of
the provisional motor cap. The old "all passed" result below applies only to
that optimistic test. Fitting and new PPO runs now require the corrected check;
the existing reference is retained as a rejected baseline, not accepted motion.

The follow-up PPO checkpoint at 131,072 steps was actually tested in the full
native model with the identified parameters and 40 ms path delay. It falls at
2.54 s and gives zero supported 2 mm liftoff on both sides. Training was stopped
and the checkpoint retained. Offline command-refinement trials also failed and
were not adopted. Reducing startup clearance and extending single support did
not yield an accepted candidate. This task is still unfinished.

Follow-up: learner and native policy evaluation now apply the same identified
response through `actuators.py`, refresh cached inertias, preserve home motor
preload, and use a fixed two-control-tick (40 ms) path delay. Checkpoints bind
the actuator signature as well as the reference. The leg action range now
includes the position offset needed to produce load torque with the identified
gain; joint, command, speed and motor-force bounds still apply. Clearance error
uses this reference's actual 2.5 mm scale, and 2 mm unloaded metrics complement
the historical 10 mm metrics. These changes passed native/learner parameter
parity and JIT regression; PPO/free-base liftoff acceptance is still pending.

The same `data/reference.json` and `data/polynomial_coefficients.pkl` now contain
standing plus a conservative **0.01 m/s, 0.5 s cycle, 2.5 mm target-clearance**
reference. This replaces the geometrically corrected but dynamically incompatible
15 mm reference; it does not establish 15 mm liftoff or walking.

Loaded soles stay flat while torso roll is allowed. Sole-roll priorities blend
through double support instead of abruptly switching at COM centre crossing.
The planner uses a verified 50 x 20 mm rectangle inside the actual bottom hull,
16 single-support nodes, 90 ms double support, C2 swing translation, contact guards,
and smaller first two swings. Reduced-model height 0.35 m is a planning proxy;
physical COM starts at 0.197683 m. CAD, 60-degree geometric home, URDF/MJCF,
physical limits, friction and motor caps remain unchanged.

With the 7.8 V S09 equivalent-response candidate applied in memory, all **1,192**
200 Hz startup/continuous-reference samples pass the optimistic actual-floor
contact-force LP, provisional torque caps, and position-command bounds. The
float32 fitted cycle, including its seam, passes **501** 1 ms samples using
analytic joint derivatives. Worst minimized peak-torque ratios are **0.997604**
for startup and **0.884520** for the fitted cycle: startup has little headroom.
Fit speed extrema are 4.043305 rad/s; maximum penetration is 0.001263 mm.
These are necessary conditions with optimistic dry friction and a 0.2 mm contact
allowance, not a tracking controller or hardware torque measurement. Fit now
rejects candidates failing either raw or fitted dynamics before writing data.
Slow-command reward/contact gating also recognizes 0.01 m/s as motion; checkpoint
evaluation rejects a different reference hash. Old policies are not validated
against this replacement reference.

A full free-base open-loop replay with identified response, home preload and
36 ms transport delay still tips at 2.164 s (reference time 0.664 s), before the
first swing request. This retains the independent feedback-control problem;
no PPO or ONNX was produced. Next: adapt feedback/PPO to the identified actuator,
then verify actual unloading, liftoff, landing and recovery.

Fixed evidence: `training/reference_repair/` (`candidate.npz`,
`fitted_candidate.npz`, `raw_audit.json`, `fitted_audit.json`, `open_loop.json`,
`open_loop.npz`, and `replay.py`). Regenerate and stage with:

```bash
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/reference.py --seconds 6
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/reference.py --fit
```

`--fit` stages checked data under `/tmp/zp10d-reference/data`; adoption must use
those checked files together. The fixed evidence directory preserves the inputs
of this adoption. Complete Chinese notes are in `SERVO_IDENTIFICATION.md`.

## 2026-10-20 identified actuator root-cause diagnosis

The earlier claim that the new candidate plus home preload stands for 8 seconds
is withdrawn. Mutating armature after compilation requires `mj_setConst` to
refresh MuJoCo's inertia and constraint constants; `replay_reference.replay` now
refreshes them before every rollout. An airborne force-balance regression drops
from 0.085609 residual with stale constants to 6.63e-16 after refresh. The original
capped baseline still reproduces its exact 2.602 s / 45.149346-degree stop.

With correct constants, the 7.8 V / 71 g S09 response candidate and fixed home
preload tips past 45 degrees at 2.168 s; Placo replay stops at 1.946 s with no
2 mm supported liftoff. Removing command limits, torque caps, or both does not
rescue it. The single-load fit remains an equivalent response candidate, not a
measurement of whole-robot heavy-load response or stall torque.

Two separate issues are isolated: the much lower position gain needs posture
stabilization beyond a constant preload, and the free-sole-roll Placo reference
is incompatible with its actual support geometry. At reference t=0.4 s, the
right sole is banked 18.31 degrees; its low contact edge lies at lateral
-111.37…-107.97 mm while the corrected ground ZMP is -15.862 mm.
The earlier -50.25 mm display is withdrawn: its horizontal-shear sign was wrong;
the independent contact-force LP is unchanged.
Only 46/118 sampled reference states admit a floor-contact force solution, even
with perfect tracking. Hypothetically doubling/quadrupling leg stiffness restores
an 8 s stand but does not establish hardware gains. Flat-foot/exact-COM candidate
constraints fail IK at t=0.100 s and are not adopted as a repaired reference.

Formal CAD, 60-degree geometric home, robot parameters, reference data and
physical limits remain unchanged. No new policy/ONNX or walking acceptance is
claimed. Full corrected results, reproducible commands and limitations are at
the top of `SERVO_IDENTIFICATION.md`; artifacts stay in the fixed
`training/calibrated_actuator/` directory (`assessment`, `diagnosis`, and
`standing_diagnosis`). Native integration uses the full 3296-geom free-base model,
2 ms physics, the shared serial lock and the import-time 4 GB address cap.

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

The superseded 2026-10-18 reference covered **standing and 0.03333333 m/s forward motion only**. Its
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

The user-requested **free-base torque-limit comparison** is now available:

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --wait \
  playground/open_duck_mini_zp10d/replay_reference.py \
  /tmp/zp10d-reference-before-dynamics/candidate.npz
```

This uses the current reference's original 200 Hz Placo startup and joint samples,
one second at home, and a 0.5-second smooth transition. Both cases use identical
position commands, the full 3296-geom scene and freely integrated base motion.
The second case disables only the 14 main servos' actuator/joint torque caps in
memory; model files, antenna caps, gains, armatures and joint limits stay intact.
The capped case stops at 45.15 degrees tilt after 1.102 reference seconds.
The uncapped case completes 5.996 reference seconds with 7.109 degrees maximum
tilt, but whole-foot clearance reaches only 0.068/0.067 mm (left/right), with no
2 mm lift; knee peaks reach 4.454/4.193 N m. Results and integrated trajectories
are in `training/reference_replay/assessment.json`, `capped.npz` and `uncapped.npz`.
Removing the caps improves this trial's stability but does not produce liftoff.
This open-loop diagnostic has no learned balance feedback; upstream PPO instead
uses the Placo reference as an imitation reward and chooses joint actions from
observations. The result is neither a walking certificate nor proof of mechanical
inability to lift. Formal model/reference hashes were unchanged.

Follow-up isolated the direct replay's failure to controller/reference response
mismatch. The 4.454 N m left-knee peak occurs at reference time 0.856 s, during
double support: target 62.843 degrees versus actual 81.931 degrees. At that instant
viscous damping opposes motion by 2.213 N m and armature acceleration accounts for
1.238 N m. Torque balance closes within 2.22e-16 N m. The inherited damping and
armature were not retuned when the torque cap changed from 3.23 to 0.980665 N m.
Ignoring external load, the constant-speed torque balance consequently drops from
5.646 to 1.630 rad/s; damping plus dry friction at the allowed 5.24 rad/s command
speed alone requires 3.002 N m. This diagnoses the simulation, not actual ZP10D
hardware capability.

Six uncapped response trials are in `training/reference_replay/diagnosis.json`.
Only increasing kp by four, with original geometry and joint limits, produces
20.022/27.011 mm whole-foot lift and completes the six-second reference without
falling. Both feet clear 10 mm while the other carries load, for 0.196/0.232 s.
Knee peaks rise to 10.760/11.446 N m: this is a causal diagnostic, not an accepted
controller. Halving replay speed or reducing armature alone does not lift; reducing
damping alone or with armature produces lift but falls and violates soft joint
limits, so those candidates are rejected. Clock scaling does not replan COM
dynamics. Independent FK verifies saved foot heights; body sag and joint lag
together cancel the nominal lift. Motor response calibration and feedback/reference
adaptation remain unresolved. Formal parameters and home were preserved.

For real-servo acquisition, follow [ZP10Dg serial capture and BAM identification](SERVO_IDENTIFICATION.md).
`scripts/collect_zp10d.py` provides a read-only probe and small-angle recordings
on Windows/Linux. Its frames were checked against the supplied ESP32/STM32 and
position-readback examples. Actual COM5 read-only queries passed after adding
CBM firmware replies (`@ ...` version and `MOD1` mode); ID0/115200/8N1 and P2071
were confirmed without movement commands. After the user confirmed a fixed case
and 7.8 V supply, a six-second T1000 recording collected 550 progressive PRAD
samples at 92.09 Hz. A 20 ms position-command gap fixed the initial query loss.
Late-window errors are nominally 2.16-2.295 degrees. The user subsequently confirmed
that nothing was attached to the output shaft (no external load or horn); the original
capture metadata retains the setup description available during acquisition.
PDST returns no ACK. These observations are not a physical angle calibration
or a BAM fit. Evidence and the response plot are under
`training/actuator_capture/slow_7p8v/`. Raw captures are not yet directly usable BAM logs.

On 2026-10-20 the user switched to COM6 and reported **71 g for the entire moving
assembly**: S09 drives its original connections and S10, with 47 g for S10 plus
24 g for the other moving connections. Read-only COM6 queries confirmed ID0,
the same firmware and mode 1. Fresh local BRep measurements give a 68.650 mm axis spacing;
distributing each measured group mass by its CAD solid volumes estimates a
48.459 mm combined COM radius and 0.000195902 kg.m2 inertia about S09. These are
geometry-based estimates, including assumed internal mass distributions, not
measured COM/inertia. Evidence, selected parts and assumptions are in
`training/actuator_capture/s09_s10_71g/rig_geometry.json`.
The user subsequently confirmed **S10/connections ABOVE S09, horizontal axis,
7.8 V, local +/-10 degree clearance, electrically held by S09 without external
support**. P1229 is the operator-aligned upright reference, not pendulum-down zero.
After the startup-movement discussion, testing resumed; no PCSM/PCSR or other
persistent configuration command was sent, and power-cycle behavior was not verified.
Runtime PULR and 20 PRAD readings were recorded without a PULR ACK.

Fixed-reference loaded T0000 training/held-out recordings collected 1180/1183
samples at 91.04/91.18 Hz. Nominal measured ranges were -5.94…+4.86 and
-2.97…+2.835 degrees. Small commands sometimes produce no motion until error
accumulates: this is evidence of a starting threshold, not an isolated diagnosis
of torque, firmware deadband or friction. A separate T2000 comparison observed
PDST cancel remaining travel (normal endpoint P1268 versus stopped P1227),
despite no ACK; this only establishes its effect in that trial.

`identify_servo.py` now uses upstream BAM dynamics with a CAD rigid-load model
and an equivalent position actuator. Preliminary M1/M2 fits remain candidates;
physical angle calibration, parameter identifiability and further load/trajectory
validation are still required before updating the robot. These are local adapter
reports, not directly loadable upstream BAM actuator parameter files. Raw data,
assessments and fit plots are under `training/actuator_capture/s09_s10_71g/`.
See `SERVO_IDENTIFICATION.md` for current errors and limitations. Formal CAD,
home pose, actuator parameters and PPO remain unchanged by these captures/fits.

After the user confirmed +/-60 degree clearance and requested larger motions,
the same rig collected a +/-30 degree slow check (555 samples), a +/-45 degree
T0000 fitting set (1187 samples), and a separate +/-35 degree T0000 validation
set (1185 samples). All motion queries passed and feedback stayed within its
configured window. The new M1 fit reached its stopping condition after 67
generations: validation MAE/RMSE/peak error 0.437/0.559/3.893 degrees.
An independent MuJoCo 3.12.0 isolated-hinge replay using the same parameters
gave 0.432/0.552/4.082 degrees with no numerical warnings. Plots and reports
are in `training/actuator_capture/s09_s10_71g/bam_wide/`.
These are usable equivalent-response candidates for further simulation checks
at 7.8 V and this 71 g load. The fixed 0.980665 N m torque cap is still a prior,
and the fit does not establish stall/continuous torque, all-load behavior or a
6.6 V calibration. No rig reorientation, formal robot parameter update or PPO
run followed this larger-motion test. The previous reorientation suggestion is
not a prerequisite for using these data.

The user reports **ZP10Dg supplied at approximately 8 V**. This does not rescale
the existing torque cap or identify motor response. `audit_actuator_response.py`
isolates the inherited parameters in a zero-gravity shaft with no external load:
the current/legacy capped models reach 90% of a 60-degree position step after
0.626/0.214 s. Constant-torque terminal speeds agree with analytic damping/friction
balance within 1e-7 rad/s. Results are in `training/actuator_response.json` and npz.
This is a model check, not a hardware calibration. A datasheet no-load speed and
its voltage can constrain an initial linear torque-speed approximation; they do
not uniquely identify armature, gain or delay. The audit accepts `--speed-60-s`
and `--speed-voltage-v` together, and never edits the formal actuator parameters.
Calibration does not add a requirement for perfect direct joint replay before
upstream Placo-reference/PPO training resumes.

The supplied 28-page manual `/root/ZP10D舵机/众灵舵机使用手..08.pdf` has SHA256
`9905405f676031379ceaa51b25bd9743c269d278e0f10fd08f538f67a532e265`.
Both selectable text and embedded image tables were reviewed. Page 12 lists
**ZP10D**, 5–8.4 V, 270 degrees and 10 kgf cm (0.980665 N m), without a torque
test voltage. It does not identify the **g** suffix. Page 14 gives 0.16 s/60 deg
at 6 V and 0.14 s/60 deg at 6.6 V, alongside **13.5/15 kgf cm**, with no exact
model label. These are generic table values, not established ZP10Dg values and
not 8 V specifications. Page 18 recommends running torque around one third of
stall torque (conditionally 0.327 N m at the existing cap); this is guidance,
not a measured continuous-duty rating. Pages 25–27 document bus commands;
the `T` field is requested movement duration, not a physical speed measurement.

The audit now verifies this PDF's hash and records page-specific applicability.
Its two **what-if** cases retain the existing torque cap, friction, armature and
gain, while matching the generic constant-speed values with equivalent damping
0.139445/0.122014 N m s/rad. Native terminal speeds match the analytic calculation
within 1e-7 rad/s; 60-degree step 90% times are 0.294/0.286 s. These hypothetical
combinations are not manufacturer parameter sets, physical calibration or walking
validation. Formal models remain unchanged. Exact-model speed at the actual
supply and transient response remain missing; the manual alone cannot resolve
the inherited armature, gain or delay. Evidence remains in the same
`training/actuator_response.json` and npz outputs.

The user subsequently selected the **6.6 V table parameters for simulation**.
`replay_reference.py --manual-6p6v MANUAL.pdf` now applies 15 kgf cm
(**1.4709975 N m**) and 0.14 s/60 deg (**7.479983 rad/s**) to the 14 main servos
in memory. With inherited friction 0.068 N m, the linear torque-speed
approximation gives damping **0.187566949 N m s/rad**; kp 13.37, armature
0.027 kg m² and SG90 parameters remain inherited. The isolated shaft reaches
the specified steady no-load speed within 1e-7 rad/s; its 60-degree step reaches
90% at 0.242 s. This selects both torque and speed from the table, unlike the
earlier speed-only sensitivity cases with the original torque cap.

Full 3296-geom free-base integration used identical Placo commands, with 1 s
home preparation and 0.5 s blending. It produced:

| Direct replay | Formal baseline | Selected 6.6 V assumption |
|---|---:|---:|
| Reference time before 45-degree tilt stop | 1.102 s | 1.264 s |
| Left/right whole-foot peak height | 0.022 / −0.004 mm | 0.091 / 1.576 mm |
| Supported clearance at least 2 mm | 0 s / 0 s | 0 s / 0 s |
| Left/right knee peak torque | 0.981 / 0.981 N m | 1.471 / 1.471 N m |
| Maximum soft joint-limit overrun | 0.463 deg | 2.238 deg |

The separate 3 s home hold under the 6.6 V assumption ran to duration: maximum
tilt 0.078 degrees, knee peaks 0.419/0.429 N m and no joint-limit overrun.
Direct motion replay still fails; this does not establish physical motor
incapability or replace PPO balance feedback. Both knees saturate during motion,
while the unmeasured transient parameters remain inherited. No learned policy
was retrained, and formal CAD/model/reference files were preserved.

Saved-state FK explains why the selected 6.6 V replay still misses lift. At the
first left planned peak (reference 0.420 s), nominal whole-foot height is
15.008 mm. Substituting actual base translation lowers it to 9.200 mm (base
5.808 mm below plan); additionally substituting actual base orientation lowers
it to 1.167 mm (tilt 4.445 degrees), then actual joints give −0.031 mm. These are
sequential geometric substitutions, not separately integrated physical states.
The left foot still carries 5.151 N. Its knee target/actual are 48.610/28.722
degrees, with knee, hip pitch and ankle torque saturated. At the first right
planned peak (0.720 s), the base is 12.249 mm below plan and tilted 13.087 degrees;
the right foot still carries 10.384 N, and its knee/hip pitch/ankle also saturate.
This identifies residual unloading, posture and tracking errors in this model;
it does not isolate physical hardware torque as the sole cause. Details are in
the report's `swing_peak_explanation`, computed from the same saved trajectory
without another physics run or parameter change.

The upstream cause audit traces this mismatch to the current planning/execution
interface. `reference.py` limits joint positions and velocities but does not
constrain joint acceleration or actuator torque. In first-swing knee reversals,
central position differences with a 20 ms half-window reach 369.57/383.08 rad/s²;
with a 50 ms half-window they still reach 155.13/152.99 rad/s². Thus the rapid
changes persist beyond a single 5 ms sample. The inherited 0.027 kg m² armature
accounts for 95.72% of each knee's home mass-matrix diagonal (rigid-body remainder
0.001207 kg m²); this is a fixed-coordinate diagonal comparison, not a full
effective inertia identification. Armature-times-reference-acceleration is only
one hypothetical dynamics term, not actual motor torque or a hardware load
requirement. With kp 13.37 and a 1.4709975 N m cap, the proportional actuator
saturates at 6.304 degrees of position error. The replay continues on its fixed
reference clock; it applies neither base/COM feedback nor a learned policy.
Placo does plan COM transfer, but the physical base is free and can fail to
follow that plan. These findings explain why changing stall torque and terminal
speed alone does not complete the dynamics/control adaptation; exact physical
motor parameters and a uniquely isolated dominant cause remain unproven.
Source trace, inertia audit and derivative windows are recorded in the same
report under `response_cause_audit`.

Evidence is in `training/reference_replay/manual_6p6v/assessment.json`, the three
named trajectory npz files and `assessment.png`. Independent FK height error is
below 6e-16 m; all samples are finite, the 1.471 N m cap holds, and all numeric
fields available in the previous capped baseline reproduce within 1e-10.
The run remained serial under the 4 GB address-space limit, peak RSS 1.111 GB
(verification/plot 1.147 GB).

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --wait \
  playground/open_duck_mini_zp10d/replay_reference.py \
  /tmp/zp10d-reference-before-dynamics/candidate.npz \
  --manual-6p6v '/root/ZP10D舵机/众灵舵机使用手..08.pdf'
```

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --wait \
  playground/open_duck_mini_zp10d/audit_actuator_response.py \
  --manual '/root/ZP10D舵机/众灵舵机使用手..08.pdf'
```

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --wait \
  playground/open_duck_mini_zp10d/diagnose_reference_replay.py \
  --candidate /path/to/the/original/candidate.npz
# Analyze existing diagnostic trajectories without rerunning physics:
.venv/bin/python scripts/cad/run_memory_limited.py --wait \
  playground/open_duck_mini_zp10d/diagnose_reference_replay.py \
  --candidate /path/to/the/original/candidate.npz --report-only
```

The script records knee peak-state force terms and checks baseline trajectory
reproduction, geometry measurements, torque identities, and joint-limit margins.
`training/reference_replay/diagnosis.png` shows the reference/actual knee angles,
motor torque and whole-foot clearance during the first two swing phases.

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

### 抬脚奖励 v7（2026-10-28，验证与训练中）

v6 在累计 13,041,664 步停止，保留其检查点；原生回放仍未验收正常抬脚。v7 将折扣势差替换为每次尝试的新卸载最大值增量，接触力先经 0.1 s 一阶滤波；只有实际合格抬脚后落地、交替目标改变才重置最大值。保持偏重、重复摇摆不重复领分，每次尝试总奖励至多 8×0.02=0.16，小于跌倒损失 2。10 mm 满额、原始接触力离地判据和动力学不变。从历史 3,014,656 步权重恢复，剩余累计预算 6,958,336 步；历史左右各 0.10 s 有效离地但 3.78 s 跌倒，不能视为步态通过。

### 抬脚奖励 v8（2026-10-28，训练中，未验收）

v7 累计至 13,369,344 步停止并保留全部检查点。其 262,144 步原生确定性回放再次退化为站立；随机策略右脚短时离地并跌倒。初始评估加权离地/腾空项合计 201.56、转向成本 169.78、滑移 72.94、跌倒 100（均为 dt 缩放前的回合累计分项），显示联合成本压过离地收益。v8 仅将转向平方惩罚限制为最大密度成本 0.2，卸载防重复、其他奖励与动力学保持。完整 JIT 物理与重置测试通过；从历史能抬脚但会跌倒的权重恢复，剩余预算 6,630,656 步。尚无改善抬脚的验证结论。

### 本地训练评估 3D 窗口（2026-10-28）

`training_window.py training_gpu/lift_10mm_current` 自动跟随根目录的 `continuation_run`，在唯一的本地 MuJoCo 窗口循环播放最新完成的 `native/*/walk/evaluation.npz`。这是检查点原生物理评估回放，不是 512 环境逐物理步直播，也不是 Placo 动画。程序只运行显示用 FK，不执行物理积分或 JAX；显示副本保留 113 个 CAD 外观网格、全部关节与惯量，去掉不可见碰撞网格，正式模型不改。红点为质量加权整机重心投影，蓝点为脚中心；显示左右全脚离地高度、承重、机身倾斜和结果。空格暂停，S 切换 0.25/0.5/1 倍速，R 重播。支持 `--profile transition/stand/walk_stochastic` 与 `--check`。

必须在导入原生库前保留脚本的 4 GB 地址空间上限，并以 systemd `MemoryMax=1000000000`、`MemorySwapMax=0` 启动；训练保留原有 7 GB 限制与串行计算锁。播放不占计算锁，不启动第二个物理工作进程。WSLg 图形环境使用 `DISPLAY=:0 MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=0 GALLIUM_DRIVER=d3d12`；强制软件渲染曾在 1 GB 硬限额内失败，不放宽限额重试。D3D12 路线已截图核验真实机器人画面，并在运行中确认自动载入后续检查点，窗口约 770 MB。

本轮累计 20,054,016 环境步结束（请求 20,000,000，PPO 评估批次向上取整）；v8 分支新增 6,684,672 步，权重从历史 3,014,656 步恢复，不等同于同一策略连续训练 2,000 万步。最终五类原生回放均未通过；行走指令期间双脚 ≥10 mm 最长连续各 0.20 s，仍在 2.12 s 跌倒。训练预算完成不代表稳定踏步完成。最后检查点为 bounded_stability/checkpoints/000006684672；本地窗口自动跟随最终原生回放并保持开启。
