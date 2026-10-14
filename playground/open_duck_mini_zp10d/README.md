# Open Duck Mini ZP10D model draft

This directory is an isolated first-pass model for replacing all 14 STS3215
servos with ZP10D double-shaft servos. It deliberately does not overwrite the
existing `open_duck_mini_v2` model.

Current state:

- joint names and axes are inherited from V2; S13 (`head_yaw`) is limited to
  ±100° in both ZP10D XML files, and its position actuator inherits this range;
- actuator force is capped at the ZP10D published stall torque of 10 kg.cm
  (`0.980665 N.m`);
- ZP10D response parameters (`kp`, damping, friction and armature) are unknown;
- mesh, mass, centre of mass and inertia were generated from the older
  `cad/local_fix/open_duck_mini_covers_forward_review.step`; they have not been
  refreshed from the current `open_duck_mini_upper_refit.step`;
- the current CAD mounting parts have been adapted, but their later geometry,
  speaker/antenna placement and body/head changes are not synchronized here;
- only the two foot soles participate in robot contact. Head, body and bracket
  self-collisions cannot be detected by this MJCF's existing contact settings.

The model is therefore suitable for checking XML compatibility and diagnosing
the older dynamics model. It cannot validate current CAD clearances or establish
that the current assembly can stand or walk. Current assembly review evidence
is recorded in `cad/local_fix/FINAL_ASSEMBLY_REVIEW.json`.

Run an existing policy against this draft with:

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py \
  -o ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
  --model_path playground/open_duck_mini_zp10d/xmls/scene_flat_terrain.xml
```

The policy was trained for the original robot. A successful playback does not
mean it will transfer to the ZP10D hardware; retraining is expected after the
CAD-derived mass properties and measured actuator parameters are installed.

## Initial torque-limit comparison

A 10-second headless run used the same existing policy and a `0.15 m/s`
forward command. The original model travelled about `0.93 m`; this draft
travelled about `0.16 m` and fell. Its actuators were at the published stall
torque limit for about `28.38%` of simulated actuator samples. This comparison
still uses the old geometry and inertias, but it shows that deploying the old
policy unchanged is not a credible path.
