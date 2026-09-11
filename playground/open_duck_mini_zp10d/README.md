# Open Duck Mini ZP10D model draft

This directory is an isolated first-pass model for replacing all 14 STS3215
servos with ZP10D double-shaft servos. It deliberately does not overwrite the
existing `open_duck_mini_v2` model.

Current state:

- joint names, axes, limits and printed-part meshes are inherited from V2;
- actuator force is capped at the ZP10D published stall torque of 10 kg.cm
  (`0.980665 N.m`);
- ZP10D response parameters (`kp`, damping, friction and armature) are unknown;
- link mass, centre of mass and inertia still describe the original assembly;
- printed mounting parts have not yet been redrawn around the ZP10D envelope.

The model is therefore suitable for checking XML compatibility and estimating
whether the lower torque limit is viable. It is not yet a manufacturing model
or a validated dynamics model.

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
