"""Simulate the current ZP10D robot, using home hold or saved liftoff commands.

Run through scripts/cad/run_memory_limited.py. No upstream ONNX policy is used.
Saved commands are replayed through physics; they are not a feedback controller.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import resource
import threading
import time


def reference_window(model, candidate, *, headless=False, duration=None):
    """Show a source-checked Placo pose sequence, without claiming physical tracking."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from reference import source_hashes

    with np.load(candidate, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        positions, bases = archive['qpos'].copy(), archive['base'].copy()
    if metadata['source_hashes'] != source_hashes():
        raise RuntimeError('Reference belongs to another model; regenerate it first')
    joints = [model.joint(name).id for name in metadata['joint_names']]
    addresses = model.jnt_qposadr[joints]
    if (positions.shape != (len(bases), len(addresses)) or bases.shape != (len(bases), 4, 4)
            or not np.isfinite(positions).all() or not np.isfinite(bases).all()):
        raise ValueError('Invalid reference positions')
    if (np.any(positions < model.jnt_range[joints, 0] - 1e-5)
            or np.any(positions > model.jnt_range[joints, 1] + 1e-5)):
        raise ValueError('Reference exceeds the current joint limits')
    period = (len(bases) - 1) / 50.
    if period <= 0:
        raise ValueError('Reference needs at least two samples')
    data = mujoco.MjData(model)
    model.geom_rgba[:, 3] = 1  # Display only: show the complete outer shape.
    if not headless:
        import mujoco.viewer
    paused = [False]
    def key_callback(key):
        if key == 32:
            paused[0] = not paused[0]
    existing_threads = set(threading.enumerate())
    context = nullcontext(None) if headless else mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback)
    viewer_threads = set(threading.enumerate()) - existing_threads
    elapsed = 0.
    with context as viewer:
        if viewer:
            viewer.opt.geomgroup[3] = 0
            viewer.cam.distance = .85
            viewer.cam.azimuth, viewer.cam.elevation = 140, -15
            viewer.set_texts((None, None, 'Placo kinematic reference / latest CAD',
                              'NOT physics or trained walking | Space: pause/resume'))
            print('MUJOCO_LOCAL_REFERENCE_WINDOW_READY', flush=True)
        while duration is None or elapsed < duration:
            if viewer and not viewer.is_running():
                break
            tick = time.monotonic()
            t = elapsed % period
            i = min(round(t * 50), len(bases) - 1)
            data.qpos[:3] = bases[i, :3, 3]
            data.qpos[3:7] = np.roll(Rotation.from_matrix(bases[i, :3, :3]).as_quat(), 1)
            data.qpos[addresses] = positions[i]
            data.qvel[:] = 0
            data.time = t
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.xpos).all():
                raise RuntimeError('Nonfinite reference pose')
            if viewer:
                viewer.cam.lookat[:] = data.xpos[model.body('base').id] + [0, 0, .03]
                viewer.sync()
                time.sleep(max(0., .02 - (time.monotonic() - tick)))
                if not paused[0]:
                    elapsed += time.monotonic() - tick
            else:
                elapsed += .02
    for thread in viewer_threads:
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError('MuJoCo reference window did not finish closing')
    print(json.dumps(dict(mode='kinematic reference, not physical walking',
                          displayed_seconds=elapsed, source_hashes=metadata['source_hashes'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', choices=('left', 'right'),
                        help='rerun saved 50 Hz liftoff commands, then hold home')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--reference', type=Path,
                        help='show a current-model Placo NPZ as looping kinematic poses, not physics')
    parser.add_argument('--duration', type=float, help='seconds after the 3 s home settling')
    args = parser.parse_args()
    if args.reference and args.replay:
        parser.error('--reference and --replay cannot be combined')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
    if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
        parser.error('run through scripts/cad/run_memory_limited.py')

    import mujoco
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    xmls = Path(__file__).resolve().parent / 'xmls'
    evidence = root / 'cad/local_fix'
    mass = json.loads((evidence / 'open_duck_mini_zp10d_mass_properties.json').read_text())
    paths = (evidence / 'open_duck_mini_upper_refit.step',
             xmls / 'open_duck_mini_zp10d.xml', xmls / 'scene_flat_terrain.xml')
    hashes = []
    for path, key in zip(paths, ('source_step_sha256', 'mjcf_sha256', 'scene_sha256')):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != mass[key]:
            raise RuntimeError(f'Published model hash mismatch: {path}')
        hashes.append(digest)

    model = mujoco.MjModel.from_xml_path(str(paths[2]))
    if model.nu != 16 or model.neq != 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError('Expected the current 16-actuator, floating-base ZP10D model')
    if args.reference:
        reference_window(model, args.reference, headless=args.headless,
                         duration=args.duration or (1. if args.headless else None))
        return
    data = mujoco.MjData(model)
    home = model.key('home').id
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.ctrl[:] = model.key_ctrl[home]
    mujoco.mj_step(model, data, nstep=round(3 / model.opt.timestep))
    mujoco.mj_forward(model, data)
    frames = []
    if args.replay:
        report = json.loads((evidence / 'CURRENT_LIFTOFF_CONTROL.json').read_text())
        trial = next(t for t in report['trials'] if t['side'] == args.replay)
        if (trial['status'] != 'passed' or trial['formal_hashes'] != hashes
                or trial['diagnostic_prefix_replayed']
                or trial['actuator_order'] != [model.actuator(i).name for i in range(model.nu)]):
            raise RuntimeError('Saved liftoff commands do not match this model')
        frames = trial['frames']
        print('Replaying accepted low-clearance position commands from controller',
              trial['simulation_script_sha256'], flush=True)
        print('This is command replay through physics, not live feedback or a trained policy.', flush=True)

    duration = args.duration
    if args.headless and duration is None:
        duration = frames[-1]['time_s'] if frames else 5.
    print(f'Model: {paths[2]}\nActuators: {model.nu}; nq={model.nq}; nv={model.nv}', flush=True)
    print('Mode:', f'{args.replay} liftoff command replay' if frames else 'live home hold', flush=True)
    if not args.headless:
        import mujoco.viewer

    existing_threads = set(threading.enumerate())
    context = (nullcontext(None) if args.headless else
               mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False))
    viewer_threads = set(threading.enumerate()) - existing_threads
    base = model.body('base').id
    start = data.time
    max_error = 0.
    max_tilt = 0.
    frame_index = 0
    phase = None
    try:
        with context as viewer:
            if viewer:
                viewer.cam.distance = .85
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -15
                viewer.cam.lookat[:] = [0., 0., .20]
            while (duration is None or data.time - start < duration - 1e-9):
                if viewer and not viewer.is_running():
                    break
                tick = time.monotonic()
                # Match the accepted controller's forward pass before each command.
                mujoco.mj_forward(model, data)
                frame = frames[frame_index] if frame_index < len(frames) else None
                data.ctrl[:] = frame['ctrl'] if frame else model.key_ctrl[home]
                new_phase = frame['phase'] if frame else 'home hold'
                if new_phase != phase:
                    phase = new_phase
                    print(f'{data.time - start:.2f} s: {phase}', flush=True)
                    if viewer:
                        viewer.set_texts((None, None, 'ZP10D / 16 actuators',
                            f'{args.replay} command replay: {phase}' if frames else 'Live home hold'))
                steps = round(.02 / model.opt.timestep)
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                tilt = math.degrees(math.acos(float(np.clip(data.xmat[base, 8], -1, 1))))
                max_tilt = max(max_tilt, tilt)
                if not np.isfinite(data.qpos).all() or any(data.warning.number) or tilt > 45:
                    raise RuntimeError('Stopped: nonfinite state, numerical warning, or tilt above 45 degrees')
                if frame:
                    max_error = max(max_error, float(np.max(np.abs(data.qpos - frame['qpos']))))
                    frame_index += 1
                if viewer:
                    viewer.sync()
                    time.sleep(max(0., .02 - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        # MuJoCo closes asynchronously; finish its GUI threads before GLFW's
        # interpreter-exit cleanup tears down the X11 display underneath them.
        for thread in viewer_threads:
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError('MuJoCo viewer did not finish closing')
    print(json.dumps({'simulated_seconds': data.time - start, 'max_tilt_deg': max_tilt,
                      'replayed_command_frames': frame_index, 'max_saved_qpos_error': max_error,
                      'warnings': data.warning.number.tolist()}), flush=True)
    if args.headless and frames and max_error > 1e-6:
        raise RuntimeError('Command replay diverged from the saved trajectory')


if __name__ == '__main__':
    main()
