"""Follow saved native PPO evaluations in one local window; no physics/JAX worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import xml.etree.ElementTree as ET

# Playback uses only forward kinematics, never the CAD/GPU computation lock.
# Bound memory before importing any native library; trainer keeps its own cap.
resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MUJOCO_GL', 'glfw')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '0')
available_cpus = os.sched_getaffinity(0)
if len(available_cpus) > 1:
    os.sched_setaffinity(0, available_cpus - {min(available_cpus)})

import mujoco
import numpy as np


def latest(run, profile):
    metadata = json.loads((run / 'metadata.json').read_text())
    selected = metadata.get('selected_checkpoint')
    if selected:
        selected = Path(selected)
        selected = selected if selected.is_absolute() else run / selected
        run = selected.parent.parent
        metadata = json.loads((run / 'metadata.json').read_text())
        path = run / 'native' / selected.name / profile / 'evaluation.npz'
    elif metadata.get('continuation_run'):
        return latest(Path(metadata['continuation_run']), profile)
    else:
        candidates = sorted((run / 'native').glob('*/' + profile + '/evaluation.npz'))
        if not candidates:
            raise FileNotFoundError('No completed native evaluation for ' + profile)
        path = candidates[-1]
    report = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as saved:
        frames = {k: saved[k].copy() for k in saved.files}
    if len(frames['qpos']) == 0 or not np.isfinite(frames['qpos']).all():
        raise ValueError('Empty or nonfinite saved trajectory')
    steps = metadata.get('prior_steps', 0) + int(path.parent.parent.name)
    return path, report, frames, steps


def training_frame(run):
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata.get('continuation_run'):
        return training_frame(Path(metadata['continuation_run']))
    path = run / 'training_state.json'
    if path.exists():
        live = json.loads(path.read_text())
        qpos, sim_time, steps = live['qpos'], live['time'], live['steps']
    else:
        scene = Path(__file__).resolve().parent / 'xmls/scene_flat_terrain.xml'
        qpos = np.fromstring(ET.parse(scene).find('.//key[@name="home"]').get('qpos'), sep=' ')
        sim_time, steps = 0., metadata.get('prior_steps', 0)
        live = dict(published_at=None, episode=0, reset=False, environments=metadata.get('num_envs', 512))
    frames = dict(qpos=np.array([qpos]), time=np.array([sim_time]))
    report = dict(source_hashes=metadata['source_hashes'], stop='live training', live=live)
    return path, report, frames, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--profile', default='walk', choices=('walk', 'transition', 'stand', 'walk_stochastic'))
    parser.add_argument('--check', action='store_true', help='validate saved FK without opening a window')
    parser.add_argument('--live-training', action='store_true', help='actual PPO training environment; never loop a saved evaluation')
    args = parser.parse_args()
    read_frame = (lambda: training_frame(args.run)) if args.live_training else (lambda: latest(args.run, args.profile))
    path, report, frames, steps = read_frame()
    scene = Path(__file__).resolve().parent / 'xmls/scene_flat_terrain.xml'
    evidence = Path(__file__).resolve().parents[2] / 'cad/local_fix/open_duck_mini_zp10d_mass_properties.json'
    expected = json.loads(evidence.read_text())
    if hashlib.sha256(scene.read_bytes()).hexdigest() != expected['scene_sha256']:
        raise ValueError('Published scene checksum mismatch')
    if report['source_hashes']['scene_flat_terrain.xml'] != expected['scene_sha256']:
        raise ValueError('Evaluation belongs to another scene')
    # Render-only copy: identical bodies/joints/visuals; omit invisible collision meshes.
    xml = ET.parse(scene).getroot()
    include = xml.find('include')
    robot = ET.parse(scene.parent / include.get('file')).getroot()
    xml.remove(include)
    for child in robot:
        xml.append(child)
    xml.find('compiler').set('meshdir', str(scene.parent / 'assets'))
    xml.find('compiler').set('usethread', 'false')
    visual = xml.find('visual')
    ET.SubElement(visual, 'quality', shadowsize='512', offsamples='0')
    visual.find('global').set('offwidth', '320')
    visual.find('global').set('offheight', '240')
    for parent in xml.iter():
        for child in list(parent):
            if child.tag == 'geom' and child.get('class') == 'collision':
                parent.remove(child)
    used = {g.get('mesh') for g in xml.iter('geom')}
    for asset in xml.findall('asset'):
        for mesh in list(asset):
            if mesh.tag == 'mesh' and mesh.get('name') not in used:
                asset.remove(mesh)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    assert model.nq == frames['qpos'].shape[1]
    assert model.nmesh == 113, 'Expected all current CAD visual parts'

    print('DISPLAY_MODEL', model.ngeom, model.nmeshvert, model.nmeshface, 'shadow', model.vis.quality.shadowsize, 'offscreen', model.vis.global_.offwidth, model.vis.global_.offheight, 'RSS_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, flush=True)
    data = mujoco.MjData(model)
    base = model.body('base').id
    feet = [model.site(side + '_foot').id for side in ('left', 'right')]
    reference = json.loads((scene.parent.parent / 'data/reference.json').read_text())
    vertices = [np.array(v) for v in reference['reward_reference']['foot_vertices_in_site']]
    model.geom_rgba[:, 3] = 1.
    def pose(index):
        data.qpos[:] = frames['qpos'][index]
        data.time = frames['time'][index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return np.sum(data.xipos * model.body_mass[:, None], axis=0) / model.body_mass.sum()
    if args.check:
        for index in (0, len(frames['qpos']) - 1):
            com = pose(index)
            assert np.isfinite(data.xpos).all() and np.isfinite(com).all()
        print('PASS live training FK:' if args.live_training else 'PASS saved native FK:',
              steps, 'steps;', len(frames['qpos']), 'frames;', report['stop'], flush=True)
        return
    import ctypes
    import gc
    del xml, robot
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    from mujoco import viewer as native_viewer
    paused, speed, restart = False, .5, False
    def key(keycode):
        nonlocal paused, speed, restart
        if keycode == 32:
            paused = not paused
        elif keycode == 83:
            speed = {.25: .5, .5: 1., 1.: .25}[speed]
        elif keycode == 82:
            restart = True
    pose(0)
    with native_viewer.launch_passive(model, data, key_callback=key,
            show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 0
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .85, 135, -15
        index, progress, last_poll, live_speed = 0, 0., 0., None
        print('MUJOCO_LIVE_TRAINING_WINDOW_READY' if args.live_training else 'MUJOCO_TRAINING_WINDOW_READY',
              steps, path, flush=True)
        while viewer.is_running():
            start = time.monotonic()
            if start - last_poll > (.1 if args.live_training else 2.) and not (args.live_training and paused):
                try:
                    new = read_frame()
                    if new[0] != path or (args.live_training and new[3] != steps):
                        if (args.live_training and new[0] == path and new[3] > steps
                                and report['live']['published_at'] is not None):
                            elapsed = new[1]['live']['published_at'] - report['live']['published_at']
                            if elapsed > 0:
                                measured = .02*(new[3]-steps)/new[1]['live']['environments']/elapsed
                                live_speed = measured if live_speed is None else .2*measured+.8*live_speed
                        elif new[0] != path:
                            live_speed = None
                        path, report, frames, steps = new
                        index, progress = 0, 0.
                        if not args.live_training:
                            print('NEW_CHECKPOINT', steps, path, flush=True)
                except (OSError, ValueError, KeyError):
                    pass  # Writer may still be publishing its latest NPZ.
                last_poll = start
            if restart:
                index, progress, restart = 0, 0., False
            com = pose(index)
            if args.live_training:
                left, right = [np.min(v @ data.site_xmat[f].reshape(3,3).T + data.site_xpos[f], axis=0)[2] * 1000
                               for f, v in zip(feet, vertices)]
                tilt = np.rad2deg(np.arccos(np.clip(data.xmat[base].reshape(3,3)[2,2], -1., 1.)))
                rotation = data.xmat[base].reshape(3, 3)
                roll = np.rad2deg(np.arctan2(rotation[2, 1], rotation[2, 2]))
                live = report['live']
                age = None if live['published_at'] is None else max(0., time.time() - live['published_at'])
            else:
                left, right = frames['foot_clearance'][index] * 1000
                loads = frames['foot_force'][index]
                tilt = frames['tilt_deg'][index]
                age = max(0., time.time() - path.stat().st_mtime)
            viewer.cam.lookat[:] = data.xpos[base] + [0., 0., .02]
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                # Red COM projection, green line to body COM, blue foot centers.
                points = [(np.array([com[0], com[1], .004]), [1., .1, .1, 1.])]
                points += [(data.site_xpos[f] * [1, 1, 0] + [0, 0, .004], [.1, .5, 1., 1.]) for f in feet]
                for point, color in points:
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.004] * 3), point, np.eye(3).ravel(), np.array(color))
                    viewer.user_scn.ngeom += 1
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.1, 1., .2, 1.]))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0015,
                    np.array([com[0], com[1], .004]), com)
                viewer.user_scn.ngeom += 1
            if args.live_training:
                status = ('WAITING: no training sample yet' if age is None else
                          f'WAITING: last training sample {age:.1f}s ago' if age > 5. else
                          f'sample age {age:.1f}s')
                if live_speed is not None and age is not None and age <= 5.:
                    status += f' | simulation speed ~{live_speed:.3f}x real time'
                viewer.set_texts((None, None,
                    f'LIVE PPO TRAINING | env 0/{live["environments"]}\n'
                    f'{steps:,} total | {live.get("phase_steps", 0):,} new steps\n{status}',
                    f'simulation t={data.time:.2f}s | episode={live["episode"]} | '
                    f'roll={roll:.1f}deg\n'
                    f'whole-foot height L/R={left:.1f}/{right:.1f}mm | tilt={tilt:.1f}deg\n'
                    + ('Episode starts from supplied Placo state; initial clearance is not learned lift\n'
                     if live.get('reference_initialized') else 'Episode starts from standing\n') +
                    'Actual sampled MJX training state; resets start a new episode\n'
                    'Space: pause display | red=COM projection, blue=feet'))
            else:
                viewer.set_texts((None, None,
                    f'PPO native evaluation | {steps:,} steps | {args.profile} | age {age:.0f}s',
                    f't={data.time:.2f}s / {frames["time"][-1]:.2f}s  speed={speed}x  '
                    f'height L/R={left:.1f}/{right:.1f}mm  load={loads[0]:.1f}/{loads[1]:.1f}N\n'
                    f'tilt={tilt:.1f}deg | red=COM projection, blue=feet | {report["stop"]}\n'
                    'Space: pause | S: speed | R: restart | Auto-follow newest checkpoint; recorded physics'))
            viewer.sync()
            time.sleep(max(0., .02 - (time.monotonic() - start)))
            if not paused and not args.live_training:
                progress += (time.monotonic() - start) * speed / .02
                if progress >= len(frames['qpos']) + 35:
                    progress = 0.
                index = min(int(progress), len(frames['qpos']) - 1)


if __name__ == '__main__':
    main()
