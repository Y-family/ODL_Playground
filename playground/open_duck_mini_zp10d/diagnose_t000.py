"""Full native paired tests of immediate position targets and lift torque onset.

T0000 is a position-command duration, not a torque command. The maximal-effort
case requests position errors large enough to reach the existing motor force
cap for the first 100 ms of lift. It never changes model files or force limits.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from playground.open_duck_mini_zp10d.actuators import configure, signature
from playground.open_duck_mini_zp10d.reference import ROOT, SCENE, source_hashes


def blend(time, start, duration):
    fraction = np.clip((time - start) / duration, 0., 1.)
    return fraction**2 * (3. - 2. * fraction)


def summarize(out):
    report = json.loads((out / 'results.json').read_text())
    trajectories = np.load(out / 'trajectories.npz')
    for label, result in report['cases'].items():
        time = trajectories[label + '_time']
        lift = (time >= 1.84) & (time < 2.6)
        hold = (time >= 2.02) & (time < 2.3)
        height, force = trajectories[label + '_foot_clearance'], trajectories[label + '_foot_force']
        result['lift_window_max_clearance_mm'] = (1000 * height[lift].max(0)).tolist()
        result['lift_hold_mean_load_n'] = force[hold].mean(0).tolist()
        swing = 0 if result['side'] == 'left' else 1
        peak_index = np.flatnonzero(lift)[np.argmax(height[lift, swing])]
        result['peak_clearance_time_s'] = float(time[peak_index])
        result['peak_clearance_foot_loads_n'] = force[peak_index].tolist()
        result['onset_100ms_near_cap_duration_s'] = {}
        torque = trajectories[label + '_torque']
        for i, (name, cap) in enumerate(report['unchanged_force_limits_nm'].items()):
            assert abs(torque[:, i]).max() <= cap + 1e-9
            window = (time >= 1.84) & (time < 1.94)
            result['onset_100ms_near_cap_duration_s'][name] = float(
                np.sum(abs(torque[window, i]) >= .99 * cap) * .002)
        if label in ('left_t000', 'right_t000'):
            knee = result['side'] + '_knee'
            assert result['onset_100ms_peak_torque_nm'][knee] >= .99 * report['unchanged_force_limits_nm'][knee]
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# T0000 整机抬脚对照', '',
        '保持相同7.8 V辨识响应、40 ms延迟、力矩上限、关节/位置限位及原安全转重/恢复轨迹。',
        'T0000只取消额外目标角度速度斜坡；max_effort另外将抬脚目标改为阶跃，并在起始100 ms请求可达的最大位置误差，不是固件力矩指令。', '',
        '| 抬脚侧 / 模式 | 起始100 ms膝峰值 N·m | 抬脚窗口整脚最高净空 mm | 保持段该脚平均承重 N | 最长2 mm卸载离地 s |',
        '|---|---:|---:|---:|---:|']
    for label, result in report['cases'].items():
        side = 0 if result['side'] == 'left' else 1
        lines.append(f"| {label} | {result['onset_100ms_peak_torque_nm'][result['side'] + '_knee']:.6f} | "
            f"{result['lift_window_max_clearance_mm'][side]:.4f} | {result['lift_hold_mean_load_n'][side]:.3f} | "
            f"{result['longest_2mm_supported_sample_run_s'][side]:.3f} |")
    lines += ['', f"{len(report['cases'])}例中通过抬脚/落脚恢复条件的有{sum(r['liftoff_and_recovery_passed'] for r in report['cases'].values())}例。结果只用于诊断，未作为正式控制采用。",
        '全部为仿真，未控制实物，未改正式CAD或控制配置。',
        '负净空表示接触求解器的轻微压入；统计排除起始0.2 mm空隙，使用1.84–2.60 s抬脚窗口。',
        '0.980665 N·m仍是当前模型未实测的力矩上限先验。此结果只检验当前转重/抬脚轨迹，不能证明所有步态都不可行。', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))


def main(support_only=False):
    out = ROOT / 'training_gpu/t000_liftoff'
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    parameters = configure(model)
    names = [model.actuator(i).name for i in range(model.nu)]
    joints = model.actuator_trnid[:, 0]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    bounds, caps = model.jnt_range[joints], np.max(abs(model.actuator_forcerange), axis=1)
    legs = np.array([any(s in name for s in ('hip', 'knee', 'ankle')) for name in names])
    qhome, uhome = model.key('home').qpos[qa].copy(), model.key('home').ctrl.copy()
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    feet = model.site_bodyid[sites]
    reference = json.loads((ROOT / 'data/reference.json').read_text())
    assert reference['source_hashes'] == source_hashes()
    vertices = [np.asarray(v) for v in reference['reward_reference']['foot_vertices_in_site']]
    weight = float(model.body_subtreemass[1] * abs(model.opt.gravity[2]))
    dt = float(model.opt.timestep)
    floor = model.geom('floor').id
    cases, arrays = {}, {}
    if support_only:
        previous = json.loads((out / 'results.json').read_text())
        assert previous['source_hashes'] == source_hashes() and previous['actuator_signature'] == signature()
        cases = previous['cases']
        with np.load(out / 'trajectories.npz') as saved:
            arrays = {key: saved[key] for key in saved.files}
    for side, roll in [('left', 20.), ('right', -20.)]:
        swing = 0 if side == 'left' else 1
        swing_ids = [names.index(side + '_' + joint) for joint in ('hip_pitch', 'knee', 'ankle')]
        for mode in ('bounded', 't000', 't000_max_effort'):
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
            mujoco.mj_forward(model, data)
            target, queue = uhome.copy(), np.tile(uhome, (3, 1))
            peak, onset_peak, cap_duration = np.zeros(model.nu), np.zeros(model.nu), np.zeros(model.nu)
            min_margin, max_target_speed, stop = np.inf, 0., 'duration'
            self_contacts, nonfoot = {}, {}
            rows, force_rows, min_heights, coms, poses, commands, torque_rows = [], [], [], [], [], [], []
            for tick in range(400):
                time = tick * .02
                prepare = blend(time, .5, .8) if time < 5. else 1. - blend(time, 5., .8)
                shift = blend(time, 1.5, .6) if time < 3. else 1. - blend(time, 3., .6)
                lift = blend(time, 1.8, .2) if time < 2.3 else 1. - blend(time, 2.3, .3)
                if mode == 't000_max_effort' and 1.8 <= time < 2.3:
                    lift = 1.
                goal = qhome.copy()
                roll_sides = (('right',) if side == 'left' else ('left',)) if support_only else ('left', 'right')
                goal[[names.index(s + '_hip_roll') for s in roll_sides]] += np.deg2rad(roll) * shift
                for leg in ('left', 'right'):
                    change = -30. * prepare + (40. * lift if leg == side else 0.)
                    goal[names.index(leg + '_knee')] += np.deg2rad(change)
                    goal[names.index(leg + '_hip_pitch')] += np.deg2rad(change / 2) * (-1 if leg == 'left' else 1)
                    goal[names.index(leg + '_ankle')] -= np.deg2rad(change / 2)
                goal = np.clip(goal, bounds[:, 0] + .002, bounds[:, 1] - .002)
                request = uhome + goal - qhome + 3. * legs * (goal - data.qpos[qa]) - .03 * legs * data.qvel[va]
                if mode == 't000_max_effort' and 1.8 <= time < 1.9:
                    direction = np.sign(goal[swing_ids] - data.qpos[qa[swing_ids]])
                    request[swing_ids] = (data.qpos[qa[swing_ids]] + direction * 1.25 * caps[swing_ids]
                                          / model.actuator_gainprm[swing_ids, 0])
                queue[1:] = queue[:-1].copy()
                queue[0] = request
                requested = np.clip(queue[2], *bounds.T)
                updated = np.clip(requested, target - 5.24 * .02, target + 5.24 * .02) if mode == 'bounded' else requested
                max_target_speed = max(max_target_speed, float(abs(updated - target).max() / .02))
                target = updated
                data.ctrl[:] = target
                for substep in range(round(.02 / dt)):
                    sample_time = time + substep * dt
                    mujoco.mj_step(model, data)
                    peak = np.maximum(peak, abs(data.actuator_force))
                    cap_duration += (abs(data.actuator_force) >= .99 * caps) * dt
                    if 1.84 <= sample_time < 1.94:
                        onset_peak = np.maximum(onset_peak, abs(data.actuator_force))
                    min_margin = min(min_margin, float(np.minimum(data.qpos[qa] - bounds[:, 0],
                                                                  bounds[:, 1] - data.qpos[qa]).min()))
                    force = np.zeros(2)
                    for ci in range(data.ncon):
                        contact = data.contact[ci]
                        pair, wrench = contact.geom, np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, wrench)
                        magnitude = float(np.linalg.norm(wrench[:3]))
                        if floor in pair:
                            other = pair[0] if pair[1] == floor else pair[1]
                            body = model.geom_bodyid[other]
                            if body in feet and contact.dist <= 0:
                                world = contact.frame.reshape(3, 3).T @ wrench[:3]
                                force[list(feet).index(body)] += max(0., world[2] * (1 if pair[0] == floor else -1))
                            elif magnitude > .05:
                                name = model.body(int(body)).name
                                nonfoot[name] = max(nonfoot.get(name, 0.), magnitude)
                        elif magnitude > .05:
                            name = ' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in pair))
                            self_contacts[name] = max(self_contacts.get(name, 0.), magnitude)
                    # mj_step's contact/site fields describe this substep's
                    # pre-integration pose: height and force share that instant.
                    height = np.array([min((v @ data.site_xmat[s].reshape(3, 3).T + data.site_xpos[s])[:, 2])
                                       for v, s in zip(vertices, sites)])
                    rows.append(sample_time)
                    force_rows.append(force)
                    min_heights.append(height)
                    torque_rows.append(data.actuator_force.copy())
                mujoco.mj_forward(model, data)
                poses.append(data.qpos.copy())
                commands.append(target.copy())
                coms.append(data.subtree_com[1].copy())
                tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1., 1.))))
                if not np.isfinite(data.qpos).all() or tilt > 45 or data.qpos[2] < .10:
                    stop = 'fall_or_nonfinite'
                    break
                if nonfoot or any(data.warning.number):
                    stop = 'nonfoot_or_warning'
                    break
            heights, forces = np.array(min_heights), np.array(force_rows)
            valid = (heights >= .002) & (forces < .5) & (forces[:, ::-1] > .5 * weight)
            longest, run = np.zeros(2), np.zeros(2)
            for row in valid:
                run = np.where(row, run + dt, 0.)
                longest = np.maximum(longest, run)
            label = ('support_only_' if support_only else '') + side + '_' + mode
            result = dict(side=side, mode=mode, seconds=len(rows) * dt, stop=stop,
                longest_2mm_supported_sample_run_s=longest.tolist(), max_whole_foot_clearance_mm=(1000 * heights.max(0)).tolist(),
                min_foot_load_n=forces.min(0).tolist(), final_tilt_deg=tilt, min_joint_margin_rad=min_margin,
                peak_torque_nm=dict(zip(names, peak.tolist())), onset_100ms_peak_torque_nm=dict(zip(names, onset_peak.tolist())),
                near_cap_duration_s=dict(zip(names, cap_duration.tolist())),
                maximum_motor_target_rate_rad_s=max_target_speed, self_contact_peak_force_n=self_contacts,
                nonfoot_contact_peak_force_n=nonfoot, warnings=data.warning.number.tolist(),
                final_home_error_rad=float(abs(data.qpos[qa] - qhome).max()),
                transfer_mode='support_hip_only' if support_only else 'coordinated_hips',
                scope='simulation only; 7.8 V T0000 identified response, 40 ms queue, original force/joint/position limits; same transfer within each mode comparison')
            result['liftoff_and_recovery_passed'] = bool(stop == 'duration' and longest[swing] >= .06
                and min_margin >= -1e-6 and not self_contacts and not nonfoot
                and not any(data.warning.number) and tilt < 5 and result['final_home_error_rad'] < .12)
            cases[label] = result
            for key, value in dict(time=rows, foot_force=forces, foot_clearance=heights,
                    torque=torque_rows, qpos=poses, motor_target=commands, com=coms).items():
                arrays[label + '_' + key] = value
            (out / 'results.json').write_text(json.dumps(dict(cases=cases, source_hashes=source_hashes(),
                actuator_signature=signature(), parameters=parameters,
                unchanged_force_limits_nm=dict(zip(names, caps.tolist())), walking_accepted=False), indent=2) + '\n')
            np.savez_compressed(out / 'trajectories.npz', **arrays)
            print(json.dumps(result), flush=True)
    summarize(out)


if __name__ == '__main__':
    if sys.argv[1:] == ['--summarize-only']:
        summarize(ROOT / 'training_gpu/t000_liftoff')
    elif sys.argv[1:] == ['--support-only']:
        main(support_only=True)
    elif sys.argv[1:]:
        raise SystemExit('Usage: diagnose_t000.py [--summarize-only | --support-only]')
    else:
        main()
