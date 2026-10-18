"""Read saved lift trials and run bounded controls without changing formal files."""
import gc
from datetime import date
import json
from pathlib import Path
import resource
import sys

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Use scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import assess_zp10d_loads as loads

ROOT, OUT = loads.ROOT, loads.OUT
CACHE = Path('/tmp/open-duck-liftoff')
PATHS = [OUT / 'open_duck_mini_upper_refit.step',
         loads.MODEL_DIR / 'open_duck_mini_zp10d.xml',
         loads.MODEL_DIR / 'scene_flat_terrain.xml']


def geometry(model, data, vertices):
    base = model.body('base').id
    feet = [model.body('foot_assembly').id, model.body('foot_assembly_2').id]
    polygons, heights, margins = [], [], []
    for body, mesh in zip(feet, vertices):
        world = mesh @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
        points = world[:, :2]
        hull = ConvexHull(points)
        polygons.append(points[hull.vertices].tolist())
        heights.append(float(world[:, 2].min() * 1000))
        signed = hull.equations[:, :2] @ data.subtree_com[base, :2] + hull.equations[:, 2]
        margins.append(float(-signed.max() * 1000))
    return {'com_world_mm': (data.subtree_com[base] * 1000).tolist(),
            'base_world_mm': (data.qpos[:3] * 1000).tolist(),
            'base_euler_deg': Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_euler('xyz', degrees=True).tolist(),
            'foot_projected_polygons_world_m': polygons,
            'com_margin_to_each_projected_sole_mm': margins,
            'sole_min_height_mm': heights,
            'foot_body_positions_world_mm': (data.xpos[feet] * 1000).tolist()}


def inspect_saved():
    source = OUT / 'CURRENT_LOAD_ASSESSMENT.json'
    saved = json.loads(source.read_text())
    hashes = [saved[k] for k in ('source_step_sha256', 'mjcf_sha256', 'scene_sha256')]
    assert [loads.digest(p) for p in PATHS] == hashes
    model = mujoco.MjModel.from_xml_path(str(PATHS[2]))
    joints = model.actuator_trnid[:, 0]; adr = model.jnt_qposadr[joints]
    feet = [model.body('foot_assembly').id, model.body('foot_assembly_2').id]
    vertices = loads.sole_vertices(model, feet)
    data = mujoco.MjData(model)
    report = {'date': '2026-10-15', 'scope': 'Diagnosis only; formal stance and files unchanged',
              'formal_hashes': hashes, 'source_load_report_sha256': loads.digest(source),
              'diagnosis_script_sha256': loads.digest(Path(__file__)),
              'actuator_order': saved['actuator_order'], 'saved_trials': [], 'ideal_waypoints': []}
    for case in saved['scenarios']:
        if case['name'] not in ('left_lift_attempt', 'right_lift_attempt'):
            continue
        side = 0 if case['name'].startswith('left') else 1
        snapshots = []
        for t in (0., 2.48, 4.5, 6.):
            f = min(case['frames'], key=lambda f: abs(f['time_s'] - t))
            data.qpos[:] = f['qpos']; data.qvel[:] = 0; data.ctrl[:] = f['ctrl']
            mujoco.mj_forward(model, data)
            row = geometry(model, data, vertices)
            row.update({'time_s': f['time_s'], 'foot_forces_world_n': f['foot_forces_world_n'],
                        'ctrl_minus_actual_deg': np.degrees(np.array(f['ctrl']) - data.qpos[adr]).tolist(),
                        'torque_nm': f['torque_nm']})
            snapshots.append(row)
        row = {'name': case['name'], 'swing_side_index': side, 'snapshots': snapshots,
               'motion_outcome': case['motion_outcome'],
               'peak_motor_torque_nm': case['peak_motor_torque_nm'],
               'max_tracking_error_deg': case['max_tracking_error_deg'],
               'warnings': case['warnings'], 'saturation_fraction': case['saturation_fraction']}
        report['saved_trials'].append(row)
        print(case['name'], 'outcome', row['motion_outcome'], flush=True)
        for f in snapshots:
            print(' t', round(f['time_s'], 2), 'COM', np.round(f['com_world_mm'], 2).tolist(),
                  'roll', round(f['base_euler_deg'][0], 2),
                  'Fz', np.round(np.array(f['foot_forces_world_n'])[:, 2], 2).tolist(),
                  'soleZ', np.round(f['sole_min_height_mm'], 2).tolist(),
                  'soleCOM margins', np.round(f['com_margin_to_each_projected_sole_mm'], 2).tolist(), flush=True)
    # Reconstruct the original 3 s initial state and geometric IK waypoints.
    settled = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, settled, 0)
    settled.ctrl[:] = model.key_ctrl[0]
    for _ in range(round(3 / model.opt.timestep)):
        mujoco.mj_step(model, settled)
    mujoco.mj_forward(model, settled)
    cases, checks = loads.trajectories(model, settled, model.key_ctrl[0].copy())
    report['ik_checks'] = checks
    for side, sign in [('left', 1), ('right', -1)]:
        for lift in (0., .008):
            request = {'roll': sign * np.deg2rad(16), side + '_lift': lift}
            ctrl, check = loads.leg_targets(model, settled, model.key_ctrl[0], **request)
            data.qpos[:] = settled.qpos
            rotation = Rotation.from_rotvec([request['roll'], 0, 0]) * Rotation.from_quat(settled.qpos[[4, 5, 6, 3]])
            data.qpos[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
            # Undo the existing gravity-equilibrium offset, recovering the angles
            # whose FK residual was actually checked by leg_targets.
            data.qpos[adr] = settled.qpos[adr] + ctrl - model.key_ctrl[0]
            data.qvel[:] = 0; mujoco.mj_forward(model, data)
            row = geometry(model, data, vertices)
            row.update({'swing_side': side, 'requested_lift_mm': lift * 1000, 'ik_checks': check,
                        'ctrl_rad': ctrl.tolist(), 'ideal_qpos': data.qpos.tolist()})
            report['ideal_waypoints'].append(row)
            print('ideal', side, lift * 1000, 'COM', np.round(row['com_world_mm'], 2).tolist(),
                  'soleCOM margins', np.round(row['com_margin_to_each_projected_sole_mm'], 2).tolist(),
                  'soleZ', np.round(row['sole_min_height_mm'], 2).tolist(), flush=True)
    del saved; gc.collect()
    assert [loads.digest(p) for p in PATHS] == hashes
    CACHE.mkdir(exist_ok=True)
    (CACHE / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def experiments():
    report = json.loads((CACHE / 'diagnosis.json').read_text())
    assert [loads.digest(p) for p in PATHS] == report['formal_hashes']
    model = mujoco.MjModel.from_xml_path(str(PATHS[2]))
    names = [model.actuator(i).name for i in range(model.nu)]
    joints = model.actuator_trnid[:, 0]; adr = model.jnt_qposadr[joints]
    base = model.body('base').id; floor = model.geom('floor').id
    feet = [model.body('foot_assembly').id, model.body('foot_assembly_2').id]
    vertices = loads.sole_vertices(model, feet)
    settled = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, settled, 0)
    settled.ctrl[:] = model.key_ctrl[0]
    for _ in range(round(3 / model.opt.timestep)):
        mujoco.mj_step(model, settled)
    mujoco.mj_forward(model, settled)
    cases, _ = loads.trajectories(model, settled, model.key_ctrl[0].copy())
    waypoints = next(c[2] for c in cases if c[0] == 'left_lift_attempt')
    report['joint_axes_initial_world'] = {n: settled.xaxis[j].tolist() for n, j in zip(names, joints)}
    # Attribute the height error using the measured saved pose, not a new trial.
    source = json.loads((OUT / 'CURRENT_LOAD_ASSESSMENT.json').read_text())
    saved_case = next(c for c in source['scenarios'] if c['name'] == 'left_lift_attempt')
    frame = min(saved_case['frames'], key=lambda f: abs(f['time_s'] - 6.))
    ideal = next(w for w in report['ideal_waypoints'] if w['swing_side'] == 'left' and w['requested_lift_mm'] == 8)
    combinations = []
    for base_pose in ('ideal', 'actual'):
        for angles in ('ideal', 'actual'):
            data = mujoco.MjData(model)
            data.qpos[:] = ideal['ideal_qpos'] if base_pose == 'ideal' else frame['qpos']
            data.qpos[adr] = np.array(ideal['ideal_qpos'] if angles == 'ideal' else frame['qpos'])[adr]
            mujoco.mj_forward(model, data)
            row = geometry(model, data, vertices)
            row.update({'base_pose': base_pose, 'joint_angles': angles})
            combinations.append(row)
    report['saved_left_lift_6s_kinematic_decomposition'] = combinations
    report['left_6s_joint_tracking_deg'] = dict(zip(names, np.degrees(np.array(frame['ctrl']) - np.array(frame['qpos'])[adr]).tolist()))
    del source; gc.collect()
    report['experiments'] = []
    original_gain = model.actuator_gainprm.copy(); original_bias = model.actuator_biasprm.copy()
    original_ranges = model.actuator_forcerange.copy()
    leg_ids = [i for i, name in enumerate(names) if name.startswith(('left_', 'right_')) and 'antenna' not in name]
    for multiplier in (1., 4.):
        # In-memory gain comparison only. Do not alter force limits or files.
        model.actuator_gainprm[:] = original_gain; model.actuator_biasprm[:] = original_bias
        model.actuator_gainprm[leg_ids, 0] *= multiplier
        model.actuator_biasprm[leg_ids, 1] *= multiplier
        data = mujoco.MjData(model); data.qpos[:] = settled.qpos; data.qvel[:] = settled.qvel
        data.ctrl[:] = waypoints[0][1]; mujoco.mj_forward(model, data)
        frames = []; peak = np.zeros(model.nu); saturated = np.zeros(model.nu)
        contact_peaks = {}; max_tilt = 0.; stop = None
        for step in range(round(8 / model.opt.timestep)):
            t = step * model.opt.timestep
            data.ctrl[:] = loads.interpolate(t, waypoints)
            mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
            tilt = np.degrees(np.arccos(np.clip(data.xmat[base].reshape(3, 3)[2, 2], -1, 1)))
            max_tilt = max(max_tilt, float(tilt))
            peak = np.maximum(peak, np.abs(data.actuator_force))
            saturated += np.abs(data.actuator_force) >= .99 * model.actuator_forcerange[:, 1]
            for ci, contact in enumerate(data.contact):
                if floor in contact.geom:
                    continue
                wrench = np.zeros(6); mujoco.mj_contactForce(model, data, ci, wrench)
                force = float(np.linalg.norm(wrench[:3]))
                if force > .05:
                    pair = ' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in contact.geom))
                    contact_peaks[pair] = max(contact_peaks.get(pair, 0.), force)
            if step % 10 == 0:
                forces, _, nonfoot = loads.ground_forces(model, data, floor, feet)
                row = geometry(model, data, vertices)
                row.update({'time_s': float(t + model.opt.timestep),
                            'qpos': data.qpos.tolist(), 'ctrl': data.ctrl.tolist(),
                            'torque_nm': data.actuator_force.tolist(),
                            'foot_forces_world_n': forces.tolist(),
                            'base_tilt_deg': float(tilt),
                            'tracking_deg': np.degrees(data.ctrl - data.qpos[adr]).tolist()})
                frames.append(row)
                if nonfoot:
                    stop = 'nonfoot_ground_contact'
            if tilt > 35 or data.xpos[base, 2] < .085:
                stop = 'excessive_tilt' if tilt > 35 else 'base_too_low'
            if stop:
                break
        fz = np.array([f['foot_forces_world_n'] for f in frames])[:, :, 2]
        clearance = np.array([f['sole_min_height_mm'] for f in frames])
        airborne = (fz[:, 0] < .5) & (clearance[:, 0] > 2.)
        row = {'name': f'left_lift_kp_x{multiplier:g}', 'kp_multiplier': multiplier,
               'simulated_seconds': float((step + 1) * model.opt.timestep), 'stop_reason': stop,
               'warnings': data.warning.number.tolist(), 'max_tilt_deg': max_tilt,
               'minimum_foot_vertical_force_n': fz.min(axis=0).tolist(),
               'max_sole_min_clearance_mm': clearance.max(axis=0).tolist(),
               'left_airborne_seconds': loads.consecutive_seconds(airborne),
               'peak_motor_torque_nm': peak.tolist(),
               'saturation_fraction': (saturated / (step + 1)).tolist(),
               'nonfloor_contact_peak_force_n': contact_peaks, 'frames': frames}
        assert not any(row['warnings'])
        report['experiments'].append(row)
        print(row['name'], 'Fz min', np.round(fz.min(axis=0), 3).tolist(),
              'soleZ max', np.round(clearance.max(axis=0), 3).tolist(), 'airborne', row['left_airborne_seconds'],
              'tilt', round(max_tilt, 2), 'peak', round(float(peak.max()), 4),
              'nonfloor contact', contact_peaks, 'stop', stop, flush=True)
        assert np.array_equal(model.actuator_forcerange, original_ranges)
        (CACHE / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    baseline = report['experiments'][0]
    for old in saved_case['frames']:
        if old['time_s'] > 8.:
            break
        new = min(baseline['frames'], key=lambda f: abs(f['time_s'] - old['time_s']))
        assert np.max(np.abs(np.array(new['qpos']) - old['qpos'])) < 1e-10
        assert np.max(np.abs(np.array(new['foot_forces_world_n']) - old['foot_forces_world_n'])) < 1e-7
    report['baseline_replay_matches_saved'] = True
    report['experiment_script_sha256'] = loads.digest(Path(__file__))
    report['resources'] = {'address_space_limit_bytes': resource.getrlimit(resource.RLIMIT_AS)[0],
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'observed_threads': len(list(Path('/proc/self/task').iterdir())), 'serial': True}
    assert [loads.digest(p) for p in PATHS] == report['formal_hashes']
    (CACHE / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def publish():
    report = json.loads((CACHE / 'diagnosis.json').read_text())
    assert [loads.digest(p) for p in PATHS] == report['formal_hashes']
    assert loads.digest(OUT / 'CURRENT_LOAD_ASSESSMENT.json') == report['source_load_report_sha256']
    assert report['baseline_replay_matches_saved']
    baseline, stiff = report['experiments']
    assert baseline['kp_multiplier'] == 1 and stiff['kp_multiplier'] == 4
    assert all(not t['nonfloor_contact_peak_force_n'] and not any(t['warnings'])
               and t['stop_reason'] is None and t['left_airborne_seconds'] == 0
               for t in report['experiments'])
    ideal = next(w for w in report['ideal_waypoints'] if w['swing_side'] == 'left' and w['requested_lift_mm'] == 8)
    assert all(c['pose_tolerance_passed'] and c['within_limits'] for c in ideal['ik_checks'])
    actual = report['saved_trials'][0]['snapshots'][-1]
    decomposition = report['saved_left_lift_6s_kinematic_decomposition']
    ideal_actual = next(c for c in decomposition if c['base_pose'] == 'ideal' and c['joint_angles'] == 'actual')
    measured = next(c for c in decomposition if c['base_pose'] == 'actual' and c['joint_angles'] == 'actual')
    loss = ideal_actual['sole_min_height_mm'][0] - measured['sole_min_height_mm'][0]
    assert ideal['com_margin_to_each_projected_sole_mm'][1] < -40
    assert actual['com_margin_to_each_projected_sole_mm'][1] < -50
    assert loss > 7 and baseline['minimum_foot_vertical_force_n'][0] > 7
    # Scan recorded joint angles against the actual compiled ranges; this checks
    # the sampled trajectory, not unsaved inter-sample joint-limit forces.
    model = mujoco.MjModel.from_xml_path(str(PATHS[2]))
    joints = model.actuator_trnid[:, 0]; adr = model.jnt_qposadr[joints]
    for case in report['experiments']:
        angles = np.array([f['qpos'] for f in case['frames']])[:, adr]
        margin = np.minimum(angles - model.jnt_range[joints, 0], model.jnt_range[joints, 1] - angles)
        case['sampled_min_joint_limit_margin_deg'] = dict(zip(report['actuator_order'], np.degrees(margin.min(axis=0)).tolist()))
        knee_actor = model.actuator('right_knee').id
        knee_angles = np.degrees(angles[:, knee_actor])
        upper = float(np.degrees(model.jnt_range[joints[knee_actor], 1]))
        exceeded = np.flatnonzero(knee_angles > upper)
        at6 = min(range(len(case['frames'])), key=lambda i: abs(case['frames'][i]['time_s'] - 6.))
        case['support_knee_limit_summary'] = {'joint': 'right_knee', 'upper_limit_deg': upper,
            'max_sampled_angle_deg': float(knee_angles.max()), 'angle_at_6s_deg': float(knee_angles[at6]),
            'first_sample_above_limit_s': case['frames'][int(exceeded[0])]['time_s'] if len(exceeded) else None}
    knee = baseline['support_knee_limit_summary']
    assert baseline['sampled_min_joint_limit_margin_deg']['right_knee'] < 0
    assert stiff['sampled_min_joint_limit_margin_deg']['right_knee'] > 1
    report['diagnosis'] = {
        'primary': 'Roll-only IK motor targets assume a base pose, but neither plan COM into the support foot nor control the floating base to that pose.',
        'sequencing': 'Fixed-time transitions lift the foot without an unloading/support-state gate.',
        'kinematic_height_loss_from_measured_base_pose_mm': loss,
        'ideal_support_margin_mm': ideal['com_margin_to_each_projected_sole_mm'][1],
        'measured_support_margin_at_6s_mm': actual['com_margin_to_each_projected_sole_mm'][1],
        'gain_only_change_did_not_achieve_liftoff': True,
        'baseline_support_knee_reaches_soft_limit': True,
        'higher_gain_trial_avoids_limit_but_still_fails_liftoff': True,
        'nonfloor_contact_reporting_threshold_n': .05,
        'formal_stance_changed': False,
        'limitations': ['Estimated dynamics/contact/actuator parameters; no hardware test.',
            'Sole projection is the convex envelope of the tread mesh, an optimistic support bound; actual loaded contact support can be smaller.',
            'Base/joint decomposition is a kinematic counterfactual, not an externally supported dynamic trial.',
            'Fourfold leg kp is an in-memory diagnostic with the same initial state and targets, not a tuning recommendation.',
            'No new gait controller or successful single-foot balance is claimed.']}
    report['report_renderer_sha256'] = loads.digest(Path(__file__))
    lines = ['# 抬脚失败定位：重心转移与自由机身控制未闭合', '',
        '**定位结论：当前诊断动作只规划了“身体侧倾16°+脚上抬8 mm”的关节目标，没有把重心转入支撑脚，也没有保证实际机身达到IK假设的位姿；动作按定时推进，在待抬脚尚未卸载时就开始抬脚。** 正式站姿保持用户已确认的当前版本；本轮只定位，不改变CAD、home、运行配置或硬件。', '',
        '## 证据一：理想IK姿态本身没有完成重心转移', '',
        f"重算原诊断目标并恢复IK实际验收的关节角。左脚抬高8 mm的理想姿态中，整张左脚底最低点达到 {ideal['sole_min_height_mm'][0]:.3f} mm，目标在0.5 mm/1°容差与关节目标限位内有解。但整机重心投影仍在右支撑脚投影边界外 {-ideal['com_margin_to_each_projected_sole_mm'][1]:.2f} mm；右脚抬起的对称理想目标同样在左支撑脚边界外约43.65 mm。对于本次慢速保持动作，这不足以建立静态单脚支撑。", '',
        '支撑边界取真实脚底网格XY投影的凸包，是比实际接触范围更宽松的上界。即使使用该宽松范围，重心仍在外侧；不能靠IK位置误差小或身体已侧倾来认定重心转移成功。动态步态可短时重心在支撑域外，但本诊断没有相应动量/落脚恢复控制。', '',
        '## 证据二：8 mm抬脚被实际机身运动抵消', '',
        '| 左脚抬起尝试，约6秒 | 规划/理想 | 实际 |', '| --- | ---: | ---: |',
        f"| 基座Z mm | {ideal['base_world_mm'][2]:.3f} | {measured['base_world_mm'][2]:.3f} |",
        f"| 基座侧倾 ° | {ideal['base_euler_deg'][0]:.3f} | {measured['base_euler_deg'][0]:.3f} |",
        f"| 左脚底最低点Z mm | {ideal['sole_min_height_mm'][0]:.3f} | {measured['sole_min_height_mm'][0]:.3f} |",
        f"| 重心到右脚投影边界 mm（负值表示在外） | {ideal['com_margin_to_each_projected_sole_mm'][1]:.3f} | {actual['com_margin_to_each_projected_sole_mm'][1]:.3f} |", '',
        f"将实际关节角保持不变，仅在运动学重算中恢复规划机身位姿，左脚底可到 {ideal_actual['sole_min_height_mm'][0]:.3f} mm；换回实际机身位姿后仅 {measured['sole_min_height_mm'][0]:.3f} mm。机身位姿差异在这组明确的分解次序下抵消约 {loss:.3f} mm 的抬高。这是FK对照，没有在仿真中固定基座或施加支撑外力。", '',
        '原控制只向关节位置执行器发送目标，没有直接规定自由基座的平移/转动。机身会在重力、双脚接触和关节扭矩作用下重新找到双脚承重的状态，因此“给脚一个世界坐标抬高目标”不等于真实脚底必然抬高。', '',
        '## 证据三：同初态、同轨迹的控制对照', '',
        '从原3秒自由站立初态出发，重跑原左脚抬起轨迹的前8秒；对照仅在内存中将10个腿关节kp改成4倍，头部和天线增益、扭矩上限、摩擦、几何和目标均保持。4倍增益在起始时刻切换，属于敏感性试验。基准轨迹逐个20 ms保存点与原轨迹qpos误差<1e-10、脚底反力误差<1e-7 N。', '',
        '| 指标 | 原增益 | 腿部kp×4 |', '| --- | ---: | ---: |',
        f"| 左脚最小竖直承重 N | {baseline['minimum_foot_vertical_force_n'][0]:.3f} | {stiff['minimum_foot_vertical_force_n'][0]:.3f} |",
        f"| 左脚底最高的最低点 mm | {baseline['max_sole_min_clearance_mm'][0]:.3f} | {stiff['max_sole_min_clearance_mm'][0]:.3f} |",
        f"| 满足卸载与整脚离地的最长时长 s | {baseline['left_airborne_seconds']:.2f} | {stiff['left_airborne_seconds']:.2f} |",
        f"| 电机峰值 N·m（这8秒） | {max(baseline['peak_motor_torque_nm']):.3f} | {max(stiff['peak_motor_torque_nm']):.3f} |",
        f"| 最大单电机饱和时间占比 | {max(baseline['saturation_fraction'])*100:.1f}% | {max(stiff['saturation_fraction'])*100:.1f}% |", '',
        '原增益约6秒时右膝位置误差2.479°，kp×4后降到约0.687°，但仍无抬脚；增益提高还引入了短时饱和。两组都没有出现>0.05 N的非地面自接触力，未见自碰撞把该动作卡住；两组均无数值警告或停止。这里不替代CAD实体碰撞检查，也不证明所有动作均无阻碍。不能把本次失败直接归因于堵转扭矩不足，或只靠提高增益解决。', '',
        f"另外发现次要限制：原增益下支撑侧右膝在保存的20 ms采样中达到 {knee['max_sampled_angle_deg']:.4f}°，触及90°软限位（采样超出约 {knee['max_sampled_angle_deg']-knee['upper_limit_deg']:.4f}°）。kp×4对照的该关节全段采样最小限位余量仍为 {stiff['sampled_min_joint_limit_margin_deg']['right_knee']:.3f}°，却同样不能抬脚。因此限位余量应在后续动作规划中保留，但不能将它当作本次失败的唯一原因；本轮未扩大关节限位。", '',
        '## 具体代码位置与后续修复边界', '',
        '- `assess_zp10d_loads.py::leg_targets`：先假定基座平移/转动，解出腿关节角；实际只返回电机目标，不约束自由基座，也不检查重心是否进入支撑脚。',
        '- `assess_zp10d_loads.py::trajectories`：用固定16°侧倾替代了原不可行的独立侧移，但未验收单脚支撑重心；2.5秒后按时间开始抬脚。',
        '- `assess_zp10d_loads.py::simulate`：执行定时关节插值，没有根据支撑脚范围、机身姿态与双脚载荷闭环修正，也没有在卸载失败时阻止进入抬脚阶段。', '',
        '下一步应保持当前home，单独修正动作规划与平衡控制：按现有每腿5轴的能力联立求解机身位姿和脚底姿态，不能重新硬塞独立侧移+双脚完全固定姿态；以重心/接触支撑、IMU机身姿态和载荷反馈完成转重，再由卸载判据允许抬脚。需要复核扭矩、限位、实际离地、单脚支撑和落脚恢复。此次没有实施或宣称已完成这套控制器。', '',
        '## 复现与范围', '',
        '读原轨迹并重算理想FK，随后只跑两组8秒对照；未重复整机CAD检查。PETG实际重量、摩擦/接触和舵机参数仍待实测，因此结论针对当前仿真与诊断控制链路。', '',
        f"串行，原生线程{report['resources']['observed_threads']}，导入前地址空间硬限额4 GB。对照进程峰值RSS采样约{report['resources']['peak_rss_bytes']/1e9:.3f} GB（最终启动器报告1.246 GB）。正式STEP/MJCF/scene哈希前后相同。实际诊断/实验脚本hash分别保存，报告渲染另记hash。", '',
        '```bash', '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/diagnose_zp10d_liftoff.py inspect',
        '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/diagnose_zp10d_liftoff.py experiments',
        '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/diagnose_zp10d_liftoff.py report', '```', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    time = [f['time_s'] for f in baseline['frames']]
    com_y = np.array([f['com_world_mm'][1] for f in baseline['frames']])
    bounds = np.array([[np.array(f['foot_projected_polygons_world_m'][1])[:, 1].min()*1000,
                       np.array(f['foot_projected_polygons_world_m'][1])[:, 1].max()*1000] for f in baseline['frames']])
    axes[0].fill_between(time, bounds[:, 0], bounds[:, 1], alpha=.2, label='Right sole Y extent')
    axes[0].plot(time, com_y, label='Actual whole-robot COM Y')
    axes[0].set_title('COM never reaches the support foot')
    axes[0].set_ylabel('Y left (mm)'); axes[0].legend(fontsize=8)
    for trial, color in [(baseline, '#307fa5'), (stiff, '#df813a')]:
        t = [f['time_s'] for f in trial['frames']]
        axes[1].plot(t, [f['foot_forces_world_n'][0][2] for f in trial['frames']], color=color, label=trial['name'])
        axes[2].plot(t, [f['sole_min_height_mm'][0] for f in trial['frames']], color=color, label=trial['name'])
    axes[1].axhline(.5, color='gray', linestyle='--', label='Unloaded criterion')
    axes[1].set_ylabel('Swing-foot vertical force (N)'); axes[1].set_title('Swing foot keeps bearing weight')
    axes[1].legend(fontsize=8)
    axes[2].axhline(2., color='gray', linestyle='--', label='Clearance criterion')
    axes[2].axhline(8., color='green', linestyle=':', label='Nominal lift target')
    axes[2].set_ylabel('Lowest sole vertex Z (mm)'); axes[2].set_title('Stronger position control is insufficient')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvspan(4.5, 6.5, color='gray', alpha=.08)
        ax.set_xlabel('Time (s)')
    fig.savefig(CACHE / 'CURRENT_LIFTOFF_DIAGNOSIS.png', dpi=170); plt.close(fig)
    (CACHE / 'CURRENT_LIFTOFF_DIAGNOSIS.md').write_text('\n'.join(lines))
    (CACHE / 'CURRENT_LIFTOFF_DIAGNOSIS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    import shutil
    for suffix in ('md', 'json', 'png'):
        name = 'CURRENT_LIFTOFF_DIAGNOSIS.' + suffix
        shutil.copyfile(CACHE / name, OUT / name)
    assert [loads.digest(p) for p in PATHS] == report['formal_hashes']
    print('Published liftoff diagnosis:', OUT / 'CURRENT_LIFTOFF_DIAGNOSIS.md', flush=True)


def control_report():
    """Review both saved feedback trials independently of their phase counters."""
    required_height_mm = 10.
    control_cache = Path('/tmp/open-duck-liftoff-control')
    trials = [json.loads((control_cache/f'{side}.json').read_text()) for side in ('left', 'right')
              if (control_cache/f'{side}.json').exists()]
    assert trials, 'No completed feedback trial has been saved'
    hashes = [loads.digest(p) for p in PATHS]
    assert all(t['formal_hashes'] == hashes for t in trials)
    assert len({t['simulation_script_sha256'] for t in trials}) == 1, 'Run both sides with the same controller'
    controller_hash = loads.digest(Path(__file__).with_name('control_zp10d_liftoff.py'))
    assert all(t['simulation_script_sha256'] == controller_hash for t in trials), 'Saved trials use an older controller'
    model = mujoco.MjModel.from_xml_path(str(PATHS[2]))
    reference = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,reference,0)
    reference.ctrl[:] = model.key_ctrl[0]
    mujoco.mj_step(model,reference,nstep=round(3/model.opt.timestep))
    adr = model.jnt_qposadr[model.actuator_trnid[:,0]]
    for trial in trials:
        frames, side = trial['frames'], trial['swing_side_index']
        time = np.array([f['time_s'] for f in frames])
        assert len(time) > 1 and np.all(np.diff(time) > 0)
        airborne = [f['Fz_n'][side] < .5 and f['sole_min_height_mm'][side] >= required_height_mm
                    and f['Fz_n'][1-side] > 10 and f['sole_min_height_mm'][1-side] < 1
                    for f in frames]
        run_start, longest = None, 0.
        for t, air in zip(time,airborne):
            run_start = t if air and run_start is None else run_start
            if air:
                longest = max(longest,t-run_start)
            else:
                run_start = None
        last = frames[-1]
        final_error = float(np.degrees(np.abs(np.array(last['qpos'])[adr]-reference.qpos[adr])).max())
        complete = [t['to'] for t in trial['transitions']] == ['transfer','lift','lower','recover','done']
        recovery_start = next((t['time_s'] for t in trial['transitions'] if t['to'] == 'recover'), None)
        stable_start, stable_span = None, 0.
        for frame in frames:
            stable = (recovery_start is not None and frame['phase'] == 'recover'
                      and frame['time_s'] >= recovery_start+3
                      and np.max(np.abs(np.array(frame['qpos'])[adr]-reference.qpos[adr])) < np.deg2rad(3)
                      and frame['tilt_deg'] < 5
                      and max(abs(v) for v in frame['qvel']) < .15
                      and min(frame['Fz_n']) > 3)
            if stable:
                if stable_start is None:
                    stable_start = frame['time_s']
                stable_span = frame['time_s']-stable_start
            else:
                stable_start, stable_span = None, 0.
        accepted = bool(trial['status'] == 'passed' and complete and longest+1e-9 >= .2
                        and not trial.get('diagnostic_prefix_replayed',False)
                        and not trial['stop'] and not trial['failures']
                        and not trial['nonfloor_contact_peaks_n'] and not any(trial['warnings'])
                        and final_error < 3 and last['tilt_deg'] < 5
                        and max(abs(v) for v in last['qvel']) < .15
                        and min(last['Fz_n']) > 3
                        and stable_span+1e-9 >= .5
                        and max(trial['loaded_slip_distance_mm']) <= 5
                        and trial['sampled_min_self_separation_mm'] >= .2
                        and min(trial['min_joint_margin_deg']) >= -1e-4)
        trial['independent_review'] = {'accepted': accepted,
            'required_whole_sole_clearance_mm': required_height_mm,
            'longest_sample_span_clear_unloaded_s': float(longest),
            'maximum_sole_min_height_mm': max(f['sole_min_height_mm'][side] for f in frames),
            'minimum_swing_load_n': min(f['Fz_n'][side] for f in frames),
            'final_joint_error_to_settled_home_deg': final_error,
            'final_stable_sample_span_s': float(stable_span),
            'maximum_tilt_deg': max(f['tilt_deg'] for f in frames),
            'acceptance_failure_reasons': [name for name,ok in (
                ('controller_did_not_pass',trial['status'] == 'passed'),
                ('incomplete_phase_sequence',complete),
                ('insufficient_continuous_whole_sole_clearance',longest+1e-9 >= .2),
                ('diagnostic_prefix_replay',not trial.get('diagnostic_prefix_replayed',False)),
                ('safety_stop_or_motion_failure',not trial['stop'] and not trial['failures']),
                ('home_joint_error',final_error < 3),
                ('unstable_final_state',last['tilt_deg'] < 5 and max(abs(v) for v in last['qvel']) < .15 and min(last['Fz_n']) > 3),
                ('insufficient_final_stable_hold',stable_span+1e-9 >= .5),
                ('loaded_slip_exceeds_5_mm',max(trial['loaded_slip_distance_mm']) <= 5),
                ('self_clearance_below_0_2_mm',trial['sampled_min_self_separation_mm'] >= .2),
                ('physical_joint_limit',min(trial['min_joint_margin_deg']) >= -1e-4)) if not ok],
            'complete_phase_sequence': complete}
        # Compare both controllers on the same saved 50 Hz samples. Native
        # per-physics-step saturation duration, when available, stays separate.
        lift_frames = [f for f in frames if f['phase'] == 'lift']
        if lift_frames:
            torque = np.array([f['torque_nm'] for f in lift_frames])
            caps = np.max(np.abs(model.actuator_forcerange),axis=1)
            trial['sampled_lift_torque_statistics'] = {
                'sample_count':len(lift_frames),
                'rms_torque_nm':np.sqrt(np.mean(torque**2,axis=0)).tolist(),
                'peak_torque_nm':np.max(np.abs(torque),axis=0).tolist(),
                'saturation_sample_fraction':np.mean(np.abs(torque)>=.999*caps,axis=0).tolist(),
                'saturation_threshold_fraction':.999,
                'scope':'50 Hz recorded samples; not continuous-time saturation duration'}
        assert trial['status'] != 'passed' or accepted, 'Controller success failed independent review'
    accepted = len(trials) == 2 and all(t['independent_review']['accepted'] for t in trials)
    pending = [s for s in ('left','right') if s not in [t['side'] for t in trials]]
    report = {'date':date.today().isoformat(), 'status':'passed' if accepted else 'failed',
        'required_whole_sole_clearance_mm': required_height_mm,
        'formal_hashes':hashes, 'trials':trials, 'pending_sides':pending,'report_renderer_sha256':loads.digest(__file__),
        'scope':'Floating-base simulation only; estimated mass/contact/actuator parameters. No hardware or trained policy validation.',
        'cad_snapshot':'Skipped: CAD geometry and formal home unchanged.'}
    baseline_path = OUT / 'CURRENT_LOAD_ASSESSMENT.json'
    baseline = json.loads(baseline_path.read_text())
    if [baseline[k] for k in ('source_step_sha256','mjcf_sha256','scene_sha256')] == hashes:
        report['baseline_load_assessment_sha256'] = loads.digest(baseline_path)
    else:
        # The load report can predate a newly accepted standing home. Bind the
        # motion review to its actual current stance instead of reusing loads.
        del baseline
        baseline_path = OUT / 'CURRENT_STANCE_ASSESSMENT.json'
        baseline = json.loads(baseline_path.read_text())
        assert [baseline[k] for k in ('source_step_sha256','mjcf_sha256','scene_sha256')] == hashes
        assert baseline['standing_passed']
        report['baseline_stance_assessment_sha256'] = loads.digest(baseline_path)
    del baseline
    lines = ['# 抬脚反馈控制验证', '',
        '**左右完整动作通过。**' if accepted else '**尚未通过左右完整抬脚、落脚与恢复验收。**', '',
        '正式 CAD、home、执行器增益、扭矩上限和关节限位保持。控制使用自由基座、原接触求解器和位置执行器，按捕获点/载荷反馈转重，通过低载荷与捕获点门槛后收腿，落脚和恢复继续闭环。实际抬脚按Fz<0.5 N及整脚最低点≥10 mm持续至少0.2秒验收；此前仅超过2 mm的低高度测试不再作为目标高度通过证据。', '',
        '当前控制接口在50 Hz优化并保持原位置执行器目标，预测与执行采用相同保持方式；没有修改力/关节限位或施加外力。该接口尚未在ZP10D实机验证，不能直接作为硬件控制方案。带诊断前缀重放的结果不能通过独立验收。', '',
        '| 待抬侧 | 独立验收 | 整脚离地连续跨度 s | 最高整脚最低点 mm | 最低承重 N | 最大倾角 ° | 最终 home 误差 ° | 左/右受载滑动 mm | 停止/未通过原因 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for t in trials:
        r = t['independent_review']
        slip = ' / '.join(f'{v:.2f}' for v in t['loaded_slip_distance_mm'])
        reason = ', '.join(([t['stop']] if t['stop'] else []) + t['failures']) or ', '.join(r['acceptance_failure_reasons']) or '无'
        if t['stop'] == 'interrupted_for_diagnosis' and 'unloading_timeout' in t['failures']:
            reason += '（转重已超时；随后中断，恢复未评估）'
        lines.append(f"| {t['side']} | {'通过' if r['accepted'] else '失败'} | {r['longest_sample_span_clear_unloaded_s']:.3f} | {r['maximum_sole_min_height_mm']:.3f} | {r['minimum_swing_load_n']:.3f} | {r['maximum_tilt_deg']:.2f} | {r['final_joint_error_to_settled_home_deg']:.2f} | {slip} | {reason} |")
    for side in pending:
        lines.append(f'| {side} | 未运行 | — | — | — | — | — | — | 等待同一控制器测试 |')
    lines += ['', '恢复验收要求：进入恢复至少3秒后，全部16轴距原settled home<3°、机身倾角<5°、所有速度<0.15且双脚Fz>3 N，连续稳定采样首末跨度至少0.5秒。离地期间支撑脚还须Fz>10 N、整脚最低点<1 mm。逐侧实际稳定跨度保存在JSON的independent_review中。', '']
    for trial in trials:
        if 'sampled_lift_torque_statistics' not in trial:
            continue
        knee = ('right' if trial['swing_side_index'] == 0 else 'left')+'_knee'
        a = trial['actuator_order'].index(knee)
        stats = trial['sampled_lift_torque_statistics']
        lines.append(f"{trial['side']} 抬脚阶段支撑膝：采样RMS {stats['rms_torque_nm'][a]:.3f} N·m，峰值 {stats['peak_torque_nm'][a]:.6f} N·m，触及99.9%扭矩上限的采样比例 {stats['saturation_sample_fraction'][a]:.1%}。逐物理步的完整饱和时长另见JSON的lift_torque_metrics。")
    peak = max(max(t['peak_torque_nm']) for t in trials)
    lines += ['', '## 全动作逐步扭矩记录', '',
        '以下统计包含准备、转重、抬脚、落脚与恢复中实际执行的阶段；失败试验未执行的阶段不计入。RMS与饱和时长按2 ms物理步积分，99.9%暂定上限作为饱和门槛。', '',
        '| 待抬侧 | 舵机 | 全程峰值 N·m | 全程RMS N·m | 饱和累计 s | 最长连续饱和 s |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for trial in trials:
        metrics = trial.get('all_phase_torque_metrics')
        if metrics is None:
            continue
        for i,name in enumerate(trial['actuator_order']):
            lines.append(f"| {trial['side']} | {loads.SID[name]} | {trial['peak_torque_nm'][i]:.6f} | {metrics['rms_torque_nm'][i]:.3f} | {metrics['saturation_duration_s'][i]:.3f} | {metrics['longest_saturation_duration_s'][i]:.3f} |")
    lines += ['', f'已保存试验的电机峰值 {peak:.6f} N·m；ZP10D 的 ±0.980665 N·m 仍是暂定堵转限制，不是已验证连续额定。没有实测 PETG 打印重量、接触参数或实机执行结果，不能作为硬件连续运行或模型训练完成的证明。', '',
        '目标高度判据为脚底真实网格所有顶点距地≥10 mm，同时该脚Fz<0.5 N，连续不少于0.2秒；独立复核采用首末合格采样的时间跨度，不把单次低载荷或脚尖翘起计作成功。允许脚掌绕真实接触点滚动，同时将受载接触点切向速度积分；每脚累计真实滑动不超过5 mm才接受。JSON保存实际轨迹、逐电机峰值/限位余量、滑动量、停止原因、源文件与执行脚本哈希。', '',
        '```bash', '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/control_zp10d_liftoff.py left',
        '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/control_zp10d_liftoff.py right',
        '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/diagnose_zp10d_liftoff.py control-report', '```', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,4,figsize=(17,7),layout='constrained')
    for row in range(len(trials),2):
        for ax in axes[row]:
            ax.text(.5,.5,'Trial pending',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
    for row,t in enumerate(trials):
        f,s = t['frames'],t['swing_side_index']
        time = [x['time_s'] for x in f]
        axes[row,0].plot(time,[x['Fz_n'][s] for x in f],label='Swing foot')
        axes[row,0].plot(time,[x['Fz_n'][1-s] for x in f],label='Support foot')
        axes[row,0].axhline(.5,color='gray',ls='--')
        axes[row,1].plot(time,[x['sole_min_height_mm'][s] for x in f])
        axes[row,1].axhline(required_height_mm,color='gray',ls='--',label='Required whole-sole height')
        axes[row,2].plot(time,[x['com_world_mm'][1] for x in f],label='COM Y')
        axes[row,2].plot(time,[x['capture_point_world_mm'][1] for x in f],label='Capture point Y')
        knee = ('right' if s == 0 else 'left')+'_knee'
        a = t['actuator_order'].index(knee)
        axes[row,3].plot(time,[abs(x['torque_nm'][a]) for x in f],label='Support knee |torque|')
        axes[row,3].axhline(max(abs(v) for v in model.actuator_forcerange[a]),color='gray',ls='--',label='Provisional stall cap')
        for col,label in enumerate(('Load (N)','Whole-sole minimum Z (mm)','World Y (mm)','Torque (N m)')):
            ax=axes[row,col];ax.set(xlabel='Time (s)',ylabel=label,title=t['side']+' — '+t['status'])
            for tr in t['transitions']:
                ax.axvline(tr['time_s'],color='gray',alpha=.2)
            ax.legend(fontsize=8)
    stem = control_cache/'CURRENT_LIFTOFF_CONTROL'
    fig.savefig(stem.with_suffix('.png'),dpi=160);plt.close(fig)
    stem.with_suffix('.md').write_text('\n'.join(lines))
    stem.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    import shutil
    for suffix in ('.md','.json','.png'):
        shutil.copyfile(stem.with_suffix(suffix),OUT/('CURRENT_LIFTOFF_CONTROL'+suffix))
    print('Saved',OUT/'CURRENT_LIFTOFF_CONTROL.md',report['status'],flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['inspect']:
        inspect_saved()
    elif sys.argv[1:] == ['experiments']:
        experiments()
    elif sys.argv[1:] == ['report']:
        publish()
    elif sys.argv[1:] == ['control-report']:
        control_report()
    else:
        raise SystemExit('usage: diagnose_zp10d_liftoff.py inspect | experiments | report | control-report')
