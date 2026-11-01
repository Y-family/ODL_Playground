"""Run directly: reference reader parity and 16-actuator imitation mapping."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['JAX_PLATFORMS'] = 'cpu'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from playground.common.poly_reference_motion import PolyReferenceMotion as JaxReference
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion as NumpyReference
from playground.open_duck_mini_v2.custom_rewards import reward_imitation
from playground.open_duck_mini_zp10d.reference import ROOT, inputs
from playground.common.utils import geoms_colliding
from playground.open_duck_mini_zp10d.evaluate import observation
from playground.open_duck_mini_zp10d.joystick import (
    Joystick, gait_reward_terms, torque_margin_cost, lateral_com_offset, sagittal_balance_cost,
    locomotion_tracking, supported_air_time_reward, overspeed_cost)


def main():
    # Air time cannot pay for standing, sideways motion, loaded feet or a jump
    # with no supporting foot, even if the contact timer reports an airborne foot.
    command = jp.array([.01, 0., 0.])
    good = (.005, .0)
    for cmd, velocity, loads, expected in [
        (command, [.01, 0., 0.], [0., 20.], 1.),
        (command, [0., 0., 0.], [0., 20.], 0.),
        (command, [0., .1, 0.], [0., 20.], 0.),
        (command, [-.01, 0., 0.], [0., 20.], 0.),
        (command, [.01, 0., 0.], [10., 10.], 0.),
        (command, [.01, 0., 0.], [0., 0.], 0.),
        (jp.zeros(3), [.01, 0., 0.], [0., 20.], 0.)]:
        np.testing.assert_allclose(supported_air_time_reward(cmd, jp.array(velocity),
            jp.array(good), jp.array(loads), jp.array([.2, 0.]), 20.), expected)
    np.testing.assert_allclose(supported_air_time_reward(command, command, jp.array(good),
        jp.array([0., 20.]), jp.array([.4, 0.]), 20.), 0.)
    # The failed policy pitched forward with both feet loaded. Penalize that
    # behavior without penalizing the roll motion needed for lateral transfer.
    from scipy.spatial.transform import Rotation
    for yaw in (0., .7, -1.8):
        for roll in (0., .35, -.35):
            for pitch in (0., np.deg2rad(10.), np.deg2rad(-10.)):
                rotation = jp.array(Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix())
                np.testing.assert_allclose(sagittal_balance_cost(rotation, pitch), 0., atol=1e-10)
                np.testing.assert_allclose(sagittal_balance_cost(rotation, 0.),
                                           0. if pitch == 0. else 1., atol=1e-6)
    # A rightward shift stays rightward after the upstream random-yaw reset.
    feet = jp.array([[0., .095, 0.], [0., -.095, 0.]])
    com = jp.array([-.02, -.025, .20])
    np.testing.assert_allclose(lateral_com_offset(com, feet), -.025, atol=1e-7)
    for angle in (0., .7, -1.8):
        rotation = jp.array([[jp.cos(angle), -jp.sin(angle), 0.],
                             [jp.sin(angle), jp.cos(angle), 0.], [0., 0., 1.]])
        translation = jp.array([.4, -.6, 0.])
        np.testing.assert_allclose(lateral_com_offset(com @ rotation.T + translation,
                                                    feet @ rotation.T + translation), -.025, atol=1e-7)
    contacts = SimpleNamespace(geom=jp.array([[2, 5], [8, 3]]), dist=jp.array([.01, -.001]))
    fake = SimpleNamespace(_impl=SimpleNamespace(contact=contacts))
    assert not bool(geoms_colliding(fake, 2, 5))
    assert bool(geoms_colliding(fake, 3, 8)) and bool(geoms_colliding(fake, 8, 3))
    for path in (ROOT / 'data/polynomial_coefficients.pkl',
                 ROOT.parent / 'open_duck_mini_v2/data/polynomial_coefficients.pkl'):
        actual, expected = JaxReference(str(path)), NumpyReference(str(path))
        # Legacy degree-15 monomials already differ between float64 NumPy and
        # float32 JAX near cycle end; the new stable bases have full-cycle parity.
        phases = np.arange(0, actual.nb_steps_in_period, .5) if actual.basis != 'monomial' else (0,)
        for x in actual.dxs:
            for i in phases:
                np.testing.assert_allclose(actual.get_reference_motion(x, 0., 0., i),
                    expected.get_reference_motion(x, 0., 0., i), atol=2e-6, rtol=2e-5)
        if actual.basis == 'periodic_cubic':
            for velocity in actual.dxs:
                coefficients = actual.data_array[actual.dxs.index(velocity), 0, 0]
                np.testing.assert_allclose(actual.sample_polynomial(jp.array(0.), coefficients),
                                           actual.sample_polynomial(jp.array(1.), coefficients), atol=2e-5)
                # The velocity reference must describe the same motion as its positions.
                for i in phases:
                    epsilon = .0001
                    minus = expected.get_reference_motion(velocity, 0., 0., i - epsilon)
                    plus = expected.get_reference_motion(velocity, 0., 0., i + epsilon)
                    frame = expected.get_reference_motion(velocity, 0., 0., i)
                    np.testing.assert_allclose((plus[:16] - minus[:16]) / (2 * epsilon / expected.fps),
                                               frame[16:32], atol=2e-5)
    names, actuators, home = inputs()
    leg_names = [n for n in names if any(x in n for x in ('hip', 'knee', 'ankle'))]
    legs = jp.array([actuators.index(n) for n in leg_names])
    q = jp.array([home[7 + names.index(n)] for n in actuators])
    ref = jp.concatenate([jp.array(home[7:]), jp.zeros(16), jp.ones(2), jp.zeros(6)])
    def reward(angles, command=jp.array([.03, 0., 0.])):
        return reward_imitation(jp.array(home[:7]), jp.zeros(6), angles, jp.zeros(16),
                                jp.ones(2), ref, command, True, legs)
    normal = reward(q)
    assert float(reward(q, jp.array([.01, 0., 0.]))) > 0.
    np.testing.assert_allclose(reward(q, jp.zeros(3)), 0., atol=1e-8)
    antenna = q.at[actuators.index('left_antenna')].add(.5)
    knee = q.at[actuators.index('right_knee')].add(.1)
    np.testing.assert_allclose(normal, reward(antenna), atol=1e-6)
    assert float(reward(knee)) < float(normal) - .1
    fitted = NumpyReference(str(ROOT / 'data/polynomial_coefficients.pkl'))
    contacts = np.array([fitted.get_reference_motion(max(fitted.dxs), 0., 0., i)[32:34] > .5
                         for i in range(fitted.nb_steps_in_period)])
    assert np.any(np.all(contacts == [1, 0], axis=1))
    assert np.any(np.all(contacts == [0, 1], axis=1))

    env = Joystick()
    assert float(env._reference_lateral_com.min()) < -.005
    assert float(env._reference_lateral_com.max()) > .005
    from playground.open_duck_mini_zp10d.actuators import configure, signature, position_feedback_action
    from playground.open_duck_mini_zp10d.reference import SCENE, apply_reference_stance
    # Exercise both loading paths, not just the shared parameter setter.
    full = mujoco.MjModel.from_xml_path(str(SCENE))
    configure(full)
    import json
    apply_reference_stance(full, json.loads((ROOT / 'data/reference.json').read_text()))
    assert env.actuator_signature == signature()
    for attr in ('dof_armature', 'dof_damping', 'dof_frictionloss', 'actuator_gainprm',
                 'actuator_biasprm', 'actuator_forcerange', 'key_qpos', 'key_ctrl'):
        np.testing.assert_allclose(getattr(full, attr), getattr(env.mj_model, attr), atol=1e-12)
    assert env._config.noise_config.action_min_delay == 2
    assert env._config.noise_config.action_max_delay == 3
    del full
    # A torque-only sensitivity run preserves command scaling, geometry and
    # identified response. Native and MJX paths must apply the same override.
    scaled = Joystick(torque_limit_scale=2.)
    full = mujoco.MjModel.from_xml_path(str(SCENE))
    prior = full.actuator_forcerange.copy()
    configure(full, torque_limit_scale=2.)
    apply_reference_stance(full, json.loads((ROOT / 'data/reference.json').read_text()))
    main_servo = np.array(['antenna' not in full.actuator(i).name for i in range(full.nu)])
    np.testing.assert_allclose(full.actuator_forcerange[main_servo], 2. * prior[main_servo])
    np.testing.assert_array_equal(full.actuator_forcerange[~main_servo], prior[~main_servo])
    np.testing.assert_array_equal(scaled._config.action_scale, env._config.action_scale)
    for attr in ('dof_armature', 'dof_damping', 'dof_frictionloss', 'actuator_gainprm',
                 'actuator_biasprm', 'key_qpos', 'key_ctrl'):
        np.testing.assert_array_equal(getattr(scaled.mj_model, attr), getattr(env.mj_model, attr))
    np.testing.assert_allclose(full.actuator_forcerange, scaled.mj_model.actuator_forcerange)
    for invalid in (0., -1., float('nan'), float('inf')):
        try:
            configure(full, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid torque limit scale accepted')
    del full, scaled
    np.testing.assert_allclose(torque_margin_cost(.5 * env._effort_limits, env._effort_limits), 0., atol=1e-6)
    np.testing.assert_allclose(torque_margin_cost(.9 * env._effort_limits, env._effort_limits), 4., atol=1e-5)
    np.testing.assert_allclose(torque_margin_cost(env._effort_limits, env._effort_limits), 16., atol=1e-5)
    target_h, target_c, target_load = jp.array([.012, 0.]), jp.array([False, True]), jp.array([0., 1.])
    cmd = jp.array([.03333333, 0., 0.])
    def terms(velocity=cmd, height=target_h, foot_v=jp.zeros((2, 3)), load=jp.array([0., 20.4]), contact=target_c,
              command=cmd):
        return gait_reward_terms(command, velocity, height, foot_v, load, contact, target_h, target_c, target_load)
    correct = terms()
    slow = terms(command=jp.array([.01, 0., 0.]), velocity=jp.array([.01, 0., 0.]))
    assert float(slow['support_contact']) > 0.
    stationary = terms(velocity=jp.zeros(3), height=jp.zeros(2), load=jp.ones(2) * 10.2, contact=jp.ones(2, dtype=bool))
    weights = dict(env._config.reward_config.scales)
    score = lambda values: float(sum(values[k] * weights[k] for k in values))
    assert score(correct) > score(stationary) + 3.
    np.testing.assert_allclose(stationary['tracking_lin_vel'], 0., atol=1e-6)
    # The 0.01 m/s command must remain active, unlike a literal >0.01 gate.
    slow_command = jp.array([.01, 0., 0.])
    for velocity in (jp.zeros(3), jp.array([-.01, 0., 0.]), jp.array([0., .01, 0.])):
        tracking, progress = locomotion_tracking(slow_command, velocity)
        np.testing.assert_allclose(tracking, 0., atol=1e-7)
        assert not bool(progress)
    np.testing.assert_allclose(locomotion_tracking(slow_command, slow_command)[0], 1., atol=1e-7)
    assert bool(locomotion_tracking(slow_command, slow_command)[1])
    np.testing.assert_allclose(locomotion_tracking(jp.zeros(3), jp.zeros(3))[0], 1., atol=1e-7)
    np.testing.assert_allclose(locomotion_tracking(slow_command, jp.array([.01, .1, 0.]))[0],
                               np.exp(-3.), atol=1e-7)
    np.testing.assert_allclose(locomotion_tracking(slow_command, jp.array([.01, .1, 0.]), .1)[0],
                               1., atol=1e-7)
    for command in (slow_command, -slow_command, jp.array([0., .01, 0.])):
        np.testing.assert_allclose(locomotion_tracking(command, command)[0], 1., atol=1e-7)
        assert bool(locomotion_tracking(command, command)[1])
        # Direct learning retains forward credit beyond twice the tiny command,
        # with a separate modest overspeed cost. Legacy v1 stays reproducible.
        velocity = command * 3.
        np.testing.assert_allclose(locomotion_tracking(command, velocity)[0], 0., atol=1e-7)
        np.testing.assert_allclose(locomotion_tracking(command, velocity, allow_overspeed=True)[0], 1., atol=1e-7)
        np.testing.assert_allclose(overspeed_cost(command, velocity), .16, atol=1e-7)
        assert 4. - .2 * float(overspeed_cost(command, velocity)) > 3.9
    for velocity in (jp.zeros(3), -slow_command, jp.array([0., .01, 0.])):
        assert float(locomotion_tracking(slow_command, velocity, allow_overspeed=True)[0]) <= 1e-7
    # Equal forward/backward motion cannot improve on stationary zero credit.
    rocking = [locomotion_tracking(slow_command, sign * slow_command,
               allow_overspeed=True)[0] for sign in (-1., 1.)]
    assert float(sum(rocking)) <= 0.
    # Nine slow forward samples and one fast backward return have zero net
    # travel. A bounded Gaussian penalty incorrectly pays for this sequence.
    asymmetric = [locomotion_tracking(slow_command, slow_command * sign,
                  allow_overspeed=True)[0] for sign in ([1.] * 9 + [-9.])]
    assert float(sum(asymmetric)) <= 1e-6
    np.testing.assert_allclose(locomotion_tracking(slow_command, jp.array([-.09, 1., 1.]),
        allow_overspeed=True)[0], -9., atol=1e-6)
    assert float(locomotion_tracking(jp.zeros(3), slow_command * 3., allow_overspeed=True)[0]) < .2
    sliding = terms(foot_v=jp.array([[0., 0., 0.], [.01, 0., 0.]]))
    assert score(sliding) < score(correct)
    moving_swing = terms(foot_v=jp.array([[.2, 0., 0.], [0., 0., 0.]]))
    np.testing.assert_allclose(moving_swing['feet_slip'], 0., atol=1e-8)
    jumping = terms(load=jp.zeros(2), contact=jp.zeros(2, dtype=bool))
    assert score(jumping) < score(correct)
    zero = terms(command=jp.zeros(3))
    assert all(float(zero[k]) == 0. for k in ('feet_clearance', 'support_load', 'support_contact'))
    shifted = gait_reward_terms(cmd, jp.array([cmd[0], .4, 0.]), target_h, jp.zeros((2, 3)),
                                jp.array([0., 20.4]), target_c, target_h, target_c, target_load, .4)
    np.testing.assert_allclose(shifted['tracking_lin_vel'], 1., atol=1e-6)
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    data.qvel[:] = np.linspace(-.1, .1, model.nv)
    mujoco.mj_forward(model, data)
    fake = SimpleNamespace(**{k: jp.array(getattr(data, k)) for k in
        ('qpos', 'qvel', 'site_xmat', 'sensordata', 'actuator_force')})
    fake.site_xmat = fake.site_xmat.reshape((-1, 3, 3))
    history = np.arange(3 * model.nu).reshape(3, model.nu) * .001
    targets = model.key('home').ctrl.copy() + .01
    command = jp.array([.03333333, 0., 0., 0., 0., 0., 0.])
    phase = jp.array([.3, .4])
    contact = jp.array([True, False])
    info = dict(rng=jax.random.PRNGKey(0), command=command,
                last_act=jp.array(history[0]), last_last_act=jp.array(history[1]),
                last_last_last_act=jp.array(history[2]), motor_targets=jp.array(targets),
                imitation_phase=phase, imu_history=jp.zeros(9), feet_air_time=jp.zeros(2),
                current_reference_motion=jp.zeros(40), imitation_i=0)
    expected = env._get_obs(fake, info, contact)['state']
    actual = observation(np.array(env.get_gyro(fake)), np.array(env.get_accelerometer(fake)),
                         command, np.array(env.get_actuator_joints_qpos(fake.qpos)),
                         np.array(env.get_actuator_joints_qvel(fake.qvel)), history,
                         targets, contact, phase, model.key('home').ctrl)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    # Compare decoded JAX foot forces with native MuJoCo on the same settled state.
    from mujoco import mjx
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    expected_force = np.zeros(2)
    for ci in range(data.ncon):
        pair = data.contact[ci].geom
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, force)
        for side, gid in enumerate(env._feet_geom_id):
            if set(pair) == {gid, env._floor_geom_id} and data.contact[ci].dist <= 0:
                expected_force[side] += max(0., (force[:3] @ data.contact[ci].frame.reshape(3, 3))[2]
                                            * (1 if pair[0] == env._floor_geom_id else -1))
    converted = mjx.put_data(model, data)
    np.testing.assert_allclose(env.foot_loads(converted), expected_force, atol=2e-4, rtol=2e-4)
    assert expected_force.sum() > 10., expected_force
    # Keep a JIT physics/reward check: eager-only success does not validate PPO.
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    assert bool(jp.isfinite(state.reward)) and bool(jp.isfinite(state.data.qpos).all())
    assert bool(jp.isfinite(state.metrics['cost/com_transfer']))
    assert bool(jp.isfinite(state.metrics['cost/sagittal_balance']))
    # Planned lateral bank is allowed; a >45-degree tip fails both backends.
    for axis in ('x', 'y'):
        for angle, should_stop in ((25., False), (46., True)):
            rotation = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
            tilted = state.data.replace(xmat=state.data.xmat.at[1].set(jp.array(rotation)))
            assert bool(env._get_termination(tilted)) == should_stop
    # Feedback must pass through the same 40 ms delay as policy residuals.
    feedback_env = Joystick(position_feedback_gain=3., position_feedback_damping=.03)
    np.testing.assert_allclose(np.asarray(feedback_env._config.action_scale),
        np.asarray(env._config.action_scale) * (1. + 3. * np.asarray(feedback_env._leg_mask)))
    feedback_state = feedback_env.reset(jax.random.PRNGKey(0))
    ankle = feedback_env.actuator_names.index('left_ankle')
    jid = feedback_env._mj_model.actuator_trnid[ankle, 0]
    address = feedback_env.actuator_joint_qpos_addr[ankle]
    feedback_state = feedback_state.replace(data=feedback_state.data.replace(
        qpos=feedback_state.data.qpos.at[address].add(.01),
        qvel=feedback_state.data.qvel.at[feedback_env._mj_model.jnt_dofadr[jid]].set(.2)))
    command = np.asarray(feedback_env.command_action(feedback_state.data, jp.zeros(16)))
    expected_delta = np.zeros(16); expected_delta[ankle] = -.036
    np.testing.assert_allclose(command * np.asarray(feedback_env._config.action_scale), expected_delta, atol=2e-7)
    native_command = position_feedback_action(
        np.zeros(16), np.asarray(feedback_env.get_actuator_joints_qpos(feedback_state.data.qpos)),
        np.asarray(feedback_env._geometric_home), np.asarray(feedback_env._config.action_scale),
        np.asarray(feedback_env._leg_mask), 3.,
        np.asarray(feedback_env.get_actuator_joints_qvel(feedback_state.data.qvel)), .03)
    np.testing.assert_allclose(command, native_command, atol=2e-7)
    step_feedback = jax.jit(feedback_env.step)
    for tick in range(3):
        feedback_state = step_feedback(feedback_state, jp.zeros(16))
        expected = np.asarray(feedback_env._default_actuator) + (expected_delta if tick == 2 else 0.)
        np.testing.assert_allclose(feedback_state.info['motor_targets'], expected, atol=3e-7)
    direct = Joystick(position_feedback_gain=3., direct_locomotion=True, position_feedback_damping=.03)
    assert direct.PRM.nb_steps_in_period == 25
    assert direct.reward_specification()['fixed_support_schedule'] is False
    assert direct.reward_specification()['moving_foot_clearance_target_m'] == .005
    np.testing.assert_array_equal(direct.PRM.get_reference_motion(.01, 0., 0., 80), np.zeros(40))
    assert direct._config.reward_config.scales.imitation == 0.
    assert direct._config.reward_config.scales.alive == 0.
    sampled = np.asarray(jax.vmap(direct.sample_command)(jax.random.split(jax.random.PRNGKey(0), 4096)))
    assert .01 < np.mean(sampled[:, 0] == 0.) < .035
    for term in ('support_load', 'support_contact', 'com_transfer', 'sagittal_balance'):
        assert direct._config.reward_config.scales[term] == 0.
    direct_state = direct.reset(jax.random.PRNGKey(0))
    direct_state.info['imitation_i'] = 24  # exercise the observation clock seam
    direct_state = jax.jit(direct.step)(direct_state, jp.zeros(16))
    assert bool(jp.isfinite(direct_state.reward))
    assert bool(jp.isfinite(direct_state.info['gait_velocity']).all())
    assert int(direct_state.info['imitation_i']) == 0
    print('PASS: reference parity, leg mapping, observations, dense reward ranking, zero-command gating, '
          'stance-only slip, native/JAX foot forces, JIT reset/physics/rewards, delayed position feedback, '
          'direct support learning and progress-gated air time', flush=True)


if __name__ == '__main__':
    main()
