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
from playground.open_duck_mini_zp10d.joystick import Joystick, gait_reward_terms, torque_margin_cost


def main():
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
    def reward(angles):
        return reward_imitation(jp.array(home[:7]), jp.zeros(6), angles, jp.zeros(16),
                                jp.ones(2), ref, jp.array([.03, 0., 0.]), True, legs)
    normal = reward(q)
    antenna = q.at[actuators.index('left_antenna')].add(.5)
    knee = q.at[actuators.index('right_knee')].add(.1)
    np.testing.assert_allclose(normal, reward(antenna), atol=1e-6)
    assert float(reward(knee)) < float(normal) - .1
    fitted = NumpyReference(str(ROOT / 'data/polynomial_coefficients.pkl'))
    contacts = np.array([fitted.get_reference_motion(.03333333, 0., 0., i)[32:34] > .5
                         for i in range(fitted.nb_steps_in_period)])
    assert np.any(np.all(contacts == [1, 0], axis=1))
    assert np.any(np.all(contacts == [0, 1], axis=1))

    env = Joystick()
    np.testing.assert_allclose(torque_margin_cost(.5 * env._effort_limits, env._effort_limits), 0., atol=1e-6)
    np.testing.assert_allclose(torque_margin_cost(.9 * env._effort_limits, env._effort_limits), 4., atol=1e-5)
    np.testing.assert_allclose(torque_margin_cost(env._effort_limits, env._effort_limits), 16., atol=1e-5)
    target_h, target_c, target_load = jp.array([.012, 0.]), jp.array([False, True]), jp.array([0., 1.])
    cmd = jp.array([.03333333, 0., 0.])
    def terms(velocity=cmd, height=target_h, foot_v=jp.zeros((2, 3)), load=jp.array([0., 20.4]), contact=target_c,
              command=cmd):
        return gait_reward_terms(command, velocity, height, foot_v, load, contact, target_h, target_c, target_load)
    correct = terms()
    stationary = terms(velocity=jp.zeros(3), height=jp.zeros(2), load=jp.ones(2) * 10.2, contact=jp.ones(2, dtype=bool))
    weights = dict(env._config.reward_config.scales)
    score = lambda values: float(sum(values[k] * weights[k] for k in values))
    assert score(correct) > score(stationary) + 3.
    np.testing.assert_allclose(stationary['tracking_lin_vel'], .06217656, atol=1e-6)
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
    print('PASS: reference parity, leg mapping, observations, dense reward ranking, zero-command gating, '
          'stance-only slip, native/JAX foot forces, JIT reset/physics/rewards', flush=True)


if __name__ == '__main__':
    main()
