"""Counterexamples and an actual JIT step for the liftoff bootstrap stage."""
import os
import json
from pathlib import Path
import sys
import tempfile
from typing import NamedTuple

os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from playground.open_duck_mini_zp10d.joystick import (
    Joystick, lift_curriculum_terms, supported_air_time_reward, unload_progress_reward,
    lift_stability_terms, body_roll_angle, body_roll_cost, placo_lift_imitation_reward, placo_lift_terminal_cost, lift_height_band,
    reference_frame_velocity, base_height_cost)
from playground.open_duck_mini_zp10d.liftoff import update_liftoff, liftoff_features, command_liftoff_runs, completed_lift_reward
from playground.open_duck_mini_zp10d.runner import wrap_training
from playground.open_duck_mini_zp10d.actuators import (coordinated_lift_action, hold_cleared_roll,
    planned_swing_roll_mask, suppress_swing_roll_residual, reference_posture_action,
    reference_torque_inputs,
    periodic_reference_torque, sampled_reference_inputs, sampled_reference_action)
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env


def check_sampled_reference(drift=False):
    gain, damping = 1.1275290085055225, .07053243703737014
    env = Joystick(position_feedback_gain=gain, position_feedback_damping=damping,
        torque_limit_scale=2., lift_curriculum=True, coordinated_lift=True,
        reference_action=True, reference_feedforward=True, sampled_reference_control=True,
        drift_observation=drift)
    model = env.mj_model
    ids = model.actuator_trnid[:, 0]
    home, default = model.key('home').qpos[model.jnt_qposadr[ids]], model.key('home').ctrl
    reference = json.loads((Path(__file__).parent/'data/reference.json').read_text())
    corrections, specification = sampled_reference_inputs(reference, env.actuator_names,
        home, default, model.actuator_gainprm[:, 0], gain, damping, stop_fade=True)
    assert specification == env.sampled_reference_specification
    assert env.coordination_guidance_version == 13
    np.testing.assert_allclose(corrections, env._sampled_reference_corrections, atol=1e-7)
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
    motion = PolyReferenceMotion(str(Path(__file__).parent/'data/polynomial_coefficients.pkl'))
    order = np.array([reference['joint_names'].index(n) for n in env.actuator_names])
    times, torques, _ = reference_torque_inputs(reference)
    raw, scale = np.linspace(-.01,.01,16), np.asarray(env._config.action_scale)
    command = np.array([max(motion.dxs),0.,0.])
    apply = jax.jit(lambda a, p, c, t:sampled_reference_action(
        a,p,c,t,env._sampled_reference_corrections,jp.array(scale),jp))
    for phase in (0,7,15,29):
        current=motion.get_reference_motion(max(motion.dxs),0.,0.,phase)
        future=motion.get_reference_motion(max(motion.dxs),0.,0.,phase+2.4)
        tau=np.mean([periodic_reference_torque(phase*.02+.054+i*.002,times,torques,np)
                     for i in range(-6,7)],axis=0)*1.1049132959059584
        expected=np.asarray(env._leg_mask)*(future[order]+tau/model.actuator_gainprm[:,0]
            +gain*(current[order]-home)+damping*current[16+order]-default)
        np.testing.assert_allclose(corrections[phase],expected,atol=3e-16)
        for elapsed,ramp in ((0.,0.),(.6,.5),(1.2,1.)):
            np.testing.assert_allclose(apply(raw,phase,command,elapsed),
                raw+ramp*expected/scale,atol=2e-7)
        np.testing.assert_array_equal(apply(raw,phase,np.zeros(3),2.),jp.array(raw))
        # Legacy v8 cuts nominal immediately; v9 keeps half amplitude halfway
        # through the fade even after the command becomes zero.
        faded = sampled_reference_action(jp.array(raw), phase, jp.zeros(3), 2.,
            env._sampled_reference_corrections, jp.array(scale), jp, blend_ticks=30)
        np.testing.assert_allclose(faded, raw+.5*expected/scale, atol=2e-7)
        np.testing.assert_array_equal(sampled_reference_action(jp.array(raw), phase,
            jp.zeros(3), 2., env._sampled_reference_corrections, jp.array(scale), jp,
            blend_ticks=0), jp.array(raw))
    state=jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.info['course_start_stop']=jp.array(False)
    state.info['command']=jp.array([max(motion.dxs),0.,0.,0.,0.,0.,0.])
    state.info['step']=jp.array(90)
    state.info['imitation_i']=jp.array(7)
    state.info['reference_motion_step']=jp.array(97)
    state.info['reference_blend_ticks']=jp.array(60)
    state.info['qualified_lift']=jp.ones(2,dtype=bool)
    state.info['swing_roll_hold']=jp.array([.3,-.3])
    # At phase 7 Placo has entered the right-foot swing block.  Only the
    # learned right-roll residual is removed; the nominal Placo path remains.
    np.testing.assert_array_equal(np.asarray(env._reference_swing_roll_mask)[7], [False, True])
    state.info['planned_swing_roll']=jp.array([False, True])
    advance=jax.jit(env.step)
    queue=[np.zeros(16),np.zeros(16),np.zeros(16)]
    from playground.open_duck_mini_zp10d.actuators import position_feedback_action
    for _ in range(3):
        phase=int(state.info['imitation_i'])
        nominal=corrections[phase]/scale
        action=suppress_swing_roll_residual(corrections[phase]/scale, nominal,
            np.array([False, True]), np)
        issued=position_feedback_action(action,np.asarray(env.get_actuator_joints_qpos(state.data.qpos)),
            home,scale,np.asarray(env._leg_mask),gain,
            np.asarray(env.get_actuator_joints_qvel(state.data.qvel)),damping)
        queue=[issued]+queue[:2]
        expected=np.clip(default+queue[2]*scale,*model.jnt_range[ids].T)
        state=advance(state,jp.zeros(16))
        np.testing.assert_allclose(state.info['motor_targets'],expected,atol=3e-7)
        assert np.isfinite(state.data.qpos).all()
    assert int(state.info['reference_motion_step']) == 100
    native = mujoco.MjData(model)
    native.qpos[:] = state.data.qpos
    native.qvel[:] = state.data.qvel
    native.ctrl[:] = state.info['motor_targets']
    mujoco.mj_forward(model, native)
    np.testing.assert_allclose(state.data.site_xpos, native.site_xpos, atol=5e-8, rtol=0)
    gyro = model.sensor('gyro').id
    adr = model.sensor_adr[gyro]
    np.testing.assert_allclose(state.data.sensordata[adr:adr+3], native.sensordata[adr:adr+3],
                               atol=3e-6, rtol=0)
    state.info['command']=jp.zeros(7)
    state=advance(state,jp.zeros(16))
    assert int(state.info['reference_blend_ticks']) == 59
    if drift:
        # Compare actual post-physics actor inputs with an independent NumPy
        # filter, including random initial heading and the advanced reference.
        probe=env.reset(jax.random.PRNGKey(17))
        probe.info['course_start_stop']=jp.array(False)
        probe.info['command']=jp.array([max(motion.dxs),0.,0.,0.,0.,0.,0.])
        previous=np.zeros(2)
        for _ in range(3):
            probe=advance(probe,jp.zeros(16))
            yaw=float(probe.info['reference_heading']); c,s=np.cos(yaw),np.sin(yaw)
            v=np.asarray(probe.data.qvel[:2]); local=np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]])
            reference=motion.get_reference_motion(max(motion.dxs),0.,0.,int(probe.info['imitation_i']))
            previous += .02/1.02*(local-reference[34:36]-previous)
            np.testing.assert_allclose(probe.info['gait_velocity'],previous,atol=2e-8)
            rotation=np.asarray(probe.data.xmat[1])
            relative=np.arctan2(rotation[1,0],rotation[0,0])-yaw
            c,s=np.cos(relative),np.sin(relative)
            expected=np.array([c*previous[0]+s*previous[1],-s*previous[0]+c*previous[1]])/.010
            for name in ('state','privileged_state'):
                np.testing.assert_allclose(probe.obs[name][11:13],expected,atol=2e-6)
        assert np.max(abs(previous)) > 1e-5, 'Nonzero drift must be visible to the actor'
        print('PASS: actual delayed MJX steps expose independently checked velocity drift; '
              'planned transfer/heading handled consistently; input shape unchanged.',flush=True)
    assert int(state.info['reference_motion_step']) == 101
    state.info['reference_blend_ticks']=jp.array(0)
    state=advance(state,jp.zeros(16))
    assert int(state.info['reference_motion_step']) == 0
    state.info['command']=jp.array([max(motion.dxs),0.,0.,0.,0.,0.,0.])
    state=advance(state,jp.zeros(16))
    assert int(state.info['reference_motion_step']) == 1
    assert int(state.info['reference_blend_ticks']) == 1
    # Exercise the actual reward path: following the planned transfer earns
    # zero drift cost; sustained backward error cannot hide in that transfer.
    base_state = env.reset(jax.random.PRNGKey(1))
    info = dict(base_state.info, command=state.info['command'], reference_heading=jp.array(0.))
    info['current_reference_motion'] = info['current_reference_motion'].at[34:36].set(jp.array([.02, .1]))
    def drift_cost(error):
        data = base_state.data.replace(qvel=base_state.data.qvel.at[:2].set(jp.array([.02, .1])+error))
        def reward(memory):
            probe = dict(info, gait_velocity=memory)
            cost = env._get_reward(data, jp.zeros(16), probe, dict(base_state.metrics),
                jp.array(False), jp.zeros(2), jp.zeros(2, dtype=bool))['planar_drift']
            return probe['gait_velocity'], cost
        memory = jax.lax.fori_loop(0, 199, lambda _, memory: reward(memory)[0], jp.zeros(2))
        return reward(memory)[1]
    score = jax.jit(drift_cost)
    np.testing.assert_allclose(score(jp.zeros(2)), 0., atol=1e-9)
    assert float(score(jp.array([-.015, 0.]))) > 2.
    assert float(score(jp.array([-.001, 0.]))) < .011
    assert float(score(jp.array([-.060, 0.]))) > 2.*float(score(jp.array([-.030, 0.]))), \
        'Faster sliding must remain more costly after the old 30mm/s cap'
    angvel = env.mj_model.sensor('global_angvel').id
    adr = env.mj_model.sensor_adr[angvel]
    def turn_cost(rate, phase):
        target = env._reference_vertical_angular_velocity[phase]
        data = base_state.data.replace(sensordata=base_state.data.sensordata.at[adr+2].set(target+rate))
        def reward(memory):
            probe = dict(info, gait_velocity=jp.zeros(2), gait_yaw_rate=memory,
                         imitation_i=jp.array(phase))
            cost = env._get_reward(data, jp.zeros(16), probe, dict(base_state.metrics),
                jp.array(False), jp.zeros(2), jp.zeros(2, dtype=bool))['tracking_ang_vel']
            return probe['gait_yaw_rate'], cost
        memory = jax.lax.fori_loop(0, 199, lambda _, memory: reward(memory)[0], jp.zeros(()))
        return reward(memory)[1]
    turn = jax.jit(turn_cost)
    for phase in (0, 7, 15, 29):
        np.testing.assert_allclose(turn(0., phase), 0., atol=1e-9)
        assert float(turn(.02, phase)) < float(turn(.01, phase)) < 0.
        assert float(turn(.2, phase)) < -2.8
    assert env.reward_specification()['recipe'] == 'placo-sampled-lift-v7'
    assert env._config.reward_config.scales.air_time == 0.
    assert env._config.reward_config.scales.completed_step == 2.
    assert 'completed_steps' in state.metrics
    assert env.liftoff_qualification_height == .010
    # Exercise command visibility and delayed fade at real JIT step boundaries.
    state=env.reset(jax.random.PRNGKey(3))
    state.info['course_start_stop']=jp.array(True)
    state.info['course_command']=jp.array([max(motion.dxs),0.,0.,0.,0.,0.,0.])
    state.info['step']=jp.array(49)
    env._schedule_training_command(state)
    np.testing.assert_array_equal(state.obs['state'][6:9],0.)
    state=advance(state,jp.zeros(16))
    assert int(state.info['step']) == 50
    for name in ('state','privileged_state'):
        np.testing.assert_allclose(state.obs[name][6:9],state.info['course_command'][:3])
    assert int(state.info['reference_motion_step']) == 0
    state=advance(state,jp.zeros(16))
    assert int(state.info['reference_motion_step']) == 1
    state.info['step']=jp.array(299)
    state.info['reference_blend_ticks']=jp.array(60)
    state=advance(state,jp.zeros(16))
    assert int(state.info['step']) == 300
    for name in ('state','privileged_state'):
        np.testing.assert_array_equal(state.obs[name][6:9],0.)
    assert int(state.info['reference_blend_ticks']) == 60
    state=advance(state,jp.zeros(16))
    assert int(state.info['reference_blend_ticks']) == 59
    print('PASS: sampled nominal matches independent native formula, NumPy/JAX, '
          'startup/stop fade and actual MJX delayed control; planned transfer is free, '
          'sustained drift/turning penalized without charging planned transfer/turning; '
          'supplied pose not used.',flush=True)
    return env


def check_randomized_course(env):
    from functools import partial
    from playground.open_duck_mini_zp10d.joystick import domain_randomize
    keys=jax.random.split(jax.random.PRNGKey(8),4)
    varied,axes=domain_randomize(env.mjx_model,keys)
    nominal=env.mjx_model
    friction=np.asarray(varied.geom_friction[:,:,0])
    base=np.asarray(nominal.geom_friction[:,0])
    ratio=friction[:,base>0]/base[base>0]
    assert ratio.min() >= .9 and ratio.max() <= 1.1 and np.ptp(ratio) > .01
    ratio=np.asarray(varied.actuator_gainprm[:,:,0])/np.asarray(nominal.actuator_gainprm[:,0])
    assert ratio.min() >= .95 and ratio.max() <= 1.05 and np.ptp(ratio) > .01
    np.testing.assert_allclose(varied.actuator_biasprm[:,:,1],-varied.actuator_gainprm[:,:,0])
    np.testing.assert_array_equal(ratio[:,-2:],1.)
    for name in ('body_mass','body_inertia','qpos0','dof_damping','dof_frictionloss','actuator_forcerange'):
        np.testing.assert_array_equal(getattr(varied,name),getattr(nominal,name))
    assert axes.body_mass is None and axes.geom_friction == 0
    wrapped=wrap_training(env,episode_length=500,
        randomization_fn=partial(domain_randomize,rng=keys))
    state=jax.jit(wrapped.reset)(keys)
    state=jax.jit(wrapped.step)(state,jp.zeros((4,16)))
    assert state.data.qpos.shape == (4,env.mj_model.nq)
    assert np.isfinite(state.data.qpos).all()
    for name in ('state','privileged_state'):
        np.testing.assert_array_equal(state.obs[name][:,6:9],state.info['command'][:,:3])
    print('PASS: bounded contact/drive variations, nominal mass/inertia/geometry/force caps preserved; '
          'actual randomized batched MJX reset/step and scheduled command observations passed.',flush=True)


def check_delayed_feedback():
    from playground.open_duck_mini_zp10d.actuators import specification
    p = specification()['parameters']
    for inertia in (.0002, .005):
        for gain in (1., 3.):
            xml = f'''<mujoco><option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
              <worldbody><body><joint name="probe" type="hinge" damping="{p['damping_Nm_s_per_rad']}"
                armature="{p['armature_kg_m2']}" frictionloss="{p['frictionloss_Nm']}"/>
              <inertial pos="0 0 0" mass=".071" diaginertia="{inertia} {inertia} {inertia}"/>
              </body></worldbody><actuator><position joint="probe" kp="{p['position_gain_Nm_per_rad']}"
              forcerange="-1.96133 1.96133"/></actuator></mujoco>'''
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            queue, positions = [0., 0., 0.], []
            for tick in range(150):
                goal = np.deg2rad(10.) if tick >= 10 else 0.
                request = goal + gain * (goal - data.qpos[0]) - .03 * data.qvel[0]
                queue = [request] + queue[:2]
                data.ctrl[0] = queue[2]
                for _ in range(10):
                    mujoco.mj_step(model, data)
                    positions.append(data.qpos[0])
            tail = np.rad2deg(positions[-500:])
            if gain == 3.:
                assert np.ptp(tail) > 40., 'Delayed high-gain instability counterexample disappeared'
            else:
                assert np.ptp(tail) < .1 and abs(tail[-1] - 10.) < .1
    print('PASS: gain-3 delayed-feedback oscillation reproduced; gain 1 settles '
          'across two load inertias, identified dynamics and force cap unchanged.', flush=True)


def check_reference_heading():
    env = Joystick(position_feedback_gain=3., position_feedback_damping=.03, torque_limit_scale=2.)
    state = env.reset(jax.random.PRNGKey(0))
    # The same motion at a different episode yaw must receive the same score.
    reference = state.info['current_reference_motion'].at[34:37].set(jp.array([.2, -.1, .03]))
    velocity = env.get_floating_base_qvel(state.data.qvel).at[:3].set(reference[34:37])
    info = dict(state.info, reference_heading=jp.array(0.), current_reference_motion=reference,
                command=jp.array([max(env.PRM.dxs), 0., 0., 0., 0., 0., 0.]))
    data = state.data.replace(qvel=state.data.qvel.at[:6].set(velocity))
    score = env._get_reward(data, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                            jp.zeros(2), jp.zeros(2, dtype=bool))['imitation']
    rotate = jax.jit(reference_frame_velocity)
    for heading in (-np.pi, -.7, .7, np.pi/2):
        c, s = np.cos(heading), np.sin(heading)
        world = velocity.at[:2].set(jp.array([c*velocity[0]-s*velocity[1],
                                             s*velocity[0]+c*velocity[1]]))
        np.testing.assert_allclose(rotate(world, heading), velocity, atol=3e-8)
        rotated_data = data.replace(qvel=data.qvel.at[:6].set(world))
        rotated_info = dict(info, reference_heading=jp.array(heading))
        actual = env._get_reward(rotated_data, jp.zeros(16), rotated_info, dict(state.metrics),
            jp.array(False), jp.zeros(2), jp.zeros(2, dtype=bool))['imitation']
        np.testing.assert_allclose(actual, score, atol=1e-6)
    # Keeping the wrong initial frame must still penalize actual heading drift.
    wrong_frame = env._get_reward(rotated_data, jp.zeros(16), info, dict(state.metrics),
        jp.array(False), jp.zeros(2), jp.zeros(2, dtype=bool))['imitation']
    assert float(score - wrong_frame) > .5
    np.testing.assert_array_equal(reference_frame_velocity(velocity, 0.), velocity)
    print('PASS: actual imitation reward is episode-yaw invariant; old fixed-world '
          'counterexample reproduced; real heading drift remains penalized.', flush=True)


def check_episode_reset():
    """Reproduce the old leak on termination and timeout, without physics."""
    class Probe:
        def reset(self, rng):
            return mjx_env.State(jp.zeros(1), jp.zeros(1), jp.array(0.), jp.array(0.),
                {'probe': jp.array(0.)}, {'rng': rng, 'motor_targets': jp.zeros(1),
                                        'action_history': jp.zeros(2)})

        def step(self, state, action):
            state.info['motor_targets'] += 1.
            state.info['action_history'] += 1.
            state.metrics['probe'] = jp.array(3.)
            return state.replace(data=state.data + 1., obs=state.obs + 1.,
                                 reward=jp.array(2.), done=(action[0] > 0).astype(float))

    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    action = jp.array([[1.], [0.]])
    legacy = wrapper.wrap_for_brax_training(Probe(), episode_length=2)
    leaked = jax.jit(legacy.step)(jax.jit(legacy.reset)(keys), action)
    np.testing.assert_array_equal(leaked.data[:, 0], [0., 1.])
    np.testing.assert_array_equal(leaked.info['motor_targets'][:, 0], [1., 1.])
    fixed = wrap_training(Probe(), episode_length=2)
    advance = jax.jit(fixed.step)
    reset = advance(jax.jit(fixed.reset)(keys), action)
    np.testing.assert_array_equal(reset.done, [1., 0.])
    np.testing.assert_array_equal(reset.info['motor_targets'][:, 0], [0., 1.])
    np.testing.assert_array_equal(reset.info['truncation'], [0., 0.])
    np.testing.assert_array_equal(reset.info['episode_metrics']['sum_reward'], [2., 2.])
    np.testing.assert_array_equal(reset.info['episode_metrics']['probe'], [3., 3.])
    reset = advance(reset, jp.zeros((2, 1)))
    np.testing.assert_array_equal(reset.done, [0., 1.])
    np.testing.assert_array_equal(reset.info['truncation'], [0., 1.])
    np.testing.assert_array_equal(reset.info['episode_metrics']['length'], [1., 2.])
    np.testing.assert_array_equal(reset.info['motor_targets'][:, 0], [1., 0.])
    print('PASS: old automatic reset demonstrably leaks motor memory; fixed mixed '
          'termination/timeout resets preserve PPO return, length and truncation.')


def main():
    check_sampled_reference()
    check_episode_reset()
    check_target_slew()
    check_delayed_feedback()
    check_reference_heading()
    check_unload_progress()
    check_lift_objective()
    check_reference_course()
    check_reference_reset()
    command = jp.array([.01, 0., 0.])
    high = jp.array([.005, 0.])
    unloaded = jp.array([0., 20.])
    timer = jp.array([.04, 0.])
    both = jp.array([True, True])
    def terms(heights=high, loads=unloaded, air_time=timer, allowed=both, cmd=command):
        return lift_curriculum_terms(cmd, heights, loads, air_time, 20., allowed)
    np.testing.assert_allclose(list(terms().values()), [1., .5])
    np.testing.assert_allclose(terms(heights=jp.array([.010, 0.]))['lift_dense'], 1.)
    np.testing.assert_allclose(terms(heights=jp.array([.020, 0.]))['lift_dense'], 1.)
    partial = terms(heights=jp.zeros(2), loads=jp.array([9., 11.]))
    assert 0. < float(partial['unload']) < 1.
    np.testing.assert_allclose(partial['lift_dense'], 0.)
    for values in (terms(loads=jp.array([10., 10.])), terms(loads=jp.zeros(2)),
                   terms(air_time=jp.array([.4, 0.])), terms(allowed=jp.array([False, True])),
                   terms(cmd=jp.zeros(3))):
        np.testing.assert_allclose(list(values.values()), [0., 0.])
    np.testing.assert_allclose(supported_air_time_reward(command, jp.zeros(3), high,
        unloaded, timer, 20.), 0.)
    np.testing.assert_allclose(supported_air_time_reward(command, jp.zeros(3), high,
        unloaded, timer, 20., require_progress=False, allowed=both, minimum_time=.02), 1.)
    env = Joystick(direct_locomotion=True, lift_curriculum=True,
                   position_feedback_gain=3., position_feedback_damping=.03)
    assert env.reward_specification()['task_stage'] == 'alternating_liftoff'
    assert env._config.reward_config.scales.tracking_lin_vel == 0.
    assert env._config.reward_config.scales.unload == 8.
    assert env._config.reward_config.scales.lift_dense == 6.
    assert env._config.reward_config.scales.air_time == 6.
    np.testing.assert_allclose(env._config.reward_config.scales.termination * env.dt, -2.)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(state.info['unload_best'], 0.)
    assert int(state.info['last_landed_foot']) == -1
    np.testing.assert_array_equal(state.obs['state'][-2:], [1., 1.])
    state = jax.jit(env.step)(state, jp.zeros(16))
    assert bool(jp.isfinite(state.reward)) and bool(jp.isfinite(state.data.qpos).all())
    assert state.obs['state'].shape == (113,)
    np.testing.assert_array_equal(state.obs['privileged_state'][111:113], state.obs['state'][-2:])
    assert 'reward/unload' in state.metrics and 'reward/lift_dense' in state.metrics
    assert bool((env.foot_loads(state.data) > .5).all())
    info = dict(state.info)
    env._get_reward(state.data, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                    jp.array([True, True]), jp.ones(2, dtype=bool))
    assert int(info['last_landed_foot']) == -1  # No preceding real liftoff.
    assert not bool(info['qualified_lift'].any())
    # Load can cross 0.5 N after the one-tick first-contact event.
    info['qualified_lift'] = jp.array([True, False])
    env._get_reward(state.data, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                    jp.zeros(2, dtype=bool), jp.ones(2, dtype=bool))
    assert int(info['last_landed_foot']) == 0
    assert not bool(info['qualified_lift'].any())
    for contact_event, expected in (([True, False], 0), ([False, False], 0), ([False, True], 1)):
        if any(contact_event):
            info['qualified_lift'] = jp.array(contact_event)
        env._get_reward(state.data, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                        jp.array(contact_event), jp.ones(2, dtype=bool))
        assert int(info['last_landed_foot']) == expected
        assert not bool(info['qualified_lift'].any())
    # Same grounded pose, different previous landing: the actor must see which
    # foot can earn the next lift. NumPy replay and JIT training advance alike.
    sequence = [([.003, 0.], [0., 20.], [3., 1.]),
                ([0., 0.], [10., 10.], [0., 1.]),
                ([0., .003], [20., 0.], [0., 3.]),
                ([0., 0.], [10., 10.], [1., 0.])]
    native_last, native_qualified = -1, np.zeros(2, dtype=bool)
    jax_last, jax_qualified = jp.array(-1), jp.zeros(2, dtype=bool)
    advance = jax.jit(lambda last, lifted, h, f: update_liftoff(last, lifted, h, f, 20., jp))
    for heights, loads, expected in sequence:
        native_last, native_qualified, _ = update_liftoff(native_last, native_qualified,
            np.array(heights), np.array(loads), 20., np)
        jax_last, jax_qualified, _ = advance(jax_last, jax_qualified, jp.array(heights), jp.array(loads))
        np.testing.assert_array_equal(liftoff_features(native_last, native_qualified, np), expected)
        np.testing.assert_array_equal(liftoff_features(jax_last, jax_qualified, jp), expected)
    state.info['qualified_lift'] = jp.array([True, False])
    state = jax.jit(env.step)(state, jp.zeros(16))
    np.testing.assert_array_equal(state.obs['state'][-2:], [0., 1.])
    np.testing.assert_array_equal(state.obs['privileged_state'][111:113], [0., 1.])
    # A timeout must clear delay/filter/landing memory while keeping
    # the terminal reward/length and truncation required by PPO bootstrapping.
    wrapped = wrap_training(env, episode_length=2, action_repeat=1)
    initial = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(7), 2))
    initial.info['steps'] = jp.array([1., 0.])
    initial.info['action_history'] += .01
    initial.info['motor_targets'] += .01
    initial.info['last_act'] += .01
    initial.info['gait_velocity'] += .01
    initial.info['last_landed_foot'] = jp.zeros(2, dtype=jp.int32)
    initial.info['unload_best'] = jp.ones(2) * .8
    advance = jax.jit(wrapped.step)
    state = advance(initial, jp.zeros((2, 16)))
    np.testing.assert_array_equal(state.done, [1., 0.])
    np.testing.assert_array_equal(state.info['truncation'], [1., 0.])
    np.testing.assert_array_equal(state.info['episode_done'], [1., 0.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [1., 1.])
    for key in ('action_history', 'last_act', 'last_last_act', 'last_last_last_act',
                'gait_velocity', 'feet_air_time', 'qualified_lift'):
        np.testing.assert_array_equal(state.info[key][0], 0.)
    np.testing.assert_array_equal(state.info['last_landed_foot'], [-1, 0])
    np.testing.assert_allclose(state.info['motor_targets'][0], env._default_actuator, atol=1e-7)
    np.testing.assert_array_equal(state.obs['state'][0, -2:], [1., 1.])
    assert bool(jp.any(state.info['action_history'][1] != 0.))
    # Unlike cached task memory, RNG streams continue across episode resets.
    assert not bool(jp.all(state.info['rng'] == initial.info['rng']))
    state = advance(state, jp.zeros((2, 16)))
    np.testing.assert_array_equal(state.done, [0., 1.])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'], [1., 2.])
    np.testing.assert_array_equal(state.info['action_history'][1], 0.)
    np.testing.assert_array_equal(state.info['last_landed_foot'][1], -1)
    np.testing.assert_array_equal(state.info['unload_best'][1], 0.)
    print('PASS: dense unloading, support/no-jump/zero-command/alternation/hover gates, '
          'observed landing memory, native/JIT parity, complete controller reset with '
          'PPO timeout accounting and actual JIT physics; not walking acceptance', flush=True)


def check_unload_progress():
    gamma, dt = .99, .02
    # The old static state earns nearly full credit without any clearance.
    stationary = lift_curriculum_terms(jp.array([.01, 0., 0.]), jp.zeros(2),
        jp.array([.1, 19.9]), jp.zeros(2), 20., jp.ones(2, dtype=bool))
    assert float(stationary['unload']) > .98 and float(stationary['lift_dense']) == 0.
    repeated, _ = unload_progress_reward(.8, .8, False, dt)
    assert float(repeated) == 0.
    for potentials in ([.8] * 500, [.8, 0.] * 50, [.2, .6, .9], [.99]):
        previous, total = 0., 0.
        for potential in potentials:
            reward, previous = unload_progress_reward(potential, previous, False, dt)
            total += float(reward)
        np.testing.assert_allclose(total, max(potentials), atol=2e-6)
        reward, previous = unload_progress_reward(1., previous, True, dt)
        assert float(reward) == 0. and float(previous) == 0.
        assert total * 8. * dt <= .16 < 2.
    print('PASS: holding and rocking cannot repeat bounded unloading credit; terminal pays zero', flush=True)


def check_reference_course():
    """Reject wrong-foot credit without losing the Placo phase observation."""
    env = Joystick(lift_curriculum=True, position_feedback_gain=3., position_feedback_damping=.03)
    state = env.reset(jax.random.PRNGKey(0))
    assert env._config.reward_config.scales.alive == 0.
    assert env._config.reward_config.scales.imitation > 0.
    assert env._allow_negative_rewards
    np.testing.assert_array_equal(state.obs['state'][-2:], state.info['imitation_phase'])
    np.testing.assert_array_equal(state.obs['state'][9:11], [1., 1.])
    # The slow 3.33 mm/s stepping command used to trigger the upstream
    # <10 mm/s standing penalty, which punishes the demonstrated joint motion.
    moving_data = state.data.replace(qvel=state.data.qvel.at[env.actuator_qvel_addr[0]].set(1.))
    for command in ([max(env.PRM.dxs), 0., 0.], [.001, 0., 0.], [0., 0., .001], [0., 0., 0.]):
        info = dict(state.info, command=jp.array(command))
        rewards = env._get_reward(moving_data, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                                  jp.zeros(2, dtype=bool), jp.ones(2, dtype=bool))
        if any(command):
            np.testing.assert_allclose(rewards['stand_still'], 0.)
        else:
            assert float(rewards['stand_still']) >= 1.
    print('PASS: active slow forward/yaw commands do not request standing; zero command still penalizes motion.', flush=True)
    foot_loads = env.foot_loads
    env.foot_loads = lambda data: jp.array([0., env._torso_mass * 9.81])
    raised = state.data.replace(site_xpos=state.data.site_xpos.at[env._feet_site_id[0], 2].add(.012))
    for desired, expected in (([False, True], 1.), ([True, False], 0.), ([True, True], 0.)):
        phase = int(np.flatnonzero(np.all(np.asarray(env._reference_contacts) == desired, axis=1))[0])
        info = dict(state.info, imitation_i=jp.array(phase), command=jp.array([max(env.PRM.dxs), 0., 0.]),
                    feet_air_time=jp.array([.1, 0.]))
        rewards = env._get_reward(raised, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                                  jp.zeros(2, dtype=bool), jp.array([False, True]))
        np.testing.assert_allclose(rewards['lift_dense'], expected, atol=1e-5)
        assert float(rewards['support_contact']) <= 0.
    # Same pose, load and phase used to conceal opposite reward eligibility.
    phase = int(np.flatnonzero(~np.asarray(env._reference_contacts)[:, 0])[0])
    snapshots, legacy = [], []
    for last_landed in (0, 1):
        info = dict(state.info, imitation_i=jp.array(phase), last_landed_foot=jp.array(last_landed),
                    command=jp.array([max(env.PRM.dxs), 0., 0., 0., 0., 0., 0.]),
                    feet_air_time=jp.array([.1, 0.]))
        obs = env._get_obs(raised, info, jp.array([False, True]))
        legacy.append(obs['state'])
        snapshot = state.replace(data=raised, info=info, obs=obs)
        env._observe_liftoff(snapshot)
        rewards = env._get_reward(raised, jp.zeros(16), info, dict(state.metrics), jp.array(False),
                                  jp.zeros(2, dtype=bool), jp.array([False, True]))
        snapshots.append(snapshot)
        np.testing.assert_allclose(rewards['lift_dense'], float(last_landed == 1), atol=1e-5)
    np.testing.assert_array_equal(legacy[0], legacy[1])
    np.testing.assert_array_equal(snapshots[0].obs['state'][9:11], [0., 1.])
    np.testing.assert_array_equal(snapshots[1].obs['state'][9:11], [1., 0.])
    for snapshot in snapshots:
        np.testing.assert_array_equal(snapshot.obs['privileged_state'][:113], snapshot.obs['state'])
        np.testing.assert_array_equal(snapshot.obs['state'][-2:], state.info['imitation_phase'])
        np.testing.assert_array_equal(snapshot.info['command'][3:], 0.)
    env.foot_loads = foot_loads
    advanced = jax.jit(env.step)(state, jp.zeros(16))
    np.testing.assert_allclose(advanced.obs['state'][-2:], advanced.info['imitation_phase'])
    np.testing.assert_array_equal(advanced.obs['state'][9:11],
        liftoff_features(advanced.info['last_landed_foot'], advanced.info['qualified_lift'], np))
    assert bool(jp.isfinite(advanced.reward))
    print('PASS: Placo lift course preserves its clock, rejects wrong-foot credit, '
          'has no survival payment and executes JIT physics.', flush=True)


def check_reference_reset(guided=False):
    env = Joystick(lift_curriculum=True, position_feedback_gain=1. if guided else 3., position_feedback_damping=.03,
                   torque_limit_scale=2., reference_reset_fraction=1., coordinated_lift=guided,
                   reference_action=guided, reference_feedforward=guided)
    speed = max(env.PRM.dxs)
    env._commands = jp.array([[speed, 0., 0.], [speed, 0., 0.]])
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(3), 32))
    phases = np.asarray(states.info['imitation_i'])
    expected_memory = jax.vmap(lambda last, qualified: liftoff_features(last, qualified, jp))(
        states.info['last_landed_foot'], states.info['qualified_lift'])
    np.testing.assert_array_equal(states.obs['state'][:, 9:11], expected_memory)
    assert len(np.unique(phases)) > 8
    qa = env.mj_model.jnt_qposadr[env.actuator_joint_ids]
    np.testing.assert_allclose(np.asarray(states.data.qpos)[:, qa],
        np.asarray(env._reference_reset_qpos)[phases][:, qa], atol=2e-7)
    np.testing.assert_allclose(np.asarray(states.obs['state'])[:, -2:],
        np.column_stack([np.cos(2*np.pi*phases/env.PRM.nb_steps_in_period),
                         np.sin(2*np.pi*phases/env.PRM.nb_steps_in_period)]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(states.data.qvel)[:, 3:],
        np.asarray(env._reference_reset_qvel)[phases, 3:], atol=2e-7)
    selected = None
    for i, phase in enumerate(phases):
        contacts = np.asarray(env._reference_contacts)[phase]
        data = mujoco.MjData(env.mj_model)
        data.qpos[:] = np.asarray(states.data.qpos[i])
        mujoco.mj_forward(env.mj_model, data)
        heights = [np.min((np.asarray(v) @ data.site_xmat[site].reshape(3,3).T
                           + data.site_xpos[site])[:, 2])
                   for site, v in zip(env._feet_site_id, env._foot_vertices)]
        np.testing.assert_allclose(heights, np.asarray(env._reference_heights)[phase], atol=1e-5)
        if not contacts.all():
            selected = i
            airborne = int(np.argmax(~contacts))
            assert int(states.info['last_landed_foot'][i]) == airborne
            allowed = np.arange(2) != airborne
            terms = lift_curriculum_terms(jp.array([speed, 0., 0.]), jp.array(heights),
                jp.array(contacts) * env._torso_mass * 9.81, jp.zeros(2),
                env._torso_mass * 9.81, jp.array(allowed))
            np.testing.assert_allclose(terms['lift_dense'], 0., atol=1e-6)
    assert selected is not None
    state = jax.tree.map(lambda value: value[selected], states)
    queued = state.info['action_history'].reshape((-1, env.action_size))
    if guided:
        from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
        metadata = json.loads((Path(__file__).parent / 'data/reference.json').read_text())
        reference = PolyReferenceMotion(env._reference_path)
        order = np.array([metadata['joint_names'].index(n) for n in env.actuator_names])
        times, torques, _ = reference_torque_inputs(metadata)
        phase = int(state.info['imitation_i'])
        def motor_at(index):
            frame = reference.get_reference_motion(speed, 0., 0., index)
            motor = frame[order] + periodic_reference_torque(index*.02, times, torques, np) / np.asarray(env._motor_gains)
            return np.clip(np.where(env._leg_mask, motor, env._default_actuator), env._target_lowers, env._target_uppers)
        assert np.asarray(states.info['reference_initialized']).all()
        assert int(state.info['step']) == 0
        np.testing.assert_allclose(state.info['motor_targets'], motor_at(phase), atol=2e-5)
        for lag in range(3):
            np.testing.assert_allclose(env._default_actuator + queued[lag] * jp.array(env._config.action_scale),
                                       motor_at(phase+2-lag), atol=2e-5)
    else:
        np.testing.assert_allclose(env._default_actuator + queued[0] * jp.array(env._config.action_scale),
                                   state.info['motor_targets'], atol=3e-7)
    advanced = jax.jit(env.step)(state, jp.zeros(env.action_size))
    assert np.isfinite(np.asarray(advanced.data.qpos)).all() and np.isfinite(float(advanced.reward))
    assert not bool(advanced.done), 'Reference reset fails on its first physics step'
    if guided:
        np.testing.assert_allclose(advanced.info['motor_targets'], motor_at(phase+1), atol=2e-5)
        np.testing.assert_allclose(env._default_actuator + advanced.info['action_history'][:16] * jp.array(env._config.action_scale),
                                   motor_at(phase+3), atol=2e-5)
        initial_airborne = int(np.argmax(~np.asarray(env._reference_contacts)[phase]))
        assert float(advanced.metrics[('left', 'right')[initial_airborne] + '_eligible_contact_10mm_s']) == 0.
    env._commands = jp.zeros((1, 3))
    standing = jax.jit(env.reset)(jax.random.PRNGKey(4))
    assert int(standing.info['imitation_i']) == 0
    np.testing.assert_allclose(standing.data.qpos[qa], env._geometric_home, atol=2e-7)
    if guided:
        assert not bool(standing.info['reference_initialized'])
    print('PASS: moving reference reset preserves geometry, phase, velocities and queued commands; '
          'initial airborne feet cannot earn lift credit; standing reset and actual JIT step remain valid.', flush=True)


def check_lift_objective():
    def trial(height=.010, hover=1, terminal=False, up_z=1., support=10., repeat=False):
        last, qualified, peak = -1, np.zeros(2, dtype=bool), np.zeros(2)
        total = 0.
        frames = [([height,0.],[0.,20.])] * hover + [([0.,0.],[10.,support])] * 2
        if repeat:
            frames *= 2
        for h, f in frames:
            h, f = np.array(h), np.array(f)
            earned, peak = completed_lift_reward(qualified, peak, h, f, up_z, terminal, np)
            total += earned
            last, qualified, _ = update_liftoff(last,qualified,h,f,20.,np,
                minimum_support_force=.5,qualification_height=.010)
        return total
    assert trial(.009) == 0.
    assert trial(.010,1) == trial(.010,10) == trial(.015) == 1.
    assert trial(.010,repeat=True) == 1., 'Repeated same-side lift cannot earn twice'
    assert trial(terminal=True) == trial(up_z=np.cos(np.deg2rad(21.))) == trial(support=0.) == 0.
    assert 0. < trial(.020) < trial(.015), 'Higher lifts cannot increase completion credit'
    complete = jax.jit(lambda q,p,h,f,z,d,active:completed_lift_reward(q,p,h,f,z,d,jp,active=active))
    for up_z, terminal, active in ((1.,False,True),(1.,True,True),(.8,False,True),(1.,False,False)):
        args=(np.array([True,False]),np.array([.011,0.]),np.zeros(2),np.ones(2)*10.,up_z,terminal)
        expected=completed_lift_reward(*args,np,active=active)
        actual=complete(*args,active)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-7)
        if not active:assert float(actual[0]) == 0.
    print('PASS: one supported 10mm sample plus stable landing earns once; hovering longer, '
          'subthreshold/repeated lifts, missing support and falls cannot improve completion credit.',flush=True)
    gamma, dt, length = .99, .02, 500
    baseline = 6. * dt * sum(gamma**i for i in range(length))
    for stop in (0, 1, 25, 100, 499):
        removed = 6. * dt * sum(gamma**i for i in range(stop+1))
        future = float(placo_lift_terminal_cost(stop, length, dt) - 2.)
        np.testing.assert_allclose(removed + gamma**stop * future, baseline, rtol=1e-6)
    # A 15/s observed loss made immediate termination cheaper under the old
    # six-unit compensation. Strengthened drift costs require a new bound.
    keep_alive=-15.*dt*sum(gamma**i for i in range(length))
    assert -float(placo_lift_terminal_cost(0,length,dt)) > keep_alive
    for cost in (6.,15.,30.):
        alive=-cost*dt*sum(gamma**i for i in range(length))
        for stop in (0,1,25,100,499):
            early=(-cost*dt*sum(gamma**i for i in range(stop+1))
                   -gamma**stop*float(placo_lift_terminal_cost(stop,length,dt,baseline=30.)))
            assert early < alive, 'Falling must not improve this bounded negative-cost return'
    last,qualified=-1,np.zeros(2,dtype=bool)
    def goal(height,force):
        nonlocal last,qualified
        last,qualified,allowed=update_liftoff(last,qualified,np.array(height),np.array(force),20.,np,
            minimum_support_force=.5,qualification_height=.010)
        return allowed
    goal([0.,.011],[20.,0.]);goal([0.,0.],[10.,10.])
    assert last == 1
    goal([.009,0.],[0.,20.]);allowed=goal([0.,0.],[10.,10.])
    assert last == 1 and not allowed[1], 'A 9mm opposite lift must not rearm the successful side'
    goal([.010,0.],[0.,20.]);allowed=goal([0.,0.],[10.,10.])
    assert last == 0 and allowed[1], 'One supported 10mm sample and landing suffice; no hover required'
    # Even perfect velocity/contact matching (upstream maximum=5) cannot pay
    # grounded rocking. 10mm supported liftoff earns the full imitation credit;
    # lifting higher cannot increase it, and pose-error gradients are retained.
    assert float(placo_lift_imitation_reward(5., 0., True)) == 0.
    for credit in (0., .25, .5, 1., 2.):
        np.testing.assert_allclose(placo_lift_imitation_reward(4., credit, True),
                                   -1. + 5. * min(credit, 1.))
        assert float(jax.grad(placo_lift_imitation_reward)(4., credit, True)) == 1.
    assert float(placo_lift_imitation_reward(0., 0., False)) == 0.
    idle = lift_stability_terms(1., 0.)
    np.testing.assert_allclose(list(idle.values()), [0., 0.])
    for rate in (0., .1, .5, 5., 50.):
        yaw = float(lift_stability_terms(1., rate)['tracking_ang_vel'])
        assert -1. <= yaw <= 0.
    def roll_matrix(angle):
        c, s = np.cos(angle), np.sin(angle)
        return jp.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    for angle in (0., np.deg2rad(10.), -np.deg2rad(10.)):
        np.testing.assert_allclose(body_roll_angle(roll_matrix(angle)), angle, atol=1e-7)
        np.testing.assert_allclose(body_roll_cost(roll_matrix(angle)), 0., atol=1e-7)
    np.testing.assert_allclose(body_roll_cost(roll_matrix(np.deg2rad(15.))), 1., atol=5e-6)
    np.testing.assert_allclose(body_roll_cost(roll_matrix(np.deg2rad(20.))), 4., atol=5e-6)
    assert float(body_roll_cost(roll_matrix(np.deg2rad(40.)))) == 9.
    stable = lift_stability_terms(jp.cos(jp.deg2rad(15.)), .1)
    balance_cost = 2. * stable['upright'] + .2 * stable['tracking_ang_vel']
    assert float(balance_cost) < 0.
    lifted = lift_curriculum_terms(jp.array([.01, 0., 0.]), jp.array([.01, 0.]),
        jp.array([0., 20.]), jp.array([.1, 0.]), 20., jp.ones(2, dtype=bool))
    air = supported_air_time_reward(jp.array([.01, 0., 0.]), jp.zeros(3),
        jp.array([.01, 0.]), jp.array([0., 20.]), jp.array([.1, 0.]), 20.,
        require_progress=False, minimum_time=.02)
    step_credit = 6. * lifted['lift_dense'] + 6. * air
    assert float(balance_cost + step_credit) > 0.
    # A terminal frame can earn at most one supported foot's 6+6 credit.
    # The -2 event loss exceeds that maximum by more than seven times.
    assert float((balance_cost + step_credit) * .02 - 2.) < 0.
    print('PASS: zero idle bonus, stable 10mm step beats idle, terminal-frame '
          'liftoff cannot outweigh the fall penalty', flush=True)


def check_target_slew():
    env = Joystick(lift_curriculum=True, position_feedback_gain=3.,
                   position_feedback_damping=.03, torque_limit_scale=2.)
    assert env._target_slew_rate is None
    initial = env.reset(jax.random.PRNGKey(0))
    initial.info['command'] = jp.zeros_like(initial.info['command'])
    action = jp.zeros(16).at[0].set(.08)
    scale = jp.array(env._config.action_scale)
    observed = {}
    for name, slew in (('T0000', None), ('legacy', 5.24)):
        env._target_slew_rate = slew
        # Distinct closures trace the two static controller configurations.
        advance = jax.jit(lambda state, action: env.step(state, action))
        state = initial
        for tick in range(3):
            previous = state.info['motor_targets']
            state = advance(state, action)
            expected = jp.clip(env._default_actuator + state.info['action_history'].reshape(-1, 16)[2] * scale,
                               env._target_lowers, env._target_uppers)
            if slew is not None:
                expected = jp.clip(expected, previous - slew*env.dt, previous + slew*env.dt)
            np.testing.assert_allclose(state.info['motor_targets'], expected, atol=1e-7)
            assert bool(jp.all(jp.abs(state.data.actuator_force) <= env._effort_limits + 1e-5))
            if tick < 2:
                assert abs(float(state.info['motor_targets'][0] - env._default_actuator[0])) < 1e-7
        observed[name] = float(state.info['motor_targets'][0] - previous[0])
    assert observed['T0000'] > .2
    np.testing.assert_allclose(observed['legacy'], 5.24*env.dt, atol=1e-7)
    print('PASS: real JIT T0000 target step after the same two-tick delay; old ramp '
          'counterexample retained; angle/force caps unchanged:', observed, flush=True)


def check_coordinated_lift():
    # A rising/falling COM changes normal force. The current Placo peak implies
    # Fz/weight=0.4355: a grounded supporting foot is not a missing support.
    command = jp.array([.00333333, 0., 0.])
    heights, loads = jp.array([.012, 0.]), jp.array([0., .4355 * 19.36])
    air = jp.array([.1, 0.])
    assert float(lift_curriculum_terms(command, heights, loads, air, 19.36, True)['lift_dense']) == 0.
    np.testing.assert_allclose(lift_curriculum_terms(command, heights, loads, air, 19.36, True,
                                minimum_support_force=.5)['lift_dense'], 1.)
    np.testing.assert_allclose(supported_air_time_reward(command, jp.zeros(3), heights, loads, air,
        19.36, require_progress=False, minimum_support_force=.5, minimum_time=.02,
        clearance_height=.010), 1.)
    for bad_heights, bad_loads in ((heights*0., loads), (heights, loads*0.),
                                  (heights, jp.array([0., .4]))):
        assert float(lift_curriculum_terms(command, bad_heights, bad_loads, air, 19.36, True,
                     minimum_support_force=.5)['lift_dense']) == 0.
    args = (-1, np.zeros(2, dtype=bool), np.array(heights), np.array(loads), 19.36)
    assert not update_liftoff(*args, np)[1].any()
    np.testing.assert_array_equal(update_liftoff(*args, np, minimum_support_force=.5)[1], [True, False])
    np.testing.assert_array_equal(update_liftoff(*args, jp, minimum_support_force=.5)[1], [True, False])
    runs = command_liftoff_runs([heights], [loads], [.00333333], 19.36)
    np.testing.assert_array_equal(runs['active_command_longest_10mm_unloaded_sample_run_s'], [0., 0.])
    runs = command_liftoff_runs([heights], [loads], [.00333333], 19.36, minimum_support_force=.5)
    np.testing.assert_array_equal(runs['active_command_longest_10mm_unloaded_sample_run_s'], [.02, 0.])
    print('PASS: half-weight gate rejects a physically grounded Placo peak; actual-contact gate '
          'restores full 10mm credit; grounded feet, flight and negligible force earn no lift. '
          'Legacy metrics and NumPy/JAX event parity remain available.', flush=True)
    for error, expected in ((0., 0.), (.010, 1.), (.020, 4.), (.040, 9.)):
        np.testing.assert_allclose(base_height_cost(.15 + error, .15), expected, atol=1e-6)
        np.testing.assert_allclose(base_height_cost(.15 - error, .15), expected, atol=1e-6)
    assert float(jax.grad(base_height_cost)(.14, .15)) < 0.
    assert float(jax.grad(base_height_cost)(.16, .15)) > 0.
    # The old cap makes 0.5 and 5 rad/s equally cheap. Keep the small-rate
    # objective, but retain a bounded restoring gradient for excessive turning.
    yaw_cost = lambda rate: -lift_stability_terms(1., rate, linear_yaw_tail=True)['tracking_ang_vel']
    np.testing.assert_allclose(lift_stability_terms(1., .5)['tracking_ang_vel'],
                               lift_stability_terms(1., 5.)['tracking_ang_vel'])
    for rate, expected in ((0., 0.), (.25, .25), (.5, 1.), (1., 3.), (3., 11.)):
        np.testing.assert_allclose(yaw_cost(rate), expected)
        np.testing.assert_allclose(yaw_cost(-rate), expected)
    for rate in (1., 3., 50.):
        np.testing.assert_allclose(jax.grad(yaw_cost)(rate), 4.)
        np.testing.assert_allclose(jax.grad(yaw_cost)(-rate), -4.)
    print('PASS: capped excessive-turning loophole reproduced; symmetric linear '
          'tail keeps small-rate cost and a nonzero bounded gradient.', flush=True)
    np.testing.assert_allclose(lift_height_band(jp.array([.010, 0.])), 1.)
    np.testing.assert_allclose(lift_height_band(jp.array([0., .015])), 1.)
    np.testing.assert_allclose(lift_height_band(jp.array([.020, 0.])), np.exp(-1.), rtol=1e-6)
    reference = json.loads((Path(__file__).parent / 'data/reference.json').read_text())['reward_reference']
    heights = np.maximum(reference['whole_foot_height_m'], 0.)
    allowed = ~np.array(reference['contacts'], dtype=bool)
    def height_return(scale, bounded):
        actual = heights * scale
        credit = np.sum(allowed * np.clip(actual / .01, 0., 1.), axis=1)
        band = np.array(jax.vmap(lift_height_band)(jp.array(actual))) if bounded else 1.
        cost = -2 * np.mean(np.minimum(((actual-heights)/.012)**2, 9.), axis=1)
        return float(np.sum(17 * credit * band + cost) * .02)
    assert height_return(2., False) > height_return(1., False)
    assert height_return(2., True) < height_return(1., True)
    print('PASS: isolated 24mm prolonged-height reward loophole reproduced and removed; '
          '10-15mm height band retains full credit.', flush=True)
    gain = 1.
    env = Joystick(lift_curriculum=True, coordinated_lift=True, position_feedback_gain=gain,
                   position_feedback_damping=.03, torque_limit_scale=2.)
    scale = np.array(env._config.action_scale)
    action = np.linspace(-.05, .05, 16)
    command = np.array([max(env.PRM.dxs), 0., 0., 0., 0., 0., 0.])
    for angle in np.linspace(0., 2*np.pi, 61):
        phase = np.array([np.cos(angle), np.sin(angle)])
        expected = coordinated_lift_action(action, phase, command, scale, gain, np)
        actual = coordinated_lift_action(jp.array(action), jp.array(phase), jp.array(command), jp.array(scale), gain, jp)
        np.testing.assert_allclose(actual, expected, atol=2e-8)
        np.testing.assert_allclose(coordinated_lift_action(action, phase, command*0., scale, gain, np), action)
    phase = np.array([np.cos(2*np.pi*6/30), np.sin(2*np.pi*6/30)])
    assert np.all(np.asarray(reference['contacts'])[6]), 'Counterexample must be double support'
    old = coordinated_lift_action(action, phase, command, scale, gain, np)
    free = coordinated_lift_action(action, phase, command, scale, gain, np, quiet_swing_roll=False)
    assert abs(old[10]) < .1 * abs(action[10]), 'Old transfer blocking counterexample disappeared'
    np.testing.assert_array_equal(free[[1, 10]], action[[1, 10]])
    held = np.array([-.1, -.2])
    for qualified in (np.array([False, False]), np.array([False, True]), np.array([True, False])):
        expected = hold_cleared_roll(free, qualified, held, np.asarray(env._geometric_home),
            np.asarray(env._default_actuator), scale, gain, np)
        actual = hold_cleared_roll(jp.array(free), jp.array(qualified), jp.array(held),
            env._geometric_home, env._default_actuator, jp.array(scale), gain, jp)
        np.testing.assert_allclose(actual, expected, atol=2e-8)
        for side, roll in enumerate((1, 10)):
            if qualified[side]:
                ctrl = env._default_actuator[roll] + expected[roll]*scale[roll] + gain*(env._geometric_home[roll]-held[side])
                np.testing.assert_allclose(ctrl, held[side], atol=1e-7)
            else:
                np.testing.assert_allclose(expected[roll], action[roll])
    print('PASS: old double-support roll suppression reproduced; transfer rolls stay free; '
          'only physically qualified swing rolls hold their captured angle, NumPy/JAX parity.', flush=True)
    for sign, indices in ((1., (10, 11, 12, 13)), (-1., (1, 2, 3, 4))):
        roll, hip, knee, ankle = indices
        issued = coordinated_lift_action(np.zeros(16), np.array([0., sign]), command, scale, gain, np)
        posture = np.rad2deg(issued * scale / (1. + gain))
        np.testing.assert_allclose(posture[[roll, hip, knee, ankle]],
                                   [0., sign*5., 10., -5.], atol=1e-7)
    state = env.reset(jax.random.PRNGKey(0))
    state.info['imitation_phase'] = jp.array([0., 1.])
    state.info['command'] = jp.array(command)
    expected = jp.array(action)
    advance = jax.jit(env.step)
    advanced = advance(state, jp.array(action))
    np.testing.assert_allclose(advanced.info['last_act'], expected, atol=1e-7)
    assert bool(jp.isfinite(advanced.reward)) and bool(jp.isfinite(advanced.data.qpos).all())
    assert 'cost/joint_margin' in advanced.metrics
    phase = (advanced.info['imitation_i'] * len(env._reference_contacts)
             // env.PRM.nb_steps_in_period) % len(env._reference_contacts)
    np.testing.assert_allclose(advanced.metrics['cost/base_height'],
        2. * base_height_cost(advanced.data.qpos[2], env._reference_base_height[phase]), atol=1e-7)
    np.testing.assert_allclose(advanced.metrics['base_height_error_mm'],
        1000. * abs(advanced.data.qpos[2] - env._reference_base_height[phase]), atol=1e-5)
    standing = env.reset(jax.random.PRNGKey(0))
    standing.info['command'] = jp.zeros(7)
    standing = advance(standing, jp.zeros(16))
    np.testing.assert_allclose(standing.metrics['cost/base_height'], 0.)
    np.testing.assert_allclose(standing.metrics['base_height_error_mm'], 0.)
    print('PASS: Placo body-height error has no stationary bonus; same-phase target, '
          'cost/metric parity and zero-command exclusion pass a real JIT step.', flush=True)
    expected_yaw = .2 * lift_stability_terms(advanced.data.xmat[1, 2, 2],
        env.get_gyro(advanced.data)[2], linear_yaw_tail=True)['tracking_ang_vel']
    np.testing.assert_allclose(advanced.metrics['reward/tracking_ang_vel'], expected_yaw, atol=1e-7)
    np.testing.assert_array_equal(advanced.obs['state'][9:11],
        liftoff_features(advanced.info['last_landed_foot'], advanced.info['qualified_lift'], np))
    state = env.reset(jax.random.PRNGKey(0))
    state.info['qualified_lift'] = jp.array([False, True])
    state.info['swing_roll_hold'] = jp.array([-.1, -.2])
    expected = jp.array(action)
    expected = hold_cleared_roll(expected, state.info['qualified_lift'], state.info['swing_roll_hold'],
        env._geometric_home, env._default_actuator, jp.array(scale), gain, jp)
    advanced = advance(state, jp.array(action))
    np.testing.assert_allclose(advanced.info['last_act'], expected, atol=1e-7)
    print('PASS: legacy fold parity, unbiased policy actions, physical roll hold, '
          'action history, observed memory, joint-margin cost and actual JIT physics.', flush=True)


def check_reference_action(feedforward=False):
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
    env = Joystick(lift_curriculum=True, coordinated_lift=True, reference_action=True,
                   reference_feedforward=feedforward, position_feedback_gain=1.,
                   position_feedback_damping=.03, torque_limit_scale=2.)
    reference = PolyReferenceMotion(env._reference_path)
    metadata = json.loads((Path(__file__).parent / 'data/reference.json').read_text())
    order = [metadata['joint_names'].index(n) for n in env.actuator_names]
    home, scale, legs = np.asarray(env._geometric_home), np.array(env._config.action_scale), np.asarray(env._leg_mask)
    command = np.array([max(reference.dxs), 0., 0., 0., 0., 0., 0.])
    residual = np.linspace(-.02, .02, 16)
    if feedforward:
        times, torques, digest = reference_torque_inputs(metadata)
        assert digest == env.reference_torque_sha256
        lookup = jax.jit(lambda t: periodic_reference_torque(t, env._reference_torque_times,
                                                            env._reference_torques, jp))
        for t in np.linspace(-.6, 1.2, 151):
            np.testing.assert_allclose(lookup(t), periodic_reference_torque(t, times, torques, np), atol=1e-4)
        # Independent phase check: the audit guard offset must not shift which
        # leg is airborne. Its 200/500ms peaks match the fitted right/left peaks.
        audit = json.loads((Path(__file__).parent / 'training/reference_repair/fitted_audit.json').read_text())
        for index in (200, 500):
            np.testing.assert_allclose(audit['rows'][index]['whole_sole_clearance_mm'],
                np.array(metadata['reward_reference']['whole_foot_height_m'][index//20])*1000, atol=.003)

    def nominal(phase):
        frame = reference.get_reference_motion(*command[:3], phase + 3)
        current = reference.get_reference_motion(*command[:3], phase)
        offset = .03 * current[16+np.array(order)]
        if feedforward:
            tau = periodic_reference_torque((phase+3)*.02, times, torques, np)
            offset += tau / np.asarray(env._motor_gains) - np.asarray(env._home_motor_preload)
        return frame[order], offset, current[order], current[16+np.array(order)]

    for phase in range(reference.nb_steps_in_period):
        target, offset, current, velocity = nominal(phase)
        np.testing.assert_allclose(env.PRM.get_reference_motion(*command[:3], phase + 3)[env._reference_actuator_order],
                                   target, atol=1e-6)
        for elapsed in (0., .6, 1.2, 2.):
            actual = reference_posture_action(jp.array(residual), jp.array(target), env._geometric_home,
                jp.array(scale), env._leg_mask, 1., jp.array(command), elapsed, jp, jp.array(offset), jp.array(current))
            expected = reference_posture_action(residual, target, home, scale, legs, 1., command, elapsed, np, offset, current)
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            np.testing.assert_array_equal(np.asarray(actual)[~legs], np.asarray(jp.array(residual))[~legs])
        for cmd, elapsed in ((command*0., 2.), (command, 0.)):
            np.testing.assert_allclose(reference_posture_action(residual, target, home, scale, legs,
                                       1., cmd, elapsed, np, offset, current), residual)
        # Moving perfect tracking has q(t), not q(t+delay). Feedback must not
        # inject a deterministic position/velocity phase-lead torque.
        issued = reference_posture_action(np.zeros(16), target, home, scale, legs, 1., command, 2., np, offset, current)
        motor = env._default_actuator + issued*scale + legs*(home-current-.03*velocity)
        if feedforward:
            desired_motor = target + periodic_reference_torque((phase+3)*.02, times, torques, np) / env._motor_gains
        else:
            desired_motor = target + env._default_actuator-home
        np.testing.assert_allclose(np.asarray(motor)[legs], desired_motor[legs], atol=2e-7)
    state = env.reset(jax.random.PRNGKey(0))
    state.info['step'] = jp.array(60)
    state.info['imitation_i'] = jp.array(7)
    state.info['command'] = jp.array(command)
    advance = jax.jit(env.step)
    first_queued = None
    for tick in range(3):
        target, offset, current, _ = nominal(7+tick)
        expected = reference_posture_action(residual, target, home, scale, legs, 1., command,
                                            (60+tick)*env.dt, np, offset, current)
        if tick == 0:
            first_queued = np.asarray(env.command_action(state.data, jp.array(expected)))
        state = advance(state, jp.array(residual))
        np.testing.assert_allclose(state.info['last_act'], expected, atol=2e-5)
        assert bool(jp.isfinite(state.reward)) and bool(jp.isfinite(state.data.qpos).all())
    np.testing.assert_allclose(state.info['motor_targets'],
        np.clip(env._default_actuator + first_queued*scale, env._target_lowers, env._target_uppers), atol=2e-5)
    assert bool(jp.all(jp.abs(state.data.actuator_force) <= env._effort_limits + 1e-5))
    print('PASS: nominal Placo postures, all-phase NumPy/JAX reference/controller parity, '
          'startup/zero-command identity, head exclusion, current-phase moving feedback, actual JIT queue delay '
          'and angle/force bounds; no physical liftoff claimed.', flush=True)
    if feedforward:
        print('PASS: audited torque phase/hash, periodic NumPy/JAX interpolation, static-preload '
              'replacement and software-damping compensation reproduce reference+tau/kp; '
              'real delayed JIT commands retain the physical caps.', flush=True)


def check_training_stream():
    class Data(NamedTuple):
        qpos: object
        time: object

    class Probe:
        def reset(self, rng):
            return mjx_env.State(Data(jp.arange(23, dtype=jp.float32), jp.array(0.)),
                jp.zeros(1), jp.array(0.), jp.array(0.), {'probe': jp.array(0.)},
                {'rng': rng, 'reference_initialized': jp.array(True)})

        def step(self, state, action):
            return state.replace(data=Data(state.data.qpos + action[0], state.data.time + .02),
                                 reward=jp.array(3.))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'training_state.json'
        env = wrap_training(Probe(), episode_length=3, stream_path=path, stream_envs=512, prior_steps=1000)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 512))
        advance = jax.jit(env.step)
        for tick in range(1, 4):
            state = advance(state, jp.ones((512, 1)) * .01)
            jax.effects_barrier()
            frame = json.loads(path.read_text())
            assert frame['steps'] == 1000 + tick * 512
            np.testing.assert_array_equal(frame['qpos'], state.data.qpos[0])
            assert frame['reset'] == (tick == 3)
            assert frame['reference_initialized']
            assert int(state.info['training_stream_tick'][0]) == tick
        before = path.read_bytes()
        evaluation = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(1), 4))
        evaluation = advance(evaluation, jp.zeros((4, 1)))
        jax.effects_barrier()
        assert path.read_bytes() == before, 'Evaluation overwrote the actual training stream'
    print('PASS: actual batched JIT training states published atomically, correct step/reset accounting, '
          'evaluation excluded; no looped policy replay.', flush=True)


if __name__ == '__main__':
    if '--completion-only' in sys.argv:
        check_lift_objective()
    elif '--drift-only' in sys.argv:
        check_randomized_course(check_sampled_reference(drift=True))
    elif '--sampled-only' in sys.argv:
        check_lift_objective()
        check_randomized_course(check_sampled_reference())
    elif '--randomization-only' in sys.argv:
        check_randomized_course(Joystick(position_feedback_gain=1.1275290085055225,
            position_feedback_damping=.07053243703737014, torque_limit_scale=2.,
            lift_curriculum=True, coordinated_lift=True, reference_action=True,
            reference_feedforward=True, sampled_reference_control=True))
    elif '--guided-reference-reset-only' in sys.argv:
        check_reference_reset(guided=True)
    elif '--reference-feedforward-only' in sys.argv:
        check_reference_action(feedforward=True)
    elif '--reference-action-only' in sys.argv:
        check_reference_action()
    elif '--stream-only' in sys.argv:
        check_training_stream()
    elif '--target-slew-only' in sys.argv:
        check_target_slew()
    elif '--coordinated-only' in sys.argv:
        check_delayed_feedback()
        check_reference_heading()
        check_lift_objective()
        check_coordinated_lift()
    elif '--reference-course-only' in sys.argv:
        check_lift_objective()
        check_reference_course()
    elif '--reference-reset-only' in sys.argv:
        check_reference_reset()
    else:
        check_episode_reset() if '--reset-wrapper-only' in sys.argv else main()
