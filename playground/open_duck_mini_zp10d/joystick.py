"""ZP10D reference task and optional direct locomotion with identified servos."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco.mjx._src import math, support
from mujoco_playground._src import mjx_env

from playground.open_duck_mini_v2 import base, joystick as original
from playground.common.mjx_compat import install_linesearch_optimization, install_static_mass_mask
from playground.open_duck_mini_zp10d.reference import ROOT, source_hashes, apply_reference_stance
from playground.open_duck_mini_zp10d.actuators import (configure, signature, position_feedback_action,
    hold_cleared_roll, planned_swing_roll_mask, suppress_swing_roll_residual,
    reference_posture_action, reference_torque_inputs,
    periodic_reference_torque, sampled_reference_inputs, sampled_reference_action)
from playground.open_duck_mini_zp10d.liftoff import update_liftoff, liftoff_features, completed_lift_reward
from playground.open_duck_mini_zp10d.contacts import install_native_sole_contacts

TRAINING_XML = Path('/tmp/zp10d-training/scene.xml')


def domain_randomize(model, rng):
    """Small contact/drive variations; keep the identified nominal model intact."""
    @jax.vmap
    def vary(key):
        contact_key, drive_key = jax.random.split(key)
        friction = model.geom_friction.at[:, 0].multiply(
            jax.random.uniform(contact_key, (), minval=.9, maxval=1.1))
        factor = jax.random.uniform(drive_key, (model.nu,), minval=.95, maxval=1.05)
        factor = jp.where(model.actuator_forcerange[:, 1] > .10001, factor, 1.)
        gain = model.actuator_gainprm.at[:, 0].multiply(factor)
        bias = model.actuator_biasprm.at[:, 1].multiply(factor)
        return friction, gain, bias
    friction, gain, bias = vary(rng)
    fields = dict(geom_friction=friction, actuator_gainprm=gain, actuator_biasprm=bias)
    axes = jax.tree_util.tree_map(lambda _: None, model)
    return model.tree_replace(fields), axes.tree_replace({name: 0 for name in fields})


def torque_margin_cost(torques, limits):
    """No cost below 80% of the provisional cap; quadratic cost approaching it."""
    return jp.sum(jp.square(jp.maximum((jp.abs(torques) / limits - .8) / .2, 0.)))


def lateral_com_offset(com, feet):
    """COM relative to the foot midpoint, invariant to translation and yaw."""
    lateral = feet[0, :2] - feet[1, :2]
    lateral /= jp.maximum(jp.linalg.norm(lateral), 1e-6)
    return jp.dot(com[:2] - jp.mean(feet[:, :2], axis=0), lateral)


def reference_frame_velocity(velocity, heading):
    """Express world translation in the planner's initial heading frame."""
    c, s = jp.cos(heading), jp.sin(heading)
    return velocity.at[:2].set(jp.array([c * velocity[0] + s * velocity[1],
                                        -s * velocity[0] + c * velocity[1]]))


def sagittal_balance_cost(rotation, target_pitch):
    """Penalize forward/backward tipping while allowing the planned roll transfer."""
    pitch = jp.arctan2(-rotation[2, 0], jp.linalg.norm(rotation[2, 1:]))
    return jp.minimum(jp.square((pitch - target_pitch) / jp.deg2rad(10.)), 9.)


BODY_ROLL_LIMIT_RAD = np.deg2rad(10.)
BODY_ROLL_SOFT_ZONE_RAD = np.deg2rad(5.)


def body_roll_angle(rotation):
    """Return torso roll around its local forward axis, independent of yaw."""
    return jp.arctan2(rotation[2, 1], rotation[2, 2])


def body_roll_cost(rotation, limit=BODY_ROLL_LIMIT_RAD, soft_zone=BODY_ROLL_SOFT_ZONE_RAD):
    """Softly limit absolute torso roll while preserving COM transfer below the limit."""
    excess = jp.maximum(jp.abs(body_roll_angle(rotation)) - limit, 0.)
    return jp.minimum(jp.square(excess / soft_zone), 9.)


def locomotion_tracking(command, velocity, reference_lateral_velocity=0., allow_overspeed=False):
    """Remove idle credit; gate gait credit on progress along the command.

    Microduck issue #46 motivates achieved-motion gating and lateral drift
    weighting. Subtracting the idle Gaussian follows microduck-lab's variant.
    Keep the planned lateral transfer; sideways drift is not forward progress.
    """
    speed = jp.linalg.norm(command[:2])
    walking = speed > 1e-6
    direction = jp.where(walking, command[:2] / jp.maximum(speed, 1e-6), jp.array([1., 0.]))
    adjusted = velocity[:2] - jp.array([0., walking * reference_lateral_velocity])
    along = jp.dot(adjusted, direction)
    sideways = jp.dot(adjusted - command[:2], jp.array([-direction[1], direction[0]]))
    # Subtracting the idle Gaussian otherwise gives zero credit above 2*speed,
    # even when the robot makes useful forward progress during a step.
    tracked = jp.where(walking & allow_overspeed, jp.minimum(along, speed), along)
    raw = jp.exp(-jp.square((tracked - speed) / .02))
    idle = jp.exp(-jp.square(speed / .02))
    advantage = (raw - idle) / jp.maximum(1. - idle, 1e-6)
    # Capped signed progress is concave: both symmetric rocking and slow
    # forward / fast backward returns earn no positive net progress. A
    # Gaussian's bounded negative tail cannot guarantee the latter.
    travel_reward = (jp.minimum(along / jp.maximum(speed, 1e-6), 1.)
                     if allow_overspeed else jp.maximum(advantage, 0.))
    tracking = jp.where(walking, travel_reward, raw)
    attenuation = jp.exp(-3. * jp.square(sideways / .1) - jp.square(velocity[2] / .1))
    # Sideways/vertical motion must not hide backward costs.
    tracking = jp.where(walking & allow_overspeed, jp.minimum(tracking, 0.)
                        + jp.maximum(tracking, 0.) * attenuation, tracking * attenuation)
    progress = walking & (jp.dot(velocity[:2], direction) >= .3 * speed)
    return tracking, progress


def overspeed_cost(command, velocity):
    speed = jp.linalg.norm(command[:2])
    along = jp.dot(velocity[:2], command[:2] / jp.maximum(speed, 1e-6))
    return jp.square(jp.maximum(along - speed, 0.) / .05)


def signed_forward_progress(command, velocity):
    """Signed command-axis speed; backward motion must carry a cost."""
    speed = jp.linalg.norm(command[:2])
    direction = command[:2] / jp.maximum(speed, 1e-6)
    return jp.where(speed > 1e-6, jp.clip(jp.dot(velocity[:2], direction)
                                           / jp.maximum(speed, 1e-6), -1., 1.), 0.)


def gait_reward_terms(command, velocity, heights, foot_velocity, loads, contacts,
                      target_heights, target_contacts, target_loads, reference_lateral_velocity=0.,
                      clearance_scale=.015, allow_overspeed=False):
    """Dense gait feedback; hard clearance/unloading thresholds are metrics only."""
    moving = jp.linalg.norm(command[:3]) > 1e-6
    load_fraction = loads / jp.maximum(jp.sum(loads), .5)
    tracking, progress = locomotion_tracking(command, velocity, reference_lateral_velocity, allow_overspeed)
    return {
        # Lateral reference motion transfers the COM between feet. Penalizing
        # its instantaneous speed against zero would fight the planned gait.
        'tracking_lin_vel': tracking,
        'feet_clearance': moving * jp.mean(jp.minimum(jp.square((heights - target_heights) / clearance_scale), 9.)),
        'support_load': moving * jp.sum(jp.square(load_fraction - target_loads)),
        'support_contact': moving * progress * jp.mean(contacts == target_contacts),
        # Use foot-site velocity, not base velocity. Normalize by the slow gait speed.
        'feet_slip': jp.sum(jp.linalg.norm(foot_velocity[:, :2], axis=1) * contacts) / .03333333,
    }


def supported_air_time_reward(command, velocity, heights, loads, air_time, weight,
                              require_progress=True, allowed=True, minimum_time=.125,
                              clearance_height=.005, minimum_support_force=None):
    """Only credit a cleared, unloaded foot with support and achieved progress."""
    _, progress = locomotion_tracking(command, velocity)
    if not require_progress:
        progress = jp.linalg.norm(command[:2]) > 1e-6
    support = .5 * weight if minimum_support_force is None else minimum_support_force
    valid = ((loads < .5) & (loads[::-1] > support)
             & (air_time > minimum_time) & (air_time < .3) & allowed)
    return progress * jp.sum(valid * jp.clip(heights / clearance_height, 0., 1.))


def lift_curriculum_terms(command, heights, loads, air_time, weight, allowed, minimum_support_force=None):
    """Bootstrap alternating unloading without a COM path or forward prerequisite."""
    support_force = .5 * weight if minimum_support_force is None else minimum_support_force
    active = ((jp.linalg.norm(command[:2]) > 1e-6) & allowed & (air_time < .3)
              & (loads[::-1] > support_force))
    unloading = jp.clip(1. - loads / (.5 * weight), 0., 1.)
    support = (jp.clip(loads[::-1] / (.8 * weight), 0., 1.)
               if minimum_support_force is None else (loads[::-1] > support_force))
    credit = active * unloading * support
    return dict(unload=jp.max(credit), lift_dense=jp.sum(credit * jp.clip(heights / .010, 0., 1.)))


def placo_lift_imitation_reward(imitation, lift_credit, moving):
    # Upstream imitation has <=5 positive velocity/contact units plus negative
    # joint tracking costs. Grounded rocking must not collect that baseline.
    return imitation - moving * 5. * (1. - jp.clip(lift_credit, 0., 1.))


def lift_height_band(heights):
    """Full height credit through 15mm, decaying above it instead of longer high lifts."""
    return jp.exp(-jp.sum(jp.square(jp.maximum(heights - .015, 0.) / .005)))


def base_height_cost(height, target):
    """Track the Placo body rise; crouching must not absorb swing clearance."""
    return jp.minimum(jp.square((height - target) / .010), 9.)


def placo_lift_terminal_cost(step, episode_length, dt, baseline=6.):
    # Compensate the discounted cost of remaining alive. Sampled training
    # includes drift costs as well as the removed imitation baseline.
    remaining = jp.maximum(episode_length - 1 - step, 0)
    return 2. + baseline * dt * .99 * (1. - .99**remaining) / (1. - .99)


def unload_progress_reward(potential, previous, terminal, dt):
    """Pay only new unloading maxima; total density credit per attempt <= 1."""
    best = jp.maximum(previous, potential)
    return jp.where(terminal, 0., best - previous), jp.where(terminal, 0., best)


def lift_stability_terms(up_z, yaw_rate, linear_yaw_tail=False):
    """Regularize stepping without paying a stationary policy to stay idle."""
    yaw = jp.minimum(jp.square(yaw_rate / .5), 1.)
    if linear_yaw_tail:
        yaw += 2. * jp.maximum(jp.abs(yaw_rate / .5) - 1., 0.)
    return dict(upright=jp.clip(up_z, -1., 1.) - 1.,
                tracking_ang_vel=-yaw)


class Joystick(original.Joystick):
    def __init__(self, noise=0., pushes=False, reward_mode='optimized', position_feedback_gain=0.,
                 direct_locomotion=False, position_feedback_damping=0., lift_curriculum=False,
                 torque_limit_scale=1., reference_reset_fraction=0., coordinated_lift=False,
                 reference_action=False, reference_feedforward=False, sampled_reference_control=False,
                 drift_observation=False, drift_cost_scale=2., forward_progress_scale=0.):
        if jax.default_backend() == 'gpu':
            install_static_mass_mask()
            install_linesearch_optimization()
        if reward_mode not in ('original', 'optimized'):
            raise ValueError(reward_mode)
        self.reward_mode = reward_mode
        self.position_feedback_gain = position_feedback_gain
        self.position_feedback_damping = position_feedback_damping
        self.direct_locomotion = direct_locomotion
        self.lift_curriculum = lift_curriculum
        if coordinated_lift and (direct_locomotion or not lift_curriculum
                                  or not position_feedback_gain or (reference_reset_fraction
                                      and not (reference_action and reference_feedforward))):
            raise ValueError('Coordinated lift requires feedback Placo; reference resets also require nominal posture/force feedforward')
        self.coordinated_lift = coordinated_lift
        if reference_action and not coordinated_lift:
            raise ValueError('Nominal Placo actions require the coordinated lift course')
        self.reference_action = reference_action
        if reference_feedforward and not reference_action:
            raise ValueError('Torque feedforward requires nominal Placo actions')
        self.reference_feedforward = reference_feedforward
        if sampled_reference_control and (not reference_feedforward or reference_reset_fraction):
            raise ValueError('Sampled Placo commands require torque feedforward and home-start episodes')
        self.sampled_reference_control = sampled_reference_control
        if not np.isfinite(drift_cost_scale) or drift_cost_scale <= 0:
            raise ValueError('drift_cost_scale must be positive and finite')
        self.drift_cost_scale = float(drift_cost_scale)
        if not np.isfinite(forward_progress_scale) or forward_progress_scale < 0:
            raise ValueError('forward_progress_scale must be finite and nonnegative')
        self.forward_progress_scale = float(forward_progress_scale)
        if drift_observation and not sampled_reference_control:
            raise ValueError('Drift observation requires the sampled Placo controller')
        self.drift_observation = drift_observation
        self._refresh_post_step_fields = sampled_reference_control
        self.minimum_support_force = .5 if coordinated_lift else None
        self.liftoff_qualification_height = .010 if sampled_reference_control else .002
        self.coordination_guidance_version = (13 if sampled_reference_control else 7 if reference_action else 4) if coordinated_lift else None
        # The identified T0000 response already models a step command. The
        # upstream target ramp would add an unmeasured second interpolation.
        self._target_slew_rate = None
        if not 0. <= reference_reset_fraction <= 1. or (reference_reset_fraction and
                (direct_locomotion or not lift_curriculum)):
            raise ValueError('Reference-state initialization requires the Placo lift course and a fraction in [0, 1]')
        self.reference_reset_fraction = reference_reset_fraction
        self._allow_negative_rewards = direct_locomotion or lift_curriculum
        if (direct_locomotion or lift_curriculum) and reward_mode != 'optimized':
            raise ValueError('Locomotion/liftoff curricula require optimized rewards')
        metadata = json.loads(TRAINING_XML.with_name('manifest.json').read_text())
        self.training_contact_metadata = dict(metadata)
        if metadata.get('training_xml_sha256') and metadata['training_xml_sha256'] != hashlib.sha256(TRAINING_XML.read_bytes()).hexdigest():
            raise ValueError('Training contact model does not match its manifest')
        if metadata['source_hashes'] != source_hashes():
            raise ValueError('Training contact model is stale; run training_model.py again')
        reference_meta = json.loads((ROOT / 'data/reference.json').read_text())
        if not direct_locomotion and reference_meta['source_hashes'] != source_hashes():
            raise ValueError('Reference is stale; regenerate against the current source model')
        self._reference_path = str(ROOT / 'data/polynomial_coefficients.pkl')
        self._feet_geom_names = ['left_foot_contact', 'right_foot_contact']
        self._reset_joint_factor_min = self._reset_joint_factor_max = 1.
        config = original.default_config()
        model = mujoco.MjModel.from_xml_path(str(TRAINING_XML))
        nominal_ctrl = model.key('home').ctrl.copy()
        nominal_effort = np.max(abs(model.actuator_forcerange), axis=1)
        self.actuator_parameters = configure(model, torque_limit_scale)
        if not direct_locomotion:
            apply_reference_stance(model, reference_meta)
        install_native_sole_contacts(model)
        contact_adapter = (ROOT / 'contacts.py').read_bytes()
        self.training_contact_signature = hashlib.sha256(TRAINING_XML.read_bytes() + contact_adapter).hexdigest()
        self.training_contact_metadata.update(
            sole_contact_method='MuJoCo 3.12 deepest hull vertex plus two distant graph neighbours; four padded slots',
            sole_contact_adapter_sha256=hashlib.sha256(contact_adapter).hexdigest())
        self.actuator_signature = signature()
        # Round the measured 36.15 ms path delay up to two 20 ms control ticks.
        # Unlike the upstream 0/20/40 ms lottery, every command sees this delay.
        delay = int(np.ceil(self.actuator_parameters['transport_delay_s'] / config.ctrl_dt))
        config.noise_config.action_min_delay = delay
        config.noise_config.action_max_delay = delay + 1
        config.noise_config.level = noise
        config.push_config.enable = pushes
        config.episode_length = 500
        if reward_mode == 'optimized':
            config.reward_config.scales.update(feet_clearance=-2., support_load=-2.,
                                               support_contact=1., feet_slip=-.25, torque_margin=0., torques=0.,
                                               com_transfer=-2., sagittal_balance=-1., tracking_lin_vel=4.)
            auxiliary = reference_meta['reward_reference']
            digest = hashlib.sha256((ROOT / 'data/polynomial_coefficients.pkl').read_bytes()).hexdigest()
            if auxiliary['reference_sha256'] != digest:
                raise ValueError('Reward targets do not match the fitted reference')
            self._reference_contacts = jp.array(auxiliary['contacts'], dtype=bool)
            self._reference_swing_roll_mask = jp.array(planned_swing_roll_mask(auxiliary['contacts']), dtype=bool)
            self._reference_heights = jp.maximum(jp.array(auxiliary['whole_foot_height_m']), 0.)
            self._clearance_scale = max(float(np.max(auxiliary['whole_foot_height_m'])), .002)
            self._reference_loads = jp.array(auxiliary['load_fraction'])
            self._reference_lateral_com = jp.array(auxiliary['lateral_com_from_foot_midpoint_m'])
            rotations = np.asarray(auxiliary['kinematic_base_transforms'])[:, :3, :3]
            rotation_dt = reference_meta['period'] / len(rotations)
            rotation_velocity = (np.roll(rotations, -1, axis=0)
                                 - np.roll(rotations, 1, axis=0)) / (2.*rotation_dt)
            angular = rotation_velocity @ rotations.transpose(0, 2, 1)
            self._reference_vertical_angular_velocity = jp.array(
                .5 * (angular[:, 1, 0] - angular[:, 0, 1]))
            self._reference_base_height = jp.array(np.asarray(auxiliary['kinematic_base_transforms'])[:, 2, 3])
            self._reference_pitch = jp.array(np.arctan2(-rotations[:, 2, 0],
                                                        np.linalg.norm(rotations[:, 2, 1:], axis=1)))
            self._com_transfer_scale = max(float(np.max(abs(np.asarray(self._reference_lateral_com)))), .005)
            self._foot_vertices = [jp.array(v) for v in auxiliary['foot_vertices_in_site']]
            if direct_locomotion:
                # Direct learning disables all trajectory targets below. Its
                # clearance measurement must still use the current CAD feet.
                from playground.open_duck_mini_zp10d.reference import foot_geometry
                self._foot_vertices = [jp.array(v) for v in foot_geometry()[0]]
                self._clearance_scale = .005
                config.reward_config.scales.update(alive=0., imitation=0., tracking_ang_vel=2.,
                    feet_clearance=-2., support_load=0., support_contact=0., com_transfer=0., sagittal_balance=0.,
                    upright=2., air_time=3., head_pose=-.2, termination=-5., overspeed=-.2)
                if lift_curriculum:
                    config.reward_config.scales.update(tracking_lin_vel=0., unload=8.,
                        lift_dense=6., air_time=6., tracking_ang_vel=.2,
                        termination=-2. / config.ctrl_dt)
            if lift_curriculum and not direct_locomotion:
                config.reward_config.scales.update(alive=0., tracking_lin_vel=0.,
                    unload=8., lift_dense=6., air_time=6., tracking_ang_vel=.2,
                    upright=2., head_pose=-.2, termination=-2. / config.ctrl_dt,
                    body_roll=-1.)
                if coordinated_lift:
                    config.reward_config.scales.update(feet_slip=-1., joint_margin=-1., base_height=-2.)
                if sampled_reference_control:
                    config.reward_config.scales.update(planar_drift=-self.drift_cost_scale, tracking_ang_vel=.5,
                                                      air_time=0., completed_step=2.,
                                                      forward_progress=self.forward_progress_scale)
        # Keep the upstream home-plus-action policy, but allow low-gain servos
        # to generate loaded torques as well as reach the kinematic trajectory.
        scale = np.array(reference_meta['action_scale'])
        leg = np.array([any(s in model.actuator(i).name for s in ('hip', 'knee', 'ankle'))
                        for i in range(model.nu)])
        scale[leg] += (abs(model.key('home').ctrl - nominal_ctrl)
                       + nominal_effort / model.actuator_gainprm[:, 0])[leg]
        # Feedback must follow the policy's desired posture rather than shrink
        # its reachable joint motion by 1/(1+gain). Physical limits still apply.
        scale[leg] *= 1. + position_feedback_gain
        with config.ignore_type():
            config.action_scale = scale.tolist()
        self._commands = jp.array(reference_meta['commands'], dtype=jp.float32)
        base.OpenDuckMiniV2Env.__init__(self, str(TRAINING_XML), config, model=model)
        self._post_init()
        if reference_action:
            self._reference_actuator_order = jp.array([reference_meta['joint_names'].index(n)
                                                       for n in self.actuator_names])
        if direct_locomotion:
            # Clock is an observation only; no fixed support/COM trajectory is
            # imposed. The official direct task learns its contact sequence.
            self.PRM.period = .5
            self.PRM.nb_steps_in_period = round(self.PRM.period / self.dt)
            self.PRM.data_array = jp.zeros_like(self.PRM.data_array)
        ids = np.array(self.actuator_joint_ids)
        self._target_lowers = jp.array(model.jnt_range[ids, 0])
        self._target_uppers = jp.array(model.jnt_range[ids, 1])
        self._effort_limits = jp.array(np.max(abs(model.actuator_forcerange), axis=1))
        self._geometric_home = jp.array(model.key('home').qpos[model.jnt_qposadr[ids]])
        self._leg_mask = jp.array(leg)
        if reference_feedforward:
            times, torques, self.reference_torque_sha256 = reference_torque_inputs(reference_meta)
            self._reference_torque_times, self._reference_torques = jp.array(times), jp.array(torques)
            self._motor_gains = jp.array(model.actuator_gainprm[:, 0])
            self._home_motor_preload = self._default_actuator - self._geometric_home
        self.sampled_reference_specification = None
        if sampled_reference_control:
            corrections, self.sampled_reference_specification = sampled_reference_inputs(
                reference_meta, self.actuator_names, model.key('home').qpos[model.jnt_qposadr[ids]],
                model.key('home').ctrl, model.actuator_gainprm[:, 0],
                position_feedback_gain, position_feedback_damping, stop_fade=True)
            self._sampled_reference_corrections = jp.array(corrections)
        if reference_reset_fraction:
            self._prepare_reference_resets(model, reference_meta)

    def _prepare_reference_resets(self, model, metadata):
        from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
        from playground.open_duck_mini_zp10d.reference import inputs

        reference = PolyReferenceMotion(self._reference_path)
        count = reference.nb_steps_in_period
        frames = np.array([reference.get_reference_motion(max(reference.dxs), 0., 0., i)
                           for i in range(count)])
        qpos = np.tile(model.key('home').qpos, (count, 1))
        names = inputs()[0]
        qpos[:, [model.joint(name).qposadr[0] for name in names]] = frames[:, :16]
        for i, transform in enumerate(metadata['reward_reference']['kinematic_base_transforms']):
            transform = np.asarray(transform)
            qpos[i, :3] = transform[:3, 3]
            mujoco.mju_mat2Quat(qpos[i, 3:7], transform[:3, :3].ravel())
        qvel = np.zeros((count, model.nv))
        for i in range(count):
            before, after = qpos[(i - 1) % count].copy(), qpos[(i + 1) % count].copy()
            # The joints/rotation repeat; forward translation advances each cycle.
            before[0] -= 2 * metadata['step_length'] if i == 0 else 0.
            after[0] += 2 * metadata['step_length'] if i == count - 1 else 0.
            mujoco.mj_differentiatePos(model, qvel[i], 2 * self.dt, before, after)
        qvel[:, [model.joint(name).dofadr[0] for name in names]] = frames[:, 16:32]
        self._reference_reset_qpos, self._reference_reset_qvel = jp.array(qpos), jp.array(qvel)
        self._reference_reset_frames = jp.array(frames)

    def _reset_from_reference(self, state, phase):
        qpos, qvel = self._reference_reset_qpos[phase], self._reference_reset_qvel[phase]
        rotation = math.quat_to_mat(state.data.qpos[3:7])
        qpos = qpos.at[:3].set(rotation @ (qpos[:3] - self._init_q[:3]) + state.data.qpos[:3])
        qpos = qpos.at[3:7].set(math.quat_mul(state.data.qpos[3:7], qpos[3:7]))
        qvel = qvel.at[:3].set(rotation @ qvel[:3])
        angles = self.get_actuator_joints_qpos(qpos)
        velocities = self.get_actuator_joints_qvel(qvel)
        target = jp.clip(angles + self._default_actuator - self._geometric_home,
                         self._target_lowers, self._target_uppers)
        action = ((target - self._default_actuator
                   - self._leg_mask * (self.position_feedback_gain * (self._geometric_home - angles)
                                      - self.position_feedback_damping * velocities))
                  / jp.asarray(self._config.action_scale))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=target)
        info = dict(state.info)
        contacts = self._reference_contacts[phase]
        # Initially airborne feet earn no lift credit. They must land, then
        # the other foot must actually lift before they become eligible again.
        info['last_landed_foot'] = jp.where(jp.all(contacts), -1, jp.argmax(~contacts)).astype(jp.int32)
        info['filtered_foot_load'] = self._reference_loads[phase] * self._torso_mass * 9.81
        info['imitation_i'] = phase
        angle = 2 * jp.pi * phase / self.PRM.nb_steps_in_period
        info['imitation_phase'] = jp.array([jp.cos(angle), jp.sin(angle)])
        info['current_reference_motion'] = self._reference_reset_frames[phase]
        info['motor_targets'] = target
        info['last_contact'] = contacts
        for key in ('last_act', 'last_last_act', 'last_last_last_act'):
            info[key] = action
        info['action_history'] = jp.tile(self.command_action(data, action),
                                         self._config.noise_config.action_max_delay)
        if self.reference_action:
            # A supplied moving state needs its periodic motor preload and
            # prior issued commands, rather than a standing startup ramp.
            scale = jp.asarray(self._config.action_scale)
            history, raw_history = [], []
            for lag in range(self._config.noise_config.action_max_delay):
                issued_phase = phase - 1 - lag
                past = self._reference_reset_frames[issued_phase % self.PRM.nb_steps_in_period]
                future = issued_phase + self._config.noise_config.action_min_delay + 1
                frame = self.PRM.get_reference_motion(*info['command'][:3], future)
                tau = periodic_reference_torque(future*self.dt, self._reference_torque_times,
                                                self._reference_torques, jp)
                motor = jp.where(self._leg_mask, frame[self._reference_actuator_order] + tau/self._motor_gains,
                                 self._default_actuator)
                command = (jp.clip(motor, self._target_lowers, self._target_uppers) - self._default_actuator) / scale
                history.append(command)
                raw_history.append(command - self._leg_mask * (
                    self.position_feedback_gain * (self._geometric_home - past[self._reference_actuator_order])
                    - self.position_feedback_damping * past[16+self._reference_actuator_order]) / scale)
            tau = periodic_reference_torque(phase*self.dt, self._reference_torque_times,
                                            self._reference_torques, jp)
            target = jp.clip(jp.where(self._leg_mask, angles + tau/self._motor_gains, self._default_actuator),
                             self._target_lowers, self._target_uppers)
            data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=target)
            info['motor_targets'] = target
            info['action_history'] = jp.concatenate(history)
            for key, previous in zip(('last_act', 'last_last_act', 'last_last_last_act'), raw_history):
                info[key] = previous
            info['reference_initialized'] = jp.array(True)
        obs = self._get_obs(data, info, contacts)
        return state.replace(data=data, info=info, obs=obs)

    def command_action(self, data, action):
        if not self.position_feedback_gain:
            return action
        return position_feedback_action(action, self.get_actuator_joints_qpos(data.qpos),
                                        self._geometric_home, jp.asarray(self._config.action_scale),
                                        self._leg_mask, self.position_feedback_gain,
                                        self.get_actuator_joints_qvel(data.qvel), self.position_feedback_damping)

    def reset(self, rng):
        state = super().reset(rng)
        rotation = math.quat_to_mat(state.data.qpos[3:7])
        state.info['reference_heading'] = jp.arctan2(rotation[1, 0], rotation[0, 0])
        if self.reward_mode == 'optimized':
            state.metrics.update(left_2mm_unloaded_s=jp.zeros(()), right_2mm_unloaded_s=jp.zeros(()),
                                 left_10mm_unloaded_s=jp.zeros(()), right_10mm_unloaded_s=jp.zeros(()),
                                 lateral_com_mm=jp.zeros(()), com_transfer_error_mm=jp.zeros(()))
        if self.direct_locomotion or self.sampled_reference_control:
            state.info['gait_velocity'] = jp.zeros(2)
        if self.sampled_reference_control:
            state.info['gait_yaw_rate'] = jp.zeros(())
            state.info['whole_foot_swing_peak'] = jp.zeros(2)
            state.metrics['completed_steps'] = jp.zeros(())
            state.info['course_command'] = state.info['command']
            state.info['rng'], key = jax.random.split(state.info['rng'])
            state.info['course_start_stop'] = jax.random.uniform(key) < .25
            self._schedule_training_command(state)
        if self.lift_curriculum:
            state.info['last_landed_foot'] = jp.array(-1, dtype=jp.int32)
            state.info['qualified_lift'] = jp.zeros(2, dtype=bool)
            state.info['unload_best'] = jp.zeros(())
            state.info['filtered_foot_load'] = jp.ones(2) * self._torso_mass * 9.81 / 2.
            if not self.direct_locomotion:
                state.metrics['body_roll_deg'] = jp.zeros(())
        if self.coordinated_lift:
            state.info['swing_roll_hold'] = self._geometric_home[jp.array([1, 10])]
            state.info['planned_swing_roll'] = jp.zeros(2, dtype=bool)
            state.metrics['base_height_error_mm'] = jp.zeros(())
            state.metrics.update(left_contact_10mm_s=jp.zeros(()), right_contact_10mm_s=jp.zeros(()))
            state.metrics.update(left_eligible_contact_10mm_s=jp.zeros(()), right_eligible_contact_10mm_s=jp.zeros(()))
        if self.reference_action:
            state.info['reference_initialized'] = jp.array(False)
        if self.sampled_reference_control:
            state.info['reference_motion_step'] = jp.array(0, dtype=jp.int32)
            state.info['reference_blend_ticks'] = jp.array(0, dtype=jp.int32)
        if self.reference_reset_fraction:
            state.info['rng'], choose, phase_key = jax.random.split(state.info['rng'], 3)
            use_reference = ((jax.random.uniform(choose) < self.reference_reset_fraction)
                             & (jp.linalg.norm(state.info['command'][:3]) > 1e-6))
            phase = jax.random.randint(phase_key, (), 0, self.PRM.nb_steps_in_period)
            state = jax.lax.cond(use_reference, lambda s: self._reset_from_reference(s, phase),
                                 lambda s: s, state)
        if self.lift_curriculum:
            self._observe_liftoff(state)
        return state

    def _schedule_training_command(self, state):
        # One quarter of episodes: 1s stand, 5s stepping, then 4s recovery.
        active = (state.info['step'] >= 50) & (state.info['step'] < 300)
        command = jp.where(state.info['course_start_stop'],
            jp.where(active, state.info['course_command'], jp.zeros_like(state.info['command'])),
            state.info['command'])
        state.info['command'] = command
        # Announce the next command before the policy chooses its next action.
        for name in ('state', 'privileged_state'):
            state.obs[name] = state.obs[name].at[6:9].set(command[:3])

    def _observe_liftoff(self, state):
        features = liftoff_features(state.info['last_landed_foot'], state.info['qualified_lift'], jp)
        size = state.obs['state'].shape[0]
        slots = slice(size-2, size) if self.direct_locomotion else slice(9, 11)
        state.obs['state'] = state.obs['state'].at[slots].set(features)
        state.obs['privileged_state'] = state.obs['privileged_state'].at[slots].set(features)
        if self.drift_observation:
            # Two unused head-command slots expose the same filtered velocity
            # error in the current body heading, so corrective actions rotate
            # with the robot rather than the episode's initial world heading.
            rotation = state.data.xmat[1]
            yaw = jp.arctan2(rotation[1, 0], rotation[0, 0])
            features = reference_frame_velocity(state.info['gait_velocity'],
                yaw - state.info['reference_heading']) / .010
            for name in ('state', 'privileged_state'):
                state.obs[name] = state.obs[name].at[11:13].set(features)

    def step(self, state, action):
        if self.sampled_reference_control:
            moving = jp.linalg.norm(state.info['command'][:3]) > 1e-6
            blend_ticks = state.info['reference_blend_ticks']
            running = moving | (blend_ticks > 0)
            motion_step = jp.where(running, state.info['reference_motion_step'], 0)
            state.info['imitation_i'] = motion_step % self.PRM.nb_steps_in_period
            state.info['reference_motion_step'] = jp.where(running, motion_step + 1, 0)
            state.info['reference_blend_ticks'] = jp.clip(blend_ticks + jp.where(moving, 1, -1), 0, 60)
            # The mask includes the preceding double-support transfer, so only
            # the loaded support hip is allowed to add a learned COM shift.
            swing = running & self._reference_swing_roll_mask[
                motion_step % self._reference_swing_roll_mask.shape[0]]
            state.info['planned_swing_roll'] = swing
            nominal = sampled_reference_action(jp.zeros_like(action), state.info['imitation_i'],
                state.info['command'], motion_step*self.dt, self._sampled_reference_corrections,
                jp.asarray(self._config.action_scale), jp, blend_ticks=blend_ticks)
            action = sampled_reference_action(action, state.info['imitation_i'], state.info['command'],
                motion_step*self.dt, self._sampled_reference_corrections,
                jp.asarray(self._config.action_scale), jp, blend_ticks=blend_ticks)
            # Keep the measured Placo swing-roll motion needed for the foot
            # path, while removing the PPO residual on that hip.  The support
            # hip remains fully controllable for COM transfer.
            action = suppress_swing_roll_residual(action, nominal, swing, jp)
        elif self.reference_action:
            # A command issued now acts two ticks later. Aim for that tick's
            # end pose, before the same feedback, queue and physical bounds.
            future = state.info['imitation_i'] + self._config.noise_config.action_min_delay + 1
            reference = self.PRM.get_reference_motion(*state.info['command'][:3], future)
            current = self.PRM.get_reference_motion(*state.info['command'][:3], state.info['imitation_i'])
            motor_offset = self.position_feedback_damping * current[16 + self._reference_actuator_order]
            if self.reference_feedforward:
                tau = periodic_reference_torque(future * self.dt, self._reference_torque_times,
                                                self._reference_torques, jp)
                motor_offset += tau / self._motor_gains - self._home_motor_preload
            action = reference_posture_action(action, reference[self._reference_actuator_order],
                self._geometric_home, jp.asarray(self._config.action_scale), self._leg_mask,
                self.position_feedback_gain, state.info['command'],
                state.info['step'] * self.dt + 1.2 * state.info['reference_initialized'],
                jp, motor_offset, current[self._reference_actuator_order])
        if self.coordinated_lift and not self.sampled_reference_control:
            # Placo supplies the imitation target. An extra sin² knee fold
            # conflicted with its stance/swing joint timing and must not bias it.
            action = hold_cleared_roll(action, state.info['qualified_lift'], state.info['swing_roll_hold'],
                self._geometric_home, self._default_actuator, jp.asarray(self._config.action_scale),
                self.position_feedback_gain, jp)
        state = super().step(state, action)
        if self.sampled_reference_control:
            self._schedule_training_command(state)
        if self.lift_curriculum:
            # The reward advances landing memory after upstream builds obs.
            self._observe_liftoff(state)
        return state

    def foot_loads(self, data):
        """Actual upward floor reaction, using MJX's supported contact-force decoder."""
        forces, indices = support.contact_force_dim(self.mjx_model, data, 3)
        contact = data._impl.contact
        pairs = contact.geom[indices]
        world_z = jp.einsum('ni,ni->n', forces[:, :3], contact.frame[indices, :, 2])
        sign = jp.where(pairs[:, 0] == self._floor_geom_id, 1., -1.)
        upward = jp.maximum(world_z * sign, 0.) * (contact.dist[indices] <= 0.)
        return jp.stack([jp.sum(upward * (
            ((pairs[:, 0] == foot) & (pairs[:, 1] == self._floor_geom_id)) |
            ((pairs[:, 1] == foot) & (pairs[:, 0] == self._floor_geom_id))))
            for foot in self._feet_geom_id])

    def whole_foot_heights(self, data):
        return jp.stack([jp.min((vertices @ data.site_xmat[site].T + data.site_xpos[site])[:, 2])
                         for site, vertices in zip(self._feet_site_id, self._foot_vertices)])

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        if self.reward_mode == 'original':
            return rewards
        if not self.direct_locomotion:
            # Free-joint translation is world-frame, but every episode starts
            # at a random yaw. The Placo velocities use its original heading.
            rewards['imitation'] = original.reward_imitation(
                self.get_floating_base_qpos(data.qpos),
                reference_frame_velocity(self.get_floating_base_qvel(data.qvel), info['reference_heading']),
                self.get_actuator_joints_qpos(data.qpos), self.get_actuator_joints_qvel(data.qvel),
                contact, info['current_reference_motion'], info['command'],
                original.USE_IMITATION_REWARD, leg_indices=self._leg_indices)
        phase = (info['imitation_i'] * len(self._reference_contacts)
                 // self.PRM.nb_steps_in_period) % len(self._reference_contacts)
        loads = self.foot_loads(data)
        heights = self.whole_foot_heights(data)
        velocity = self.get_local_linvel(data)
        if self.direct_locomotion:
            # Penalize sustained lateral drift while allowing periodic transfer.
            alpha = self.dt / (jp.array([.2, 2.]) + self.dt)
            info['gait_velocity'] += alpha * (velocity[:2] - info['gait_velocity'])
            velocity = velocity.at[:2].set(info['gait_velocity'])
        rewards.update(gait_reward_terms(
            info['command'], velocity, heights,
            data.sensordata[self._foot_linvel_sensor_adr], loads, contact,
            self._reference_heights[phase], self._reference_contacts[phase], self._reference_loads[phase],
            info['current_reference_motion'][35], self._clearance_scale, self.direct_locomotion))
        if self.sampled_reference_control:
            rewards['forward_progress'] = signed_forward_progress(info['command'], velocity)
        if self.sampled_reference_control:
            world_velocity = self.get_floating_base_qvel(data.qvel)
            reference_velocity = reference_frame_velocity(world_velocity, info['reference_heading'])
            error = reference_velocity[:2] - info['current_reference_motion'][34:36]
            info['gait_velocity'] += self.dt/(1. + self.dt) * (error - info['gait_velocity'])
            drift = jp.abs(info['gait_velocity']/.010)
            # Keep distinguishing fast drift after 30 mm/s. The previous
            # cap made every faster backward/sideways slide equally costly.
            rewards['planar_drift'] = (jp.linalg.norm(info['command'][:3]) > 1e-6) * jp.sum(
                jp.where(drift <= 3., drift**2, 6.*drift - 9.))
        if self.direct_locomotion:
            # Official-style moving-foot clearance cost, without a phase
            # target that requires a particular transfer or step duration.
            foot_speed = jp.linalg.norm(data.sensordata[self._foot_linvel_sensor_adr][:, :2], axis=1)
            rewards['feet_clearance'] = (jp.linalg.norm(info['command'][:3]) > 1e-6) * jp.sum(
                jp.abs(heights - .005) * foot_speed)
        offset = lateral_com_offset(data.subtree_com[1], data.site_xpos[self._feet_site_id])
        target_com = self._reference_lateral_com[phase]
        transfer_scale = self._com_transfer_scale
        if self.direct_locomotion:
            rewards['imitation'] = jp.zeros(())
            roll = jp.arctan2(data.xmat[1, 2, 1], data.xmat[1, 2, 2])
            rewards['upright'] = jp.exp(-sagittal_balance_cost(data.xmat[1], 0.) - jp.square(roll / .6))
            rewards['air_time'] = supported_air_time_reward(info['command'], velocity, heights, loads,
                info['feet_air_time'], self._torso_mass * 9.81)
            rewards['head_pose'] = jp.sum(jp.square((self.get_actuator_joints_qpos(data.qpos)
                - self._geometric_home) * ~self._leg_mask))
            rewards['termination'] = done.astype(jp.float32)
            rewards['overspeed'] = overspeed_cost(info['command'], velocity)
        if self.lift_curriculum:
            rewards.update(lift_stability_terms(data.xmat[1, 2, 2], self.get_gyro(data)[2],
                                                linear_yaw_tail=self.coordinated_lift))
            if not self.direct_locomotion:
                # Keep the planned COM transfer, but stop the torso from using
                # excessive roll as a substitute for lifting the swing foot.
                rewards['body_roll'] = body_roll_cost(data.xmat[1])
                metrics['body_roll_deg'] = jp.rad2deg(jp.abs(body_roll_angle(data.xmat[1])))
            if self.sampled_reference_control:
                # The reference itself turns periodically during transfer.
                # Penalize sustained excess rotation, allowing that motion.
                target_yaw = jp.where(jp.linalg.norm(info['command'][:3]) > 1e-6,
                                      self._reference_vertical_angular_velocity[phase], 0.)
                yaw_error = self.get_global_angvel(data)[2] - target_yaw
                info['gait_yaw_rate'] += self.dt/(1. + self.dt) * (yaw_error - info['gait_yaw_rate'])
                yaw = jp.abs(info['gait_yaw_rate'] / .1)
                rewards['tracking_ang_vel'] = -jp.where(yaw <= 1., yaw**2, 2.*yaw - 1.)
            # Contact flicker while rocking a loaded sole is not a step.
            weight = self._torso_mass * 9.81
            previous_landed = info['last_landed_foot']
            previous_qualified = info['qualified_lift']
            if self.sampled_reference_control:
                completed, info['whole_foot_swing_peak'] = completed_lift_reward(
                    previous_qualified, info['whole_foot_swing_peak'], heights, loads,
                    data.xmat[1, 2, 2], done, jp,
                    active=(jp.linalg.norm(info['command'][:3]) > 1e-6)
                           | (info['reference_blend_ticks'] > 0))
                # Reward aggregation multiplies by dt: retain a fixed amount
                # per completed lift instead of paying for hovering longer.
                rewards['completed_step'] = completed / self.dt
                metrics['completed_steps'] = completed
            info['last_landed_foot'], info['qualified_lift'], allowed = update_liftoff(
                info['last_landed_foot'], info['qualified_lift'], heights, loads, weight, jp,
                minimum_support_force=self.minimum_support_force,
                qualification_height=self.liftoff_qualification_height)
            if self.coordinated_lift and not self.sampled_reference_control:
                angles = self.get_actuator_joints_qpos(data.qpos)[jp.array([1, 10])]
                info['swing_roll_hold'] = jp.where(info['qualified_lift'] & ~previous_qualified,
                                                  angles, info['swing_roll_hold'])
            if not self.direct_locomotion:
                allowed &= ~self._reference_contacts[phase]
                rewards['support_contact'] = (jp.linalg.norm(info['command'][:3]) > 1e-6) * jp.mean(
                    (contact == self._reference_contacts[phase]).astype(jp.float32) - 1.)
            rewards.update(lift_curriculum_terms(info['command'], heights, loads,
                info['feet_air_time'], weight, allowed, self.minimum_support_force))
            if self.coordinated_lift:
                rewards['lift_dense'] *= lift_height_band(heights)
            if not self.direct_locomotion:
                rewards['imitation'] = placo_lift_imitation_reward(
                    rewards['imitation'], rewards['lift_dense'],
                    jp.linalg.norm(info['command'][:3]) > 1e-6)
            info['filtered_foot_load'] += self.dt / (.1 + self.dt) * (loads - info['filtered_foot_load'])
            potential = lift_curriculum_terms(info['command'], heights, info['filtered_foot_load'],
                info['feet_air_time'], weight, allowed, self.minimum_support_force)['unload']
            info['unload_best'] = jp.where(info['last_landed_foot'] != previous_landed, 0., info['unload_best'])
            terminal = done | (info['step'] >= self._config.episode_length - 1)
            rewards['unload'], info['unload_best'] = unload_progress_reward(
                potential, info['unload_best'], terminal, self.dt)
            rewards['air_time'] = supported_air_time_reward(info['command'], velocity, heights, loads,
                info['feet_air_time'], weight, require_progress=False, allowed=allowed, minimum_time=.02,
                clearance_height=.005 if self.direct_locomotion else .010,
                minimum_support_force=self.minimum_support_force)
            if self.coordinated_lift:
                rewards['air_time'] *= lift_height_band(heights)
            rewards['head_pose'] = jp.sum(jp.square((self.get_actuator_joints_qpos(data.qpos)
                - self._geometric_home) * ~self._leg_mask))
            rewards['termination'] = done.astype(jp.float32)
            if not self.direct_locomotion:
                moving = jp.linalg.norm(info['command'][:3]) > 1e-6
                event_cost = placo_lift_terminal_cost(info['step'], self._config.episode_length, self.dt,
                                                     baseline=30. if self.sampled_reference_control else 6.)
                rewards['termination'] *= jp.where(moving, event_cost / 2., 1.)
        transfer_error = offset - target_com
        moving = jp.linalg.norm(info['command'][:3]) > 1e-6
        rewards['com_transfer'] = (moving & (not self.direct_locomotion)) * jp.minimum(jp.square(transfer_error / transfer_scale), 9.)
        target_pitch = jp.where(moving & (not self.direct_locomotion), self._reference_pitch[phase], 0.)
        rewards['sagittal_balance'] = sagittal_balance_cost(data.xmat[1], target_pitch)
        metrics['lateral_com_mm'] = offset * 1000
        metrics['com_transfer_error_mm'] = (moving & (not self.direct_locomotion)) * jp.abs(transfer_error) * 1000
        # A preload is a motor command, not the desired unloaded joint geometry.
        # The upstream <0.01 m/s deadband includes our active 0.00333 m/s gait.
        # Only a zero task command should request a stationary pose.
        rewards['stand_still'] = original.cost_stand_still(
            info['command'], self.get_actuator_joints_qpos(data.qpos),
            self.get_actuator_joints_qvel(data.qvel), self._geometric_home, ignore_head=False) * ~moving
        rewards['torque_margin'] = torque_margin_cost(data.actuator_force, self._effort_limits)
        if self.coordinated_lift:
            rewards['base_height'] = moving * base_height_cost(data.qpos[2], self._reference_base_height[phase])
            metrics['base_height_error_mm'] = moving * jp.abs(data.qpos[2] - self._reference_base_height[phase]) * 1000
            contact_supported = (heights >= .01) & (loads < .5) & (loads[::-1] > self.minimum_support_force)
            metrics['left_contact_10mm_s'] = contact_supported[0] * self.dt
            metrics['right_contact_10mm_s'] = contact_supported[1] * self.dt
            metrics['left_eligible_contact_10mm_s'] = (contact_supported & allowed)[0] * self.dt
            metrics['right_eligible_contact_10mm_s'] = (contact_supported & allowed)[1] * self.dt
            angles = self.get_actuator_joints_qpos(data.qpos)
            margin = jp.minimum(angles - self._target_lowers, self._target_uppers - angles)
            rewards['joint_margin'] = jp.sum(jp.square(jp.maximum(1. - margin / jp.deg2rad(3.), 0.)))
        valid = ((heights >= .01) & (loads < .5) &
                 (loads[::-1] > .5 * self._torso_mass * 9.81))
        metrics['left_10mm_unloaded_s'] = valid[0] * self.dt
        metrics['right_10mm_unloaded_s'] = valid[1] * self.dt
        valid_2mm = ((heights >= .002) & (loads < .5) &
                     (loads[::-1] > .5 * self._torso_mass * 9.81))
        metrics['left_2mm_unloaded_s'] = valid_2mm[0] * self.dt
        metrics['right_2mm_unloaded_s'] = valid_2mm[1] * self.dt
        return rewards

    def reward_specification(self):
        if self.reward_mode == 'original':
            return dict(mode='original', scales=dict(self._config.reward_config.scales),
                        tracking_sigma=.01, lateral_deadband_m_s=.1)
        return dict(mode=self.reward_mode, scales=dict(self._config.reward_config.scales),
                    recipe=('placo-sampled-lift-v7' if self.sampled_reference_control else 'placo-coordinated-lift-v7' if self.coordinated_lift else 'placo-lift-curriculum-v5' if self.lift_curriculum and not self.direct_locomotion else ('microduck-plato-imitation-v3-torque-unconstrained' if not self.direct_locomotion else ('microduck-lift-curriculum-v9' if self.lift_curriculum else 'microduck-progress-v6'))),
                    training_command_profiles='25%: stand 1s, step 5s, recover 4s; 75%: original continuous/idle episode; command appears in actor and critic before control' if self.sampled_reference_control else None,
                    planar_drift=(f'-{self.drift_cost_scale:g}*sum(Huber3(lowpass_1s(world_velocity_in_initial_heading-Placo_velocity)/.010)); quadratic through 30mm/s, linear beyond; active commands only'
                                  if self.sampled_reference_control else None),
                    planar_drift_cost_scale=self.drift_cost_scale if self.sampled_reference_control else None,
                    forward_progress_scale=self.forward_progress_scale if self.sampled_reference_control else None,
                    sampled_yaw='-0.5*Huber1(lowpass_1s(world_vertical_angular_velocity-Placo_world_vertical_angular_velocity)/.1); planned turning is free' if self.sampled_reference_control else None,
                    planar_drift_observation='lowpass_1s_reference_velocity_error_in_current_yaw_frame_over_.010_in_slots_11_13' if self.drift_observation else None,
                    imitation_translation_frame='world velocity rotated into initial episode heading; reference yaw not randomized',
                    stand_still_command_threshold=1e-6,
                    task_stage='alternating_liftoff' if self.lift_curriculum else 'locomotion',
                    lift_dense_full_height_m=.010 if self.lift_curriculum else None,
                    base_height_tracking='-2*min(((base_z-Placo_base_z)/.010)^2,9); active commands only; no idle payment' if self.coordinated_lift else None,
                    minimum_support_force_n=self.minimum_support_force,
                    liftoff_qualification_height_m=self.liftoff_qualification_height,
                    completed_step_reward='2 per supported landing after alternating supported >=10mm lift; active command or stop fade only; <=20deg landing tilt; no terminal payment; peak-height band full through 15mm, decays above' if self.sampled_reference_control else None,
                    air_time_payment='disabled; completion event replaces duration payment' if self.sampled_reference_control else 'legacy height-weighted air-time density',
                    support_credit='actual decoded opposite-foot floor contact >0.5N; no static weight-fraction multiplier; conservative half-weight metrics retained' if self.coordinated_lift else 'legacy half-weight gate and 80%-weight saturation',
                    lift_height_band='multiply lift/air-time and imitation lift credit by exp(-sum((max(h-.015,0)/.005)^2)); full at 10-15mm' if self.coordinated_lift else None,
                    unloading_reward='positive new unloading maximum only; <=1 per attempt; 0.1s load filter; rearm after qualified landing' if self.lift_curriculum else None,
                    lift_stability_formulas=('upright=up_z-1; yaw as sampled_yaw; zero idle bonus' if self.sampled_reference_control else 'upright=up_z-1; yaw=-Huber(yaw_rate/0.5), quadratic within 0.5rad/s and linear beyond; zero idle bonus' if self.coordinated_lift else 'upright=up_z-1; yaw=-min((yaw_rate/0.5)^2,1); zero idle bonus') if self.lift_curriculum else None,
                    terminal_event_cost=2. if self.lift_curriculum else None,
                    terminal_baseline_compensation=('30*dt*gamma*(1-gamma**remaining_steps)/(1-gamma); gamma=.99; covers strengthened sampled drift costs' if self.sampled_reference_control else '6*dt*gamma*(1-gamma**remaining_steps)/(1-gamma); gamma=.99') if self.lift_curriculum and not self.direct_locomotion else None,
                    air_time_requires_progress=not self.lift_curriculum,
                    liftoff_task_observation=('allowed_plus_2qualified_in_clock_slots' if self.direct_locomotion else 'allowed_plus_2qualified_in_head_command_slots') if self.lift_curriculum else None,
                    imitation_positive_credit='imitation - 5*(1-supported_lift_credit); zero command excluded; lift saturates at 10mm' if self.lift_curriculum and not self.direct_locomotion else None,
                    contact_matching_credit='nonpositive mismatch cost' if self.lift_curriculum and not self.direct_locomotion else None,
                    liftoff_alternation=('exclude last landed foot until opposite foot reaches supported 10mm then lands; one sample suffices; rewards end after 0.3s airborne' if self.sampled_reference_control else 'exclude last landed foot only after supported 2 mm whole-foot liftoff; rewards end after 0.3 s airborne') if self.lift_curriculum else None,
                    liftoff_phase='Placo swing foot only; preserve phase observations' if self.lift_curriculum and not self.direct_locomotion else None,
                    forward_progress_formula='min(signed_forward_speed/command_speed,1); backward cost unbounded' if self.direct_locomotion else None,
                    overspeed_progress_credit=self.direct_locomotion,
                    overspeed_cost_normalizer_m_s=.05 if self.direct_locomotion else None,
                    trajectory_imitation=not self.direct_locomotion,
                    torque_limit_reward='disabled for Placo liftoff training' if not self.direct_locomotion else 'physical actuator range retained',
                    support_schedule_only=False,
                    fixed_support_schedule=False if self.direct_locomotion else None,
                    standing_command_probability=.02 if self.direct_locomotion or self.lift_curriculum else .5,
                    gait_period_s=self.PRM.period,
                    negative_rewards=self._allow_negative_rewards,
                    lateral_drift_filter_s=2. if self.direct_locomotion else None,
                    forward_progress_filter_s=.2 if self.direct_locomotion else None,
                    supported_air_time_window_s=[.02 if self.lift_curriculum else .125, .3] if self.direct_locomotion or self.lift_curriculum else None,
                    supported_air_time_progress_fraction=.3 if self.direct_locomotion else None,
                    supported_air_time_clearance_scale_m=.005 if self.direct_locomotion else .010 if self.lift_curriculum else None,
                    forward_velocity_std_m_s=None if self.direct_locomotion else .02,
                    lateral_velocity_std_m_s=.1,
                    lateral_error_multiplier=3., vertical_velocity_std_m_s=.1,
                    velocity_idle_baseline_subtracted=not self.direct_locomotion,
                    support_contact_progress_fraction=None if self.direct_locomotion else .3,
                    progress='signed horizontal projection along command; no backward/sideways credit',
                    lateral_target='filtered cycle drift' if self.direct_locomotion else 'command plus phase-dependent Placo base lateral velocity',
                    clearance_normalizer_m=None if self.direct_locomotion else self._clearance_scale, slip_normalizer_m_s=.03333333,
                    moving_foot_clearance_target_m=.005 if self.direct_locomotion else None,
                    support='learned contact sequence; supported air-time feedback' if self.direct_locomotion else 'exact Placo contacts; smooth load transfer during double support',
                    com_transfer='disabled' if self.direct_locomotion else 'phase-dependent COM lateral offset from foot midpoint; yaw/translation invariant',
                    com_transfer_normalizer_m=None if self.direct_locomotion else self._com_transfer_scale,
                    sagittal_balance='Placo pitch plus upright cost; no idle bonus' if self.lift_curriculum and not self.direct_locomotion else 'vertical-axis deviation cost; no upright idle bonus' if self.lift_curriculum else 'upright pitch; roll tolerance .6 rad' if self.direct_locomotion else 'pitch error from Placo; zero-command target upright; roll remains free',
                    body_roll=('soft quadratic cost on absolute torso roll; free through ±10°, 5° soft zone, capped at 9; active Placo lift only'
                               if self.lift_curriculum and not self.direct_locomotion else None),
                    body_roll_limit_deg=10. if self.lift_curriculum and not self.direct_locomotion else None,
                    body_roll_soft_zone_deg=5. if self.lift_curriculum and not self.direct_locomotion else None,
                    sagittal_balance_normalizer_deg=None if self.lift_curriculum else 10.,
                    termination_tilt_deg=45.,
                    contact_force='MJX decoded floor reaction, N',
                    clearance='minimum over full CAD-derived foot collision vertices',
                    torque_margin_start_fraction=.8,
                    action_smoothing_and_yaw_weights='original smoothing; excess world vertical yaw cost .5' if self.sampled_reference_control else 'original smoothing; yaw-rate cost .2' if self.lift_curriculum else 'original smoothing; yaw 2' if self.direct_locomotion else 'original; relaxed candidate regressed in full-model test',
                    survival_constant_unchanged=not (self.direct_locomotion or self.lift_curriculum))

    def sample_command(self, rng):
        if (self.direct_locomotion or self.lift_curriculum) and len(self._commands) > 1:
            stand_rng, move_rng = jax.random.split(rng)
            index = jp.where(jax.random.uniform(stand_rng) < .02, 0,
                             jax.random.randint(move_rng, (), 1, len(self._commands)))
        else:
            index = jax.random.randint(rng, (), 0, len(self._commands))
        command = self._commands[index]
        return jp.concatenate([command, jp.zeros(4)])

    def _get_termination(self, data):
        # Match native evaluation; the upstream 90-degree cutoff rewards
        # survival after a pose has already failed our walking criterion.
        return (super()._get_termination(data) | (data.qpos[2] < .10)
                | (data.xmat[1, 2, 2] < jp.cos(jp.pi / 4)))
