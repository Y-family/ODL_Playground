"""Current ZP10D adapter for the original reference-imitation walking task."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco.mjx._src import support

from playground.open_duck_mini_v2 import base, joystick as original
from playground.common.mjx_compat import install_static_mass_mask
from playground.open_duck_mini_zp10d.reference import ROOT, source_hashes

TRAINING_XML = Path('/tmp/zp10d-training/scene.xml')


def torque_margin_cost(torques, limits):
    """No cost below 80% of the provisional cap; quadratic cost approaching it."""
    return jp.sum(jp.square(jp.maximum((jp.abs(torques) / limits - .8) / .2, 0.)))


def gait_reward_terms(command, velocity, heights, foot_velocity, loads, contacts,
                      target_heights, target_contacts, target_loads, reference_lateral_velocity=0.):
    """Dense gait feedback; hard clearance/unloading thresholds are metrics only."""
    moving = jp.linalg.norm(command[:3]) > .01
    load_fraction = loads / jp.maximum(jp.sum(loads), .5)
    return {
        # Lateral reference motion transfers the COM between feet. Penalizing
        # its instantaneous speed against zero would fight the planned gait.
        'tracking_lin_vel': jp.exp(-jp.square((velocity[0] - command[0]) / .02)
                                  - jp.square((velocity[1] - command[1] - moving * reference_lateral_velocity) / .1)),
        'feet_clearance': moving * jp.mean(jp.minimum(jp.square((heights - target_heights) / .015), 9.)),
        'support_load': moving * jp.sum(jp.square(load_fraction - target_loads)),
        'support_contact': moving * jp.mean(contacts == target_contacts),
        # Use foot-site velocity, not base velocity. Normalize by the slow gait speed.
        'feet_slip': jp.sum(jp.linalg.norm(foot_velocity[:, :2], axis=1) * contacts) / .03333333,
    }


class Joystick(original.Joystick):
    def __init__(self, noise=0., pushes=False, reward_mode='optimized'):
        if jax.default_backend() == 'gpu':
            install_static_mass_mask()
        if reward_mode not in ('original', 'optimized'):
            raise ValueError(reward_mode)
        self.reward_mode = reward_mode
        metadata = json.loads(TRAINING_XML.with_name('manifest.json').read_text())
        if metadata['source_hashes'] != source_hashes():
            raise ValueError('Training contact model is stale; run training_model.py again')
        reference_meta = json.loads((ROOT / 'data/reference.json').read_text())
        if reference_meta['source_hashes'] != source_hashes():
            raise ValueError('Reference is stale; regenerate against the current source model')
        self._reference_path = str(ROOT / 'data/polynomial_coefficients.pkl')
        self._feet_geom_names = ['left_foot_contact', 'right_foot_contact']
        self._reset_joint_factor_min = self._reset_joint_factor_max = 1.
        config = original.default_config()
        config.noise_config.level = noise
        config.push_config.enable = pushes
        config.episode_length = 500
        if reward_mode == 'optimized':
            config.reward_config.scales.update(feet_clearance=-2., support_load=-2.,
                                               support_contact=1., feet_slip=-.25, torque_margin=-.2)
            auxiliary = reference_meta['reward_reference']
            digest = hashlib.sha256((ROOT / 'data/polynomial_coefficients.pkl').read_bytes()).hexdigest()
            if auxiliary['reference_sha256'] != digest:
                raise ValueError('Reward targets do not match the fitted reference')
            self._reference_contacts = jp.array(auxiliary['contacts'], dtype=bool)
            self._reference_heights = jp.maximum(jp.array(auxiliary['whole_foot_height_m']), 0.)
            self._reference_loads = jp.array(auxiliary['load_fraction'])
            self._foot_vertices = [jp.array(v) for v in auxiliary['foot_vertices_in_site']]
        # Same position-offset policy as upstream; scale covers this reference's
        # measured joint excursion. Gains/torque/physical limits are unchanged.
        with config.ignore_type():
            config.action_scale = reference_meta['action_scale']
        self._commands = jp.array(reference_meta['commands'], dtype=jp.float32)
        model = mujoco.MjModel.from_xml_path(str(TRAINING_XML))
        base.OpenDuckMiniV2Env.__init__(self, str(TRAINING_XML), config, model=model)
        self._post_init()
        ids = np.array(self.actuator_joint_ids)
        self._target_lowers = jp.array(model.jnt_range[ids, 0])
        self._target_uppers = jp.array(model.jnt_range[ids, 1])
        self._effort_limits = jp.array(np.max(abs(model.actuator_forcerange), axis=1))

    def reset(self, rng):
        state = super().reset(rng)
        if self.reward_mode == 'optimized':
            state.metrics.update(left_10mm_unloaded_s=jp.zeros(()), right_10mm_unloaded_s=jp.zeros(()))
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
        phase = info['imitation_i'] % len(self._reference_contacts)
        loads = self.foot_loads(data)
        heights = self.whole_foot_heights(data)
        rewards.update(gait_reward_terms(
            info['command'], self.get_local_linvel(data), heights,
            data.sensordata[self._foot_linvel_sensor_adr], loads, contact,
            self._reference_heights[phase], self._reference_contacts[phase], self._reference_loads[phase],
            info['current_reference_motion'][35]))
        rewards['torque_margin'] = torque_margin_cost(data.actuator_force, self._effort_limits)
        valid = ((heights >= .01) & (loads < .5) &
                 (loads[::-1] > .5 * self._torso_mass * 9.81))
        metrics['left_10mm_unloaded_s'] = valid[0] * self.dt
        metrics['right_10mm_unloaded_s'] = valid[1] * self.dt
        return rewards

    def reward_specification(self):
        if self.reward_mode == 'original':
            return dict(mode='original', scales=dict(self._config.reward_config.scales),
                        tracking_sigma=.01, lateral_deadband_m_s=.1)
        return dict(mode=self.reward_mode, scales=dict(self._config.reward_config.scales),
                    forward_velocity_std_m_s=.02 if self.reward_mode == 'optimized' else .1,
                    lateral_velocity_std_m_s=.1,
                    lateral_target='command plus phase-dependent Placo base lateral velocity',
                    clearance_normalizer_m=.015, slip_normalizer_m_s=.03333333,
                    support='exact Placo contacts; smooth load transfer during double support',
                    contact_force='MJX decoded floor reaction, N',
                    clearance='minimum over full CAD-derived foot collision vertices',
                    torque_margin_start_fraction=.8,
                    action_smoothing_and_yaw_weights='original; relaxed candidate regressed in full-model test',
                    survival_constant_unchanged=True)

    def sample_command(self, rng):
        command = self._commands[jax.random.randint(rng, (), 0, len(self._commands))]
        return jp.concatenate([command, jp.zeros(4)])

    def _get_termination(self, data):
        return super()._get_termination(data) | (data.qpos[2] < .10)
