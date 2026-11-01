"""The identified response used by both the learner and native validation.

The fit is bound to the adopted reference. Its bench mounting offset is not a
robot zero correction; motor force limits remain the unmeasured MJCF prior.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def position_feedback_action(action, angles, home, scale, legs, gain, velocities=0., damping=0.):
    """Motor-position residual plus home feedback, queued before transport delay."""
    return action + legs * (gain * (home - angles) - damping * velocities) / scale


def coordinated_lift_action(action, phase, command, scale, feedback_gain, xp, *, quiet_swing_roll=True):
    """Placo-phase swing guidance: quiet roll, shorten the leg, preserve pitch."""
    right = phase[1] >= 0.
    indices = xp.arange(16)
    roll, hip, knee, ankle = [xp.where(right, r, l) for l, r in ((1, 10), (2, 11), (3, 12), (4, 13))]
    blend = phase[1]**2 * (xp.linalg.norm(command[:3]) > 1e-6)
    # sin² joins continuously at the support exchange; the standing 1° inward
    # pose remains the neutral roll. Account for the existing feedback scaling.
    delta = xp.deg2rad(10.) * blend * (1. + feedback_gain)
    offset = ((indices == knee) + xp.where(right, .5, -.5) * (indices == hip)
              - .5 * (indices == ankle)) * delta / scale
    return action * (1. - blend * (indices == roll) * quiet_swing_roll) + offset


def hold_cleared_roll(action, qualified, held_angles, home, default, scale, gain, xp):
    """Hold the measured hip-roll angle after supported physical liftoff."""
    indices = xp.arange(16)
    for side, roll in enumerate((1, 10)):
        hold = (held_angles[side] - default[roll] + gain * (held_angles[side] - home[roll])) / scale[roll]
        action = xp.where((indices == roll) & qualified[side], hold, action)
    return action


def planned_swing_roll_mask(reference_contacts):
    """Choose the next swing foot during double support.

    Placo's contact schedule contains double-support samples before each
    swing.  During those samples the future swing hip must already be quiet so
    the opposite, loaded hip can transfer the COM.  For a single-support
    sample the non-contact foot is the swing foot directly.
    """
    contacts = np.asarray(reference_contacts, dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != 2 or not contacts.shape[0]:
        raise ValueError('reference_contacts must have shape (period, 2)')
    result = np.zeros_like(contacts)
    for phase in range(len(contacts)):
        for offset in range(len(contacts)):
            candidate = contacts[(phase + offset) % len(contacts)]
            if not bool(candidate.all()):
                result[phase] = ~candidate
                break
    return result


def suppress_swing_roll_residual(action, nominal, swing, xp):
    """Keep the Placo swing-roll command and remove only learned residual."""
    indices = xp.arange(16)
    for side, roll in enumerate((1, 10)):
        action = xp.where((indices == roll) & swing[side], nominal[roll], action)
    return action


def reference_posture_action(action, reference, home, scale, legs, gain, command, elapsed, xp,
                             motor_offset=0., feedback_reference=None):
    """Use Placo as the nominal leg posture; PPO supplies the loaded correction."""
    u = xp.clip(elapsed / 1.2, 0., 1.)
    ramp = u**3 * (10. - 15. * u + 6. * u**2)
    moving = xp.linalg.norm(command[:3]) > 1e-6
    action = action + moving * ramp * legs * (1. + gain) * (reference - home) / scale
    if feedback_reference is not None:
        # Predict the nominal motor command through the transport delay, but
        # compare measured angles with the reference at the measurement time.
        action = action + moving * ramp * legs * gain * (feedback_reference - reference) / scale
    return action + moving * ramp * legs * motor_offset / scale


def reference_torque_inputs(metadata):
    """Load the force allocation audited against this exact reference and model."""
    path = Path(__file__).parent / 'training/reference_repair/fitted_audit.json'
    payload = path.read_bytes()
    audit = json.loads(payload)
    if {k: v for k, v in audit.items() if k != 'rows'} != metadata['fitted_dynamics_audit']:
        raise ValueError('Nominal torque audit does not match the current reference')
    rows = audit['rows']
    if not all(r['results']['floor_command_bounds']['feasible'] for r in rows):
        raise ValueError('Nominal torque reference contains infeasible floor contacts')
    times = np.array([r['reference_time_s'] for r in rows])
    # The audit trims its four derivative guard samples. Its first retained
    # row is polynomial phase zero, irrespective of that guard's timestamp.
    times -= times[0]
    torques = np.array([r['results']['floor_command_bounds']['motor_torque_nm'] for r in rows])
    if (torques.shape != (len(times), 16) or not np.isfinite(torques).all()
            or not np.all(np.diff(times) > 0) or abs(times[-1] - metadata['period']) > 1e-6):
        raise ValueError('Invalid nominal torque cycle')
    np.testing.assert_allclose(torques[0], torques[-1], atol=1e-5)
    return times, torques, hashlib.sha256(payload).hexdigest()


def periodic_reference_torque(time, times, torques, xp):
    phase = time % times[-1]
    index = xp.clip(xp.searchsorted(times, phase, side='right') - 1, 0, len(times) - 2)
    fraction = (phase - times[index]) / (times[index + 1] - times[index])
    return torques[index] + fraction * (torques[index + 1] - torques[index])


def sampled_reference_inputs(metadata, names, home, default, motor_gains, feedback_gain, damping, stop_fade=False):
    """Precompute the 50 Hz nominal command, including its two-tick queue.

    The LP force allocation has fast changes that cannot be issued at 50 Hz.
    Average it over the control interval; posture and force need separate leads.
    These are software command settings, not changes to the identified motor.
    """
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

    reference = PolyReferenceMotion(str(Path(__file__).parent / 'data/polynomial_coefficients.pkl'))
    times, torques, torque_hash = reference_torque_inputs(metadata)
    order = np.array([metadata['joint_names'].index(name) for name in names])
    legs = np.array([any(s in name for s in ('hip', 'knee', 'ankle')) for name in names])
    specification = dict(control_dt_s=.02, delay_ticks=2, posture_advance_s=.048,
                         torque_advance_s=.054, torque_average_half_width_s=.012,
                         torque_average_sample_s=.002, torque_scale=1.1049132959059584,
                         startup_s=1.2, physical_liftoff_roll_hold=False,
                         startup_clock='active command ticks; reset on zero command',
                         feedback_gain=feedback_gain, feedback_damping=damping,
                         torque_source_sha256=torque_hash)
    if stop_fade:
        specification.update(startup_clock='active or fading ticks; reset after blend reaches zero',
                             stop_transition='1.2s quintic fade, continuing reference phase')
    offsets = np.arange(-6, 7) * .002
    commands = []
    for phase in range(reference.nb_steps_in_period):
        current = reference.get_reference_motion(max(reference.dxs), 0., 0., phase)
        future = reference.get_reference_motion(max(reference.dxs), 0., 0., phase + .048/.02)
        tau = np.mean([periodic_reference_torque(phase*.02+.054+t, times, torques, np)
                       for t in offsets], axis=0) * specification['torque_scale']
        commands.append(legs * (future[order] - home + feedback_gain*(current[order] - home)
            + damping*current[16 + order] + tau/motor_gains - (default - home)))
    commands = np.array(commands)
    if commands.shape != (reference.nb_steps_in_period, len(names)) or not np.isfinite(commands).all():
        raise ValueError('Invalid sampled nominal commands')
    specification['motor_correction_sha256'] = hashlib.sha256(commands.tobytes()).hexdigest()
    return commands, specification


def sampled_reference_action(action, phase, command, elapsed, corrections, scale, xp, blend_ticks=None):
    """PPO residual plus the sampled nominal, before measured feedback/delay."""
    u = xp.clip(elapsed/1.2 if blend_ticks is None else blend_ticks/60., 0., 1.)
    ramp = u**3 * (10. - 15.*u + 6.*u*u)
    active = (xp.linalg.norm(command[:3]) > 1e-6) if blend_ticks is None else True
    return action + active * ramp * corrections[phase] / scale


def specification():
    reference = json.loads((Path(__file__).parent / 'data/reference.json').read_text())
    audit = reference['fitted_dynamics_audit']
    return dict(parameters=audit['parameters'], fit_sha256=audit['parameter_source_sha256'],
                torque_limits='unchanged MJCF prior, not a measured continuous rating')


def signature():
    return hashlib.sha256(json.dumps(specification(), sort_keys=True).encode()).hexdigest()


def set_identified(model, parameters):
    main = np.array([i for i in range(model.nu) if 'antenna' not in model.actuator(i).name])
    dofs = model.jnt_dofadr[model.actuator_trnid[main, 0]]
    gain = parameters['position_gain_Nm_per_rad']
    model.actuator_gainprm[main, 0] = gain
    model.actuator_biasprm[main, 1] = -gain
    model.dof_armature[dofs] = parameters['armature_kg_m2']
    model.dof_damping[dofs] = parameters['damping_Nm_s_per_rad']
    model.dof_frictionloss[dofs] = parameters['frictionloss_Nm']
    return main


def configure(model, torque_limit_scale=1.):
    """Apply once to a freshly loaded MJCF, preserving the home motor preload."""
    import mujoco

    if not np.isfinite(torque_limit_scale) or torque_limit_scale <= 0:
        raise ValueError('Torque limit scale must be positive and finite')
    home = model.key('home')
    q = home.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    preload = model.actuator_gainprm[:, 0] * (home.ctrl - q)
    parameters = specification()['parameters']
    main = set_identified(model, parameters)
    model.actuator_forcerange[main] *= torque_limit_scale
    home.ctrl[:] = q + preload / model.actuator_gainprm[:, 0]
    # Armature affects cached mass constants as well as the visible model array.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return parameters
