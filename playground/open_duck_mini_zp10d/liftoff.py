"""Shared liftoff memory for JAX training and native NumPy replay."""


def update_liftoff(last_landed, qualified, heights, loads, weight, xp, minimum_support_force=None,
                   qualification_height=.002):
    landed = (loads > .5) & qualified
    last_landed = xp.where(xp.any(landed), xp.argmax(landed), last_landed)
    allowed = xp.arange(2) != last_landed
    support = .5 * weight if minimum_support_force is None else minimum_support_force
    cleared = (heights >= qualification_height) & (loads < .5) & (loads[::-1] > support) & allowed
    qualified = (qualified | cleared) & ~landed
    return last_landed, qualified, allowed


def liftoff_features(last_landed, qualified, xp):
    # 0=excluded, 1=eligible, 3=qualified; Placo keeps its clock separately.
    return 1. * (xp.arange(2) != last_landed) + 2. * qualified


def completed_lift_reward(qualified, peak, heights, loads, up_z, terminal, xp, active=True):
    """Pay once on a supported landing; duration above 10 mm is irrelevant."""
    peak = xp.maximum(peak, heights)
    landed = qualified & (loads > .5)
    stable = (up_z >= xp.cos(xp.pi / 9.)) & xp.logical_not(terminal)  # 20 degrees
    credit = landed & (loads[::-1] > .5) & stable & active
    band = xp.exp(-xp.square(xp.maximum(peak - .015, 0.) / .005))
    return xp.sum(credit * band), xp.where(loads > .5, 0., peak)


def command_liftoff_runs(heights, forces, commands, weight, dt=.02, minimum_support_force=None):
    """Keep commanded liftoff separate from startup/stop transients in replay."""
    import numpy as np

    heights, forces = np.asarray(heights), np.asarray(forces)
    moving = np.abs(np.asarray(commands)) > 1e-6
    support = .5 * weight if minimum_support_force is None else minimum_support_force
    valid = (heights >= .01) & (forces < .5) & (forces[:, ::-1] > support)
    result = {}
    for name, phase in (('active', moving), ('inactive', ~moving)):
        longest, current = np.zeros(2), np.zeros(2)
        for frame in valid & phase[:, None]:
            current = np.where(frame, current + dt, 0.)
            longest = np.maximum(longest, current)
        result[name + '_command_longest_10mm_unloaded_sample_run_s'] = longest.tolist()
    return result
