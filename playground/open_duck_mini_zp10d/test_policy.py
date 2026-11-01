"""Rare-event normalization counterexample and actual checkpoint round trip."""

import functools
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['JAX_PLATFORMS'] = 'cpu'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
from playground.open_duck_mini_zp10d.policy import make_ppo_networks, with_policy_std, with_liftoff_observation


def check_liftoff_migration(params, network, config, slots=slice(9, 11)):
    stats = params[0]
    obs = {k: jp.asarray(v) for k, v in stats.mean.items()}
    before_actor = network.policy_network.apply(stats, params[1], obs)
    before_value = network.value_network.apply(stats, params[2], obs)
    migrated = with_liftoff_observation(params, slots=slots)
    for features in ([1., 1.], [0., 1.], [1., 0.], [3., 0.], [0., 3.]):
        updated = {k: v.at[slots].set(jp.array(features)) for k, v in obs.items()}
        np.testing.assert_allclose(network.policy_network.apply(migrated[0], migrated[1], updated),
                                   before_actor, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(network.value_network.apply(migrated[0], migrated[2], updated),
                                   before_value, rtol=1e-6, atol=1e-7)
        for value in running_statistics.normalize(updated, migrated[0]).values():
            np.testing.assert_allclose(value[slots], features)
    # Other input rows and their accumulated statistics must not change.
    for key in stats.mean:
        for name in ('mean', 'std', 'summed_variance'):
            keep = np.ones(stats.mean[key].shape, dtype=bool)
            keep[slots] = False
            np.testing.assert_array_equal(getattr(migrated[0], name)[key][keep],
                                          getattr(stats, name)[key][keep])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint.save(directory, 1, migrated, config)
        loaded = checkpoint.load(str(Path(directory) / '000000000001'))
        np.testing.assert_allclose(network.policy_network.apply(loaded[0], loaded[1], updated), before_actor,
                                   rtol=1e-6, atol=1e-7)
    print('PASS: liftoff input migration preserves actor/value outputs, unit scaling, '
          'other normalization statistics and actual checkpoint persistence.', flush=True)


def main():
    sizes = {'state': 113, 'privileged_state': 230}
    shapes = {key: jax.ShapeDtypeStruct((size,), jp.float32)
              for key, size in sizes.items()}
    stats = running_statistics.init_state(shapes)
    stats = stats.replace(std=jax.tree.map(lambda x: jp.full_like(x, 1e-6), stats.std))
    obs = jax.tree.map(jp.zeros_like, stats.mean)
    obs = {key: value.at[111].set(2.) for key, value in obs.items()}
    assert float(running_statistics.normalize(obs, stats)['state'][111]) == 2e6
    factory = functools.partial(make_ppo_networks, observation_clip=5.,
        mean_kernel_init_fn=jax.nn.initializers.constant, mean_kernel_init_kwargs={'value': .01})
    config = checkpoint.network_config(sizes, 16, True, factory)
    bounded = checkpoint._get_ppo_network(config, make_ppo_networks)
    key = jax.random.PRNGKey(0)
    params = (stats, bounded.policy_network.init(key), bounded.value_network.init(key))
    check_liftoff_migration(params, bounded, config)
    check_liftoff_migration(params, bounded, config, slots=slice(11, 13))
    original_logits = bounded.policy_network.apply(stats, params[1], obs)
    explored = with_policy_std(params[1], .05)
    explored_logits = bounded.policy_network.apply(stats, explored, obs)
    np.testing.assert_array_equal(explored_logits[0], original_logits[0])
    np.testing.assert_allclose(explored_logits[1], .05, rtol=1e-6)
    np.testing.assert_allclose(original_logits[1], .0125, rtol=1e-6)
    for bad_std in (0., -1., float('inf'), float('nan')):
        try:
            with_policy_std(params[1], bad_std)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid exploration standard deviation accepted')
    value = float(bounded.value_network.apply(stats, params[2], obs))
    plain_config = checkpoint.network_config(sizes, 16, True,
        functools.partial(networks.make_ppo_networks, policy_hidden_layer_sizes=(64, 64),
            value_hidden_layer_sizes=(128, 128), value_obs_key='privileged_state',
            distribution_type='normal', noise_std_type='log', init_noise_std=.0125,
            mean_kernel_init_fn=jax.nn.initializers.constant, mean_kernel_init_kwargs={'value': .01}))
    plain = checkpoint._get_ppo_network(plain_config, networks.make_ppo_networks)
    legacy = checkpoint._get_ppo_network(plain_config, make_ppo_networks)
    plain_value = float(plain.value_network.apply(stats, params[2], obs))
    assert abs(plain_value) > 100. and abs(value) < 100., (plain_value, value)
    np.testing.assert_array_equal(plain.policy_network.apply(stats, params[1], obs),
                                  legacy.policy_network.apply(stats, params[1], obs))
    with tempfile.TemporaryDirectory() as directory:
        checkpoint.save(directory, 1, (stats, explored, params[2]), config)
        path = str(Path(directory) / '000000000001')
        loaded = checkpoint.load_policy(path, network_factory=make_ppo_networks)
        roundtrip = checkpoint.load(path)
        np.testing.assert_allclose(bounded.policy_network.apply(
            roundtrip[0], roundtrip[1], obs)[1], .05, rtol=1e-6)
        expected, _ = networks.make_inference_fn(bounded)(params, deterministic=True)(obs, key)
        actual, _ = loaded(obs, key)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        assert checkpoint.load_config(path).network_factory_kwargs.observation_clip == 5.
    print(f'PASS: rare event normalized to 2e6; value {plain_value:g} -> {value:g}; '
          'exploration override preserves mean and checkpoint round trip; legacy preprocessing unchanged.')
    if '--checkpoint' in sys.argv:
        path = str(Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve())
        actual_config = checkpoint.load_config(path)
        actual_network = checkpoint._get_ppo_network(actual_config, make_ppo_networks)
        check_liftoff_migration(checkpoint.load(path), actual_network, actual_config)
    if '--drift-checkpoint' in sys.argv:
        path = str(Path(sys.argv[sys.argv.index('--drift-checkpoint') + 1]).resolve())
        actual_config = checkpoint.load_config(path)
        actual_network = checkpoint._get_ppo_network(actual_config, make_ppo_networks)
        params = checkpoint.load(path)
        metadata = json.loads((Path(path).parent.parent / 'metadata.json').read_text())
        assert metadata['reward_specification'].get('planar_drift_observation')
        assert any(np.max(np.abs(np.asarray(params[0].mean[key][11:13]))) > 0.
                   for key in ('state', 'privileged_state'))
        obs = {key: value for key, value in params[0].mean.items()}
        output = actual_network.policy_network.apply(params[0], params[1], obs)
        assert np.isfinite(np.asarray(output[0])).all()
        print('PASS: trained drift-observation checkpoint loads with nonzero migrated '
              'slots and finite actor output.', flush=True)


if __name__ == '__main__':
    main()
