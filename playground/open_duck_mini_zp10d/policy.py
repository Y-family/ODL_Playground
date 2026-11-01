"""Checkpointed PPO preprocessing shared by training and native replay."""

import math

import jax
import jax.numpy as jp
from brax.training import types
from brax.training.agents.ppo import networks


def with_liftoff_observation(params, slots=slice(9, 11)):
    """Reuse two formerly zero head-command inputs without changing the policy."""
    stats, actor, critic = jax.tree.map(jp.asarray, params)
    for key in ('state', 'privileged_state'):
        if not bool(jp.all(stats.mean[key][slots] == 0.)):
            raise ValueError('Task observation slots were not unused zero commands')
    count = (jp.float32(stats.count.hi) * jp.float32(2.**32) + jp.float32(stats.count.lo)
             if hasattr(stats.count, 'hi') else jp.float32(stats.count))
    stats = stats.replace(
        mean={k: v.at[slots].set(0.) for k, v in stats.mean.items()},
        std={k: v.at[slots].set(1.) for k, v in stats.std.items()},
        summed_variance={k: v.at[slots].set(count * (1. - stats.std_eps))
                         for k, v in stats.summed_variance.items()})
    for network in (actor, critic):
        layers = network['params'].get('MLP_0', network['params'])
        layers['hidden_0']['kernel'] = layers['hidden_0']['kernel'].at[slots].set(0.)
    return stats, actor, critic


def with_policy_std(params, std):
    """Reset normal-policy exploration while preserving the learned mean."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError('Policy standard deviation must be positive and finite')
    updated = jax.tree.map(lambda value: value, params)
    previous = updated['params']['std_logparam']['log_value']
    updated['params']['std_logparam']['log_value'] = jp.full_like(previous, math.log(std))
    return updated


def make_ppo_networks(
        observation_size, action_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(128, 128),
        policy_obs_key='state', value_obs_key='privileged_state',
        distribution_type='normal', noise_std_type='log', init_noise_std=.0125,
        mean_clip_scale=None, mean_kernel_init_fn=None, mean_kernel_init_kwargs=None,
        observation_clip=None, **kwargs):
    # Defaults are explicit so Brax records every configured argument. None
    # preserves historical checkpoints; new runs opt into the bound explicitly.
    preprocess = preprocess_observations_fn
    if observation_clip is not None:
        if not math.isfinite(observation_clip) or observation_clip <= 0:
            raise ValueError('observation_clip must be positive and finite')

        def preprocess(observation, statistics):
            return jax.tree.map(lambda value: jp.clip(value, -observation_clip, observation_clip),
                                preprocess_observations_fn(observation, statistics))

    return networks.make_ppo_networks(
        observation_size, action_size, preprocess_observations_fn=preprocess,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        policy_obs_key=policy_obs_key, value_obs_key=value_obs_key,
        distribution_type=distribution_type, noise_std_type=noise_std_type,
        init_noise_std=init_noise_std, mean_clip_scale=mean_clip_scale,
        mean_kernel_init_fn=mean_kernel_init_fn, mean_kernel_init_kwargs=mean_kernel_init_kwargs,
        **kwargs)
