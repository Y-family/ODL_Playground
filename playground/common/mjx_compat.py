"""MJX GPU compatibility and forward-simulation optimizations."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco.mjx._src import solver, support
from mujoco.mjx._src.types import ConeType, DisableBit


_original_make_m = support.make_m
_original_linesearch_point = solver._LSPoint.create
_original_linesearch_create = solver._LSPoint.__dict__['create']


def _pyramidal_point(cls, m, d, ctx, alpha, jv, quad, quad_gauss, uu, v0, uv, vv):
    """Evaluate the same quadratics, restricting friction work to friction rows.

    Equations follow MuJoCo MJX solver.py (Apache-2.0). Contact/limit rows
    have zero frictionloss; the upstream full-width temporary selects their
    original quadratic back unchanged. Avoid constructing that temporary.
    """
    if m.opt.cone != ConeType.PYRAMIDAL:
        return _original_linesearch_point(m, d, ctx, alpha, jv, quad, quad_gauss, uu, v0, uv, vv)
    start, stop = d._impl.ne, d._impl.ne + d._impl.nf
    x = ctx.Jaref + alpha * jv
    active = (x < 0).at[:stop].set(True)
    if stop > start and not m.opt.disableflags & DisableBit.FRICTIONLOSS:
        f = d._impl.efc_frictionloss[start:stop]
        diagonal = d._impl.efc_D[start:stop]
        r = 1. / (diagonal + (diagonal == 0.) * mujoco.mjMINVAL)
        rf, z = r * f, jp.zeros_like(f)
        linear_neg = (x[start:stop] <= -rf)[:, None]
        linear_pos = (x[start:stop] >= rf)[:, None]
        qf = linear_neg * jp.array([f * (-.5 * rf - ctx.Jaref[start:stop]), -f * jv[start:stop], z]).T
        qf += linear_pos * jp.array([f * (-.5 * rf + ctx.Jaref[start:stop]), f * jv[start:stop], z]).T
        friction = jp.where((linear_neg | linear_pos) & (f[:, None] > 0), qf, quad[start:stop])
        quad = jax.lax.dynamic_update_slice(quad, friction, (start, 0))
    quad_total = quad_gauss + jp.sum(jax.vmap(jp.multiply)(quad, active), axis=0)
    cost = alpha * alpha * quad_total[2] + alpha * quad_total[1] + quad_total[0]
    deriv_0 = 2 * alpha * quad_total[2] + quad_total[1]
    deriv_1 = 2 * quad_total[2] + (quad_total[2] == 0) * mujoco.mjMINVAL
    return cls(alpha=alpha, cost=cost, deriv_0=deriv_0, deriv_1=deriv_1)


def install_linesearch_optimization():
    """Keep the original scan, tolerances, formulas and differentiation."""
    solver._LSPoint.create = classmethod(_pyramidal_point)


def _make_m(m, a, b, d=None):
    if support.is_sparse(m):
        return _original_make_m(m, a, b, d)
    matrix = a @ b.T
    if d is not None:
        matrix += jp.diag(d)
    # The topology is static. A GPU bool scatter can emit atom.sys.global.cas,
    # which the Windows/WSL Pascal driver rejects when loading the CUBIN.
    mask = np.zeros((m.nv, m.nv), dtype=bool)
    mask[np.repeat(np.arange(m.nv), m.M_rownnz), m.M_colind] = True
    matrix *= jp.asarray(mask)
    return matrix + jp.tril(matrix, -1).T


def install_static_mass_mask():
    """Keep MJX's mass-matrix arithmetic, constructing its mask on the host."""
    support.make_m = _make_m


if __name__ == '__main__':
    import os
    from types import SimpleNamespace
    from mujoco.mjx._src.types import JacobianType

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    gpu, cpu = jax.devices('gpu')[0], jax.devices('cpu')[0]
    model = SimpleNamespace(
        nv=4, M_rownnz=np.array([1, 2, 1, 3]),
        M_colind=np.array([0, 0, 1, 2, 0, 1, 3]),
        M_rowadr=np.array([0, 1, 3, 4]), opt=SimpleNamespace(jacobian=None))
    rng = np.random.default_rng(0)
    a, b = (rng.normal(size=(4, 4, 6)).astype(np.float32) for _ in range(2))
    for jacobian in (JacobianType.DENSE, JacobianType.SPARSE):
        model.opt.jacobian = jacobian
        for diagonal in (None, np.arange(1, 5, dtype=np.float32)):
            reference = jax.jit(jax.vmap(lambda x, y: _original_make_m(model, x, y, diagonal)))
            actual = jax.jit(jax.vmap(lambda x, y: _make_m(model, x, y, diagonal)))
            expected = reference(jax.device_put(a, cpu), jax.device_put(b, cpu))
            result = actual(jax.device_put(a, gpu), jax.device_put(b, gpu))
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
            assert result.device.platform == 'gpu'
    print('PASS: batched GPU dense/sparse mass matrices match original CPU, with/without diagonal')
