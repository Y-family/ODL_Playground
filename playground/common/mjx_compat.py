"""MJX GPU compatibility and forward-simulation optimizations."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco.mjx._src import support


_original_make_m = support.make_m


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
