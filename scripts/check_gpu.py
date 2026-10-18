"""Check CUDA and JAX under the launcher's verified kernel memory limit."""

import argparse
import ctypes
import ctypes.util
import os
from pathlib import Path
import resource

from cad.run_memory_limited import gpu_memory_limit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--driver-only', action='store_true',
                    help='test CUDA initialization without importing JAX or CUDA math libraries')
args = parser.parse_args()

ram_limit = gpu_memory_limit()
if ram_limit is None and resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

print(f'RAM cgroup hard limit: {ram_limit} bytes including children; swap disabled'
      if ram_limit else 'Address-space hard limit: 4,000,000,000 bytes', flush=True)
print('; '.join(line for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith(('VmSize:', 'VmRSS:'))), flush=True)
driver_path = ctypes.util.find_library('cuda')
if not driver_path:
    wsl_driver = Path('/usr/lib/wsl/lib/libcuda.so.1')
    driver_path = str(wsl_driver) if wsl_driver.is_file() else 'libcuda.so.1'
try:
    driver = ctypes.CDLL(driver_path)
except OSError as error:
    raise SystemExit(f'Cannot load the CUDA driver: {error}')
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuInit.restype = ctypes.c_int
status = driver.cuInit(0)
print('CUDA driver:', driver_path, 'cuInit(0):', status, flush=True)
if status == 2:
    raise SystemExit(
        'CUDA_ERROR_OUT_OF_MEMORY during driver initialization, before importing JAX. '
        'This can reflect host virtual-address reservations under RLIMIT_AS, '
        'or GPU/driver resource pressure; it does not establish that VRAM is full. '
        'Check nvidia-smi. Use the launcher --gpu mode for CUDA virtual-address reservations.')
if status:
    raise SystemExit(
        f'CUDA driver initialization failed (code {status}); JAX was not imported. '
        'Check /dev/dxg and nvidia-smi in your own WSL terminal; '
        'the agent sandbox may not expose GPU devices.')
if args.driver_only:
    print('PASS: CUDA driver initialization completed under the active memory cap.', flush=True)
    raise SystemExit(0)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import jax
import jax.numpy as jnp

print('JAX:', jax.__version__, 'devices:', jax.devices(), flush=True)
gpus = [device for device in jax.devices() if device.platform == 'gpu']
if not gpus:
    raise SystemExit('GPU unavailable: JAX currently exposes no GPU device.')

with jax.default_device(gpus[0]):
    matrix = jnp.ones((4, 4), dtype=jnp.float32)
    result = (matrix @ matrix).block_until_ready()
assert result.device.platform == 'gpu' and float(result[0, 0]) == 4.
print('PASS: matrix computation completed on', result.device, flush=True)
