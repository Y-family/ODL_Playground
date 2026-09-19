"""Run one CAD script with an address-space ceiling before native imports."""

import fcntl
import os
from pathlib import Path
import resource
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: run_memory_limited.py SCRIPT [ARGS ...]')
    with open('/tmp/open_duck_cad.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another memory-limited CAD task is running')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CADGEN_CACHE_DIR'] = '/tmp/cadgen'
        os.environ['CADGEN_DAEMON'] = '0'
        os.environ['CADGEN_COMPONENT_WORKERS'] = '1'
        os.environ['CADGEN_VALIDATE_WORKERS'] = '1'
        resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
        if sys.argv[1] == '--check':
            try:
                allocation = bytearray(4_000_000_000)
            except MemoryError:
                print('PASS: allocation at the address-space ceiling was rejected')
                return
            raise AssertionError(f'Address-space limit failed: {len(allocation)} bytes allocated')
        script = Path(sys.argv[1]).resolve()
        sys.argv = sys.argv[1:]
        sys.path.insert(0, str(script.parent))
        try:
            runpy.run_path(str(script), run_name='__main__')
        finally:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            print(f'CAD peak process RSS: {peak / 1_000_000:.1f} MB; address-space cap: 4000 MB', flush=True)


if __name__ == '__main__':
    main()
