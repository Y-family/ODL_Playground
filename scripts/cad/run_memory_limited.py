"""Run CAD with a 4 GB address cap, or GPU work with a 7 GB cgroup RAM cap."""

import fcntl
import os
from pathlib import Path
import resource
import runpy
import sys


GPU_MEMORY_BYTES = 7_000_000_000


def gpu_memory_limit():
    """Return the current cgroup's verified RAM limit, or None outside the scope."""
    relative = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    group = Path('/sys/fs/cgroup') / relative.lstrip('/')
    try:
        limit = int((group / 'memory.max').read_text())
        swap = int((group / 'memory.swap.max').read_text())
    except (FileNotFoundError, ValueError):
        return None
    return limit if 0 < limit <= GPU_MEMORY_BYTES and swap == 0 else None


def main():
    wait = gpu = False
    while sys.argv[1:2] in (['--wait'], ['--gpu']):
        wait |= sys.argv[1] == '--wait'
        gpu |= sys.argv[1] == '--gpu'
        del sys.argv[1]
    if len(sys.argv) < 2:
        raise SystemExit('usage: run_memory_limited.py [--wait] [--gpu] SCRIPT [ARGS ...]')
    if gpu and gpu_memory_limit() is None:
        # A scope inherits this terminal, environment and cwd; the kernel accounts
        # for all descendants before the Python target imports native libraries.
        command = ['systemd-run', '--user', '--scope', '--quiet', '--collect',
                   '--property=MemoryMax=7000000000', '--property=MemorySwapMax=0',
                   '--', sys.executable,
                   str(Path(__file__).resolve()), '--gpu']
        if wait:
            command.append('--wait')
        os.execvp(command[0], command + sys.argv[1:])
    with open('/tmp/open_duck_cad.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            holder = lock.read().strip()
            detail = f' ({holder})' if holder else ''
            if not wait:
                raise SystemExit('Another memory-limited CAD task is running' + detail
                                 + '. Add --wait before SCRIPT to queue until it exits.')
            print('Waiting for the running memory-limited task to exit' + detail
                  + '; Ctrl-C cancels this wait.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print('Serial lock acquired; starting requested script.', flush=True)
        lock.seek(0)
        lock.truncate()
        lock.write(' '.join(sys.argv[1:]) + '\n')
        lock.flush()
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CADGEN_CACHE_DIR'] = '/tmp/cadgen'
        os.environ['CADGEN_DAEMON'] = '0'
        os.environ['CADGEN_COMPONENT_WORKERS'] = '1'
        os.environ['CADGEN_VALIDATE_WORKERS'] = '1'
        if gpu:
            if resource.getrlimit(resource.RLIMIT_AS)[1] != resource.RLIM_INFINITY:
                raise SystemExit('GPU task inherited an address-space hard limit; start from a fresh terminal')
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.0')
            print(f'GPU task RAM hard limit: {gpu_memory_limit()} bytes including children; '
                  'swap disabled; virtual address space unrestricted; workspace budget 10 GB', flush=True)
        else:
            resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
        if sys.argv[1] == '--check':
            if gpu:
                import mmap
                with mmap.mmap(-1, 8_000_000_000):
                    print('PASS: 8 GB virtual reservation accepted under the 7 GB cgroup RAM limit')
                return
            try:
                allocation = bytearray(4_000_000_000)
            except MemoryError:
                print('PASS: allocation at the address-space ceiling was rejected')
                return
            raise AssertionError(f'Address-space limit failed: {len(allocation)} bytes allocated')
        # build123d requests parallel OCC meshing even with OMP_NUM_THREADS=1.
        # Cap OCC's own native pool before the target imports or reads geometry.
        if not gpu:
            from OCP.OSD import OSD_Parallel, OSD_ThreadPool
            OSD_Parallel.SetUseOcctThreads_s(True)
            OSD_ThreadPool.DefaultPool_s().Init(1)
        script = Path(sys.argv[1]).resolve()
        sys.argv = sys.argv[1:]
        sys.path.insert(0, str(script.parent))
        try:
            runpy.run_path(str(script), run_name='__main__')
        finally:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            limit = 'RAM cgroup cap: 7000 MB including children' if gpu else 'address-space cap: 4000 MB'
            print(f'Peak process RSS: {peak / 1_000_000:.1f} MB; {limit}', flush=True)


if __name__ == '__main__':
    main()
