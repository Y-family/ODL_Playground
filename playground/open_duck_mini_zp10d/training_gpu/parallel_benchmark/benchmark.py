"""Serial, cgroup-contained measurements of the existing ZP10D PPO runner."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT = Path('/root/work/Open_Duck_Playground')
OUT = PROJECT / 'playground/open_duck_mini_zp10d/training_gpu/parallel_benchmark'
OUT.mkdir(exist_ok=True)
REPORT = OUT / 'results.json'
report = dict(status='running', gpu='NVIDIA GeForce MX350', vram_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
              workspace_ram_budget_bytes=10_000_000_000, task_cgroup_ram_limit_bytes=7_000_000_000,
              model='current ZP10D; existing 18-box floor-contact approximation',
              configuration='existing runner; optimized rewards; unroll=16; batch_size=num_envs; 2 updates per batch; no eval',
              scope='short capacity/throughput test, not walking acceptance or long-run stability', runs=[])
if REPORT.exists():
    report = json.loads(REPORT.read_text())
    report['status'] = 'running'
    for previous in report['runs']:
        if previous['status'] == 'running':
            previous.update(status='CANCELLED', cancellation_reason='User requested jumping directly to 128 environments')
        elif previous['status'] == 'PASS':
            rows = [json.loads(x) for x in (Path(previous['output']) / 'metrics.jsonl').read_text().splitlines()]
            rates = [x['training/sps'] for x in rows[2:]]
            previous['post_warmup_sps'] = rates
            previous['median_post_warmup_sps'] = statistics.median(rates)
            previous['estimated_300m_days_no_eval'] = 300_000_000 / previous['median_post_warmup_sps'] / 86400
report['timing_method'] = 'Discard first two epochs: both contain JAX compilation. Subsequent trials run six epochs.'
group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
assert int((group / 'memory.max').read_text()) <= 7_000_000_000
assert int((group / 'memory.swap.max').read_text()) == 0

def save():
    temporary = REPORT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(REPORT)

def sample(pid):
    values = {}
    try:
        status = Path(f'/proc/{pid}/status').read_text().splitlines()
        for label, key in [('VmRSS:', 'process_rss_bytes'), ('VmHWM:', 'process_peak_rss_bytes')]:
            line = next((x for x in status if x.startswith(label)), '')
            if line:
                values[key] = int(line.split()[1]) * 1024
        values['cgroup_ram_bytes'] = int((group / 'memory.current').read_text())
    except (FileNotFoundError, ProcessLookupError):
        pass
    try:
        result = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            memory, utilization, temperature = result.stdout.strip().split(',')
            values.update(gpu_total_used_mib=float(memory), gpu_util_percent=float(utilization), gpu_temp_c=float(temperature))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return values

def test(envs, epochs=6, label=None):
    steps_per_epoch = envs * 16
    output = OUT / (label or f'envs_{envs}')
    if output.exists():
        output = OUT / f'envs_{envs}_confirmation'
    output.mkdir(exist_ok=False)
    command = [sys.executable, '-u', 'playground/open_duck_mini_zp10d/runner.py', '--envs', str(envs),
               '--steps', str(steps_per_epoch * epochs), '--checkpoint-every', str(steps_per_epoch), '--no-eval',
               '--reward-mode', 'optimized', '--output', str(output)]
    run = dict(envs=envs, epochs=epochs, command=command, output=str(output), status='running')
    report['runs'].append(run)
    save()
    print(f'START envs={envs} epochs={epochs}', flush=True)
    started = time.monotonic()
    peaks = {}
    samples = []
    timed_out = False
    with (output / 'train.log').open('w') as log, (output / 'resources.jsonl').open('w') as resources:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            reading = sample(process.pid)
            reading['elapsed_seconds'] = time.monotonic() - started
            resources.write(json.dumps(reading) + '\n')
            resources.flush()
            for key, value in reading.items():
                if key != 'elapsed_seconds':
                    peaks[key] = max(peaks.get(key, 0), value)
            if 'gpu_util_percent' in reading:
                samples.append(reading['gpu_util_percent'])
            if time.monotonic() - started > 900:
                timed_out = True
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                break
            time.sleep(2)
        returncode = process.wait()
    run.update(returncode=returncode, elapsed_seconds=time.monotonic() - started, resource_peaks=peaks)
    if samples:
        run['gpu_util_mean_percent'] = statistics.mean(samples)
    meta_path = output / 'metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows_path = output / 'metrics.jsonl'
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    rates = [r['training/sps'] for r in rows[2:] if 'training/sps' in r]
    run.update(completed_steps=meta.get('completed_steps', 0), policy_parameter_max_delta=meta.get('policy_parameter_max_delta'),
               source_hashes=meta.get('source_hashes'), reward_code_sha256=meta.get('reward_code_sha256'),
               reference_sha256=meta.get('reference_sha256'), reward_reference_sha256=meta.get('reward_reference_sha256'),
               post_warmup_sps=rates)
    if rates:
        run['median_post_warmup_sps'] = statistics.median(rates)
        run['estimated_300m_days_no_eval'] = 300_000_000 / run['median_post_warmup_sps'] / 86400
    success = (returncode == 0 and meta.get('backend') == 'gpu' and
               meta.get('completed_steps', 0) >= steps_per_epoch * epochs and
               meta.get('policy_parameter_max_delta', 0) > 1e-8 and len(rows) == epochs)
    run['status'] = 'PASS' if success else 'FAIL'
    if not success:
        content = (output / 'train.log').read_text(errors='replace')
        if timed_out:
            reason = 'test timed out; capacity not determined'
        elif 'RESOURCE_EXHAUSTED' in content or 'Out of memory' in content or 'out of memory' in content:
            reason = 'GPU/device allocation failed; inspect train.log'
        elif returncode == -9:
            reason = 'process killed; inspect cgroup memory.events for RAM limit'
        else:
            reason = meta.get('error', 'runner failed; inspect train.log')
        run['failure_reason'] = reason
        run['log_tail'] = '\n'.join(content.splitlines()[-18:])
    run['cgroup_memory_events_after'] = (group / 'memory.events').read_text()
    save()
    print(f'END envs={envs} {run["status"]} sps={run.get("median_post_warmup_sps")} peaks={peaks}', flush=True)
    if not success:
        print(run['failure_reason'], run['log_tail'], sep='\n', flush=True)
    return success

save()
low = max((r['envs'] for r in report['runs'] if r['status'] == 'PASS'), default=0)
high = None
for envs in (128, 256, 512, 1024, 2048, 4096, 8192):
    if test(envs):
        low = envs
    else:
        high = envs
        break
if low and high:
    while high - low > 32:
        middle = ((high + low) // 2 // 32) * 32
        if test(middle):
            low = middle
        else:
            high = middle
report.update(status='completed', largest_short_test_pass_envs=low,
              smallest_test_failure_envs=high, boundary_resolution_envs=32)
passed = [r for r in report['runs'] if r['status'] == 'PASS']
if passed:
    fastest = max(passed, key=lambda r: r['median_post_warmup_sps'])
    report['fastest_short_test_envs'] = fastest['envs']
    report['fastest_short_test_sps'] = fastest['median_post_warmup_sps']
    fingerprints = {(r['reward_code_sha256'], r['reference_sha256'], r['reward_reference_sha256'], json.dumps(r['source_hashes'], sort_keys=True)) for r in passed}
    report['all_passed_inputs_identical'] = len(fingerprints) == 1
save()
print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2), flush=True)
