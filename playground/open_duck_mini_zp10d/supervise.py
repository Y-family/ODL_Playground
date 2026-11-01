"""Read-only training watchdog. No JAX, model changes, or automatic restarts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def inspect(run, service, now):
    meta = json.loads((run / 'metadata.json').read_text())
    if meta.get('continuation_run'):
        return inspect(Path(meta['continuation_run']), service, now)
    # Metadata advances at checkpoints; metrics also record intermediate updates.
    steps = meta.get('completed_steps', 0)
    metrics = run / 'metrics.jsonl'
    if metrics.exists():
        with metrics.open('rb') as stream:
            stream.seek(max(0, metrics.stat().st_size - 65536))
            for line in stream.read().splitlines():
                try:
                    steps = max(steps, int(json.loads(line).get('steps', 0)))
                except (ValueError, TypeError):
                    pass  # A partial concurrent write is retried next minute.
    profiles = {}
    for label in ('stand', 'walk', 'transition', 'stand_stochastic', 'walk_stochastic',
                  'walk_stochastic_seed1', 'walk_yaw'):
        files = sorted(run.glob(f'native/*/{label}/evaluation.json'))
        if files:
            report = json.loads(files[-1].read_text())
            profiles[label] = dict(checkpoint=meta.get('prior_steps', 0) + int(files[-1].parent.parent.name),
                stop=report['stop'], seconds=report['elapsed_sim_seconds'],
                clearance_mm=[1000 * v for v in report['max_whole_foot_clearance_m']],
                supported_10mm_s=report.get('active_command_longest_10mm_unloaded_sample_run_s',
                                            report['longest_10mm_unloaded_sample_run_s']),
                command_filtered='active_command_longest_10mm_unloaded_sample_run_s' in report,
                contact_supported_10mm_s=report.get('contact_supported_active_command_longest_10mm_unloaded_sample_run_s'),
                supported_2mm_s=report['longest_2mm_unloaded_sample_run_s'],
                completed_lifts=report.get('stable_completed_lift_count'),
                completion_height_m=report.get('completion_qualification_height_m'),
                displacement_m=report.get('displacement_m'))
    heartbeat = max(p.stat().st_mtime for p in (run / 'train.log', metrics, run / 'metadata.json') if p.exists())
    complete = meta.get('status', '').startswith('requested PPO updates completed')
    alerts = []
    if not complete and service.get('ActiveState') != 'active':
        alerts.append('training_service_not_active')
    if meta.get('error'):
        alerts.append('training_exception: ' + meta['error'])
    if not complete and now - heartbeat > 900:
        alerts.append('no_training_or_evaluation_output_for_15_minutes')
    for label, report in profiles.items():
        if report['stop'] != 'duration':
            alerts.append(label + ': ' + report['stop'])
    walk = profiles.get('walk')
    if walk and min(walk['supported_2mm_s']) == 0:
        alerts.append('bilateral_supported_2mm_liftoff_not_demonstrated')
    if walk and min(walk['supported_10mm_s']) == 0:
        alerts.append('bilateral_supported_10mm_liftoff_not_demonstrated')
    if (walk and walk.get('completion_height_m') == .010
            and min(walk.get('completed_lifts') or [0, 0]) == 0):
        alerts.append('bilateral_10mm_lift_and_landing_not_demonstrated')
    return dict(checked_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        environment_steps=steps, requested_steps=meta['requested_steps'],
        budget_completed=complete, training_status=meta['status'],
        service=service, seconds_since_output=now - heartbeat,
        profiles=profiles, alerts=alerts, liftoff_accepted=False,
        acceptance='10mm heights alone do not establish alternation, landing and recovery')


def format_status(status, previous_steps=None):
    """Plain terminal output; no monitoring artifacts are written."""
    steps = status['environment_steps']
    lines = [f"[{status['checked_at']}] 抬脚训练监测",
        f"训练：{steps:,} / {status['requested_steps']:,} 步；进程：{status['service'].get('ActiveState')}"]
    if previous_steps is not None:
        lines.append(f"距上次监测增加 {steps - previous_steps:,} 步")
    for label, p in status['profiles'].items():
        height = ' / '.join(f'{v:.2f}' for v in p['clearance_mm'])
        duration = ' / '.join(f'{v:.2f}' for v in p['supported_10mm_s'])
        lines.append(f"{label}（{p['checkpoint']:,}步）：左右最高 {height} mm；"
                     f"{'指令期间' if p['command_filtered'] else '全程'}旧半体重支撑判据≥10mm持续 {duration} 秒；结束：{p['stop']}")
        if p.get('contact_supported_10mm_s') is not None:
            contact = ' / '.join(f'{v:.2f}' for v in p['contact_supported_10mm_s'])
            lines.append(f"  指令期间实际接触支撑（另一脚地面反力>0.5N）≥10mm持续 {contact} 秒")
        if p.get('completed_lifts') is not None:
            counts = ' / '.join(str(v) for v in p['completed_lifts'])
            lines.append(f"  达到{p['completion_height_m']*1000:g}mm后稳定落地：左右 {counts} 次")
        if p.get('displacement_m') is not None:
            displacement = ' / '.join(f'{v*1000:.1f}' for v in p['displacement_m'][:2])
            lines.append(f"  身体水平位移 X/Y：{displacement} mm")
    lines.append('异常或未达标项：' + ('；'.join(status['alerts']) or '无已记录异常'))
    lines.append('状态：尚未验收正常抬脚，仍需验证交替、落脚和恢复。')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--unit', default='zp10d-lift-20m')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--report-every', type=int, default=600,
                        help='print a training-result summary every N seconds')
    args = parser.parse_args()
    if args.report_every < 60:
        parser.error('report-every must be at least 60 seconds')
    next_report = 0.
    previous_steps = None
    previous_health = None
    while True:
        try:
            result = subprocess.run(['systemctl', '--user', 'show', args.unit,
                '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'MemoryCurrent', '-p', 'Result'],
                text=True, capture_output=True, timeout=10, check=True)
            service = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
            status = inspect(args.run, service, time.time())
        except Exception as error:
            status = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                          alerts=[f'watchdog_read_error: {type(error).__name__}: {error}'])
        if 'environment_steps' in status and (args.once or time.monotonic() >= next_report or status.get('budget_completed')):
            print(format_status(status, previous_steps), flush=True)
            previous_steps = status['environment_steps']
            next_report = time.monotonic() + args.report_every
        # Keep minute health checks; print newly changed process/read failures.
        health = [a for a in status['alerts'] if not (a.startswith('bilateral_') or ': fall_' in a)]
        if health and health != previous_health:
            print(status['checked_at'] + ' 进程巡检异常：' + '；'.join(health), flush=True)
        previous_health = health
        if args.once or status.get('budget_completed'):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
