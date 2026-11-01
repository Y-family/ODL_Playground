"""Watchdog counterexamples without loading physics or interrupting training."""
import json
from pathlib import Path
import tempfile
import time

from supervise import inspect, format_status
from liftoff import command_liftoff_runs


def main():
    phases = command_liftoff_runs([[.012, 0.], [0., 0.], [0., .015], [0., .015]],
        [[0., 20.], [10., 10.], [20., 0.], [20., 0.]], [.01, .01, 0., 0.], 20.)
    assert phases['active_command_longest_10mm_unloaded_sample_run_s'] == [.02, 0.]
    assert phases['inactive_command_longest_10mm_unloaded_sample_run_s'] == [0., .04]
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        meta = dict(status='training; walking acceptance pending',
                    completed_steps=131072, requested_steps=20000000)
        (run / 'metadata.json').write_text(json.dumps(meta))
        (run / 'train.log').touch()
        (run / 'metrics.jsonl').write_text('{"steps":163840}\n{"steps":')
        output = run / 'native/000000131072/walk'
        output.mkdir(parents=True)
        report = dict(stop='duration', elapsed_sim_seconds=8.,
            max_whole_foot_clearance_m=[.012, .012],
            longest_10mm_unloaded_sample_run_s=[0., 0.],
            longest_2mm_unloaded_sample_run_s=[0., 0.])
        (output / 'evaluation.json').write_text(json.dumps(report))
        state = inspect(run, {'ActiveState': 'active'}, time.time())
        assert state['environment_steps'] == 163840
        assert 'bilateral_supported_10mm_liftoff_not_demonstrated' in state['alerts']
        assert not state['liftoff_accepted']  # Peak height alone cannot pass.
        files_before = sorted(run.rglob('*'))
        rendered = format_status(state, state['environment_steps'] - 8192)
        assert '8,192' in rendered and 'walk' in rendered
        report['contact_supported_active_command_longest_10mm_unloaded_sample_run_s'] = [.1, .12]
        (output / 'evaluation.json').write_text(json.dumps(report))
        rendered = format_status(inspect(run, {'ActiveState': 'active'}, time.time()))
        assert '旧半体重支撑判据' in rendered and '0.10 / 0.12' in rendered
        assert sorted(run.rglob('*')) == files_before  # Screen output creates no reports.
        report.update(stable_completed_lift_count=[3,0],completion_qualification_height_m=.010)
        (output / 'evaluation.json').write_text(json.dumps(report))
        assert 'bilateral_10mm_lift_and_landing_not_demonstrated' in inspect(
            run, {'ActiveState':'active'}, time.time())['alerts']
        report['stable_completed_lift_count']=[3,2]
        (output / 'evaluation.json').write_text(json.dumps(report))
        rendered=format_status(inspect(run, {'ActiveState':'active'}, time.time()))
        assert '左右 3 / 2 次' in rendered
        yaw_output=run/'native/000000131072/walk_yaw'
        yaw_output.mkdir()
        (yaw_output/'evaluation.json').write_text(json.dumps(dict(report,stop='fall_or_nonfinite')))
        assert 'walk_yaw: fall_or_nonfinite' in inspect(run,{'ActiveState':'active'},time.time())['alerts']
        stalled = inspect(run, {'ActiveState': 'failed'}, time.time() + 1000)
        assert 'training_service_not_active' in stalled['alerts']
        assert 'no_training_or_evaluation_output_for_15_minutes' in stalled['alerts']
        meta['status'] = 'requested PPO updates completed; walking acceptance pending'
        (run / 'metadata.json').write_text(json.dumps(meta))
        done = inspect(run, {'ActiveState': 'inactive'}, time.time() + 1000)
        assert done['budget_completed'] and not done['liftoff_accepted']
        assert 'training_service_not_active' not in done['alerts']
        continuation = run / 'unload_progress'
        continuation.mkdir()
        meta.update(status='training; walking acceptance pending', completed_steps=139264,
                    prior_steps=131072)
        (continuation / 'metadata.json').write_text(json.dumps(meta))
        (continuation / 'train.log').touch()
        next_output = continuation / 'native/000000008192/walk'
        next_output.mkdir(parents=True)
        (next_output / 'evaluation.json').write_text(json.dumps(report))
        (run / 'metadata.json').write_text(json.dumps(dict(continuation_run=str(continuation))))
        continued = inspect(run, {'ActiveState': 'active'}, time.time())
        assert continued['environment_steps'] == 139264
        assert continued['profiles']['walk']['checkpoint'] == 139264
    print('PASS: partial metrics, progress, height-only false pass, service failure, stall, completion')


if __name__ == '__main__':
    main()
