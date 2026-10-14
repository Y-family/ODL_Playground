"""Recheck known remaining head-chain collisions in the current STEP.

Millimetres; MJCF angle signs and STEP-measured axes. Check only previously
identified S12/S13/S14 pairs at their sampled onset, preceding degree and
historical maximum. Also check the repaired S13 speaker pair and the S11
head/body limit interface. This is a targeted audit, not a full motion scan.
The s13-limits mode checks the two known S13 pairs every degree inside the
current MJCF limits, separating original shaft contact from rear-wall clearance.
Run serially through run_memory_limited.py; geometry is never changed.
"""

import gc
import hashlib
import json
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step
from OCP.OSD import OSD_Parallel, OSD_ThreadPool

from exact_motion_current import _overlap
from full_motion_current import TARGET, MJCF, LIMITS, _leaves, measured_axes, is_moving
from repair_motion_clearance_current import detached

REPORT = TARGET.parent / 'REMAINING_COLLISIONS_REPORT.json'
OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)


def main():
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else TARGET
    s13_limited = len(sys.argv) > 1 and sys.argv[1] == "s13-limits"
    s13_only = len(sys.argv) > 1 and sys.argv[1] in ("s13", "s13-limits")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    robot = read_step(target)
    leaves = _leaves(robot)
    by_path = {row['path']: row for row in leaves}
    axes = measured_axes(robot)
    historical = json.loads((TARGET.parent / 'FULL_MOTION_EXACT_REPORT.json').read_text())
    report = dict(target_sha256=digest,
                  mjcf_sha256=hashlib.sha256(MJCF.read_bytes()).hexdigest(),
                  historical_report_sha256=hashlib.sha256(
                      (TARGET.parent / 'FULL_MOTION_EXACT_REPORT.json').read_bytes()).hexdigest(),
                  axes=axes, limits_deg=LIMITS, volume_threshold_mm3=1e-6,
                  scope='Known head-chain pairs at selected angles only; other joints neutral.',
                  checks=[])
    if REPORT.exists():
        previous = json.loads(REPORT.read_text())
        for key in ('optimization_assessment', 'stretch_only_assessment', 'latest_user_constraint', 'hardware_limit_write'):
            if key in previous:
                report[key] = previous[key]
        report['previous_audit'] = {key: previous[key] for key in (
            'target_sha256', 'mjcf_sha256', 'scope', 'limits_deg', 'checks',
            'completed', 'remaining_pair_count') if key in previous}
    rear_wall = None
    if s13_limited:
        import refit_upper_servo_interfaces_local as refit
        frame = next(row['shape'] for row in leaves if row['label']=='head_pitch_to_yaw')
        rear_wall = refit._shape(frame.intersect(refit._world_box(
            -8.100001, -5.1, -10.900001, 11.100001,
            171.57593881241, 225.65093881235)))
        report['scope'] = 'Two known S13 pairs at 1 degree steps inside current MJCF limits; other joints neutral.'
        report['clearance_requirement_mm'] = .2
        report['body_clearance_reference'] = 'Original rear wall only; complete C-frame also checked for zero volume overlap. Original coaxial shaft contact is retained and not used as running clearance.'
        report['geometry_changed'] = False

    def check(servo, moving, fixed, angles, purpose):
        mover, anchor = by_path[moving], by_path[fixed]
        assert is_moving(mover, servo) and not is_moving(anchor, servo)
        axis = bd.Axis(*axes[servo])
        left, right = detached(mover['shape']), detached(anchor['shape'])
        result = dict(joint=servo, moving=moving, fixed=fixed,
                      purpose=purpose, samples=[])
        for angle in sorted(set(angles)):
            posed = left.rotate(axis, angle)
            volume = _overlap(posed, right)
            gap = posed.distance_to(right)
            sample = dict(angle_deg=angle, overlap_mm3=volume, gap_mm=gap,
                          in_operating_range=LIMITS[servo][0]<=angle<=LIMITS[servo][1])
            if s13_limited:
                running_gap = posed.distance_to(rear_wall) if mover['label']=='ZP10D_vendor_body' else gap
                sample.update(running_gap_mm=running_gap,
                              passed=volume<=1e-6 and running_gap>=.2-1e-7)
            result['samples'].append(sample)
            if not s13_limited or angle % 20 == 0 or not sample['passed']:
                print(servo, mover['label'], anchor['label'], angle,
                      'overlap', round(volume, 6), 'gap', round(gap, 6),
                      'running_gap', sample.get('running_gap_mm'), flush=True)
            del posed
            gc.collect()
        report['checks'].append(result)
        if s13_limited:
            result['passed'] = all(sample['passed'] for sample in result['samples'])
            result['minimum_running_gap_mm'] = min(sample['running_gap_mm'] for sample in result['samples'])
        REPORT.write_text(json.dumps(report, indent=2)+'\n')
        del left, right
        gc.collect()

    if s13_limited:
        frame_path, = [row['path'] for row in leaves if row['label'] == 'head_pitch_to_yaw']
        for label in ('ZP10D_vendor_body', 'head_yaw_to_roll'):
            moving_path, = [row['path'] for row in leaves if row['label'] == label
                           and (label != 'ZP10D_vendor_body' or row['servo'] == 'S13')]
            check('S13', moving_path, frame_path,
                  range(LIMITS['S13'][0], LIMITS['S13'][1] + 1), 'remaining_collision')
        assert len(report['checks']) == 2, 'Both known S13 pairs must be checked'

    for old in (() if s13_limited else historical['checks']):
        servo = old['joint']
        if servo not in ('S12', 'S13', 'S14') or (s13_only and (servo != 'S13' or 'speaker_stand' in old['moving'])):
            continue
        hits = [row for row in old['exact'] if row['overlap_mm3'] > .05]
        if not hits:
            continue
        angles = {0, max(hits, key=lambda row: row['overlap_mm3'])['angle_deg']}
        for sign in (-1, 1):
            side = [row for row in hits if row['angle_deg']*sign > 0]
            if side:
                first = min(side, key=lambda row: abs(row['angle_deg']))['angle_deg']
                angles.update((first-sign, first))
        if s13_only:
            angles.update((-160,160))
            if 'head_yaw_to_roll' in old['moving']: angles.update((-106,106))
        if s13_limited:
            angles = range(LIMITS['S13'][0], LIMITS['S13'][1]+1)
        purpose = 'remaining_collision'
        if servo == 'S13' and 'speaker_stand' in old['moving']:
            angles.update((-125, -65))
            purpose = 'repaired_speaker_pair'
        check(servo, old['moving'], old['fixed'], angles, purpose)

    def unique(label):
        matches = [row['path'] for row in leaves if row['label']==label]
        assert len(matches)==1, (label, matches)
        return matches[0]

    for moving, fixed, angles in (() if s13_only else (
        ('head', 'body_middle_top', (39, 39.25, 39.5, 40)),
        ('head_bot_sheet', 'body_middle_top', (39, 39.25, 39.5, 40)),
        ('head', 'body_back', (40, 41)),
    )):
        check('S11', unique(moving), unique(fixed), angles, 'current_limit_and_beyond')
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    report['completed']=True
    report['remaining_pair_count']=sum(
        row['purpose']=='remaining_collision' and
        any(sample['overlap_mm3']>1e-6 and sample['in_operating_range'] for sample in row['samples'])
        for row in report['checks'])
    if s13_limited:
        assert len(report['checks']) == 2, 'Expected exactly the two known S13 pairs'
        report['passed'] = all(row['passed'] for row in report['checks'])
        report['latest_user_constraint'] = {
            'date': '2026-10-14', 'retain_rear_wall': True, 'retain_upper_lower_arms': True,
            'retain_servo_pose': True, 's13_limits_deg': list(LIMITS['S13']),
            'formal_geometry_changed': False,
            'single_end_and_flipped_servo': 'rejected by user',
        }
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    if s13_limited:
        assert report['passed'], 'S13 in-range clearance audit failed; see report'


if __name__ == '__main__':
    main()
