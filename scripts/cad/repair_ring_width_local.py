"""Restore the accepted 12.25 mm radial ring width without moving servos.

Millimetres, STEP world coordinates. Candidate R32.85/R20.60, original Z
thickness and screw seats. Only the existing ring is resized. The installed
S12 connector is fixed input. All CAD operations run serially under the
memory-limited launcher. The user stopped this candidate. Further dimensional
changes require explicit user confirmation; only the original-ring rollback
is currently authorized.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import repair_head_motion_local as repair

repair.WORK = Path('/tmp/ring-width')
repair.BASE_SHA256 = '6c96c621a3738f266108e399af66406a93b59ce7488af9bd7a39be0123c54d12'
repair.R_OUTER, repair.R_INNER = 32.85, 20.60


def restore():
    """User-requested rollback of the ring only; S12 remains unchanged."""
    bd=repair.bd
    assert hashlib.sha256(repair.TARGET.read_bytes()).hexdigest()==repair.BASE_SHA256
    old_cache=json.loads(Path('/tmp/head-motion/cache.json').read_text())
    assert old_cache['source_sha256']=='579174b3c4ec4748bdf7a085cce242e13e990bb4326f991f284d9c480c675115'
    old=repair.part(old_cache,'head_yaw_to_roll')
    robot=repair.read_step(repair.TARGET);before=repair.leaves(robot)
    axes=repair.measured_axes(robot)
    def rebuild(node):
        if node.children:
            result=bd.Compound(children=[rebuild(child) for child in node.children],label=node.label)
            result.location=node.location
        else:
            result=repair.detached(old if node.label=='head_yaw_to_roll' else node)
            result.label=node.label
        result.color=node.color
        return result
    candidate=repair.WORK/'review.step'
    bd.export_step(repair.refit._encode_colours(rebuild(robot)),candidate)
    reloaded=repair.read_step(candidate);after=repair.leaves(reloaded)
    assert len(before)==len(after)==135
    for (p,a),(q,b) in zip(before,after):
        assert p==q
        repair.compare(old if a.label=='head_yaw_to_roll' else a,b)
        assert (a.color is None)==(b.color is None)
        if a.color is not None:assert max(abs(x-y) for x,y in zip(a.color,b.color))<1e-6
    fresh_axes=repair.measured_axes(reloaded)
    for servo in axes:
        assert max(abs(a-b) for x,y in zip(axes[servo],fresh_axes[servo])
                   for a,b in zip(x,y))<1e-5
    validity=repair.check_occurrence_shape(old.wrapped)
    assert validity['solidCount']==1 and not validity['reasons']
    bd.export_brep(old,repair.WORK/'ring-candidate.brep')
    repair.review()
    report=json.loads((repair.WORK/'source-report.json').read_text())
    report['previous_repair']=dict(output_sha256=report['output_sha256'],
        ring_radii_mm=[30.5,24.5],user_rejected=True,
        report_backup=str(repair.WORK/'source-report.json'))
    report.update(source_sha256=repair.BASE_SHA256,
        output_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        scope='User-requested ring rollback; installed S12 connector retained.',
        ring_radii_mm=[50.75,38.5],ring_radial_width_mm=12.25,
        installed=str(repair.TARGET),export_leaf_count=135,unchanged_leaf_count=134,
        ring_rollback_geometry_passed=True,ring_validity=validity,
        local_motion_passed=False,interfaces_passed=False,added_material_passed=False,
        rollback='Original ring restored exactly. S14 five known pairs unresolved again. S13 remains unresolved.',
        next_changes_require_user_confirmation=True,
        stopped_candidate=dict(ring_radii_mm=[32.85,20.6],installed=False,
                               status='Stopped by user; interfaces and additions not verified.'))
    for key in ('viewer','viewer_url','review_viewed','review_sha256','remaining_audit',
                'workspace_rss_final_sample_bytes','modified','motion_checks',
                'bearing_checks','static_checks','added_material_motion_checks',
                'added_material_mm3','added_envelope_covered_mm3','process_peak_rss_mb'):
        report.pop(key,None)
    report.update(candidate=str(candidate),user_acceptance='Shrunk ring rejected; original ring restored on user instruction.',
        restored_ring=dict(bounds=repair.bounds(old),volume_mm3=repair.volume(old)),
        retained_s12='Per-face identical to installed 6c96c621 input; no S12 geometry change during rollback.')
    shutil.copyfile(repair.TARGET,repair.WORK/'source.step')
    temp=repair.TARGET.with_suffix('.step.tmp');shutil.copyfile(candidate,temp)
    assert hashlib.sha256(repair.TARGET.read_bytes()).hexdigest()==repair.BASE_SHA256
    os.replace(temp,repair.TARGET)
    repair.REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('Restored original ring',report['output_sha256'],flush=True)

if __name__ == '__main__':
    {'build': repair.build, 'verify': repair.verify,
     'interfaces': repair.interfaces, 'additions': repair.additions,
     'export': repair.export, 'review': repair.review,
     'install': repair.install, 'snapshot': repair.snapshot,
     'restore': restore}[sys.argv[1]]()
