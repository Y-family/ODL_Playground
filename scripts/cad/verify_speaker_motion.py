"""Check only the changed speaker assembly and its direct neighbours, mm."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from cadgen import build123d as bd, read_step
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

from full_motion_current import _leaves, measured_axes
from repair_minor_motion_clearance import _mesh as bounded_mesh, detached
from repair_speaker_layout_local import SPEAKER_LABELS, WORK, overlap

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path('/tmp/speaker-rework/review.step')
OUT = ROOT / 'cad/local_fix/SPEAKER_LAYOUT_MOTION_REPORT.json'


def main():
    robot = read_step(TARGET)
    parts = {r['path']: r for r in _leaves(robot)}
    geometry = {p: detached(r['shape']) for p, r in parts.items()}
    axes = measured_axes(robot)
    one = lambda label: next(p for p, r in parts.items() if r['label'] == label)
    changed = [one(label) for label in (*SPEAKER_LABELS,'speaker_stand')]
    head_path=one('head')
    added_path=head_path+'::added_wall_material'
    parts[added_path]={**parts[head_path],'label':'head_added_material'}
    geometry[added_path]=detached(bd.import_brep(WORK/'head-added.brep'))
    changed.append(added_path)
    result = dict(source_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                  static=[], motion=[], completed=False, passed=False,
                  sampling='single joint, other joints at zero, 1 degree steps',
                  antenna_range_diagnostic_only=True,
                  head_scope='Full new head used against speaker leaves; newly added wall material checked against direct neighbours and motion. Removed material cannot introduce collisions.')
    if OUT.exists():
        previous=json.loads(OUT.read_text())
        if previous.get('source_sha256')==result['source_sha256']:
            result['static']=previous['static']
            result['motion']=previous['motion']

    def save():
        OUT.write_text(json.dumps(result, indent=2) + '\n')

    neighbours = [p for p, r in parts.items() if r['root'] == 'head_assembly <1>'
                  or 'antenna' in r['root'] or r['servo'] == 'S13'
                  or r['label'] in ('head_pitch_to_yaw', 'head_yaw_to_roll')]
    for p in changed:
        for q in neighbours:
            if p == q or q==added_path or (p==added_path and q==head_path) or (q in changed and changed.index(q) < changed.index(p)):
                continue
            if any(row['moving']==p and row['fixed']==q and row['passed'] for row in result['static']):
                continue
            a, b = geometry[p], geometry[q]
            gap = a.distance_to(b)
            common = overlap(a, b) if gap < 1e-7 else 0.0
            contact = (p in changed[:-1] and q in changed[:-1]) or (parts[p]['label'] == 'speaker_stand'
                                       and parts[q]['label'] == 'head')
            row = dict(moving=p, fixed=q, gap_mm=gap, overlap_mm3=common,
                       intended_contact=contact,
                       passed=common < 1e-6 and (contact or gap >= .2 - 1e-7))
            result['static'].append(row)
            save()
            assert row['passed'], row
        print('Static passed', parts[p]['label'], flush=True)

    meshes = {}

    def mesh(p):
        if p not in meshes:
            meshes[p] = bounded_mesh(geometry[p], .01)
        return meshes[p]

    fixed_roll = [one('head_yaw_to_roll'), one('head_pitch_to_yaw')]
    fixed_roll += [p for p, r in parts.items() if r['servo'] == 'S13']
    jobs = [('S14', p, q, -30, 30, axes['S14'])
            for p, q in itertools.product(changed, fixed_roll)]
    jobs += [('S13', p, one('head_pitch_to_yaw'), -160, 160, axes['S13'])
             for p in changed]
    for joint, low, high in (('S11', -20, 39), ('S12', -45, 45)):
        fixed = [p for p, r in parts.items() if
                 (joint == 'S11' and r['label'] in
                  ('trunk_top', 'body_front', 'body_back', 'body_middle_top')) or
                 (joint == 'S12' and (r['label'] in ('neck_left_sheet', 'neck_right_sheet')
                  or (r['servo'] == 'S12' and r['label'] == 'ZP10D_vendor_body')))]
        jobs += [(joint, p, q, low, high, axes[joint])
                 for p, q in itertools.product(changed, fixed)]
    for p, r in parts.items():
        if r['label'] not in ('antenna', 'left_antenna_holder', 'right_antenna_holder'):
            continue
        positive = geometry[p].center().Y > .1
        servo = next(q for q, rr in parts.items() if rr['label'] == 'sg90'
                     and (geometry[q].center().Y > .1) == positive)
        shaft = []
        for face in geometry[servo].faces():
            ad = BRepAdaptor_Surface(face.wrapped, True)
            if ad.GetType() == GeomAbs_Cylinder and abs(ad.Cylinder().Radius() - 2.425) < 1e-6:
                cy = ad.Cylinder()
                shaft.append((list(cy.Location().Coord()), list(cy.Axis().Direction().Coord())))
        assert shaft
        jobs += [('antenna', p, q, -90, 90, shaft[0]) for q in changed]

    for joint, p, q, low, high, axis in jobs:
        if any(row['joint']==joint and row['moving']==p and row['fixed']==q
               and row['range_deg']==[low,high] for row in result['motion']):
            continue
        moving, fixed = mesh(p), mesh(q)
        manager = trimesh.collision.CollisionManager()
        manager.add_object('fixed', fixed)
        error = moving.metadata['maximum_deflection_mm'] + fixed.metadata['maximum_deflection_mm']
        minimum = (float('inf'), 0)
        fallbacks = []
        corners=trimesh.bounds.corners(moving.bounds)
        for angle in range(low, high + 1):
            trans = trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis[1], axis[0])
            posed_box=trimesh.transform_points(corners,trans)
            separation=np.maximum(np.maximum(posed_box.min(axis=0)-fixed.bounds[1],
                        fixed.bounds[0]-posed_box.max(axis=0)),0)
            box_gap=float(np.linalg.norm(separation))-error
            gap=box_gap if box_gap>.2 else manager.min_distance_single(moving,transform=trans)-error
            if gap < .2:
                posed = geometry[p].rotate(bd.Axis(*axis), angle)
                gap = posed.distance_to(geometry[q])
                common = overlap(posed, geometry[q]) if gap < 1e-7 else 0.0
                sample = dict(angle_deg=angle, gap_mm=gap, overlap_mm3=common)
                fallbacks.append(sample)
                assert gap >= .2 - 1e-7 and common < 1e-6, (joint, p, q, sample)
            minimum = min(minimum, (gap, angle))
        gap, angle = minimum
        exact = geometry[p].rotate(bd.Axis(*axis), angle).distance_to(geometry[q])
        assert exact >= .2 - 1e-7, (joint, p, q, angle, exact)
        result['motion'].append(dict(joint=joint, moving=p, fixed=q,
            range_deg=[low, high], step_deg=1, min_gap_lower_bound_mm=gap,
            minimum_angle_deg=angle, exact_minimum_sample_gap_mm=exact,
            overlap_mm3=0.0, mesh_error_mm=error, exact_fallback=fallbacks,
            broad_phase='Conservative transformed mesh bounds; FCL and BRep where bounds do not establish clearance'))
        save()
        print('Motion passed', joint, parts[p]['label'], parts[q]['label'], gap, angle, flush=True)
    result.update(completed=True, passed=True)
    save()
    print('Speaker local checks PASSED', len(result['static']), len(result['motion']), flush=True)


if __name__ == '__main__':
    main()
