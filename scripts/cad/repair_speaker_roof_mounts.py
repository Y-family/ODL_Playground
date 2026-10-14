"""Move the two existing speaker roof fastenings to opposite plate ends.

CAD brief: mm, original world frame; edit only speaker_stand and head from the
accepted STEP. Keep the 3 mm plate, tangent R6/R3 bend and speaker seats. Make
the roof flange rectangular, with one original-size fastening at each X end.
Reposition the original two tapered shell seats; preserve all other mounts,
outer roof, speaker pose, grille, servo axes and assembly structure. Temporary
files stay in /tmp; install the same formal STEP only after local validation.
Run serially via run_memory_limited.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder

import refit_upper_servo_interfaces_local as refit
from repair_speaker_layout_local import leaves, rebuild, overlap
from repair_speaker_stand_continuity import fixing_cylinders, verify_tangent_bend
from remove_s01_s06_upper_horns import compare
from full_motion_current import measured_axes

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
REPORT = ROOT / 'cad/local_fix/SPEAKER_STAND_CONTINUITY_REPORT.json'
WORK = Path('/tmp/speaker-end-fix')
SOURCE, CANDIDATE = WORK/'source.step', WORK/'review.step'
BASE_SHA = 'cea4d998a4f6b5aa205a78dad82a0f327a34b820c1e1a91b6daf3fe2c2ac722d'
TOP = 228.90093881236
ROOF = 233.90093881236
X0, X1, YEND = 38.5, 93.641783, -38.8
OLD = [(42.120660268393, -38.37627502199), (44.286624883739, -30.67506750076)]
NEW = [(44.0, -43.8), (88.141783, -43.8)]
UP = bd.Vector(0, 0, 1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rod(x, y, z0, z1, radius):
    return refit._rod(bd.Vector(x, y, 0), UP, z0, z1, radius)


def cone(x, y, z0, z1, r0, r1):
    return refit._cone(bd.Vector(x, y, 0), UP, z0, z1, r0, r1)


def mapping(robot):
    rows = list(leaves(robot))
    return {r['label']: r['shape'] for r in rows}, rows


def verify_bearing_material(old, new):
    lower=refit._world_box(30,100,-100,0,200,225.9)
    compare(refit._shape(old.intersect(lower)),refit._shape(new.intersect(lower)))
    checks=[]
    for face in fixing_cylinders(old):
        cy=BRepAdaptor_Surface(face.wrapped,True).Cylinder()
        origin,direction=bd.Vector(cy.Location().Coord()),bd.Vector(cy.Axis().Direction().Coord())
        ends=[(v.center()-origin).dot(direction) for v in face.vertices()]
        # Imported compound cylindrical trims have an intermediate seam.
        # Split the guard there; OCCT otherwise silently returns an empty common.
        stations=sorted(set(round(t,8) for t in ends))
        material_volume=lost=blocked=0
        for lo,hi in zip(stations,stations[1:]):
            guard=refit._rod(origin,direction,lo+1e-5,hi-1e-5,1.575)
            material=refit._shape(old.intersect(guard))
            new_material=refit._shape(new.intersect(guard))
            # Independent clips avoid OCCT's false disjoint classification
            # when subtracting this coaxial annulus from its own parent.
            compare(material,new_material)
            material_volume+=material.volume
            lost+=max(0,material.volume-new_material.volume)
            blocked+=overlap(new,refit._rod(origin,direction,lo+1e-5,hi-1e-5,1.025-1e-5))
        assert lost<1e-5 and blocked<1e-6,(lost,blocked)
        checks.append(dict(axis_origin=list(origin),axis_direction=list(direction),
                           axial_span_mm=max(ends)-min(ends),guard_radius_mm=1.575,
                           original_bearing_volume_mm3=material_volume,lost_material_mm3=lost,
                           bore_obstruction_mm3=blocked,
                           verification='Unchanged complete lower geometry and independent annular clip face/volume comparisons'))
    assert len(checks)==2
    return checks


def removal_mask():
    # Enclose the two original seats and their old blended root. Preserve the
    # neighbouring roll mount's original analytic tapered seat explicitly.
    mask = refit._world_box(36.9, 49.5, -43.6, -25.4, TOP-.01, ROOF)
    protected = cone(54, -27.26975329173, 226.90093881236, ROOF, 5, 7)
    return refit._cut(mask, [protected])


def build():
    assert sha(SOURCE) == sha(FORMAL) == BASE_SHA
    robot = read_step(SOURCE)
    old, _ = mapping(robot)
    # Trim only the old roof wing, then extend the original plate in its own
    # plane. The tiny internal overlap removes a coincident boolean seam.
    stand = refit._cut(old['speaker_stand'], [refit._world_box(30, 100, -41.7, 0, TOP-3.00001, TOP+.01)])
    stand = refit._fuse(stand, refit._world_box(X0, X1, -41.701, YEND, TOP-3, TOP)).clean().fix()
    stand = refit._cut(stand, [rod(x, y, TOP-4, TOP+1, 1.35) for x, y in NEW]).clean().fix()
    stand.label, stand.color = old['speaker_stand'].label, old['speaker_stand'].color
    assert check_occurrence_shape(stand.wrapped)['reasons'] == []
    verify_tangent_bend(stand)
    verify_bearing_material(old['speaker_stand'], stand)
    head = refit._cut(old['head'], [removal_mask()]).clean().fix()
    for x, y in NEW:
        # Same original tapered root, continued 0.001 mm into the roof for a
        # robust fuse; the exterior roof and 3 mm blind-hole closure stay put.
        seat = cone(x, y, TOP, ROOF+.001, 3, 5.0004)
        seat = refit._cut(seat, [rod(x, y, TOP-.01, ROOF, 1)])
        head = refit._fuse(head, seat).clean().fix()
    head.label, head.color = old['head'].label, old['head'].color
    assert len(head.solids()) == 1
    added = refit._shape(head.cut(old['head']))
    bd.export_brep(added, WORK/'head-added.brep')
    bd.export_brep(stand, WORK/'stand-new.brep')
    bd.export_brep(head, WORK/'head-new.brep')
    fresh = rebuild(robot, {'head': head, 'speaker_stand': stand})
    bd.export_step(refit._encode_colours(fresh), CANDIDATE)
    report = dict(status='candidate; local verification pending', source_sha256=BASE_SHA,
                  candidate_sha256=sha(CANDIDATE), modified_labels=['head', 'speaker_stand'],
                  previous=json.loads(REPORT.read_text()), roof_hole_centers_xy_mm=NEW,
                  roof_hole_spacing_mm=NEW[1][0]-NEW[0][0], plate_thickness_mm=3,
                  straight_flange_bounds_mm=[X0, X1, -48.92324703103, YEND, TOP-3, TOP])
    (WORK/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('built', CANDIDATE, report['candidate_sha256'], flush=True)


def verify():
    report = json.loads((WORK/'report.json').read_text())
    assert sha(CANDIDATE) == report['candidate_sha256'] and sha(FORMAL) == BASE_SHA
    before, after = read_step(SOURCE), read_step(CANDIDATE)
    old, old_rows = mapping(before)
    new, new_rows = mapping(after)
    assert len(old_rows) == len(new_rows) == 135
    assert [r['path'] for r in old_rows] == [r['path'] for r in new_rows]
    for a, b in zip(old_rows, new_rows):
        if a['label'] not in ('speaker_stand', 'head'): compare(a['shape'], b['shape'])
        if a['shape'].color is not None:
            assert max(abs(x-y) for x,y in zip(a['shape'].color,b['shape'].color)) < 1e-6
    print('135 leaves, 133 unchanged geometry/colour/path checks passed',flush=True)
    a, b = measured_axes(before), measured_axes(after)
    assert len(a) == len(b) == 14
    for joint in a:
        for av,bv in zip(a[joint],b[joint]): assert max(abs(x-y) for x,y in zip(av,bv)) < 1e-5
    print('14 main servo axes preserved',flush=True)
    stand, head = new['speaker_stand'], new['head']
    validity = check_occurrence_shape(stand.wrapped)
    assert validity['solidCount'] == 1 and not validity['reasons'], validity
    report['smooth_surface'] = verify_tangent_bend(stand)
    report['speaker_bearing_material'] = verify_bearing_material(old['speaker_stand'], stand)
    axes=[]
    for face in stand.faces():
        ad=BRepAdaptor_Surface(face.wrapped,True)
        if (ad.GetType()==GeomAbs_Cylinder and abs(ad.Cylinder().Radius()-1.35)<1e-6
                and abs(ad.Cylinder().Axis().Direction().Z())>.999):
            cy=ad.Cylinder(); axes.append((cy.Location().X(),cy.Location().Y()))
    assert len(axes)==2
    for xy in NEW: assert min(math.dist(xy,p) for p in axes)<1e-6
    fixing=[]
    for x,y in NEW:
        annulus = refit._cut(rod(x,y,TOP,TOP+.0001,3), [rod(x,y,TOP-.001,TOP+.001,1.35)])
        # Actual coincident planar bearing faces, independently on both parts.
        area=[]
        for part,normal in ((stand,1),(head,-1)):
            candidates=[f for f in part.faces() if f.geom_type==bd.GeomType.PLANE
                        and abs(f.center().Z-TOP)<1e-6 and f.normal_at().Z*normal>.999]
            bearing=0
            for f in candidates:
                common=f.intersect(annulus)
                bearing+=sum(i.area for i in refit._items(common))
            area.append(bearing)
        assert min(area)>math.pi*(3**2-1.35**2)-1e-4, area
        open_stand=overlap(stand,rod(x,y,TOP-3.001,TOP+.001,1.35-1e-5))
        open_head=overlap(head,rod(x,y,TOP-.001,ROOF,1-1e-5))
        assert open_stand<1e-6 and open_head<1e-6
        for z in (TOP+.001,TOP+2.5,ROOF-.001):
            outer=3+.4*(z-TOP)
            for i in range(32):
                t=2*math.pi*i/32
                assert head.is_inside((x+(outer-.05)*math.cos(t),y+(outer-.05)*math.sin(t),z),tolerance=1e-6)
        # Screwdriver/head access from below: no neighbouring part within a
        # 6 mm diameter, 18 mm long approach cylinder.
        access=rod(x,y,TOP-21,TOP-3.0001,3)
        blocked=[]
        for row in new_rows:
            label,part=row['label'],row['shape']
            if label=='speaker_stand':continue
            vol=overlap(access,part)
            if vol>1e-6:blocked.append((label,vol))
        assert not blocked, (x,y,blocked)
        fixing.append(dict(center_xy_mm=[x,y],axis=[0,0,1],stand_hole_diameter_mm=2.7,
                           shell_pilot_diameter_mm=2,stand_grip_mm=3,shell_pilot_depth_mm=5,
                           retained_roof_closure_mm=3,bearing_areas_mm2=area,
                           bore_obstruction_mm3=[open_stand,open_head],tool_access_diameter_mm=6,
                           tool_access_length_mm=18,tool_access_clear=True))
        print('roof fixing verified',x,y,flush=True)
    # No old seat material remains, other than the original neighbouring mount
    # explicitly excluded from the removal mask. No shell exterior was altered.
    old_mask=refit._cut(removal_mask(),[cone(x,y,TOP,ROOF+.001,3,5.0004) for x,y in NEW])
    old_seat_remainder=overlap(head,old_mask)
    assert old_seat_remainder<1e-5,old_seat_remainder
    upper=refit._world_box(-300,300,-300,300,ROOF+.01,300)
    unchanged_roof=refit._shape(old['head'].intersect(upper))
    assert overlap(unchanged_roof,head)>unchanged_roof.volume-1e-5
    checks=[]
    for label in ('head','speaker_interface','speaker_bottom','speaker_top','head_roll_mount','head_yaw_to_roll','head_pitch_to_yaw'):
        gap=stand.distance_to(new[label]);vol=overlap(stand,new[label])
        passed=vol<1e-6 and (label in ('head','speaker_interface') or gap>=.2-1e-7)
        checks.append(dict(label=label,gap_mm=gap,overlap_mm3=vol,passed=passed));assert passed,checks[-1]
    old_valid=check_occurrence_shape(old['head'].wrapped)
    new_valid=check_occurrence_shape(head.wrapped)
    assert old_valid['reasons']==new_valid['reasons'],(old_valid,new_valid)
    report.update(status='local geometry verified; motion/visual pending',leaf_count=135,
                  unchanged_leaves=133,main_axes_preserved=14,stand_validity=validity,
                  head_validity_before=old_valid,head_validity_after=new_valid,
                  roof_fixing_checks=fixing,old_seat_remainder_mm3=old_seat_remainder,
                  local_clearance=checks,stand_volume_mm3=stand.volume,
                  plate_x_edge_distances_mm=[5.5,5.5])
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('geometry verified',fixing,flush=True)


def install():
    report=json.loads((WORK/'report.json').read_text())
    assert report['status']=='local geometry verified; motion/visual pending'
    assert sha(FORMAL)==BASE_SHA and sha(CANDIDATE)==report['candidate_sha256']
    motion=json.loads((WORK/'motion.json').read_text())
    assert motion['passed'] and motion['source_sha256']==report['candidate_sha256']
    assert report['visual_review_sha256']==report['candidate_sha256']
    assert report['standard_validation_no_new_faults']
    assert report['head_fault_locations_unchanged']
    fd,temporary=tempfile.mkstemp(prefix='.speaker-end-',suffix='.tmp',dir=FORMAL.parent)
    os.close(fd)
    try:
        shutil.copyfile(CANDIDATE,temporary)
        os.replace(temporary,FORMAL)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    reread,rows=mapping(read_step(FORMAL))
    assert len(rows)==135
    validity=check_occurrence_shape(reread['speaker_stand'].wrapped)
    assert validity['solidCount']==1 and not validity['reasons']
    report.update(status='installed and reread',installed_sha256=sha(FORMAL),
                  installed_leaf_count=len(rows),installed_stand_validity=validity,motion_checks=motion)
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print('installed',report['installed_sha256'],flush=True)


if __name__=='__main__':
    import sys
    {'build':build,'verify':verify,'install':install}[sys.argv[1]]()
