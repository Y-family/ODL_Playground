"""Reseat the speaker against the shell and shorten its original hanger.

CAD brief: millimetres, +X toward the eyes, +Z up. Edit only the three existing
speaker leaves, original orange support and original head wall, using the accepted STEP as input.
Retain the nearer two head fixing holes and the two upper speaker fixing holes.
Retain the short upper hanger, reshape its original curved transition,
and keep the original head wall as one solid with aligned sound holes. Require 0.2 mm local clearance, original mating faces and screw
material, unchanged assembly occurrences/colours and unchanged servo axes.
Run serially via run_memory_limited.py. Install only after local checks pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.OSD import OSD_Parallel, OSD_ThreadPool

import refit_upper_servo_interfaces_local as refit

OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
WORK = Path('/tmp/speaker-rework')
REPORT = ROOT / 'cad/local_fix/SPEAKER_LAYOUT_REPAIR_REPORT.json'

BASE_SHA256 = '953778525984d6df474c1df6250235028db87239db882a3d41f012e2d022ddd5'
SPEAKER_DX_MM = 11.757119393403315
SPEAKER_DY_MM = -10.604389057992051
SPEAKER_DZ_MM = 3.743918615470619
SPEAKER_CENTER = (58.242880606596685, -61.85036226764758, 206.25608138452938)
SPEAKER_AXIS = (-0.09795978200223937, 0.08997993848163868, 0.2472178491822183)
SPEAKER_ANGLE_DEG = 16.303761228070062
FRONT_CENTER = (70.0, -72.45475132563963, 210.0)
FRONT_NORMAL = (0.0, -0.9396926207859084, 0.3420201433256687)
SEAT_OFFSET_MM = 140.159425259972
STAND_HEAD_X_MIN_MM = 38.5
STAND_TOP_BAND_Z_MM = 214.0
STAND_ROOF_Z_MM = 223.5
ARM_X0_MM = 48.0
ARM_X1_MM = 81.0
ARM_Y_SPLIT_MM = -41.8
SPEAKER_LABELS = ('speaker_interface', 'speaker_bottom', 'speaker_top')
MODIFIED_LABELS = (*SPEAKER_LABELS, 'speaker_stand', 'head')
LOCAL_LABELS = (*MODIFIED_LABELS, 'head_roll_mount', 'head_yaw_to_roll',
                'head_pitch_to_yaw', 'head_bot_sheet')


def speaker_pose(shape):
    return shape.rotate(bd.Axis(SPEAKER_CENTER, SPEAKER_AXIS), SPEAKER_ANGLE_DEG).moved(
        bd.Pos(SPEAKER_DX_MM, SPEAKER_DY_MM, SPEAKER_DZ_MM))


def seat_plane():
    return bd.Plane(origin=bd.Vector(FRONT_CENTER)+bd.Vector(FRONT_NORMAL)*.25,
                    x_dir=(1, 0, 0), z_dir=FRONT_NORMAL)


def grille_positions():
    import math
    return [(3.2*(a+.5*b), 3.2*math.sqrt(3)/2*b)
            for a in range(-6,7) for b in range(-6,7)
            if (3.2*(a+.5*b))**2+(3.2*math.sqrt(3)/2*b)**2 <= 16.3**2]


def reshape_wall(head, posed):
    """Re-dimension the original wall from its exterior; no new mounting part."""
    from compact_head_local import clip as clip_axis, affine
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.TopTools import TopTools_ListOfShape
    faces=head.faces(); normal=bd.Vector(-.2630837400590191,-.935408853542068,.2362143569550172)
    assert len(faces[395].inner_wires())==92 and len(faces[531].inner_wires())==28
    yc=-70.74202; factor=(16-11.15798)/16
    boxes=[wire.bounding_box() for i in (395,531) for wire in faces[i].inner_wires()]
    guard=refit._world_box(min(b.min.X for b in boxes)-2,max(b.max.X for b in boxes)+2,
                           -100,-60,min(b.min.Z for b in boxes)-2,max(b.max.Z for b in boxes)+2)
    # A hole cut by the compression seam becomes a notch in both outer wires.
    # Reconstruct the original analytic wall, including those open notches.
    wall_plane=bd.Plane(origin=faces[531].center(),z_dir=normal)
    wall=wall_plane*refit._world_box(-300,300,-300,300,-3.01,0)
    xz=refit._world_box(guard.bounding_box().min.X,guard.bounding_box().max.X,
                       -200,0,guard.bounding_box().min.Z,guard.bounding_box().max.Z)
    wall=refit._shape(wall.intersect(xz))
    low=clip_axis(wall,1,-200,yc); middle=clip_axis(wall,1,yc,yc+16)
    middle=affine(middle,[[1,0,0,0],[0,factor,0,yc*(1-factor)],[0,0,1,0],[0,0,0,1]])
    high=clip_axis(wall,1,yc+16,0)
    if high: high=high.moved(bd.Pos(0,-11.15798,0))
    patch=refit._shape(refit._fuse(*[piece for piece in (low,middle,high) if piece is not None]).intersect(guard))
    result=refit._fuse(head,patch).clean()
    print('Old grille restored from both original exterior contours',flush=True)
    plane=seat_plane()
    region=refit._world_box(48,94,-100,-60,190,231)
    taper=plane*bd.Cone(20.5,26.5,10,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
    outside=plane*refit._world_box(-400,400,-400,400,0,400)
    material=[]
    for face in result.faces():
        box=face.bounding_box()
        if (box.min.X<94 and box.max.X>48 and box.min.Z<231 and box.max.Z>190
            and box.max.Y<-60 and face.normal_at().Y<-.7 and face.area>20):
            swept=bd.extrude(face,dir=(0,1,0),amount=12)
            try:
                piece=refit._shape(refit._shape(swept.intersect(region)).intersect(outside)).intersect(taper)
            except ValueError:
                continue
            material.extend(item for item in refit._items(piece) if item.volume>1e-6)
    result=refit._fuse(result,*material).clean()
    # Shorten the upper, unused parts of the two original floor screw pillars.
    # Retain the original entry bores, threaded portion and exterior wall.
    inside=plane*refit._world_box(-400,400,-400,400,-400,0)
    cutters=[refit._shape(refit._world_box(x-4.001,x+4.001,y-6.5,y+6.5,
                           190.500939,230.900939).intersect(inside))
             for x,y in ((69,-72.74202),(94,-67.9))]
    box=bd.Compound(list(posed.values())).moved(plane.location.inverse()).bounding_box()
    cutters.append(plane*refit._world_box(box.min.X-.2,box.max.X+.2,
                   box.min.Y-.2,box.max.Y+.2,-3.2,0))
    result=refit._cut(result,cutters).clean()
    # Reopen circular bores along the actual acoustic normal. One serial cut.
    holes=[plane*refit._rod(bd.Vector(x,y,0),bd.Vector(0,0,1),-.01,12,1)
           for x,y in grille_positions()]
    args=TopTools_ListOfShape(); args.Append(result.wrapped)
    tools=TopTools_ListOfShape()
    for hole in holes: tools.Append(hole.wrapped)
    operation=BRepAlgoAPI_Cut(); operation.SetArguments(args); operation.SetTools(tools)
    operation.SetRunParallel(False); operation.Build(); assert operation.IsDone()
    result=bd.Compound(operation.Shape()).clean()
    result.label, result.color=head.label,head.color
    print('Speaker wall seating and',len(holes),'aligned round holes built',flush=True)
    return result


def reshape_stand(stand, interface):
    from compact_head_local import clip as clip_axis, affine
    roof=clip_axis(stand,2,223.5,260)
    fixed=clip_axis(roof,1,-41.8,150); arm=clip_axis(roof,1,-150,-41.8)
    low=clip_axis(arm,0,-200,48); middle=clip_axis(arm,0,48,81)
    high=speaker_pose(clip_axis(arm,0,77,200))
    lower=speaker_pose(clip_axis(stand,2,-200,223.5))
    point=bd.Vector(81,-48,225.5)
    change=speaker_pose(bd.Vertex(*point)).center()-point; span=33
    matrix=[[1+change.X/span,0,0,-48*change.X/span],
            [change.Y/span,1,0,-48*change.Y/span],
            [change.Z/span,0,1,-48*change.Z/span],[0,0,0,1]]
    result=refit._fuse(fixed,low,affine(middle,matrix),high,lower).clean().fix()
    # Trim only the small transition interference; retain both rigidly moved
    # original speaker screw seats and both unchanged roof seats.
    result=refit._cut(result,[interface]).clean().fix()
    # Rebuild the original low transition through its actual sections. The
    # former spline seam at X40 gained a self-interference flag after STEP
    # round-trip; a ruled transition exports with consistent shared edges.
    root=refit._shape(result.intersect(refit._world_box(38,48,-100,-41.8,223.5,260)))
    sections=[]
    for x in (38.5001,40,44,47.9999):
        sliced=root.intersect(bd.Plane(origin=(x,0,0),x_dir=(0,1,0),z_dir=(1,0,0)))
        faces=[f for item in refit._items(sliced) for f in item.faces()]
        sections.append(max(faces,key=lambda f:f.area).outer_wire())
    transition=bd.Solid.make_loft(sections,ruled=True)
    result=refit._fuse(refit._cut(result,[refit._world_box(38,47.9998,-100,-41.8,223.5,260)]),transition).clean().fix()
    result.label,result.color=stand.label,stand.color
    return result


def children(node):
    return list(getattr(node, 'children', None) or [])


def leaves(node, path=''):
    kids = children(node)
    if not kids:
        yield dict(path=path, label=node.label, shape=node)
        return
    for index, child in enumerate(kids):
        yield from leaves(child, path + '/' + child.label + f'[{index}]')


def shape_only(shape, label=None):
    solids = list(shape.solids())
    if len(solids) != 1:
        raise ValueError(f'{label or shape.label}: expected one solid, got {len(solids)}')
    result = solids[0]
    result.label = label or shape.label
    result.color = getattr(shape, 'color', None)
    return result


def overlap(left, right):
    if left.distance_to(right) > 1e-7:
        return 0.0
    result = left.intersect(right)
    return sum(piece.volume for piece in refit._items(result) if piece.volume > 0.0)


def roof_fixing_axes(shape):
    """Return the head-side fixing axes present in the roof."""
    rows = []
    for face in shape.faces():
        adaptor = BRepAdaptor_Surface(face.wrapped, True)
        if adaptor.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = adaptor.Cylinder()
        direction = cylinder.Axis().Direction()
        if abs(cylinder.Radius() - 1.35) > 1e-6 or abs(direction.Z()) < .999999:
            continue
        point = cylinder.Location()
        rows.append((round(point.X(), 6), round(point.Y(), 6), round(point.Z(), 6)))
    return sorted(rows)


def bbox_near(left, right, margin=0.2):
    a, b = left.bounding_box(), right.bounding_box()
    return not any((a.max.X < b.min.X - margin or b.max.X < a.min.X - margin,
                    a.max.Y < b.min.Y - margin or b.max.Y < a.min.Y - margin,
                    a.max.Z < b.min.Z - margin or b.max.Z < a.min.Z - margin))


def by_label(rows, label):
    found = [row for row in rows if row['label'] == label]
    if len(found) != 1:
        raise ValueError(f'{label}: expected one leaf, found {len(found)}')
    return found[0]


def clip(shape, x0, x1, z0, z1):
    """Keep one axis-aligned region of an existing solid."""
    return refit._shape(shape.intersect(refit._world_box(x0, x1, -100, 50, z0, z1)))


def make_replacements(rows):
    replacements={label:shape_only(speaker_pose(by_label(rows,label)['shape']),label)
                  for label in SPEAKER_LABELS}
    stand=by_label(rows,'speaker_stand')['shape']
    replacements['speaker_stand']=shape_only(reshape_stand(stand,replacements['speaker_interface']),'speaker_stand')
    head=by_label(rows,'head')['shape']
    replacements['head']=shape_only(reshape_wall(head,{label:replacements[label] for label in SPEAKER_LABELS}),'head')
    for label,new in replacements.items():
        bd.export_brep(new,WORK/(label+'-built.brep'))
    added=refit._shape(replacements['head'].cut(head))
    bd.export_brep(added,WORK/'head-added.brep')
    return replacements


def rebuild(node, replacements):
    kids = children(node)
    if kids:
        result = bd.Compound(children=[rebuild(child, replacements) for child in kids],
                             label=node.label)
        result.location = node.location
    else:
        result = replacements.get(node.label, node)
    result.label = node.label
    result.color = getattr(node, 'color', None) if node.label not in replacements else replacements[node.label].color
    return result


def build():
    if hashlib.sha256(TARGET.read_bytes()).hexdigest() != BASE_SHA256:
        raise SystemExit('formal STEP changed; refuse to build on an unexpected input')
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    rows = list(leaves(robot))
    if len(rows) != 135:
        raise ValueError(f'expected 135 leaves, got {len(rows)}')
    replacements = make_replacements(rows)
    candidate = WORK / 'review.step'
    bd.export_step(refit._encode_colours(rebuild(robot, replacements)), candidate)
    report = {
        'status': 'candidate built; exact local verification pending',
        'source_sha256': BASE_SHA256,
        'candidate': str(candidate),
        'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'speaker_translation_mm': [SPEAKER_DX_MM, SPEAKER_DY_MM, SPEAKER_DZ_MM],
        'speaker_rotation': {'angle_deg':SPEAKER_ANGLE_DEG,'axis':SPEAKER_AXIS,'center_mm':SPEAKER_CENTER},
        'face_seating': {'center_mm':FRONT_CENTER,'normal':FRONT_NORMAL,'normal_gap_mm':.25},
        'grille': {'hole_diameter_mm':2.0,'hole_count':len(grille_positions()),'center_mm':list(seat_plane().origin),'normal':FRONT_NORMAL,'old_holes_restored':True},
        'speaker_stand_refit': {
            'original_transition_x_span_mm': [48.0,81.0],
            'transition_reference_displacement_mm': [8.960830439030701,-3.476964289408855,.04172194309109045],
            'head_fixings_retained':2,'speaker_fixings_retained':2,
            'head_fixing_positions_unchanged':True,
            'original_upper_band_follows_speaker_rigid_pose':True,
            'transition_interface_overlap_removed_mm3':.3550886091786761,
        },
        'modified': {},
    }
    for label in MODIFIED_LABELS:
        old = by_label(rows, label)['shape']
        new = replacements[label]
        box = new.bounding_box()
        validity = check_occurrence_shape(new.wrapped)
        if validity['solidCount'] != 1 or (validity['reasons'] and not (label=='head' and validity['reasons']==['selfIntersecting'])):
            raise ValueError((label, validity))
        report['modified'][label] = {
            'volume_before_mm3': old.volume,
            'volume_after_mm3': new.volume,
            'bounds_after_mm': list(box.min) + list(box.max),
            'validity': validity,
        }
    report['leaf_count'] = len(rows)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('built', candidate, report['candidate_sha256'], flush=True)


def verify():
    from full_motion_current import measured_axes
    from remove_s01_s06_upper_horns import compare
    report = json.loads(REPORT.read_text())
    if report['source_sha256'] != BASE_SHA256:
        raise ValueError('report source does not match formal input')
    candidate = Path(report['candidate'])
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != report['candidate_sha256']:
        raise ValueError('candidate changed after build')
    original_robot = read_step(TARGET)
    candidate_robot = read_step(candidate)
    before = list(leaves(original_robot))
    after = list(leaves(candidate_robot))
    if len(before) != 135 or len(after) != 135:
        raise ValueError((len(before), len(after)))
    for index,(old, new) in enumerate(zip(before, after)):
        if old['path'] != new['path'] or old['label'] != new['label']:
            raise ValueError(('leaf order changed', old['path'], new['path']))
        if old['label'] not in MODIFIED_LABELS:
            compare(old['shape'], new['shape'])
        assert (old['shape'].color is None) == (new['shape'].color is None)
        if old['shape'].color is not None:
            assert max(abs(a-b) for a,b in zip(old['shape'].color,new['shape'].color)) < 1e-6
        if index%30==0: print('Preserved assembly leaves checked',index+1,flush=True)
    old_axes, new_axes = measured_axes(original_robot), measured_axes(candidate_robot)
    assert len(old_axes) == len(new_axes) == 14
    for joint in old_axes:
        for expected, actual in zip(old_axes[joint],new_axes[joint]):
            assert max(abs(a-b) for a,b in zip(expected,actual)) < 1e-5
    old_by = {r['label']: r['shape'] for r in before}
    new_by = {r['label']: r['shape'] for r in after}
    replacements = {label:bd.import_brep(WORK/(label+'-built.brep')) for label in MODIFIED_LABELS}
    original_roof_axes = [axis for axis in roof_fixing_axes(old_by['speaker_stand'])
                          if axis[0] > STAND_HEAD_X_MIN_MM]
    retained_roof_axes = roof_fixing_axes(new_by['speaker_stand'])
    if len(retained_roof_axes)!=2 or original_roof_axes != retained_roof_axes:
        raise ValueError(('head-side roof fixing axes changed', original_roof_axes, retained_roof_axes))
    for label in MODIFIED_LABELS:
        expected, actual = replacements[label], new_by[label]
        if abs(expected.volume - actual.volume) > 0.01:
            raise ValueError((label, 'volume mismatch', expected.volume, actual.volume))
        if max(abs(a - b) for a, b in zip(
                list(expected.bounding_box().min) + list(expected.bounding_box().max),
                list(actual.bounding_box().min) + list(actual.bounding_box().max))) > 1e-4:
            raise ValueError((label, 'bounds mismatch'))
        validity = check_occurrence_shape(actual.wrapped)
        assert validity['solidCount'] == 1 and (not validity['reasons'] or (label=='head' and validity['reasons']==['selfIntersecting'])), (label,validity)

    # Preserve the upper two original speaker screw cylinders and surrounding
    # material. The lower two screws belonged to the removed lower hanger.
    mounting = []
    oldstand, stand = old_by['speaker_stand'], new_by['speaker_stand']
    for face in oldstand.faces():
        ad = BRepAdaptor_Surface(face.wrapped, True)
        if ad.GetType() != GeomAbs_Cylinder:
            continue
        cy = ad.Cylinder()
        if abs(cy.Radius() - 1.025) > 1e-6 or abs(cy.Axis().Direction().Z()) > .99 or cy.Location().Z() < 214:
            continue
        expected = speaker_pose(face)
        matches = [f for f in stand.faces() if f.geom_type == bd.GeomType.CYLINDER
                   and abs(f.area-expected.area)<1e-5 and f.distance_to(expected)<1e-6]
        assert matches, ('upper speaker fixing cylinder lost', list(cy.Location().Coord()))
        # A short band made from the original solid is retained verbatim.
        # Check actual material around the screw, not only the hole axis.
        reference = speaker_pose(oldstand)
        point = expected.center()
        moved_cylinder=BRepAdaptor_Surface(expected.wrapped,True).Cylinder()
        direction=bd.Vector(moved_cylinder.Axis().Direction().Coord())
        axis_origin=bd.Vector(moved_cylinder.Location().Coord())
        axial = axis_origin + direction * (point-axis_origin).dot(direction)
        radial = (point-axial).normalized()
        tangent = direction.cross(radial).normalized()
        material_samples = []
        import math
        for index in range(16):
            angle = 2*math.pi*index/16
            test = axial + (radial*math.cos(angle)+tangent*math.sin(angle))*(cy.Radius()+.55)
            old_inside = reference.solids()[0].is_inside(test)
            new_inside = stand.solids()[0].is_inside(test)
            assert old_inside and new_inside, (index,list(test),old_inside,new_inside)
            material_samples.append(list(test))
        mounting.append(dict(original_axis=list(cy.Location().Coord()),
                             translation_mm=[SPEAKER_DX_MM,SPEAKER_DY_MM,SPEAKER_DZ_MM],
                             rotation_deg=SPEAKER_ANGLE_DEG,
                             moved_axis=list(moved_cylinder.Location().Coord()),
                             moved_axis_direction=list(moved_cylinder.Axis().Direction().Coord()),
                             bearing_radial_material_checked_mm=.55,
                             material_samples=material_samples))
    assert len(mounting)==2, mounting
    print('Assembly preservation and upper speaker screws passed',flush=True)

    head = new_by['head']
    oldhead = old_by['head']
    plane = seat_plane()
    normal = bd.Vector(FRONT_NORMAL)
    seatfaces = bd.Compound([f for f in head.faces()
        if f.geom_type == bd.GeomType.PLANE
        and abs(f.normal_at().dot(normal)) > .999999
        and abs(normal.dot(f.center())-SEAT_OFFSET_MM) < 1e-5])
    import math
    positions = grille_positions()
    points = [(18.8*math.cos(i*math.pi/32),18.8*math.sin(i*math.pi/32)) for i in range(64)]
    points += [(x,y) for x in range(-18,19,3) for y in range(-18,19,3)
               if x*x+y*y <= 18.8**2
               and min((x-a)**2+(y-b)**2 for a,b in positions) > 1.01**2]
    gaps = [bd.Vertex(*plane.from_local_coords(bd.Vector(x,y,-.25))).distance_to(seatfaces)
            for x,y in points]
    assert gaps and max(abs(g-.25) for g in gaps) < 1e-5, (min(gaps),max(gaps))
    # Check every former opening, including holes changed into ellipses by the
    # old compression. A sample lies inside the actual restored wall.
    restored=[]
    for index in (395,531):
        face=oldhead.faces()[index]
        for wire in face.inner_wires():
            sample=wire.center()-face.normal_at()*.05
            assert head.solids()[0].is_inside(sample), ('old grille still open',list(sample))
            restored.append(list(sample))
    restored_wall_samples=[]
    for face in oldhead.faces():
        box=face.bounding_box()
        if (box.min.X>5 and box.max.X<49 and box.min.Y<-60 and box.max.Y<-59
            and box.min.Z>188 and box.max.Z<226
            and face.geom_type!=bd.GeomType.PLANE and face.area<20):
            assert head.solids()[0].is_inside(face.center()), ('old split opening remains',list(face.center()))
            restored_wall_samples.append(list(face.center()))
    cylinders=[]
    for face in head.faces():
        ad=BRepAdaptor_Surface(face.wrapped,True)
        if ad.GetType()!=GeomAbs_Cylinder: continue
        cy=ad.Cylinder()
        if abs(cy.Radius()-1)>1e-6 or abs(bd.Vector(cy.Axis().Direction().Coord()).dot(normal))<.999999:
            continue
        local=plane.to_local_coords(bd.Vector(cy.Location().Coord()))
        cylinders.append((round(local.X,5),round(local.Y,5)))
    expected=sorted((round(x,5),round(y,5)) for x,y in positions)
    assert sorted(set(cylinders))==expected, (len(set(cylinders)),len(expected))
    front=max((f for f in new_by['speaker_top'].faces()
               if f.geom_type==bd.GeomType.PLANE and f.normal_at().dot(normal)>.999999),
              key=lambda f:f.area)
    mean_x=sum(x for x,y in set(cylinders))/len(set(cylinders))
    mean_y=sum(y for x,y in set(cylinders))/len(set(cylinders))
    actual_grille_center=plane.from_local_coords(bd.Vector(mean_x,mean_y,0))
    lateral=actual_grille_center-front.center()
    lateral-=normal*lateral.dot(normal)
    angle_error=math.degrees(math.acos(max(-1,min(1,front.normal_at().dot(normal)))))
    assert lateral.length<1e-5 and angle_error<1e-5
    coverage=[]
    for x,y in positions:
        for angle in range(0,360,45):
            a=x+math.cos(math.radians(angle)); b=y+math.sin(math.radians(angle))
            sample=plane.from_local_coords(bd.Vector(a,b,-.25))
            coverage.append(bd.Vertex(*sample).distance_to(front))
    assert max(coverage)<1e-6, max(coverage)
    def floor_screws(shape):
        rows=[]
        for face in shape.faces():
            ad=BRepAdaptor_Surface(face.wrapped,True)
            if ad.GetType()!=GeomAbs_Cylinder: continue
            cy=ad.Cylinder(); b=face.bounding_box()
            if min(abs(cy.Radius()-r) for r in (1.6,2.0))<1e-6 and b.min.X>60 and b.max.Y<0 and b.max.Z<190.501:
                rows.append([round(cy.Radius(),5),round(face.area,5),
                             *[round(v,5) for v in list(b.min)+list(b.max)]])
        return sorted(rows)
    original_screws,actual_screws=floor_screws(oldhead),floor_screws(head)
    assert original_screws==actual_screws and original_screws
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Check
    def faults(shape):
        checker=BRepAlgoAPI_Check(shape.wrapped,True,True)
        return [dict(status=str(err.GetCheckStatus()),boxes=[
            [round(v,5) for v in list(bd.Shape(q).bounding_box().min)+list(bd.Shape(q).bounding_box().max)]
            for q in err.GetFaultyShapes1()]) for err in checker.Result()]
    old_faults,new_faults=faults(oldhead),faults(head)
    assert all(row in old_faults for row in new_faults), (old_faults,new_faults)
    critical=[]
    for joint,fixed_label,angles in (
            ('S14','head_yaw_to_roll',(-30,0,30)),
            ('S11','body_middle_top',(-20,39))):
        for angle in angles:
            posed_head=head.rotate(bd.Axis(*new_axes[joint]),angle)
            fixed=new_by[fixed_label]
            gap=posed_head.distance_to(fixed)
            common=overlap(posed_head,fixed) if gap<1e-7 else 0
            assert gap>=.2-1e-7 and common<1e-6,(joint,angle,gap,common)
            critical.append(dict(joint=joint,fixed=fixed_label,angle_deg=angle,
                                 gap_mm=gap,overlap_mm3=common))
    report['head_geometry_checks']=dict(no_new_self_intersections=True,
        original_faults=old_faults,current_faults=new_faults,
        speaker_face_sample_count=len(gaps),face_gap_range_mm=[min(gaps),max(gaps)],
        restored_old_opening_sample_count=len(restored),
        restored_old_hole_wall_sample_count=len(restored_wall_samples),
        aligned_circular_hole_count=len(expected),hole_axis_radius_mm=1,
        grille_center_error_mm=lateral.length,grille_axis_angle_error_deg=angle_error,
        actual_grille_center_mm=list(actual_grille_center),
        actual_speaker_front_center_mm=list(front.center()),
        grille_edge_coverage_sample_count=len(coverage),
        grille_edge_to_speaker_face_max_distance_mm=max(coverage),
        original_floor_screw_surfaces_preserved=actual_screws,
        complete_head_critical_samples=critical)
    print('Speaker face, all former openings, new grille and floor screws passed',flush=True)

    # These current-candidate checks run once in verify_speaker_motion.py.
    # Bind them to the exact bytes just reread instead of repeating booleans.
    motion = json.loads((ROOT/'cad/local_fix/SPEAKER_LAYOUT_MOTION_REPORT.json').read_text())
    assert motion['completed'] and motion['passed'] and motion['source_sha256']==report['candidate_sha256']
    static = motion['static']
    assert len(static)>=126 and all(row['passed'] for row in static)
    paths = {label: by_label(after,label)['path'] for label in LOCAL_LABELS}
    def pair(a,b):
        return next(row for row in static if {row['moving'],row['fixed']}=={paths[a],paths[b]})
    speaker_head = [dict(part=label,gap_mm=pair(label,'head')['gap_mm'],
                         overlap_mm3=pair(label,'head')['overlap_mm3']) for label in SPEAKER_LABELS]
    stand_head_overlap = pair('speaker_stand','head')['overlap_mm3']
    stand_speaker = [dict(part=label,gap_mm=pair(label,'speaker_stand')['gap_mm'],
                         overlap_mm3=pair(label,'speaker_stand')['overlap_mm3']) for label in SPEAKER_LABELS]

    report.update({
        'status': 'candidate verified; formal STEP unchanged',
        'static_local_checks': static,
        'speaker_head_clearance': speaker_head,
        'speaker_stand_head_overlap_mm3': stand_head_overlap,
        'speaker_stand_clearance': stand_speaker,
        'roof_fixing_axes_before_after': {
            'before': original_roof_axes,
            'after': retained_roof_axes,
        },
        'candidate_leaf_count': len(after),
        'unchanged_leaf_count': 130,
        'main_axes_preserved': 14,
        'upper_speaker_fixings': mounting,
        'formal_sha256_before_install': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        'verified': True,
    })
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('verified candidate', json.dumps({
        'speaker_head_clearance': speaker_head,
        'stand_bounds': report['modified']['speaker_stand']['bounds_after_mm'],
        'stand_volume_mm3': report['modified']['speaker_stand']['volume_after_mm3'],
    }), flush=True)


def install():
    from full_motion_current import measured_axes
    report = json.loads(REPORT.read_text())
    if not report.get('verified'):
        raise SystemExit('run verify before install')
    motion = json.loads((ROOT/'cad/local_fix/SPEAKER_LAYOUT_MOTION_REPORT.json').read_text())
    assert motion['completed'] and motion['passed']
    assert motion['source_sha256']==report['candidate_sha256']
    assert report['unchanged_leaf_count']==130 and report['main_axes_preserved']==14
    validation=json.loads((WORK/'step-validation.json').read_text())
    assert validation['occurrenceCount']==5 and validation['failureCount']==1
    assert len(validation['parts'])==1 and validation['parts'][0]['name']=='head'
    assert validation['parts'][0]['reasons']==['selfIntersecting'] and not validation['errors']
    assert report['head_geometry_checks']['no_new_self_intersections']
    snapshot=ROOT/'cad/local_fix/SPEAKER_LAYOUT_REVIEW.png'
    visual=json.loads((WORK/'visual-review.json').read_text())
    assert visual['source_sha256']==report['candidate_sha256']
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==visual['snapshot_sha256']
    if hashlib.sha256(TARGET.read_bytes()).hexdigest() != BASE_SHA256:
        raise SystemExit('formal STEP changed since candidate build')
    candidate = Path(report['candidate'])
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != report['candidate_sha256']:
        raise SystemExit('candidate changed since verification')
    fd, temporary = tempfile.mkstemp(prefix='.speaker-layout-', suffix='.tmp', dir=TARGET.parent)
    try:
        with os.fdopen(fd, 'wb') as out, candidate.open('rb') as src:
            shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, TARGET)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    installed = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if installed != report['candidate_sha256']:
        raise ValueError('installed SHA mismatch')
    reread=read_step(TARGET)
    installed_rows=list(leaves(reread))
    assert len(installed_rows)==135 and len(measured_axes(reread))==14
    for label in MODIFIED_LABELS:
        actual=by_label(installed_rows,label)['shape']
        check=check_occurrence_shape(actual.wrapped)
        assert check['solidCount']==1 and (not check['reasons'] or (label=='head' and check['reasons']==['selfIntersecting'])),(label,check)
    report.update(status='installed and locally verified; appearance awaiting user review',
                  installed=True, installed_sha256=installed,
                  installed_reread_leaf_count=len(installed_rows),
                  step_validation=validation, visual_review=visual,
                  motion_report=str(ROOT/'cad/local_fix/SPEAKER_LAYOUT_MOTION_REPORT.json'))
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('installed', installed, flush=True)


def main():
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in {'build', 'verify', 'install'}:
        raise SystemExit('usage: repair_speaker_layout_local.py build|verify|install')
    {'build': build, 'verify': verify, 'install': install}[sys.argv[1]]()


if __name__ == '__main__':
    main()
