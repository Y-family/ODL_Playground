"""Local review: +X is forward, millimetres; original complete STEP is baseline.

Seat S03 case tail at yellow X=7.625 by translating its entire downstream
chain +12.185. Remap four existing case screws and remove cavity interference.
Restore original covers; trim only right cover's upper inner skirt on a circular
joint envelope. Keep component tree, colours, labels, and all other placements.
Validate contact, case/horn clearance, bores, single solids and retained cover.
"""
import json
import math
from pathlib import Path
from cadgen import read_step, step, report, build123d as bd
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from build_left_leg_from_approved import remap_group, cylinder

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'cad/local_fix/open_duck_mini_complete_14_zp10d_numbered.step'
FORWARD_MM = 12.185
SKIRT_CLEARANCE_MM = 1.0
MOVING = {'right_cache', 'knee_and_ankle_assembly <3>',
          'knee_and_ankle_assembly <1>', 'foot_assembly <1>', 'Frame <1>'}

def move_tree(part, shift):
    if part.children:
        return bd.Compound(children=[move_tree(c,shift) for c in part.children],label=part.label)
    return part.moved(shift)

@step(out='../../cad/local_fix/open_duck_mini_right_leg_forward_review.step')
def right_leg_forward():
    report('Read approved complete model; restore full cover profile')
    robot = read_step(SOURCE)
    def tree(p): return (p.label,[tree(c) for c in p.children])
    original_tree = tree(robot)
    group = m._one_child(robot, 'right_roll_to_pitch_assembly <1>')
    yellow = m._one_child(group, 'right_roll_to_pitch')
    servo = m._one_child(group, 'S03_RIGHT_HIP_PITCH_ZP10D')
    body, passive, active = v._servo_parts(servo)
    shift = bd.Pos(FORWARD_MM, 0, 0)
    moved_servo = move_tree(servo,shift)
    moved_body, moved_passive, moved_active = v._servo_parts(moved_servo)
    vendor = max(m._children(read_step(v.VENDOR)), key=lambda p:p.volume)
    old_lines = v._case_mount_lines(body, vendor)
    new_lines = v._case_mount_lines(moved_body, vendor)
    report('Relocate existing yellow case screw bores')
    adapted = yellow
    for old, new in zip(old_lines, new_lines):
        assert (new[0]-old[0]-bd.Vector(FORWARD_MM,0,0)).length < .001
        adapted = remap_group(adapted, [old], [new], old[1],
                              old[0].dot(old[1]), (1.0, 2.5))
    # Shorten the existing mounting cheeks by the same forward offset, keeping
    # their original free-edge clearance from the moving black thigh plates.
    end_x = -18.46+FORWARD_MM
    end_trim = bd.Box(end_x+200,140.9,200,align=(bd.Align.MIN,)*3).moved(bd.Pos(-200,-200,-100))
    adapted = adapted.cut(end_trim).clean()
    cut_result = adapted.cut(moved_body).clean()
    print('yellow cut solids', [(s.volume,s.is_valid) for s in cut_result.solids()],flush=True)
    adapted = m._require_one_solid(cut_result, yellow.label,
                                   max_disconnected_trim=4.0)
    adapted.color = yellow.color
    assert adapted.is_valid
    overlap = m._intersection_volume(adapted,moved_body)
    assert overlap < .05, overlap
    assert abs(moved_body.bounding_box().max.X-7.625) < .001
    assert adapted.distance_to(moved_body) < .001
    horn_gaps = [adapted.distance_to(h) for h in (moved_passive,moved_active)]
    assert min(horn_gaps) > .5, horn_gaps
    host_lines = v._cylinder_lines(adapted, 1.0, bd.Vector(0,1,0))
    for p,d in new_lines:
        assert any(v._line_distance(p,d,q,e)<.001 for q,e in host_lines), (p,d)
    report('Local circular relief in upper inner cover skirt')
    original_cover = m._one_child(robot, 'right_cache')
    cover = original_cover.moved(shift)
    axis, center = v._axis(moved_servo)
    # Circular swept envelope is conservative over every pitch angle. Cut only
    # the inner depth occupied by yellow, retaining the external cover face.
    radius = max(math.hypot(p.X-center.X,p.Z-center.Z)
                 for p in (vertex.center() for vertex in adapted.vertices()))
    radius += SKIRT_CLEARANCE_MM
    ylimit = adapted.bounding_box().min.Y-SKIRT_CLEARANCE_MM
    tool = cylinder(center, bd.Vector(0,1,0), ylimit, 200, radius)
    trimmed = m._require_one_solid(cover.cut(tool).clean(), cover.label,
                                   max_disconnected_trim=0)
    trimmed.color = original_cover.color
    assert trimmed.is_valid
    assert abs(trimmed.bounding_box().min.Z-cover.bounding_box().min.Z)<.001
    gaps=[]
    for angle in range(-30,71,5):
        posed=trimmed.rotate(bd.Axis(center,axis),angle)
        gap=posed.distance_to(adapted)
        assert gap >= SKIRT_CLEARANCE_MM-.001, (angle,gap)
        gaps.append({'angle_deg':angle,'gap_mm':gap})
    children=[]
    for child in robot.children:
        if child.label==group.label:
            children.append(bd.Compound(children=[adapted if p.label==yellow.label
                            else moved_servo if p.label==servo.label else p
                            for p in group.children], label=group.label))
        elif child.label=='right_cache': children.append(trimmed)
        elif child.label in MOVING: children.append(move_tree(child,shift))
        else: children.append(child)
    result=bd.Compound(children=children,label=robot.label)
    assert tree(result)==original_tree, (tree(result),original_tree)
    data=dict(forward_mm=FORWARD_MM,tail_x_mm=moved_body.bounding_box().max.X,
              case_yellow_overlap_mm3=overlap,horn_yellow_gaps_mm=horn_gaps,
              case_holes=[list(p) for p,d in new_lines],
              cover_retained_percent=100*trimmed.volume/cover.volume,
              cover_relief_radius_mm=radius,cover_relief_y_start_mm=ylimit,
              cover_yellow_sweep=gaps, yellow_valid=adapted.is_valid,
              cover_valid=trimmed.is_valid,tree_preserved=True)
    (ROOT/'cad/review/right_leg_forward_measurements.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)
    return result

if __name__=='__main__': right_leg_forward()
