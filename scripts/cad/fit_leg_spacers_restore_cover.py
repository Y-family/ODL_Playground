"""Local mm edit: retain forward-shift review and restore uncut right cover.

Move all four leg_spacer down 16.5 mm along world Z, retaining their thickness.
Relocate matching sheet bores. Extend the original right yellow bridge 7 mm
outwards and translate S03 and its downstream leg together; S02 stays fixed.
"""
import json
from cadgen import read_step, step, build123d as bd
from fit_right_leg_forward import ROOT, SOURCE, FORWARD_MM, move_tree, MOVING
from build_right_link_outward import right_link_outward, OUTWARD_MM
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from build_left_leg_from_approved import remap_group

INPUT=ROOT/'cad/local_fix/open_duck_mini_right_leg_forward_review.step'
LOWER_MM=16.5

@step(out='../../cad/local_fix/open_duck_mini_spacers_lower_review.step')
def spacers_lower():
    robot=read_step(INPUT)
    outward_yellow=right_link_outward()
    original=read_step(SOURCE)
    cover=m._one_child(original,'right_cache').moved(bd.Pos(FORWARD_MM,0,0))
    records=[]
    def signature(p):return (p.label,[signature(c) for c in p.children])
    old_tree=signature(robot)
    def edit(p):
        if p.label=='right_cache': return cover
        spacers=[c for c in p.children if c.label=='leg_spacer']
        if spacers:
            assert len(spacers)==1
            spacer=spacers[0]; b=spacer.bounding_box()
            shift=bd.Vector(0,0,-LOWER_MM)
            new_spacer=spacer.moved(bd.Pos(0,0,-LOWER_MM))
            old_lines=v._cylinder_lines(spacer,1.0,bd.Vector(0,1,0))
            assert len(old_lines)==2
            new_lines=[(q+shift,d) for q,d in old_lines]
            children=[]
            for child in p.children:
                if child.label=='leg_spacer':children.append(new_spacer)
                elif 'sheet' in child.label:
                    mid=child.bounding_box().center().Y
                    new=remap_group(child,old_lines,new_lines,bd.Vector(0,1,0),mid,(1.6,3.25))
                    v._require_host_lines('spacer', (new,), new_lines, 1.6)
                    assert new.is_valid and len(new.solids())==1
                    print('spacer/sheet',p.label,child.label,m._intersection_volume(new,new_spacer),flush=True)
                    if m._intersection_volume(new,new_spacer)>.05:
                        # Use the wall before drilling the relocated holes, so
                        # no artificial pins remain inside the new screw bores.
                        new_spacer=m._require_one_solid(new_spacer.cut(child),spacer.label,max_disconnected_trim=0)
                        new_spacer.color=spacer.color
                    assert m._intersection_volume(new,new_spacer)<.05
                    children.append(new)
                else: children.append(child)
            children=[new_spacer if c.label=='leg_spacer' else c for c in children]
            records.append(dict(group=p.label,center_before=list(b.center()),
                                center_after=list(new_spacer.bounding_box().center()),
                                lowered_mm=LOWER_MM,height_mm=b.size.Z,
                                end_faces_fitted_to_existing_sheets=True,
                                removed_end_volume_mm3=spacer.volume-new_spacer.volume))
            return bd.Compound(children=children,label=p.label)
        if p.children:
            return bd.Compound(children=[edit(c) for c in p.children],label=p.label)
        return p
    result=edit(robot)
    final_children=[]
    shift=bd.Pos(0,-OUTWARD_MM,0)
    for child in result.children:
        if child.label=='right_roll_to_pitch_assembly <1>':
            yellow=outward_yellow
            servo=move_tree(m._one_child(child,'S03_RIGHT_HIP_PITCH_ZP10D'),shift)
            body,_,_=v._servo_parts(servo)
            vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
            v._require_host_lines('outward S03 case',(yellow,),v._case_mount_lines(body,vendor),1.)
            assert m._intersection_volume(body,yellow)<.05
            final_children.append(bd.Compound(children=[yellow if p.label==yellow.label else
                                  servo if p.label==servo.label else p for p in child.children],label=child.label))
        elif child.label in MOVING:final_children.append(move_tree(child,shift))
        else:final_children.append(child)
    result=bd.Compound(children=final_children,label=result.label)
    assert len(records)==4
    assert signature(result)==old_tree
    (ROOT/'cad/review/spacers_lower_measurements.json').write_text(json.dumps(dict(
        right_leg_outward_mm=OUTWARD_MM,spacers=records),indent=2))
    print(json.dumps(records),flush=True)
    return result

if __name__=='__main__':spacers_lower()
