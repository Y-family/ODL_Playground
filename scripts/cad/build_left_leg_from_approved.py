"""Mirror approved printed parts and remap bores to rigidly placed real servos."""
from cadgen import read_step, step, report, build123d as bd
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from package_open_duck_symmetric_10_zp10d import _aligned_left_servo

APPROVED=m.PROJECT_ROOT/'cad/local_fix/open_duck_mini_right_leg_approved.step'
BRANCH_PAIRS={
    'hip_roll_assembly <2>':'hip_roll_assembly <1>',
    'left_roll_to_pitch_assembly <1>':'right_roll_to_pitch_assembly <1>',
    'knee_and_ankle_assembly <2>':'knee_and_ankle_assembly <1>',
    'knee_and_ankle_assembly <4>':'knee_and_ankle_assembly <3>',
    'foot_assembly <2>':'foot_assembly <1>',
}
MOTORS=[('roll','hip_roll_assembly <2>','S02_RIGHT_HIP_ROLL_ZP10D','S07_LEFT_HIP_ROLL_ZP10D'),
        ('pitch','left_roll_to_pitch_assembly <1>','S03_RIGHT_HIP_PITCH_ZP10D','S08_LEFT_HIP_PITCH_ZP10D'),
        ('knee','knee_and_ankle_assembly <2>','S04_RIGHT_KNEE_ZP10D','S09_LEFT_KNEE_ZP10D'),
        ('ankle','knee_and_ankle_assembly <4>','S05_RIGHT_ANKLE_ZP10D','S10_LEFT_ANKLE_ZP10D')]

def mirror_point(p):
    return bd.Vector(p.X,-p.Y,p.Z)

def cylinder(point,axis,lo,hi,radius):
    start=point+axis*(lo-point.dot(axis))
    return bd.Plane(origin=start,z_dir=axis).location*bd.Cylinder(radius,hi-lo,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))

def radius_of(face):
    if face.radius is not None: return float(face.radius)
    return next((float(e.radius) for e in face.edges() if 'CIRCLE' in str(e.geom_type) and e.radius is not None),None)

def remap_group(part,old_lines,new_lines,axis,axial_reference,radii):
    """Fill only existing bore volumes; preserve the mounting wall outline."""
    if len(old_lines)==len(new_lines) and all(
        any(v._line_distance(p,d,q,e)<1e-5 for q,e in new_lines) for p,d in old_lines):
        return part
    faces=[]
    for f in part.faces():
        if 'CYLINDER' not in str(f.geom_type): continue
        radius=radius_of(f)
        if radius is None or not any(abs(radius-r)<.001 for r in radii): continue
        line=f.axis_of_rotation
        if abs(abs(line.direction.dot(axis))-1)>.001: continue
        lo,hi=m._bbox_projection(f,axis)
        if axial_reference<lo-8 or axial_reference>hi+8: continue
        if not any(v._line_distance(line.position,line.direction,p,d)<.01 for p,d in old_lines): continue
        faces.append((f,radius,lo,hi,line.position))
    if not faces: return part
    fillers=[cylinder(p,axis,lo,hi,r) for f,r,lo,hi,p in faces]
    result=part.fuse(*fillers,tol=.001).clean()
    lo=min(row[2] for row in faces); hi=max(row[3] for row in faces)
    tools=[cylinder(p,axis,lo-.01,hi+.01,min(radii)) for p,_ in new_lines]
    counter=[row for row in faces if len(radii)>1 and abs(row[1]-max(radii))<.001]
    if counter:
        cb_lo=min(row[2] for row in counter); cb_hi=max(row[3] for row in counter)
        tools.extend(cylinder(p,axis,cb_lo,cb_hi,max(radii)) for p,_ in new_lines)
    result=m._require_one_solid(m._cut_sequential(result,tools),part.label,max_disconnected_trim=0)
    result.color=part.color
    return result

def adapt_mirrored_part(source,right_servos,left_servos,vendor_body,label=None):
    part=source.mirror(bd.Plane.XZ)
    part.label=label or source.label; part.color=source.color
    for name,right in right_servos.items():
        left=left_servos[name]
        rb,rp,ra=v._servo_parts(right); lb,lp,la=v._servo_parts(left)
        old_case=[(mirror_point(p),mirror_point(d)) for p,d in v._case_mount_lines(rb,vendor_body)]
        new_case=v._case_mount_lines(lb,vendor_body)
        for old_point,old_axis in old_case:
            new_line=min(new_case,key=lambda line:(line[0]-old_point).length)
            part=remap_group(part,[(old_point,old_axis)],[new_line],old_axis,
                             old_point.dot(old_axis),(1.0,))
        axis,center=v._axis(right); axis=mirror_point(axis); center=mirror_point(center)
        left_axis,left_center=v._axis(left)
        for rh,lh in [(rp,lp),(ra,la)]:
            mirrored=rh.mirror(bd.Plane.XZ)
            old_lines=v._horn_pattern_lines(mirrored,axis,center)
            new_lines=v._horn_pattern_lines(lh,left_axis,left_center)
            part=remap_group(part,old_lines,new_lines,axis,
                             mirrored.bounding_box().center().dot(axis),(1.6,3.25))
        if m._intersection_volume(part,lb)>.05:
            part=m._require_one_solid(part.cut(lb),part.label,max_disconnected_trim=0)
            part.color=source.color
    return part

@step(out='../../cad/local_fix/open_duck_mini_left_leg_s07_s10.step')
def left_leg():
    right=read_step(APPROVED); original=read_step(m.SOURCE_ROBOT)
    vendor_body=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    right_servos={}; left_servos={}
    for name,left_branch,right_id,left_id in MOTORS:
        source=m._one_child(m._one_child(right,BRANCH_PAIRS[left_branch]),right_id)
        old=m._one_prefix_child(m._one_child(original,left_branch),'STS32155')
        placement=_aligned_left_servo(old,source,'left_'+name)
        right_servos[name]=source
        left_servos[name]=m._servo_compound(placement,left_id)
    result=[]
    for old_branch in original.children:
        if old_branch.label=='left_cache':
            p=m._one_child(right,'right_cache').mirror(bd.Plane.XZ)
            p.label='left_cache'; p.color=m._one_child(right,'right_cache').color
            result.append(p)
        elif old_branch.label in BRANCH_PAIRS:
            report('left '+old_branch.label)
            right_branch=m._one_child(right,BRANCH_PAIRS[old_branch.label])
            parts=[]
            for old in old_branch.children:
                if old.label.startswith('STS32155'):
                    name=next(row[0] for row in MOTORS if row[1]==old_branch.label)
                    parts.append(left_servos[name])
                else:
                    source_label='right_roll_to_pitch' if old.label=='left_roll_to_pitch' else old.label
                    source=m._one_child(right_branch,source_label)
                    parts.append(adapt_mirrored_part(source,right_servos,left_servos,vendor_body,old.label))
            result.append(bd.Compound(children=parts,label=old_branch.label))
    return bd.Compound(children=result,label='OPEN_DUCK_MINI_LEFT_LEG_S07_S10')

if __name__=='__main__':
    left_leg()
