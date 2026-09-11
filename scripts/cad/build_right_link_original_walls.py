"""Resize original yellow walls; no mounting pads or horn pockets."""
from cadgen import read_step, step, build123d as bd
import build_zp10d_left_leg_complete as m

BRIDGE_SHORTENING_MM = 3.8

def remap_holes(link, roll):
    # Fill only the old bore volumes, never cylinders spanning an entire plate.
    fillers=[]
    for face in link.faces():
        if 'CYLINDER' not in str(face.geom_type): continue
        axis=face.axis_of_rotation
        p=axis.position
        if abs(abs(axis.direction.X)-1)>.001: continue
        if abs(((p.Y-roll.center.Y)**2+(p.Z-roll.center.Z)**2)**.5-7)>.01: continue
        box=face.bounding_box()
        radius=face.radius
        if radius is None:
            radius=next(e.radius for e in face.edges() if 'CIRCLE' in str(e.geom_type) and e.radius is not None)
        fillers.append(m._local_x_cylinder(box.min.X,box.max.X,p.Y,p.Z,float(radius)))
    link=link.fuse(*fillers).clean()
    box=link.bounding_box()
    for horn,lo,hi,outer in [
        (roll.passive_horn,box.min.X,roll.passive_horn.bounding_box().min.X,'min'),
        (roll.active_horn,roll.active_horn.bounding_box().max.X,box.max.X,'max')]:
        for p in m._x_axis_pattern_centers(horn,roll.center,7):
            link=link.cut(m._local_x_cylinder(lo-1,hi+1,p.Y,p.Z,1.6))
            a,b=(lo-1,lo+2) if outer=='min' else (hi-2,hi+1)
            link=link.cut(m._local_x_cylinder(a,b,p.Y,p.Z,3.25))
    return m._require_one_solid(link,'right_roll_to_pitch',max_disconnected_trim=0)

@step(out='../../cad/local_fix/right_link_original_walls.step')
def right_link_original_walls():
    original = read_step(m.SOURCE_ROBOT)
    hip = m._one_child(original, 'hip_roll_assembly <1>')
    pitch = m._one_child(original, 'right_roll_to_pitch_assembly <1>')
    roll = m._servo_placement(m._one_prefix_child(hip, 'STS32155'), 'hip_roll')
    servo = m._servo_placement(m._one_prefix_child(pitch, 'STS32155'), 'hip_pitch')
    d=BRIDGE_SHORTENING_MM
    shift=bd.Pos(0,d,0)
    servo=m.ServoPlacement(shift*servo.pose,servo.axis,servo.center+bd.Vector(0,d,0),
                          servo.body.moved(shift),servo.passive_horn.moved(shift),servo.active_horn.moved(shift))
    link = m._stretch_original_roll_link(m._one_child(pitch, 'right_roll_to_pitch'), roll.axis, roll.center)
    link = remap_holes(link, roll)
    # Move existing mounting walls to the case faces; stretch only the joining spans.
    bands = [(-200., -107.1, -204.125+d, -111.225+d),
             (-107.1, -72.1, -111.225+d, -71.225+d),
             (-72.1, -67.37, -71.225+d, -59.1+d),
             (-67.37, -50., -59.1+d, -50.),
             (-50., 100., -50., 100.)]
    pieces = []
    for lo, hi, new_lo, new_hi in bands:
        clip = m._axis_aligned_clip_box(link, bd.Vector(0,1,0), lo, hi)
        for piece in m._intersect_solids(link, clip):
            factor = (new_hi-new_lo)/(hi-lo)
            if abs(factor-1) > 1e-10:
                piece = piece.scale((1,factor,1), about=(0,lo,0))
            pieces.append(piece.moved(bd.Pos(0,new_lo-lo,0)))
    link = m._require_one_solid(pieces[0].fuse(*pieces[1:]), 'right_roll_to_pitch', max_disconnected_trim=0)
    # Shorten both free ends across their full thickness, open to the exterior.
    # Retain the original transverse bridge, shifted clear of the passive horn.
    stop_x = -18.46  # black plate edge -19.71 plus 1.25 mm free-edge clearance
    exterior = bd.Box(stop_x+200,200-59.1+d,200,align=(bd.Align.MIN,)*3).moved(bd.Pos(-200,-200,-100))
    link = link.cut(exterior)
    # Extend the two EXISTING end walls as full rectangular wall sections.
    # This is continuous wall material, not individual screw pads or new parts.
    wall_box = link.bounding_box()
    for y0,y1 in [(-114.225+d,-111.225+d),(-71.225+d,-59.1+d)]:
        wall = bd.Box(wall_box.max.X-stop_x,y1-y0,wall_box.size.Z,
                      align=(bd.Align.MIN,)*3).moved(bd.Pos(stop_x,y0,wall_box.min.Z))
        link = link.fuse(wall)
    # Shorten the projecting outer end through its WHOLE Z thickness.
    # Black sheet sweep reaches X=-9.5663; its inner face is Y=-64.5.
    outer_end = bd.Box(200-8.0,6.6,200,align=(bd.Align.MIN,)*3).moved(
        bd.Pos(-200,-65.7+d,-100))
    link = link.cut(outer_end)
    # Original continuous walls carry the screws. Cylinders here REMOVE holes only.
    holes = [servo.pose * m._local_x_cylinder(x-5,x+5,y,z,m.CASE_HOLE_DIAMETER_MM/2)
             for x,y,z in m.ZP10D_CASE_HOLES]
    # Relocate the original left leg's diameter-5 screwdriver channels to
    # the ZP10D rear case screws. Leave 2 mm of wall under each screw head.
    # These are coaxial case-screw counterbores, not horn relief pockets.
    for x,y,z in m.ZP10D_CASE_HOLES[2:]:
        holes.append(servo.pose * m._local_x_cylinder(x+2,x+45,y,z,2.5))
    link = m._cut_sequential(link, [servo.body, *holes])
    for solid in link.solids():
        print('remaining solid', solid.volume, solid.is_valid, solid.bounding_box(), flush=True)
    link = m._require_one_solid(link, 'right_roll_to_pitch', max_disconnected_trim=0)
    link.color = m.srgb(m.COLOURS['roll_link'])
    return link

if __name__ == '__main__':
    right_link_original_walls()
