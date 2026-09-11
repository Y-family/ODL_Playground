"""Check the new trunk-to-hip screw and bearing interfaces only."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

TARGET='cad/local_fix/open_duck_mini_both_legs_trunk_s01_s10.step'

def main():
    r=read_step(TARGET); trunk=m._one_child(r,'trunk_assembly <1>')
    top=m._one_child(trunk,'trunk_top'); bottom=m._one_child(trunk,'trunk_bottom')
    vb=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    for index,sid in [(1,'S01'),(2,'S06')]:
        hip=m._one_child(r,f'hip_roll_assembly <{index}>')
        servo=next(p for p in trunk.children if p.label.startswith(sid+'_'))
        upper=m._one_child(hip,'roll_motor_top'); journal=m._one_child(hip,'roll_motor_bottom')
        v._validate_joint(sid,v.JointSpec(servo,(top,),(),active_host=upper),vb)
        body,_,_=v._servo_parts(servo)
        for point,_ in v._case_mount_lines(body,vb):
            side=-1 if point.Z<60 else 1
            for dz in (.25,1.0,1.4):
                for dx,dy in ((1.5,0),(-1.5,0),(0,1.5),(0,-1.5)):
                    assert top.is_inside(point+m.bd.Vector(dx,dy,side*dz)),(sid,'case screw bearing material',point,dx,dy,dz)
        axis,center=v._axis(servo)
        bearing=min((p for p in trunk.children if p.label=='roll_bearing'),
                    key=lambda p:abs(p.bounding_box().center().Y-center.Y))
        for part,radius in [(bearing,10),(journal,9.95)]:
            axes=v._cylinder_lines(part,radius,m.bd.Vector(0,0,1))
            assert any(v._line_distance(p,d,center,axis)<.001 for p,d in axes),(sid,'bearing axis')
        overlap=m._intersection_volume(bearing,journal)
        radial_gap=bearing.distance_to(journal)
        assert overlap<.001 and abs(radial_gap-.05)<.001,(overlap,radial_gap)
        assert bearing.distance_to(bottom)<.001
        assert m._intersection_volume(bearing,bottom)<.001
        assert m._intersection_volume(upper,top)<.05
        assert m._intersection_volume(journal,bottom)<.05
        journal_faces=[f for f in journal.faces() if 'CYLINDER' in str(f.geom_type)
                       and any('CIRCLE' in str(e.geom_type) and abs(e.radius-9.95)<.001 for e in f.edges())]
        journal_low=min(f.bounding_box().min.Z for f in journal_faces)
        journal_high=max(f.bounding_box().max.Z for f in journal_faces)
        axial_overlap=min(journal_high,bearing.bounding_box().max.Z)-max(journal_low,bearing.bounding_box().min.Z)
        assert axial_overlap>6.5
        print(sid,f'bearing coaxial; radial clearance 0.05 mm; engagement {axial_overlap:.3f} mm; outer race seated',flush=True)
        body,p,a=v._servo_parts(servo)
        placement=m.ServoPlacement(m.bd.Location(),axis,center,body,p,a)
        # These are the newly attached hip frames, not a whole-robot test.
        roll=next(p for p in hip.children if p.label.startswith(('S02_','S07_')))
        for angle in (-30,0,30):
            for part in [upper,journal,roll]:
                posed=m._rotate_about_servo_axis(part,placement,angle)
                for host in (top,bottom):
                    if not m._bounding_boxes_overlap(posed,host): continue
                    volume=m._intersection_volume(posed,host)
                    if volume>.05:
                        print('YAW COLLISION',sid,angle,part.label,host.label,volume,flush=True)
                        raise SystemExit(1)
        print(sid,'hip yaw endpoint and neutral clearance PASS',flush=True)

if __name__=='__main__':
    main()
