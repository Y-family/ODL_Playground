"""Check the relocated rear case-screw channels, not whole-robot motion."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
from build_right_link_original_walls import BRIDGE_SHORTENING_MM

def main():
    original=read_step(m.SOURCE_ROBOT)
    pitch=m._one_child(original,'right_roll_to_pitch_assembly <1>')
    p=m._servo_placement(m._one_prefix_child(pitch,'STS32155'),'hip_pitch')
    shift=m.bd.Pos(0,BRIDGE_SHORTENING_MM,0)
    p=m.ServoPlacement(shift*p.pose,p.axis,p.center+m.bd.Vector(0,BRIDGE_SHORTENING_MM,0),
                      p.body.moved(shift),p.passive_horn.moved(shift),p.active_horn.moved(shift))
    link=read_step('cad/local_fix/right_link_original_walls.step')
    for i,(x,y,z) in enumerate(m.ZP10D_CASE_HOLES[2:],1):
        tool=p.pose*m._local_x_cylinder(x+2.01,x+45,y,z,2.49)
        overlap=m._intersection_volume(tool,link)
        assert overlap<.001,(i,overlap)
        # Bearing material around the bore must remain, along the 2 mm seat.
        for dx in (.1,1,1.9):
            for dy,dz in [(1.5,0),(-1.5,0),(0,1.5),(0,-1.5)]:
                point=m.bd.Vertex(x+dx,y+dy,z+dz).moved(p.pose).center()
                assert link.is_inside(point),('missing screw seat',i,dx,dy,dz)
        print(f'S03 rear screw {i}: diameter 5 channel clear; 2 mm bearing seat retained',flush=True)
    # The case of S02 lies in the straight screwdriver approach. Assembly order
    # therefore matters: fasten S03 into yellow before installing S02.
    tool=p.pose*m._local_x_cylinder(m.ZP10D_CASE_HOLES[2][0]+2,m.ZP10D_CASE_HOLES[2][0]+100,
                                  m.ZP10D_CASE_HOLES[2][1],m.ZP10D_CASE_HOLES[2][2],2.49)
    r=read_step('cad/local_fix/open_duck_mini_right_leg_s02_s05.step')
    s02=m._one_child(m._one_child(r,'hip_roll_assembly <1>'),'S02_RIGHT_HIP_ROLL_ZP10D')
    print('S02 blocks straight screwdriver after assembly:',m._intersection_volume(tool,s02)>.01)

if __name__=='__main__':
    main()
