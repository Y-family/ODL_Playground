"""Targeted mating and rigid-placement checks for the local forward review."""
from cadgen import read_step
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
from fit_right_leg_forward import ROOT, SOURCE, FORWARD_MM, MOVING
from build_left_leg_from_approved import cylinder

def main():
    before=read_step(SOURCE)
    robot=read_step(ROOT/'cad/local_fix/open_duck_mini_right_leg_forward_review.step')
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    pitch=v._one(robot,'right_roll_to_pitch_assembly <1>')
    upper=v._one(robot,'knee_and_ankle_assembly <1>')
    lower=v._one(robot,'knee_and_ankle_assembly <3>')
    foot=v._one(robot,'foot_assembly <1>')
    yellow=v._one(pitch,'right_roll_to_pitch')
    servo=next(p for p in pitch.children if p.label.startswith('S03_'))
    body,_,_=v._servo_parts(servo)
    for i,(p,d) in enumerate(v._case_mount_lines(body,vendor)):
        lo,hi=(p.Y-2,p.Y) if i<2 else (p.Y,p.Y+2)
        ring=cylinder(p,m.bd.Vector(0,1,0),lo,hi,2).cut(
             cylinder(p,m.bd.Vector(0,1,0),lo-.01,hi+.01,1.1))
        ratio=m._intersection_volume(ring,yellow)/ring.volume
        print('S03 screw seat',i+1,'2mm annulus coverage',ratio,flush=True)
        assert ratio>.95,(i,ratio)
    sheets=lambda g:tuple(v._one(g,'left_knee_to_ankle_'+s+'_sheet') for s in ('left','right'))
    for sid,g,case,hosts in [('S03',pitch,(yellow,),sheets(upper)),
                              ('S04',upper,sheets(upper),sheets(lower)),
                              ('S05',lower,sheets(lower),tuple(v._one(foot,s) for s in ('foot_top','foot_side')))]:
        servo=next(p for p in g.children if p.label.startswith(sid+'_'))
        v._validate_joint(sid,v.JointSpec(servo,case,hosts),vendor)
    for sheet in sheets(upper):
        overlap=m._intersection_volume(sheet,yellow)
        assert overlap<.05, (sheet.label,overlap)
        print('S03 black/yellow',sheet.label,'overlap',overlap,'gap',sheet.distance_to(yellow),flush=True)
    # Compare leaf placements so a parent-only transform cannot hide a missed move.
    def leaves(p):
        return [s for c in p.children for s in leaves(c)] if p.children else [p]
    for old,new in zip(before.children,robot.children):
        if old.label in ('right_cache','right_roll_to_pitch_assembly <1>'): continue
        dx=FORWARD_MM if old.label in MOVING else 0
        for a,b in zip(leaves(old),leaves(new)):
            delta=b.bounding_box().center()-a.bounding_box().center()
            assert (delta-m.bd.Vector(dx,0,0)).length<.001,(a.label,delta)
            assert abs(a.volume-b.volume)<.001,(a.label,'volume')
    print('PASS rigid forward placements; unrelated geometry unchanged',flush=True)

if __name__=='__main__': main()
