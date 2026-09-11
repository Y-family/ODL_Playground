"""Check exported hip interfaces and yellow-to-fixed-part roll clearances."""
import json,math
import trimesh
from cadgen import read_step,build123d as bd
from fit_right_leg_forward import ROOT
from scan_s01_s14_motion_collisions import _mesh
from review_all_servo_motion import walk
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

TARGET=ROOT/'cad/local_fix/open_duck_mini_symmetric_hips_review.step'

def main():
    robot=read_step(TARGET);nodes=dict(walk(robot))
    vendor=max(m._children(read_step(v.VENDOR)),key=lambda p:p.volume)
    joints=json.loads(TARGET.with_name(TARGET.name+'.js').read_text().splitlines()[0][15:-1])
    report={'range_deg':[-25,25],'step_deg':1,'interfaces':[],'roll_sweeps':[]}
    left=nodes['o1.26.1'];right=nodes['o1.17.1']
    assert left.volume>37000 and right.volume>37000, 'Export lost modified bridge geometry'
    lb=left.bounding_box();rb=right.bounding_box()
    assert abs(lb.min.Y+rb.max.Y)<.001 and abs(lb.max.Y+rb.min.Y)<.001
    assert (lb.size-rb.size).length<.001
    report['yellow_dimensions_mm']=list(lb.size)
    report['yellow_envelopes_mirror_symmetric']=True
    report['yellow_volumes_mm3']={'left':left.volume,'right':right.volume}
    for sid in ('S03','S04','S05','S08','S09','S10'):
        motor=next(p for p in nodes.values() if p.label.startswith(sid+'_'))
        _,c=v._axis(motor)
        j=next(j for j in joints if j['id']==sid)
        assert (bd.Vector(j['origin'])-c).length<1e-5,(sid,j['origin'],list(c))
    for side,roll_id,pitch_id,hip_name,pitch_name,upper_name in [
        ('right','S02','S03','hip_roll_assembly <1>','right_roll_to_pitch_assembly <1>','knee_and_ankle_assembly <1>'),
        ('left','S07','S08','hip_roll_assembly <2>','left_roll_to_pitch_assembly <1>','knee_and_ankle_assembly <2>')]:
        hip=m._one_child(robot,hip_name);pitch=m._one_child(robot,pitch_name)
        yellow=m._one_child(pitch,side+'_roll_to_pitch')
        roll=next(p for p in hip.children if p.label.startswith(roll_id+'_'))
        motor=next(p for p in pitch.children if p.label.startswith(pitch_id+'_'))
        upper=m._one_child(robot,upper_name)
        for sid,spec in [(roll_id,v.JointSpec(roll,tuple(p for p in hip.children if 'ZP10D' not in p.label),(yellow,))),
                         (pitch_id,v.JointSpec(motor,(yellow,),tuple(p for p in upper.children if 'sheet' in p.label)))]:
            v._validate_joint(sid,spec,vendor)
            report['interfaces'].append({'servo':sid,'case_holes':4,'horn_holes':8,'passed':True})
        leg_hosts=[p for p in upper.children if 'sheet' in p.label or p.label=='leg_spacer']
        v._require_no_interference(side+' yellow/upper leg',yellow,tuple(leg_hosts))
        print(side,'yellow/black-sheet gap',min(yellow.distance_to(p) for p in leg_hosts),flush=True)
        j=next(j for j in joints if j['id']==roll_id)
        axis,center=v._axis(roll)
        assert (bd.Vector(j['origin'])-center).length<1e-5
        moving=set(j['refs'])
        b=yellow.bounding_box()
        # Exact extrema for every bbox corner over the roll interval, avoiding
        # meshing unrelated head parts while keeping a conservative enclosure.
        swept=[]
        for y in (b.min.Y,b.max.Y):
            for z in (b.min.Z,b.max.Z):
                dy,dz=y-center.Y,z-center.Z
                angles=[math.radians(-25),math.radians(25)]
                for base in (math.atan2(-dz,dy),math.atan2(dy,dz)):
                    angles.extend(a for k in range(-2,3) if -math.pi*25/180<=(a:=base+k*math.pi)<=math.pi*25/180)
                swept.extend((center.Y+dy*math.cos(a)-dz*math.sin(a),
                              center.Z+dy*math.sin(a)+dz*math.cos(a)) for a in angles)
        ymin=min(p[0] for p in swept)-.01;ymax=max(p[0] for p in swept)+.01
        zmin=min(p[1] for p in swept)-.01;zmax=max(p[1] for p in swept)+.01
        fixed={}
        for ref,p in nodes.items():
            if p.children or ref in moving or int(ref.split('.')[1]) in {3,4,8,12,19}:continue
            pb=p.bounding_box()
            if pb.max.X<b.min.X or pb.min.X>b.max.X:continue
            if pb.max.Y<ymin or pb.min.Y>ymax:continue
            if pb.max.Z<zmin or pb.min.Z>zmax:continue
            fixed[ref]=p
        print(roll_id,'fixed candidates',[(r,p.label) for r,p in fixed.items()],flush=True)
        fm=trimesh.collision.CollisionManager();ym=trimesh.collision.CollisionManager()
        for ref,p in fixed.items():fm.add_object(ref,_mesh(p,.12,.2))
        ym.add_object('yellow',_mesh(yellow,.12,.2))
        samples=[];hits={}
        for angle in range(-25,26):
            ym.set_transform('yellow',trimesh.transformations.rotation_matrix(math.radians(angle),list(axis),list(center)))
            hit,pairs=ym.in_collision_other(fm,return_names=True)
            for pair in pairs:
                ref=next(r for r in pair if r!='yellow');hits.setdefault(ref,[]).append(angle)
            samples.append((ym.min_distance_other(fm),angle))
        exact_hits=[]
        for ref,angles in hits.items():
            for angle in angles:
                posed=yellow.rotate(bd.Axis(center,axis),angle)
                volume=m._intersection_volume(posed,fixed[ref])
                if volume>.05:exact_hits.append({'ref':ref,'name':fixed[ref].label,'angle':angle,'overlap_mm3':volume})
        _,angle=min(samples);posed=yellow.rotate(bd.Axis(center,axis),angle)
        distances=[(posed.distance_to(p),ref) for ref,p in fixed.items()]
        gap,closest=min(distances)
        row={'servo':roll_id,'closest_angle_deg':angle,'minimum_gap_mm':gap,
             'closest_part':fixed[closest].label,'fixed_parts_checked':len(fixed),'confirmed_clashes':exact_hits}
        report['roll_sweeps'].append(row)
        (ROOT/'cad/review/symmetric_hips_checks.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
        assert not exact_hits,row
    print('Both exported hip interfaces and roll sweeps passed',flush=True)

if __name__=='__main__':main()
