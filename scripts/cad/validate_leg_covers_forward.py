"""Validate the exported covers, mounting axes and the four requested motions."""
import json
from cadgen import read_step,build123d as bd
from fit_right_leg_forward import ROOT
from review_all_servo_motion import walk
from scan_cover_forward import scan
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

TARGET=ROOT/'cad/local_fix/open_duck_mini_covers_forward_review.step'

def main():
    rows=scan(TARGET,[0.],['right','left'])
    nodes=dict(walk(read_step(TARGET)))
    joints=json.loads(TARGET.with_name(TARGET.name+'.js').read_text().splitlines()[0][15:-1])
    for row in rows:
        assert not row['hits'],row
        j=next(j for j in joints if j['id']==row['joint'])
        cref='o1.6' if row['joint'] in ('S03','S04') else 'o1.1'
        other=next(ref for ref in row['closest_parts'] if ref!=cref)
        pivot=bd.Axis(j['origin'],j['axis']);cover_moves=cref in j['refs']
        base=row['closest_angle_deg'];samples=[]
        for i in range(-4,5):
            angle=max(j['limits'][0],min(j['limits'][1],base+i*.25))
            a=nodes[cref].rotate(pivot,angle) if cover_moves else nodes[cref]
            b=nodes[other] if cover_moves else nodes[other].rotate(pivot,angle)
            samples.append((a.distance_to(b),angle))
        gap,angle=min(samples)
        a=nodes[cref].rotate(pivot,angle) if cover_moves else nodes[cref]
        b=nodes[other] if cover_moves else nodes[other].rotate(pivot,angle)
        overlap=m._intersection_volume(a,b)
        row.update(exact_minimum_gap_mm=gap,refined_closest_angle_deg=angle,overlap_mm3=overlap,
                   range_deg=j['limits'],step_deg=1,closest_part_label=nodes[other].label)
        assert gap>.5 and overlap<.05,row
        print('EXACT',json.dumps(row),flush=True)
    mounts=[]
    for side,ref,group in [('right','o1.6','o1.20'),('left','o1.1','o1.27')]:
        upper=nodes[group];cover=nodes[ref];spacer=m._one_child(upper,'leg_spacer')
        lines=v._cylinder_lines(spacer,1.,bd.Vector(0,1,0))
        host=max([p for p in upper.children if 'sheet' in p.label],key=lambda p:abs(p.bounding_box().center().Y))
        v._require_host_lines(side+' cover',(cover,),lines,1.75)
        v._require_host_lines(side+' sheet',(host,),lines,1.6)
        v._require_no_interference(side+' cover',cover,tuple(m._children(upper)))
        assert cover.distance_to(host)<.001
        mounts.append({'side':side,'aligned_holes':len(lines),'contact_gap_mm':cover.distance_to(host)})
    report={'forward_mm':11.,'motions':rows,'mounts':mounts,'external_profile_trimmed':False}
    (ROOT/'cad/review/covers_forward_checks.json').write_text(json.dumps(report,indent=2))
    print('All four motions and both cover attachments passed',flush=True)

if __name__=='__main__':main()
