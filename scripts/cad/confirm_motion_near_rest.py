"""Confirm the nearest-to-rest mesh warning for each servo, using STEP solids."""
import json
from cadgen import read_step, build123d as bd
from review_all_servo_motion import ROOT, TARGET, walk
from confirm_zp10d_only_motion_candidates_brep import _intersection_facts

def main():
    report=json.loads((ROOT/'cad/review/all_servos_motion_report.json').read_text())
    nodes=dict(walk(read_step(TARGET)))
    results=[]
    for j in report['joints']:
        if not j['candidates']: continue
        p=min(j['candidates'],key=lambda p:min(abs(a) for a in p['angles_deg']))
        angle=min(p['angles_deg'],key=abs)
        axis=bd.Axis(j['origin'],j['axis'])
        shapes=[nodes[r].rotate(axis,angle) if r in j['refs'] else nodes[r] for r in p['refs']]
        volume,bounds=_intersection_facts(*shapes)
        neutral,_=_intersection_facts(*(nodes[r] for r in p['refs']))
        result=dict(id=j['id'],angle_deg=angle,refs=p['refs'],labels=p['labels'],
                    volume_mm3=volume,neutral_volume_mm3=neutral,bounds_mm=bounds)
        results.append(result)
        print(j['id'],angle,p['labels'],'volume=',round(volume,6),'neutral=',round(neutral,6),flush=True)
        (ROOT/'cad/review/near_rest_brep_report.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
