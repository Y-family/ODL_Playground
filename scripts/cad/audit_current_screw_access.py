"""Read-only hole alignment and straight Ø5 x 60 mm driver approach audit.

Tool size is an explicit screening assumption, not a known screwdriver spec.
No model changes. An obstructed approach in a completed assembly may be handled
by assembly order; this report does not certify thread engagement or tool access.
"""
import json,sys,math,hashlib
import trimesh
from cadgen import read_step, build123d as bd
from fit_right_leg_forward import ROOT
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v
import validate_s11_s14_interfaces_local as upper
from review_all_servo_motion import walk
from scan_s01_s14_motion_collisions import _mesh
from build_left_leg_from_approved import cylinder

TARGET=ROOT/'cad/local_fix/open_duck_mini_covers_forward_review.step'
REPORT=ROOT/'cad/review/current_screw_access.json'

def main():
    fresh='--fresh' in sys.argv
    model_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest()
    specs={}
    # Reuse the existing, explicit host mapping without stopping on the first
    # unrelated support-gap failure. Capture it, then audit every mounting hole.
    original_validate=v._validate_joint
    v._validate_joint=lambda sid,spec,vendor:specs.update({sid:(spec,vendor)})
    sys.argv=[sys.argv[0],'--robot',str(TARGET)]
    v.main();upper.main()
    v._validate_joint=original_validate
    robot=read_step(TARGET)
    nodes=dict(walk(robot))
    leaves={r:p for r,p in nodes.items() if not p.children and
            int(r.split('.')[1]) not in {3,4,8,12,19}}
    # Nearby BRep checks only: bbox reject prevents a whole-robot boolean sweep.
    bounds={}
    def cached_bbox(shape):
        key=id(shape)
        if key not in bounds:bounds[key]=(shape,shape.bounding_box())
        return bounds[key][1]
    def bbox_hit(a,b):
        x,y=cached_bbox(a),cached_bbox(b)
        return all(getattr(x.max,k)>=getattr(y.min,k) and getattr(y.max,k)>=getattr(x.min,k)
                   for k in ('X','Y','Z'))
    rows=json.loads(REPORT.read_text()).get('servos',[]) if REPORT.exists() and not fresh else []
    completed={row['servo'] for row in rows}
    manager=trimesh.collision.CollisionManager()
    loaded=set(); fallback=set()
    for sid,(spec,vendor) in specs.items():
        if sid in completed:continue
        body,passive,active=v._servo_parts(spec.servo)
        axis,center=v._axis(spec.servo)
        groups=[('case',body,spec.case_hosts,v._case_mount_lines(body,vendor),1.0)]
        for name,h in [('passive',passive),('active',active)]:
            if spec.active_host is not None:
                if name=='passive':continue
                hosts=(spec.active_host,)
            else:hosts=(v._nearest_host(h,spec.horn_hosts),)
            groups.append((name,h,hosts,v._horn_pattern_lines(h,axis,center),1.6))
        holes=[]
        for kind,part,hosts,lines,radius in groups:
            for index,(point,direction) in enumerate(lines,1):
                found=[]
                for host in hosts:
                    for f in host.faces():
                        if f.geom_type!=bd.GeomType.CYLINDER or f.radius is None:continue
                        if abs(f.radius-radius)>.001:continue
                        a=f.axis_of_rotation
                        error=v._line_distance(point,direction,a.position,a.direction)
                        if error<.01:found.append((host,f,error))
                item=dict(kind=kind,hole=index,aligned=bool(found),
                          axis_error_mm=min((x[2] for x in found),default=None))
                if found:
                    outward=direction if (point-body.bounding_box().center()).dot(direction)>0 else -direction
                    if kind!='case':outward=axis if (part.bounding_box().center()-center).dot(axis)>0 else -axis
                    # Use the end of the small bore, retaining any counterbore
                    # approach beyond it as part of the obstruction test.
                    end=max(m._bbox_projection(f,outward)[1] for _,f,_ in found)
                    tool=cylinder(point,outward,end+.05,end+60,2.49)
                    candidates=[]
                    for ref,other in leaves.items():
                        if not bbox_hit(tool,other):continue
                        candidates.append((ref,other))
                    blocks=[]
                    for ref,other in candidates:
                        if ref in loaded or ref in fallback:continue
                        try:
                            manager.add_object(ref,_mesh(other,.1,.15));loaded.add(ref)
                        except (AttributeError,ValueError):fallback.add(ref)
                    hit,names=manager.in_collision_single(_mesh(tool,.08,.15),return_names=True)
                    for ref in names:
                        blocks.append(dict(ref=ref,part=leaves[ref].label,method='mesh screening'))
                    for ref,other in candidates:
                        if ref not in fallback:continue
                        volume=m._intersection_volume(tool,other)
                        if volume>.05:blocks.append(dict(ref=ref,part=other.label,overlap_mm3=volume))
                    item['driver_blockers']=blocks
                    item['driver_clear']=not blocks
                holes.append(item)
        row=dict(servo=sid,holes=holes,
                 aligned=sum(h['aligned'] for h in holes),total=len(holes),
                 driver_clear=sum(h.get('driver_clear',False) for h in holes))
        rows.append(row)
        REPORT.write_text(json.dumps(dict(model=str(TARGET),model_sha256=model_sha256,driver_diameter_mm=4.98,
                         driver_length_mm=60,servos=rows),indent=2))
        print(sid,'aligned',row['aligned'],'/',row['total'],'driver clear',row['driver_clear'],flush=True)
    print('REPORT',REPORT,flush=True)

if __name__=='__main__':main()
