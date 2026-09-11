"""Move both existing leg covers +11 mm X; preserve profiles and attachment.

The previous spacer drop left the cover bores at the old height. Close those
partial bores within their original wall, then place the two original bores
on the current spacer/sheet screw axes. No new component or outline trimming.
Units mm, world +X forward, +/-Y left/right, +Z up.
"""
from cadgen import read_step,step,report,build123d as bd
from fit_right_leg_forward import ROOT
from build_left_leg_from_approved import cylinder,radius_of
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

SOURCE=ROOT/'cad/local_fix/open_duck_mini_symmetric_hips_review.step'
FORWARD_MM=11.0
BORE_RADIUS=1.75
COUNTERBORE_RADIUS=3.25
MIN_WEB_MM=2.0
EXISTING_INNER_WALL_Y=130.25

def cover_forward(cover,upper,side):
    q=cover.moved(bd.Pos(FORWARD_MM,0,0))
    axis=bd.Vector(0,1,0)
    old_lines=v._cylinder_lines(q,BORE_RADIUS,axis)
    assert len(old_lines)==2
    fills=[]
    for face in q.faces():
        if 'CYLINDER' not in str(face.geom_type):continue
        radius=radius_of(face)
        if radius is None or min(abs(radius-r) for r in (BORE_RADIUS,COUNTERBORE_RADIUS))>.001:continue
        a=face.axis_of_rotation
        if not any(v._line_distance(a.position,a.direction,p,d)<.001 for p,d in old_lines):continue
        b=face.bounding_box()
        fill=cylinder(a.position,axis,b.min.Y,b.max.Y,radius)
        # Old bores terminate at the existing Z=-20 step. Clipping the fill
        # there prevents cylinder stubs on the thinner lower wall.
        above=bd.Box(400,400,400,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Pos(0,0,-20.))
        fills.extend(fill.intersect(above).solids())
    closed=q.fuse(*fills,tol=.00001).clean()
    closed=m._require_one_solid(closed,cover.label,max_disconnected_trim=0)
    spacer=m._one_child(upper,'leg_spacer')
    new_lines=v._cylinder_lines(spacer,1.,axis)
    assert len(new_lines)==2
    sheets=[p for p in upper.children if 'sheet' in p.label]
    host=max(sheets,key=lambda p:abs(p.bounding_box().center().Y))
    contact_y=host.bounding_box().min.Y if side=='right' else host.bounding_box().max.Y
    b=q.bounding_box();tools=[]
    # Extend the existing flat mounting wall inward at the relocated screws.
    # The lower shell is thinner than the old mounting step; without this
    # thickness adjustment the screws would span an unsupported cavity.
    xmin=min(p.X for p,d in new_lines)-6.;xmax=max(p.X for p,d in new_lines)+6.
    zmid=sum(p.Z for p,d in new_lines)/len(new_lines)
    ymin,ymax=(-EXISTING_INNER_WALL_Y-.25,contact_y) if side=='right' else (contact_y,EXISTING_INNER_WALL_Y+.25)
    support=bd.Box(xmax-xmin,ymax-ymin,12.,align=(bd.Align.MIN,)*3).moved(bd.Pos(xmin,ymin,zmid-6.))
    outer_y=ymin if side=='right' else ymax
    assert all(q.is_inside(bd.Vector(x,outer_y,z)) for x in (xmin,xmax) for z in (zmid-6.,zmid+6.))
    closed=m._require_one_solid(closed.fuse(support,tol=.00001).clean(),cover.label,max_disconnected_trim=0)
    for p,d in new_lines:
        lo,hi=(b.min.Y-1,contact_y+.01) if side=='right' else (contact_y-.01,b.max.Y+1)
        tools.append(cylinder(p,axis,lo,hi,BORE_RADIUS))
        cb_lo,cb_hi=(lo,contact_y-MIN_WEB_MM) if side=='right' else (contact_y+MIN_WEB_MM,hi)
        tools.append(cylinder(p,axis,cb_lo,cb_hi,COUNTERBORE_RADIUS))
    result=m._require_one_solid(closed.cut(*tools).clean(),cover.label,max_disconnected_trim=0)
    # Strip imported occurrence metadata after changing the solid.
    fresh=bd.Solid(result.solids()[0].wrapped);fresh.label=cover.label;fresh.color=cover.color
    # The translated inner rear lip clips the fixed thigh sheet near its
    # bottom corner. Relieve only that local inner patch, with 0.3 mm margin;
    # the outer skin and the lower rounded silhouette stay intact.
    for fixed in m._children(upper):
        common=fresh.intersect(fixed)
        common_solids=[] if common is None else [p for p in common.solids() if p.volume>.05]
        if common_solids:
            before=fresh.volume
            for piece in common_solids:
                ib=piece.bounding_box();margin=.3
                assert ib.max.Z<zmid-6. and abs(ib.center().Y)<130.,ib
                tool=bd.Box(ib.size.X+2*margin,ib.size.Y+2*margin,ib.size.Z+2*margin).moved(bd.Pos(ib.center()))
                fresh=m._require_one_solid(fresh.cut(tool).clean(),cover.label,max_disconnected_trim=0)
            fresh.color=cover.color
            print(side,'inner lip relief mm3',fixed.label,before-fresh.volume,flush=True)
    v._require_host_lines(side+' cover',(fresh,),new_lines,BORE_RADIUS)
    v._require_host_lines(side+' sheet',(host,),new_lines,1.6)
    v._require_no_interference(side+' cover',fresh,tuple(m._children(upper)))
    assert fresh.distance_to(host)<.001
    assert (fresh.bounding_box().size-q.bounding_box().size).length<.001
    print(side,'cover forward',FORWARD_MM,'mm; two mounting holes aligned; profile retained',flush=True)
    return fresh

@step(out='../../cad/local_fix/open_duck_mini_covers_forward_review.step')
def covers_forward():
    r=read_step(SOURCE)
    def signature(p):return p.label,[signature(c) for c in p.children]
    original_tree=signature(r)
    replacements={}
    for side,group in [('right','knee_and_ankle_assembly <1>'),('left','knee_and_ankle_assembly <2>')]:
        report(side+' cover: translate and align existing fasteners')
        replacements[side+'_cache']=cover_forward(m._one_child(r,side+'_cache'),m._one_child(r,group),side)
    out=bd.Compound(children=[replacements.get(p.label,p) for p in r.children],label=r.label)
    assert signature(out)==original_tree
    return out

if __name__=='__main__':covers_forward()
