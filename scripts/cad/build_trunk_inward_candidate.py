"""Move complete hip frames inward; extend original links to fixed lower legs."""
from pathlib import Path
from cadgen import step, read_step, build123d as bd
import build_zp10d_left_leg_complete as m

SOURCE = Path(__file__).resolve().parents[2] / 'cad/local_fix/open_duck_mini_both_legs_trunk_s01_s10.step'
INWARD_MM = 4.75
BLACK_YELLOW_GAP_DELTA_MM = 1.500075
BRIDGE_INWARD_THICKEN_MM = 2.0
CONNECTOR_WINDOW_X_PAD_MM = 2.25
CONNECTOR_WINDOW_Z_PAD_MM = 2.125

def clip(p,axis,lo,hi):
    return m._intersect_solids(p,m._axis_aligned_clip_box(p,axis,lo,hi))

def join(p,pieces):
    result=m._require_one_solid(pieces[0].fuse(*pieces[1:],tol=1e-4).clean().fix(),p.label,max_disconnected_trim=0)
    result.color=p.color
    return result

def narrow(p):
    pieces=[]
    if p.label=='trunk_top':
        # Shorten the central span with rigid halves; avoid affine distortion
        # of the original small bores and fillets at the sewing boundary.
        for lo,hi,dy in [(-200,-INWARD_MM,INWARD_MM),(INWARD_MM,200,-INWARD_MM)]:
            pieces.extend(s.moved(bd.Pos(0,dy,0)) for s in clip(p,bd.Vector(0,1,0),lo,hi))
        return join(p,pieces)
    for lo,hi,dy in [(-200,-16,INWARD_MM),(16,200,-INWARD_MM)]:
        pieces.extend(s.moved(bd.Pos(0,dy,0)) for s in clip(p,bd.Vector(0,1,0),lo,hi))
    pieces.extend(s.scale((1,(16-INWARD_MM)/16,1)) for s in clip(p,bd.Vector(0,1,0),-16,16))
    return join(p,pieces)

def open_serial_ports(housing,servo):
    """Open both original three-pin connector pockets through the housing."""
    body=m._one_child(servo,'ZP10D_vendor_body')
    caps=[f for f in body.faces()
          if f.geom_type==bd.GeomType.PLANE and .7<f.area<.9
          and abs(f.normal_at().Y)>.99]
    groups={-1:[],1:[]}
    for face in caps:
        groups[1 if face.normal_at().Y>0 else -1].append(face)
    assert all(len(group)==3 for group in groups.values()), 'two three-pin ports required'
    housing_box=housing.bounding_box();cutters=[]
    for direction,group in groups.items():
        boxes=[face.bounding_box() for face in group]
        xmin=min(box.min.X for box in boxes)-CONNECTOR_WINDOW_X_PAD_MM
        xmax=max(box.max.X for box in boxes)+CONNECTOR_WINDOW_X_PAD_MM
        zmin=min(box.min.Z for box in boxes)-CONNECTOR_WINDOW_Z_PAD_MM
        zmax=max(box.max.Z for box in boxes)+CONNECTOR_WINDOW_Z_PAD_MM
        opening_y=sum(face.center().Y for face in group)/len(group)
        if direction<0:
            ymin,ymax=housing_box.min.Y-1.0,opening_y+.05
        else:
            ymin,ymax=opening_y-.05,housing_box.max.Y+1.0
        cutters.append(bd.Box(xmax-xmin,ymax-ymin,zmax-zmin,
                              align=(bd.Align.MIN,bd.Align.MIN,bd.Align.MIN))
                       .moved(bd.Pos(xmin,ymin,zmin)))
    result=m._require_one_solid(m._cut_sequential(housing,cutters),housing.label,
                                max_disconnected_trim=0)
    result.label=housing.label;result.color=housing.color
    return result

def extend_link(p,right,obstacles):
    # Work in a right-leg frame. Both original bore groups remain rigid.
    source=p if right else p.mirror(bd.Plane.XZ)
    pieces=clip(source,bd.Vector(0,1,0),-200,-48)
    pieces.extend(s.scale((1,(3+INWARD_MM)/3,1),about=(0,-48,0))
                  for s in clip(source,bd.Vector(0,1,0),-48,-45))
    pieces.extend(s.moved(bd.Pos(0,INWARD_MM,0))
                  for s in clip(source,bd.Vector(0,1,0),-45,200))
    result=join(source,pieces)
    # Thicken the existing inward-facing transverse sheet, not the X cheeks.
    # Work in the right-leg frame; mirror after extruding so both grow inward.
    faces=[f for f in result.faces()
           if abs(f.center().Y+53.77430584028)<.001 and f.normal_at().Y>.99 and f.area>1000]
    assert len(faces)==1, 'inward sheet face must be unique'
    inward_growth=BRIDGE_INWARD_THICKEN_MM
    reinforcement=bd.Solid.extrude(faces[0],bd.Vector(0,inward_growth,0))
    # Continue the inward growth through both original rounded transitions.
    corners=[f for f in result.faces() if f.geom_type==bd.GeomType.BSPLINE
             and -54<f.center().Y<-52 and f.area>100
             and f.normal_at().Y>.9 and abs(f.normal_at().X)>.1]
    assert len(corners)==2, 'two inward rounded transitions required'
    rounded=[bd.Solid.extrude(f,bd.Vector(0,inward_growth,0)) for f in corners]
    result=join(source,[result,reinforcement,*rounded])
    if not right:result=result.mirror(bd.Plane.XZ)
    result.label=p.label;result.color=p.color
    return result

@step(out='../../cad/local_fix/open_duck_mini_trunk_inward_candidate.step')
def candidate():
    root=read_step(SOURCE);result=[]
    for branch in root.children:
        name=branch.label.replace('_',' ')
        if name=='trunk assembly <1>':
            parts=[]
            for p in branch.children:
                if p.label in ('trunk_top','trunk_bottom'):
                    p=narrow(p)
                    if p.label=='trunk_bottom':
                        pieces=[]
                        for lo,hi,half,xlo,xhi in [(-200,15.9,None,0,0),(15.9,39.4,10,-27,-11),(39.4,200,None,0,0)]:
                            for s in clip(p,bd.Vector(0,0,1),lo,hi):
                                if half is None:pieces.append(s)
                                else:
                                    s=s.scale((1,half/(10*(16-INWARD_MM)/16),1))
                                    pieces.extend(clip(s,bd.Vector(1,0,0),xlo,xhi))
                        p=join(p,pieces)
                elif p.label=='roll_bearing' or p.label.startswith(('S01_','S06_')):
                    dy=INWARD_MM if p.bounding_box().center().Y<0 else -INWARD_MM
                    p=p.moved(bd.Pos(0,dy,0))
                parts.append(p)
            branch=bd.Compound(children=parts,label=branch.label)
        elif name in ('hip roll assembly <1>','hip roll assembly <2>'):
            dy=INWARD_MM if branch.bounding_box().center().Y<0 else -INWARD_MM
            # Relieve only the old top-plate perimeter; the central web stays
            # 20 mm wide. Envelop the fixed web at sampled yaw orientations.
            center_y=-35+dy if dy>0 else 35+dy
            axis=bd.Axis((-19,center_y,0),(0,0,1))
            guard=bd.Box(17.5,21.5,4.2,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Pos(-19,0,35.1))
            cutters=[guard.rotate(axis,angle) for angle in range(-30,31,5)]
            parts=[]
            for p in branch.children:
                if p.label=='roll_motor_bottom':
                    servo=next(child for child in branch.children
                               if child.label.startswith(('S02_','S07_')))
                    p=open_serial_ports(p,servo)
                p=p.moved(bd.Pos(0,dy,0))
                if p.label=='roll_motor_top':
                    original=p
                    p=m._require_one_solid(m._cut_sequential(p,cutters),p.label,max_disconnected_trim=0)
                    p.color=original.color
                parts.append(p)
            branch=bd.Compound(children=parts,label=branch.label)
        elif name in ('right roll to pitch assembly <1>','left roll to pitch assembly <1>'):
            right=name.startswith('right')
            obstacles=[p for p in branch.children if p.label.startswith('S')]
            branch=bd.Compound(children=[extend_link(p,right,obstacles) if p.label in ('right_roll_to_pitch','left_roll_to_pitch') else p for p in branch.children],label=branch.label)
        elif name in ('knee and ankle assembly <1>','knee and ankle assembly <2>'):
            # Double only the black-to-yellow clearance. Keep the horn-side
            # material and its screw pattern fixed.
            right=branch.bounding_box().center().Y<0
            parts=[]
            for p in branch.children:
                if p.label=='left_knee_to_ankle_left_sheet':
                    box=p.bounding_box()
                    pieces=(clip(p,bd.Vector(0,1,0),-200,
                                 box.max.Y-BLACK_YELLOW_GAP_DELTA_MM)
                            if right else
                            clip(p,bd.Vector(0,1,0),
                                 box.min.Y+BLACK_YELLOW_GAP_DELTA_MM,200))
                    p=join(p,pieces)
                parts.append(p)
            branch=bd.Compound(children=parts,label=branch.label)
        result.append(branch)
    return bd.Compound(children=result,label=root.label)

if __name__ == '__main__':
    candidate()
