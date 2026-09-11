"""Extend the existing yellow bridge outside the fixed S02 horn interface."""
from copy import deepcopy
from cadgen import build123d as bd
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter,BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.TopTools import TopTools_ListOfShape
from OCP.gp import gp_Pln,gp_Pnt,gp_Dir
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
import build_zp10d_left_leg_complete as m

def extend_bridge(link, distance):
    link=deepcopy(link)
    args=TopTools_ListOfShape();args.Append(link.wrapped)
    tools=TopTools_ListOfShape()
    for level in (-51.,-50.):
        tools.Append(BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0,level,0),gp_Dir(0,1,0))).Face())
    split=BRepAlgoAPI_Splitter();split.SetArguments(args);split.SetTools(tools)
    split.SetNonDestructive(True);split.Build()
    parts=list(bd.Compound(split.Shape()).solids())
    def volume(shape):
        props=GProp_GProps()
        BRepGProp.VolumeProperties_s(shape.wrapped,props,1e-9,True,False)
        return props.Mass()
    partition_volume=sum(volume(p) for p in parts)
    source_volume=volume(link)
    # Independent BRep coverage check; mass integration across split curved
    # surfaces is less stable than the geometric difference itself.
    rest=BRepAlgoAPI_Cut(link.wrapped,bd.Compound(children=parts).wrapped)
    rest.Build()
    missing=sum(abs(volume(p)) for p in bd.Compound(rest.Shape()).solids())
    assert missing<.01,(missing,partition_volume,source_volume)
    pieces=[]
    for part in parts:
        y=part.center().Y
        if y < -51.:part=part.moved(bd.Pos(0,-distance,0))
        elif y < -50.:
            part=part.scale((1,1+distance,1),about=(0,-51,0)).moved(bd.Pos(0,-distance,0))
        pieces.append(part)
    result=m._require_one_solid(pieces[0].fuse(*pieces[1:],tol=.001).clean(),link.label,max_disconnected_trim=0)
    result.color=link.color
    return result
