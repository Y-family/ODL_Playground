"""Shorten existing cover skirts inward; retain outer silhouette and screw posts."""
import json
from pathlib import Path
from cadgen import read_step, step, build123d as bd
import build_zp10d_left_leg_complete as m
import validate_s01_s10_interfaces_local as v

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'cad/local_fix/open_duck_mini_complete_14_zp10d_numbered.step'
CLEARANCE_MM=3.0

@step(out='../../cad/local_fix/open_duck_mini_cover_depth_review.step')
def cover_depth_review():
    robot=read_step(SOURCE)
    replacements={}; results=[]
    for side,sid,angles in [('right','S03',range(-30,71,5)),('left','S08',range(-70,31,5))]:
        cover=m._one_child(robot,side+'_cache')
        group=m._one_child(robot,side+'_roll_to_pitch_assembly <1>')
        yellow=m._one_child(group,side+'_roll_to_pitch')
        servo=next(p for p in group.children if p.label.startswith(sid+'_'))
        bound=yellow.bounding_box()
        limit=bound.min.Y-CLEARANCE_MM if side=='right' else bound.max.Y+CLEARANCE_MM
        lo,hi=(-300,limit) if side=='right' else (limit,300)
        pieces=m._intersect_solids(cover,m._axis_aligned_clip_box(cover,bd.Vector(0,1,0),lo,hi))
        assert len(pieces)==1
        new=m._require_one_solid(pieces[0],cover.label,max_disconnected_trim=0)
        new.color=cover.color
        # Every cylindrical screw feature away from the skirt remains intact.
        for face in cover.faces():
            if face.geom_type!=bd.GeomType.CYLINDER: continue
            a=face.axis_of_rotation
            if abs(a.direction.Y)<.999: continue
            if face.radius is not None and face.radius<4:
                assert any(f.geom_type==bd.GeomType.CYLINDER and
                           abs(f.area-face.area)<1e-5 and (f.center()-face.center()).length<1e-5
                           for f in new.faces()), 'screw feature changed'
        before,after=cover.bounding_box(),new.bounding_box()
        assert abs(before.min.Z-after.min.Z)<.001 and abs(before.max.Z-after.max.Z)<.001
        assert abs(before.size.X-after.size.X)<.001
        axis,center=v._axis(servo)
        gaps=[]
        for angle in angles:
            posed=new.rotate(bd.Axis(center,axis),angle)
            gap=posed.distance_to(yellow)
            assert gap>=CLEARANCE_MM-.001,(sid,angle,gap)
            gaps.append({'angle_deg':angle,'gap_mm':gap})
        overlap_before=m._intersection_volume(cover,yellow)
        overlap_after=m._intersection_volume(new,yellow)
        assert overlap_after<1e-5
        results.append(dict(part=cover.label,before_depth_mm=before.size.Y,after_depth_mm=after.size.Y,
                            cut_plane_y_mm=limit,neutral_overlap_before_mm3=overlap_before,
                            neutral_overlap_after_mm3=overlap_after,min_sampled_gap_mm=min(g['gap_mm'] for g in gaps),
                            sweep=gaps,valid=new.is_valid,solid_count=len(new.solids())))
        replacements[cover.label]=new
        print(results[-1],flush=True)
    (ROOT/'cad/review/cover_depth_measurements.json').write_text(json.dumps(results,indent=2))
    return bd.Compound(children=[replacements.get(p.label,p) for p in robot.children],label=robot.label)

if __name__=='__main__': cover_depth_review()
