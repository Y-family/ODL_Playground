"""Fit three 6800 envelopes in original seats and narrow complete hip chains."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
import refit_upper_servo_interfaces_local as refit
from full_motion_current import TARGET, _leaves, MOVING_ROOTS
from repair_motion_clearance_current import detached, all_leaves, face_facts, volume

WORK = Path('/tmp/duck-6800')
BASELINE = '945755d7426bbf28b3c6a4287605ddcdbf206a2ae3e8879e663378aa50a43976'
HIP_Y = 30.25
HIP_TOP = -13.38406118762
HEAD_Z = 196.85093881236


def rod(origin, direction, lower, upper, radius):
    return refit._rod(bd.Vector(origin), bd.Vector(direction), lower, upper, radius)


def clip(shape, xmin=-200, xmax=200, ymin=-200, ymax=200, zmin=-300, zmax=400):
    result = shape.intersect(refit._world_box(xmin, xmax, ymin, ymax, zmin, zmax))
    return list(result.solids()) if result is not None else []


def join(pieces):
    assert pieces
    result = pieces[0]
    for piece in pieces[1:]:
        result = refit._fuse(result, piece)
    return result.clean()


def narrow_span(shape, inward):
    pieces = clip(shape, ymin=-14, ymax=14)
    pieces += [piece.translate((0,inward,0)) for piece in clip(shape,ymax=-14-inward)]
    pieces += [piece.translate((0,-inward,0)) for piece in clip(shape,ymin=14+inward)]
    return join(pieces)


def narrow_top(shape, inward):
    middle = join(clip(shape,xmin=-55,xmax=-8))
    return join(clip(shape,xmax=-55)+[narrow_span(middle,inward)]+clip(shape,xmin=-8))


def hip_socket(shape):
    for sign in (-1,1):
        centre = (-19,sign*HIP_Y,0)
        shape = refit._fuse(shape,rod(centre,(0,0,1),-23.88406118762,-13.88406118762,16.15))
        shape = refit._cut(shape,[rod(centre,(0,0,1),-24,HIP_TOP,9.6)])
    return shape.clean()


def hip_shaft(shape):
    sign = 1 if shape.bounding_box().center().Y>0 else -1
    centre = (-19,sign*HIP_Y,0)
    shell = refit._cut(rod(centre,(0,0,1),-20.185,-13.18406118762,10),
                      [rod(centre,(0,0,1),-20.186,-13.18,4.95)])
    return refit._cut(shape,[shell,rod(centre,(0,0,1),-20.185,HIP_TOP-4.8,10)]).clean()


def trunk_column(shape,inward):
    shape=refit._cut(shape,[refit._world_box(-200,200,-200,200,44.79,400)]).clean()
    if not inward:
        return shape
    half=5
    bands=[(-27,-11,15.9,39.4),(-40,1,-13.5,15.9)]
    faces=[refit._world_box(xmin,xmax,-10,-half,zmin,zmax) for xmin,xmax,zmin,zmax in bands]
    faces+=[refit._world_box(xmin,xmax,half,10,zmin,zmax) for xmin,xmax,zmin,zmax in bands]
    shape=refit._cut(shape,faces).clean()
    column=refit._world_box(-38,0,-half,half,15.9,39.4)
    shape=refit._fuse(shape,column).clean()
    ends=[refit._world_box(-6.5,5,-5.1,5.1,-13.6,39.5),
          refit._world_box(-45,-31.5,-5.1,5.1,-13.6,39.5)]
    return refit._cut(shape,ends).clean()


def motor_wall_relief(shape):
    """Halve the two side walls that grip the servo (2.9 mm -> 1.45 mm).
    Cut from the outside so the servo channel faces (|y|=14.25 and 34.25) and
    the seat plane stay exactly where they were."""
    sign=1 if shape.bounding_box().center().Y>0 else -1
    ylo,yhi=sorted((sign*18.8,sign*41.7))
    cuts=[refit._world_box(-40,0,yhi,60,-40,60),refit._world_box(-40,0,-60,ylo,-40,60)]
    return refit._cut(shape,cuts).clean()


def head_socket(shape):
    centre = (0,.1,HEAD_Z)
    shape = refit._fuse(shape,rod(centre,(1,0,0),-41,-32,16.15))
    return refit._cut(shape,[rod(centre,(1,0,0),-41.01,-31.99,9.6)]).clean()


def head_shaft(shape):
    centre=(0,.1,HEAD_Z)
    sleeve=refit._cut(rod(centre,(1,0,0),-39.11,-31,9.91),
                      [rod(centre,(1,0,0),-39.12,-30.99,4.9)])
    return refit._cut(shape,[sleeve]).clean()


def bearing(shape):
    bounds=shape.bounding_box()
    if bounds.size.X<8:
        origin,direction,lower,upper=(0,.1,HEAD_Z),(1,0,0),-37.1,-32.1
    else:
        origin,direction,lower,upper=(-19,bounds.center().Y,0),(0,0,1),HIP_TOP-5,HIP_TOP
    return refit._cut(rod(origin,direction,lower,upper,9.5),
                      [rod(origin,direction,lower-.01,upper+.01,5)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inward',type=float,default=3)
    args=parser.parse_args()
    assert 0<=args.inward<=6
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==BASELINE
    WORK.mkdir(exist_ok=True)
    robot=read_step(TARGET)
    items=_leaves(robot)
    replacements={}
    records=[]
    for item in items:
        original=detached(item['shape'])
        fresh=original
        name=item['label']
        changed=False
        if name=='roll_bearing':
            fresh=bearing(original)
            changed=True
        elif name=='roll_motor_bottom':
            fresh=motor_wall_relief(hip_shaft(original))
            changed=True
        elif name=='trunk_bottom':
            fresh=trunk_column(narrow_span(hip_socket(original),args.inward),args.inward)
            changed=True
        elif name=='trunk_top' and args.inward:
            fresh=narrow_top(original,args.inward)
            changed=True
        elif name=='head':
            fresh=head_socket(original)
            changed=True
        elif name=='head_yaw_to_roll':
            fresh=head_shaft(original)
            changed=True
        side=0
        if item['root'] in MOVING_ROOTS['S01'] or item['servo']=='S01':
            side=-1
        if item['root'] in MOVING_ROOTS['S06'] or item['servo']=='S06':
            side=1
        if name=='roll_bearing' and item['root']=='trunk_assembly <1>':
            side=1 if original.bounding_box().center().Y>0 else -1
        shift=-side*args.inward
        if shift:
            fresh=fresh.translate((0,shift,0))
        if not changed and not shift:
            continue
        if changed:
            print('validate',item['path'],flush=True)
            validity=check_occurrence_shape(fresh.wrapped,check_self_intersection=name!='head')
            assert validity['solidCount']==1 and not validity['reasons'],(name,validity)
            if name=='head':
                local=bd.Compound(clip(fresh,xmin=-42,xmax=-30,ymin=-20,ymax=20.2,zmin=176,zmax=218))
                local_validity=check_occurrence_shape(local.wrapped)
                assert not local_validity['reasons'],('head bearing region',local_validity)
        replacements[id(item['shape'])]=fresh
        filename=WORK/f'leaf_{len(records)}.step'
        bd.export_step(fresh,filename)
        records.append(dict(path=item['path'],label=name,changed=changed,shift_y_mm=shift,
                            file=str(filename),before_volume=original.volume,after_volume=fresh.volume,
                            bounds=list(fresh.bounding_box().min)+list(fresh.bounding_box().max)))
        print('built',name,'shift',shift,flush=True)
        gc.collect()
    def rebuild(node):
        if node.children:
            result=bd.Compound(children=[rebuild(child) for child in node.children],label=node.label)
            result.location=node.location
        else:
            result=detached(replacements.get(id(node),node))
            result.label=node.label
        result.color=node.color
        return result
    output=WORK/'review.step'
    bd.export_step(refit._encode_colours(rebuild(robot)),output)
    exported=read_step(output)
    before,after=all_leaves(robot),all_leaves(exported)
    assert len(before)==len(after)==137
    for original,actual in zip(before,after):
        expected=replacements.get(id(original),original)
        assert actual.label==original.label and actual.is_valid
        assert abs(actual.volume-expected.volume)<.02,(original.label,actual.volume,expected.volume)
        first,second=face_facts(expected),face_facts(actual)
        assert len(first)==len(second),original.label
        for left,right in zip(first,second):
            assert left[0]==right[0]
            assert max(abs(one-two) for one,two in zip(left[1],right[1]))<.0002,original.label
    report=dict(input_sha256=BASELINE,output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                inward_each_side_mm=args.inward,bearing_dimensions_mm=[10,19,5],
                old_hip_yaw_spacing_mm=60.5,new_hip_yaw_spacing_mm=60.5-2*args.inward,
                old_hip_pitch_spacing_mm=187.85,new_hip_pitch_spacing_mm=187.85-2*args.inward,
                parts=records,export_leaf_count=137,status='candidate; interfaces and motion not yet validated',
                bearing_source='6800 and 61800 returned zero catalog matches; nominal annular envelope, no seal/race detail')
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Export verified',report['output_sha256'],flush=True)


if __name__=='__main__':
    main()
