import json
from pathlib import Path
from cadgen import build123d as bd, read_step
import refit_upper_servo_interfaces_local as refit

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'

def walk(node):
    for child in refit._children(node):
        if refit._children(child):
            yield from walk(child)
        else:
            yield child

def clip(shape, limits):
    tool=refit._world_box(*limits)
    result=shape.intersect(tool)
    solids=[s for s in refit._items(result) if s.volume>1e-5]
    return refit._shape(result) if solids else None

def main():
    robot=read_step(TARGET)
    stand=next(x for x in walk(robot) if x.label=='speaker_stand')
    rows=[]
    for name,lim in [
        ('z_upper',(-200,200,-200,200,223.5,260)),
        ('z_lower',(-200,200,-200,200,170,223.5)),
        ('y_head',(-200,200,-41.8,200,170,260)),
        ('y_front',(-200,200,-200,-41.8,170,260)),
        ('x_head',(-200,47.8,-200,200,170,260)),
        ('x_speaker',(47.8,200,-200,200,170,260)),
        ('z_220',(-200,200,-200,200,220,230)),
        ('z_215',(-200,200,-200,200,210,220)),
        ('z_200',(-200,200,-200,200,190,210)),
        ('z_190',(-200,200,-200,200,180,190)),
    ]:
        s=clip(stand,lim)
        if s is not None:
            b=s.bounding_box();rows.append(dict(name=name,volume=s.volume,bounds=list(b.min)+list(b.max),solids=len(s.solids())))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
