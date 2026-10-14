import json, math
from pathlib import Path
import numpy as np
import trimesh
from cadgen import read_step
import refit_upper_servo_interfaces_local as refit

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'

def walk(n):
    ch=refit._children(n)
    if not ch: yield n
    else:
        for c in ch: yield from walk(c)

def mesh(shape):
    v,f=shape.tessellate(.4,.25)
    return trimesh.Trimesh(vertices=np.array([[p.X,p.Y,p.Z] for p in v]),faces=np.array(f),process=False)

def main():
    r=read_step(TARGET); leaves=list(walk(r))
    by={}
    for x in leaves:
        if x.label not in by: by[x.label]=x
    speakers=[x for x in leaves if x.label in ('speaker_interface','speaker_bottom','speaker_top')]
    fixed=[x for x in leaves if x.label in ('head','head_pitch_to_yaw','head_yaw_to_roll','head_roll_mount','head_bot_sheet')]
    managers=[]
    for x in fixed:
        m=trimesh.collision.CollisionManager();m.add_object(x.label,mesh(x));managers.append((x.label,m))
    sm=[mesh(x) for x in speakers]
    rows=[]
    for dx in np.arange(-30,31,5):
      for dy in np.arange(-15,16,5):
        trans=np.eye(4);trans[:3,3]=[dx,dy,0]
        hits=[]
        for name,man in managers:
          if any(man.in_collision_single(m,transform=trans) for m in sm): hits.append(name)
        if not hits:
          gaps=[]
          for m in sm:
            gaps.append(min(man.min_distance_single(m, transform=trans) for _,man in managers))
          rows.append(dict(dx=float(dx),dy=float(dy),mesh_gap=float(min(gaps)),score=float(dx*dx+dy*dy)))
    rows.sort(key=lambda q:(q['score'],-q['mesh_gap']))
    print(json.dumps(rows[:80],indent=2))

if __name__=='__main__':main()
