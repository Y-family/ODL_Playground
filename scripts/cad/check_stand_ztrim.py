import json
from pathlib import Path
from cadgen import read_step
import refit_upper_servo_interfaces_local as refit
ROOT=Path(__file__).resolve().parents[2]; TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'
def walk(n):
 ch=refit._children(n)
 if not ch: yield n
 else:
  for c in ch: yield from walk(c)
def ov(a,b): return sum(s.volume for s in refit._items(a.intersect(b)))
def main():
 r=read_step(TARGET);ls=list(walk(r));by={}
 for x in ls:by.setdefault(x.label,[]).append(x)
 old=by['speaker_stand'][0];head=by['head'][0]; mount=by['head_roll_mount'][0]
 sp=[by[x][0].moved(refit.bd.Pos(15,5,0)) for x in ('speaker_interface','speaker_bottom','speaker_top')]
 for x0,z0 in [(40,215),(40,218),(40,220),(40,221),(45,218),(50,218),(55,218)]:
  base=refit._shape(old.intersect(refit._world_box(x0,200,-100,50,z0,240))).moved(refit.bd.Pos(0,5,0))
  rows=[dict(label=s.label,overlap=ov(s,base),gap=s.distance_to(base)) for s in sp]
  b=base.bounding_box()
  print(json.dumps(dict(x0=x0,z0=z0,vol=base.volume,solids=len(base.solids()),bounds=list(b.min)+list(b.max),head_gap=base.distance_to(head),mount_gap=base.distance_to(mount),rows=rows),default=float))
if __name__=='__main__':main()
