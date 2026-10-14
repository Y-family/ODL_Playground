import json
from pathlib import Path
from cadgen import read_step
import refit_upper_servo_interfaces_local as refit
ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'
def walk(n):
 ch=refit._children(n)
 if not ch: yield n
 else:
  for c in ch: yield from walk(c)
def ov(a,b): return sum(s.volume for s in refit._items(a.intersect(b)))
def main():
 r=read_step(TARGET); ls=list(walk(r)); by={}
 for x in ls: by.setdefault(x.label,[]).append(x)
 stand0=by['speaker_stand'][0]; head=by['head'][0]; mount=by['head_roll_mount'][0]
 speakers=[by[x][0] for x in ('speaker_interface','speaker_bottom','speaker_top')]
 for x0 in [40,42,43,44,45]:
  base=refit._shape(stand0.intersect(refit._world_box(x0,200,-100,50,223.5,240)))
  for dx,dy in [(0,5)]:
   stand=base.moved(refit.bd.Pos(dx,dy,0))
   sp=[s.moved(refit.bd.Pos(15,5,0)) for s in speakers]
   rows=[]
   for s in sp:
    rows.append(dict(label=s.label,overlap=ov(s,stand),gap=s.distance_to(stand)))
   print(json.dumps(dict(x0=x0,dx=dx,dy=dy,head_gap=stand.distance_to(head),mount_gap=stand.distance_to(mount),rows=rows),default=float))
if __name__=='__main__':main()
