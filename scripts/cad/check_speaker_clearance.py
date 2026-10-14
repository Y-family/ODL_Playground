from pathlib import Path
from cadgen import read_step
import refit_upper_servo_interfaces_local as refit
ROOT=Path(__file__).resolve().parents[2]; TARGET=ROOT/'cad/local_fix/open_duck_mini_upper_refit.step'
def walk(n):
 ch=refit._children(n)
 if not ch: yield n
 else:
  for c in ch: yield from walk(c)
def main():
 r=read_step(TARGET); by={}
 for x in walk(r):by.setdefault(x.label,[]).append(x)
 for dy in [5.0,5.2,5.4,5.5,5.8,6.0,6.5,7.0]:
  print(dy, by['speaker_top'][0].moved(refit.bd.Pos(15,dy,0)).distance_to(by['head'][0]))
if __name__=='__main__':main()
