// Compare the render module's handle operations to independently calculated
// CAD matrices, including simultaneous rotations, pause, reverse and looping.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const [modulePath, assemblyPath, expectedPath] = process.argv.slice(2);
const source = readFileSync(modulePath, 'utf8');
const module = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
assert.deepEqual(Object.keys(module), ['clips']);
const refs = new Set(JSON.parse(readFileSync(assemblyPath, 'utf8')).occurrences.map(o=>o.id));
const expected = JSON.parse(readFileSync(expectedPath, 'utf8'));
const identity = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
function multiply(a,b) {
  return Array.from({length:16}, (_,i)=>{
    const row=Math.floor(i/4), col=i%4;
    return [0,1,2,3].reduce((sum,k)=>sum+a[row*4+k]*b[k*4+col],0);
  });
}
function rotation(axis,degrees,origin) {
  const norm=Math.hypot(...axis), [x,y,z]=axis.map(v=>v/norm);
  const a=degrees*Math.PI/180, c=Math.cos(a), s=Math.sin(a), d=1-c;
  const m=[c+x*x*d,x*y*d-z*s,x*z*d+y*s,0,
           y*x*d+z*s,c+y*y*d,y*z*d-x*s,0,
           z*x*d-y*s,z*y*d+x*s,c+z*z*d,0,0,0,0,1];
  for(let row=0;row<3;++row) m[row*4+3]=origin[row]-[0,1,2].reduce((sum,k)=>sum+m[row*4+k]*origin[k],0);
  return m;
}
let matrixCount=0, maxError=0;
for(const frame of expected) {
  const states=new Map([...refs].map(ref=>[ref,{matrix:identity(),opacity:1,visible:true}]));
  const handles={get(target) {
    const targets=target.replace(/^#/,'').split(',');
    for(const ref of targets) assert(refs.has(ref),`Unknown occurrence: ${ref}`);
    return {
      rotate(axis,degrees,origin=[0,0,0]) {
        const m=rotation(axis,degrees,origin);
        for(const ref of targets) states.get(ref).matrix=multiply(m,states.get(ref).matrix);
      },
      opacity(value) {
        assert(value>=0 && value<=1);
        for(const ref of targets) states.get(ref).opacity=value;
      },
      visible(value) {for(const ref of targets) states.get(ref).visible=Boolean(value);},
    };
  }};
  const clip=module.clips[frame.clip];
  assert(clip && clip.duration===12 && clip.loop===true);
  clip.update(frame.time,handles);
  for(const [ref,wanted] of Object.entries(frame.matrices)) {
    const actual=states.get(ref).matrix;
    for(let i=0;i<16;++i) maxError=Math.max(maxError,Math.abs(actual[i]-wanted[i]));
    assert(states.get(ref).visible,`Physical part hidden: ${ref}`);
    ++matrixCount;
  }
  for(const ref of refs) if(!(ref in frame.matrices)) assert(!states.get(ref).visible,`Reference geometry still visible: ${ref}`);
}
assert(maxError<1e-8,`Animation differs from CAD joint transforms: ${maxError}`);
console.log(JSON.stringify({passed:true,clips:Object.keys(module.clips),frame_count:expected.length,
  part_matrix_count:matrixCount,max_element_error:maxError,unknown_targets:0,
  scope:'Render-module operations checked against CAD matrices; not a browser UI test.'}));
