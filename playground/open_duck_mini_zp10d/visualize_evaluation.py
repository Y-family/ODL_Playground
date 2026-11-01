"""Render a recorded native evaluation as a self-contained, seekable HTML replay.

Run through scripts/cad/run_memory_limited.py; this never reruns the policy or
integrates physics. Every rendered pose comes from evaluation.npz.
"""

import argparse
import atexit
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LP_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from playground.open_duck_mini_zp10d.reference import SCENE, source_hashes


PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZP10D · 实际动力学仿真回放</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101721;color:#e8edf4;font:15px system-ui,sans-serif}
main{max-width:1480px;margin:auto;padding:20px}h1{font-size:24px;margin:0 0 8px}.muted{color:#aebdce;line-height:1.6}
.warning{background:#40291f;color:#ffcdad;border-radius:8px;padding:12px;margin:14px 0}
#scene{display:block;width:100%;background:#eee;border-radius:10px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}
button,select{background:#25364a;color:white;border:1px solid #52687f;border-radius:6px;padding:8px 14px;cursor:pointer}
input{flex:1;min-width:160px;accent-color:#57c5f7}.metrics,.plots{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.card{background:#1b2736;border-radius:8px;padding:14px}.card strong{font-size:23px;display:block;margin-top:6px}
canvas{width:100%;height:145px}#time{font-variant-numeric:tabular-nums;min-width:112px}.plots{margin-top:14px}
@media(max-width:750px){.plots{grid-template-columns:1fr}.metrics{gap:5px}.card{padding:9px}.card strong{font-size:17px}main{padding:10px}}
</style><main>
<h1>实际动力学仿真回放</h1>
<div class="muted">已记录的自由基座运动 · 131,072 步训练检查点 · 实测舵机响应参数 · 40 ms 控制延迟<br>
三个视角同步：整体斜视 / 正侧视 / 双脚近景。紫色地面点是整机重心投影；下方蓝 / 橙曲线分别表示左 / 右脚的数据。</div>
<div class="warning">本次结果：未有效抬脚，2.54 秒时机身倾斜超过 45°，评估停止。动画停在最后一帧，未补造后续跌落。</div>
<img id="scene" alt="实际仿真三视角回放">
<div class="controls"><button id="play">播放</button><button id="reset">回到开头</button>
<select id="speed" aria-label="播放速度"><option value="0.25">0.25 倍慢放</option><option value="0.5" selected>0.5 倍慢放</option><option value="1">正常速度</option></select>
<input id="seek" type="range" min="0" step="1" value="0" aria-label="仿真时间"><span id="time"></span></div>
<div class="metrics"><div class="card">机身倾斜<strong id="tilt"></strong></div><div class="card">整脚最低点：左 / 右<strong id="height"></strong></div>
<div class="card">脚部支撑力：左 / 右<strong id="force"></strong></div></div>
<div class="plots"><div class="card">机身倾斜（°）<canvas id="tiltplot" width="600" height="180"></canvas></div>
<div class="card">整脚最低点离地（mm）<canvas id="heightplot" width="600" height="180"></canvas></div>
<div class="card">脚部支撑力（N）<canvas id="forceplot" width="600" height="180"></canvas></div></div>
<p class="muted">0–1 秒为站立指令，1 秒后为前进 0.01 m/s 指令。蓝线为左脚，橙线为右脚。离地高度为完整脚部碰撞网格最低点；负值表示接触模型的微小压入。渲染按原始 20 ms 采样回放，没有插值或夸大抬脚幅度。此处展示实际动力学响应，不能作为 Placo 规划已可执行的证明。</p>
<p class="muted">可直接保存本 HTML 离线观看。记录 SHA256：<span id="digest"></span></p>
</main><script>
const record=__RECORD__, frames=__FRAMES__, $=id=>document.getElementById(id);
const seek=$('seek'), scene=$('scene'), play=$('play');seek.max=frames.length-1;
let index=0,playing=false,last=0,elapsed=0;
function plot(id,series,min,max,threshold){
 const c=$(id),x=c.getContext('2d'),w=c.width,h=c.height,L=48,R=w-12,T=12,B=h-28;
 x.clearRect(0,0,w,h);x.font='14px sans-serif';
 const py=v=>B-(v-min)/(max-min)*(B-T),px=i=>L+i/(frames.length-1)*(R-L);
 for(let j=0;j<4;j++){const v=min+(max-min)*j/3,y=py(v);x.strokeStyle='#34475b';x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.fillStyle='#bac9da';x.fillText(v.toFixed(1),2,y+5)}
 if(threshold!==undefined){x.strokeStyle='#d8dfe8';x.setLineDash([6,5]);x.beginPath();x.moveTo(L,py(threshold));x.lineTo(R,py(threshold));x.stroke();x.setLineDash([])}
 series.forEach((s,j)=>{x.strokeStyle=['#55c7ff','#ffac60'][j];x.lineWidth=2;x.beginPath();s.forEach((v,i)=>i?x.lineTo(px(i),py(v)):x.moveTo(px(i),py(v)));x.stroke()});
 x.strokeStyle='#e9edfa';x.lineWidth=1;x.beginPath();x.moveTo(px(index),T);x.lineTo(px(index),B);x.stroke();
 x.fillStyle='#bac9da';x.fillText('0.02 s',L,h-5);x.fillText(record.time.at(-1).toFixed(2)+' s',R-48,h-5);
}
function show(i){index=Math.max(0,Math.min(frames.length-1,i));seek.value=index;scene.src='data:image/jpeg;base64,'+frames[index];
 $('time').textContent=record.time[index].toFixed(2)+' / '+record.time.at(-1).toFixed(2)+' s';
 $('tilt').textContent=record.tilt[index].toFixed(1)+'°';
 $('height').textContent=record.height[index].map(v=>v.toFixed(2)).join(' / ')+' mm';
 $('force').textContent=record.force[index].map(v=>v.toFixed(1)).join(' / ')+' N';
 plot('tiltplot',[record.tilt],0,Math.max(50,...record.tilt),45);
 plot('heightplot',[record.height.map(v=>v[0]),record.height.map(v=>v[1])],Math.min(-.5,...record.height.flat()),3,2);
 plot('forceplot',[record.force.map(v=>v[0]),record.force.map(v=>v[1])],0,Math.max(25,...record.force.flat()));
}
function pause(){playing=false;play.textContent='播放'}
play.onclick=()=>{if(playing){pause();return}if(index===frames.length-1)show(0);playing=true;play.textContent='暂停';last=0;elapsed=record.time[index]};
seek.oninput=()=>{pause();show(+seek.value)};$('reset').onclick=()=>{pause();show(0)};
function tick(now){if(playing){if(last)elapsed+=(now-last)/1000*+$('speed').value;last=now;
 let next=index;while(next<frames.length-1&&record.time[next+1]<=elapsed+1e-8)next++;
 if(next!==index)show(next);if(index===frames.length-1)pause()}requestAnimationFrame(tick)}
$('digest').textContent=record.sha256;show(0);requestAnimationFrame(tick);
</script></html>'''


LIVE_PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ZP10D 实时物理仿真</title>
<style>body{margin:20px;background:#101721;color:#e8edf4;font:16px system-ui}
main{max-width:1200px;margin:auto}img{width:100%;background:#edf1f5}pre{white-space:pre-wrap;line-height:1.8}
a{color:#7fd4ff}.muted{color:#adbed0}</style><main><h2>实时物理仿真</h2>
<p id="status">等待仿真任务启动</p><p id="label" class="muted"></p>
<p class="muted">左右均以机器人自身为准。绿色地面箭头指向机器人右侧；紫色点为整机重心及其地面投影。</p>
<img id="scene" alt="等待真实物理仿真图像"><pre id="metrics"></pre>
<button id="review" hidden>回看本次过程（0.25倍）</button>
<button id="resume" hidden>返回实时</button>
<input id="seek" type="range" min="0" value="0" step="1" hidden aria-label="回看进度">
<p class="muted">紫点：整机重心及地面投影；蓝/橙箭头：左/右脚支撑力。S02为右髋侧摆，S07为左髋侧摆。
画面随计算更新，计算较慢时播放也会变慢。任务结束后保留最后一帧，时间不继续前进。</p>
<a href="replay.html">查看此前失败轨迹的离线回放</a></main><script>
const el=id=>document.getElementById(id);
let reviewing=false,reviewTimer=null,reviewFrames=[];
function show(s){
 el('status').textContent=({waiting:'等待计算资源',running:'正在执行物理仿真',finished:'本次仿真已结束',error:'仿真异常结束',recorded:'正在回看已记录的物理仿真'}[s.status]||s.status)+(s.reason?'：'+s.reason:'');
 el('label').textContent=s.label||'';
 if(s.image){el('scene').src=s.image;
 el('metrics').textContent='仿真时间 '+s.time_s.toFixed(3)+' s · 帧 '+s.sequence+'\\n'
 +'脚底最低点 左/右：'+s.clearance_mm.map(x=>x.toFixed(2)).join(' / ')+' mm\\n'
 +'支撑力 左/右：'+s.foot_force_n.map(x=>x.toFixed(2)).join(' / ')+' N\\n'
 +'相对初始位置（机器人向右为正）：重心 '+s.com_right_shift_mm.toFixed(2)+' mm；机身 '+s.body_right_shift_mm.toFixed(2)+' mm\\n'
 +'俯仰/侧倾：'+s.pitch_roll_deg.map(x=>x.toFixed(2)).join(' / ')+'°\\n'
 +'S02实际/指令：'+s.s02_deg.map(x=>x.toFixed(2)).join(' / ')+'°；S07实际/指令：'+s.s07_deg.map(x=>x.toFixed(2)).join(' / ')+'°';}}
function reviewFrame(i){el('seek').value=i;show(reviewFrames[i]);}
el('review').onclick=async()=>{try{
 const response=await fetch('live_frames.json',{cache:'no-store'});if(!response.ok)throw Error(response.status);
 reviewFrames=await response.json();if(!reviewFrames.length)return;
 clearTimeout(reviewTimer);reviewing=true;el('seek').hidden=false;el('resume').hidden=false;
 el('seek').max=reviewFrames.length-1;reviewFrame(0);let index=0;
 function next(){if(index>=reviewFrames.length-1)return;
 reviewTimer=setTimeout(()=>{reviewFrame(++index);next();},(reviewFrames[index+1].time_s-reviewFrames[index].time_s)*4000);}next();
 }catch(error){el('status').textContent='本次记录暂不可用';}};
el('seek').oninput=()=>{clearTimeout(reviewTimer);reviewFrame(+el('seek').value);};
el('resume').onclick=()=>{clearTimeout(reviewTimer);reviewing=false;el('seek').hidden=true;el('resume').hidden=true;};
async function update(){try{
 const response=await fetch('live.json',{cache:'no-store'});if(!response.ok)throw Error(response.status);
 const s=await response.json();if(!reviewing)show(s);el('review').hidden=!s.recording_available;
 }catch(error){el('status').textContent='实时数据暂不可用，正在重连';}
 setTimeout(update,100);}update();</script></html>'''


class LiveView:
    """Render observer snapshots in a separate MjData; never integrate or edit the robot state."""
    def __init__(self, model, directory, label):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'live.html').write_text(LIVE_PAGE)
        self.model, self.data = model, mujoco.MjData(model)
        self.original_rgba = model.geom_rgba.copy()
        model.geom_rgba[:, 3] = 1.
        self.renderer = mujoco.Renderer(model, width=480, height=480)
        self.camera, self.options = mujoco.MjvCamera(), mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.options.sitegroup[:] = 0
        self.sites = [model.site(side + '_foot').id for side in ('left', 'right')]
        mujoco.mj_resetDataKeyframe(model, self.data, model.key('home').id)
        mujoco.mj_kinematics(model, self.data)
        mujoco.mj_comPos(model, self.data)
        self.initial_com = self.data.subtree_com[1].copy()
        self.initial_body = self.data.xpos[1].copy()
        self.right_axis = self.data.site_xpos[self.sites[1]] - self.data.site_xpos[self.sites[0]]
        self.right_axis[2] = 0.
        self.right_axis /= np.linalg.norm(self.right_axis)
        self.motors = [model.actuator(side + '_hip_roll').id for side in ('right', 'left')]
        self.addresses = model.jnt_qposadr[model.actuator_trnid[self.motors, 0]]
        self.started, self.last_time = time.monotonic(), -1.
        self.state = dict(status='running', label=label, sequence=0)
        self.frames = []
        self._publish()
        atexit.register(self._on_exit)

    def _on_exit(self):
        if self.state['status'] == 'running':
            self.close('任务未正常结束', failed=True)

    def _publish(self):
        path = self.directory / 'live.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, allow_nan=False))
        temporary.replace(path)

    def update(self, time_s, qpos, heights, loads, target, *, force=False):
        if not force and time_s - self.last_time < .1 - 1e-8:
            return
        # Render a private pose snapshot. Physics, contact forces and commands
        # remain those supplied by the caller; no extra forward/step on its data.
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        com = self.data.subtree_com[1].copy()
        canvas = Image.new('RGB', (960, 480))
        for view, azimuth in enumerate((-35, 0)):
            self.camera.azimuth, self.camera.elevation, self.camera.distance = azimuth, -15, .88
            self.camera.lookat[:] = [-.015, 0., .18]
            self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
            start = np.array([.10, 0., .008])
            marker(self.renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, [.003, 0, 0], start,
                   [.15, .8, .25, 1], start + .09*self.right_axis)
            marker(self.renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.005, 0, 0], com, [.85, .1, .8, 1])
            marker(self.renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.004, 0, 0], [*com[:2], .004], [.85, .1, .8, 1])
            for side, site in enumerate(self.sites):
                start = self.data.site_xpos[site].copy()
                start[2] = .004
                foot_point = start.copy()
                start[1] += .07 if side == 0 else -.07  # Clear the opaque foot/leg.
                end = start + [0, 0, max(.001, loads[side] * .003)]
                marker(self.renderer.scene, mujoco.mjtGeom.mjGEOM_LINE, [.002, 0, 0], foot_point,
                       ([.1, .65, 1, 1], [1, .4, .03, 1])[side], start)
                marker(self.renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, [.003, 0, 0], start,
                       ([.1, .65, 1, 1], [1, .4, .03, 1])[side], end)
            canvas.paste(Image.fromarray(self.renderer.render()), (view * 480, 0))
        buffer = BytesIO()
        canvas.save(buffer, format='JPEG', quality=80)
        feet = self.data.site_xpos[self.sites, :2]
        lateral = feet[0] - feet[1]
        lateral /= max(np.linalg.norm(lateral), 1e-8)
        rotation = self.data.xmat[1].reshape(3, 3)
        pitch_roll = np.rad2deg([np.arctan2(-rotation[2, 0], np.linalg.norm(rotation[2, 1:])),
                                np.arctan2(rotation[2, 1], rotation[2, 2])])
        self.state.update(time_s=float(time_s), sequence=self.state['sequence'] + 1,
            image='data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'),
            clearance_mm=(np.asarray(heights) * 1000).tolist(), foot_force_n=np.asarray(loads).tolist(),
            lateral_com_mm=float((com[:2] - feet.mean(0)) @ lateral * 1000),
            com_right_shift_mm=float((com - self.initial_com) @ self.right_axis * 1000),
            body_right_shift_mm=float((self.data.xpos[1] - self.initial_body) @ self.right_axis * 1000),
            pitch_roll_deg=pitch_roll.tolist(),
            s02_deg=np.rad2deg([qpos[self.addresses[0]], target[self.motors[0]]]).tolist(),
            s07_deg=np.rad2deg([qpos[self.addresses[1]], target[self.motors[1]]]).tolist())
        self.frames.append(dict(self.state, status='recorded'))
        self._publish()
        self.last_time = time_s
        # Keep short native tests observable; slow computation is never skipped.
        time.sleep(max(0., min(.1, time_s - (time.monotonic() - self.started))))

    def close(self, reason, *, failed=False):
        if self.frames:
            path = self.directory / 'live_frames.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self.frames, allow_nan=False))
            temporary.replace(path)
        self.state.update(status='error' if failed else 'finished', reason=reason,
                          recording_available=bool(self.frames))
        self._publish()
        self.renderer.close()
        self.model.geom_rgba[:] = self.original_rgba
        atexit.unregister(self._on_exit)


def marker(scene, kind, size, position, color, end=None):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float),
                       np.asarray(position, dtype=float), np.eye(3).ravel(),
                       np.asarray(color, dtype=np.float32))
    if end is not None:
        mujoco.mjv_connector(geom, kind, size[0], np.asarray(position), np.asarray(end))
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='contains evaluation.json and evaluation.npz')
    args = parser.parse_args()
    report = json.loads((args.directory / 'evaluation.json').read_text())
    if report['source_hashes'] != source_hashes():
        raise ValueError('The scene differs from the recorded evaluation')
    path = args.directory / 'evaluation.npz'
    with np.load(path) as archive:
        recorded = {key: archive[key].copy() for key in archive.files}
    count = len(recorded['time'])
    assert recorded['qpos'].shape[0] == count > 1
    assert np.all(np.diff(recorded['time']) > 0)
    assert np.isfinite(recorded['qpos']).all()
    assert abs(recorded['time'][-1] - report['elapsed_sim_seconds']) < 1e-8
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    width, height = 480, 520
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    # Display-only sRGB conversion; source files and recorded dynamics stay intact.
    rgb = model.geom_rgba[:, :3].copy()
    model.geom_rgba[:, :3] = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    model.geom_rgba[:, 3] = 1
    model.vis.headlight.ambient[:] = .5
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 21)
    feet = [model.site(side + '_foot').id for side in ('left', 'right')]
    views = [('整体斜视', -35, -16, .88, (-.015, 0., .20)),
             ('侧视 · 前方在右侧', 90, -12, .88, (-.015, 0., .20)),
             ('双脚近景', -5, -20, .48, (-.015, 0., .035))]
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index in range(count):
            data.qpos[:] = recorded['qpos'][index]
            data.qvel[:] = recorded['qvel'][index]
            data.ctrl[:] = recorded['motor_target'][index]
            mujoco.mj_forward(model, data)
            com = data.subtree_com[1].copy()
            canvas = Image.new('RGB', (width * len(views), height + 44), '#edf1f5')
            draw = ImageDraw.Draw(canvas)
            for view, (title, azimuth, elevation, distance, lookat) in enumerate(views):
                camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
                camera.lookat[:] = lookat
                renderer.update_scene(data, camera=camera, scene_option=options)
                marker(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.005, 0, 0], com, [.85, .1, .8, 1])
                marker(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.004, 0, 0], [com[0], com[1], .004], [.85, .1, .8, 1])
                for side, site in enumerate(feet):
                    start = data.site_xpos[site].copy()
                    start[2] = .004
                    # Arrow length: 3 mm per newton, using saved contact forces.
                    end = start + [0, 0, max(.001, recorded['foot_force'][index, side] * .003)]
                    color = ([.1, .65, 1, 1], [1, .4, .03, 1])[side]
                    marker(renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, [.003, 0, 0], start, color, end)
                canvas.paste(Image.fromarray(renderer.render()), (view * width, 44))
                draw.text((view * width + 12, 9), title, font=font, fill='#182534')
            buffer = BytesIO()
            canvas.save(buffer, format='JPEG', quality=83)
            frames.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
            if index == 0:
                canvas.save(args.directory / 'replay.png')
            if index == count - 1:
                canvas.save(args.directory / 'replay_final.png')
            if index % 25 == 0 or index == count - 1:
                print(f'Rendered {index + 1}/{count} recorded poses, three views', flush=True)
    record = dict(time=recorded['time'].tolist(), tilt=recorded['tilt_deg'].tolist(),
                  height=(recorded['foot_clearance'] * 1000).tolist(),
                  force=recorded['foot_force'].tolist(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    page = PAGE.replace('__RECORD__', json.dumps(record)).replace('__FRAMES__', json.dumps(frames))
    target = args.directory / 'replay.html'
    target.write_text(page)
    print(f'PASS: {count} original samples; self-contained replay saved to {target}', flush=True)


if __name__ == '__main__':
    main()
