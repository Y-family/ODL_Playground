"""Render every recorded mesh warning and the available BRep confirmations."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]/'cad/review'
scan=json.loads((root/'all_servos_motion_report.json').read_text())
confirmed=json.loads((root/'near_rest_brep_report.json').read_text())
lines=['# 整机运动检查问题清单', '',
       '逐关节测试，其余关节保持零位；采样步长 5°。角度相对 STEP 零位，正负按各舵机轴方向。',
       '声明范围不是已验证的可用行程。未测试多关节组合、受力、线缆和实物螺丝装配。', '',
       '## 实体复核', '',
       '| 舵机 | 检查角度 | 零件对 | 交集 mm³ | 零位交集 mm³ |',
       '|---|---:|---|---:|---:|']
for p in confirmed:
    lines.append(f"| {p['id']} | {p['angle_deg']:+g}° | {' ↔ '.join(p['labels'])} | {p['volume_mm3']:.6f} | {p['neutral_volume_mm3']:.6f} |")
lines+=['', 'S02/S07 在 ±10° 的交集低于 0.05 mm³ 判定阈值，但不代表有足够运动间隙；其他行已确认超过阈值。',
        '只复核了每个关节最接近零位的一对报警，下面其他零件对仍为候选。', '', '## 全部网格候选（含零件引用）', '']
count=0
for j in scan['joints']:
    lines += [f"### {j['id']} — {j['name']}；范围 {j['limits'][0]}° 至 {j['limits'][1]}°", '']
    if not j['candidates']: lines += ['未发现随运动增加的网格干涉候选。', '']
    for p in j['candidates']:
        count+=1
        angles=', '.join(f'{a:+g}' for a in p['angles_deg'])
        lines += [f"- {' ↔ '.join(p['labels'])}（{' ↔ '.join(p['refs'])}）；报警采样角度：{angles}°。"]
    lines += ['', f"零位网格报警数量：{len(j['neutral_candidates'])}。这些是跨运动组的零位报警，不能一概当成合法接触。", '']
    for p in j['neutral_candidates']:
        lines += [f"- 零位：{' ↔ '.join(p['refs'])}；网格深度 {p['depth_mm']:.6f} mm（未经逐项实体确认）。"]
    lines += ['']
lines += ['## 其他已知模型问题', '',
          '- 左右 antenna_holder 在之前的自相交检查中报警；原版同样存在。',
          '- 原运动定义的髋部轴位置已过时；当前动画使用本版 STEP 实测轴线。',
          '- 此报告不能作为整机全行程无碰撞或可制造的结论。']
(root/'motion_issues.md').write_text('\n'.join(lines)+'\n')
print('candidate pairs:',count,'; report:',root/'motion_issues.md')
