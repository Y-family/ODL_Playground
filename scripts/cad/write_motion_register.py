"""Write the per-servo repair register from fingerprinted motion reports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'cad/local_fix'
NAMES = ['右髋偏航', '右髋侧摆', '右髋俯仰', '右膝', '右踝',
         '左髋偏航', '左髋侧摆', '左髋俯仰', '左膝', '左踝',
         '颈部俯仰', '头部俯仰', '头部偏航', '头部侧摆']


def main():
    broad = json.loads((FOLDER / 'FULL_MOTION_CURRENT_REPORT.json').read_text())
    exact = json.loads((FOLDER / 'FULL_MOTION_EXACT_REPORT.json').read_text())
    digest = hashlib.sha256((FOLDER / 'open_duck_mini_upper_refit.step').read_bytes()).hexdigest()
    assert broad['target_sha256'] == exact['target_sha256']
    repair_path = FOLDER / 'FRONT_HIP_CLEARANCE_REPORT.json'
    repair = json.loads(repair_path.read_text()) if repair_path.exists() else None
    flush_path = FOLDER / 'LOWER_BRACKET_FLUSH_REPORT.json'
    flush = json.loads(flush_path.read_text()) if flush_path.exists() else None
    revised = digest != broad['target_sha256']
    if revised:
        assert repair and repair['input_sha256'] == broad['target_sha256']
        if repair['output_sha256'] != digest:
            assert flush and flush['output_sha256'] == digest
            predecessor = flush.get('previous_outer_repair', flush)
            assert predecessor['input_sha256'] == repair['output_sha256']
            if predecessor is not flush:
                assert predecessor['output_sha256'] == flush['input_sha256']
    assert exact['broad_sha256'] == hashlib.sha256((FOLDER / 'FULL_MOTION_CURRENT_REPORT.json').read_bytes()).hexdigest()
    lines = ['# 舵机运动限位、碰撞与逐项修复台账', '',
             f'全面检查输入 STEP SHA256：`{broad["target_sha256"]}`。当前预览 STEP SHA256：`{digest}`。', '',
             '限位来源：`playground/open_duck_mini_zp10d/xmls/open_duck_mini_zp10d.xml` 的 joint range（弧度转度）。',
             '正负角度按 MJCF 的关节轴及父级姿态定义；轴心取当前 STEP 实测值。旧报告曾使用相反轴向，其正负角度不得直接引用。',
             '每次只转动一个关节，其他关节保持当前 STEP 零位；全范围 5°网格初筛，检测到的首个干涉附近追加 1°实体检查。',
             '“首个碰撞角”表示本次采样中距零位最近的干涉角（交集体积 >0.05 mm³），不是连续运动安全限位；未发现不等于全姿态无碰撞。',
             '同一采购舵机内部壳体/舵盘配合及 Frame 坐标参考实体不计外部碰撞；不同实例均独立编号。', '',
             '|舵机|功能|仿真下限～上限|最大绝对角 / 总行程|本次测试范围|负向首个碰撞角|正向首个碰撞角|状态|',
             '|---|---|---|---|---|---|---|---|']
    details = []
    for index, joint in enumerate(broad['joints']):
        servo = joint['id']
        lower, upper = joint['limits']
        checks = [row for row in exact['checks'] if row['joint'] == servo]
        hits = [sample['angle_deg'] for row in checks for sample in row['collision_samples']]
        completed = servo in exact.get('completed_joints', [])
        negative = [angle for angle in hits if angle < 0]
        positive = [angle for angle in hits if angle > 0]
        minus = f'{max(negative)}°' if negative else ('未发现' if completed else '待复核')
        plus = f'+{min(positive)}°' if positive else ('未发现' if completed else '待复核')
        status = ('有碰撞，待修复' if hits else '本轮采样未发现') if completed else '实体复核中'
        if 0 in hits:
            status += '；零位亦有干涉'
        if revised and servo in ('S01', 'S06'):
            status = '已修复，当前全范围抽样通过；左列为修复前首碰角'
        lines.append(f'|{servo}|{NAMES[index]}|{lower}°～+{upper}°|{max(abs(lower),abs(upper))}° / {upper-lower}°|{lower}°～+{upper}°|{minus}|{plus}|{status}|')
        details.extend(['', f'## {servo} {NAMES[index]}', '', f'仿真关节：`{joint["name"]}`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。'])
        for row in checks:
            samples = row['collision_samples']
            if not samples:
                continue
            angles = ', '.join(str(sample['angle_deg']) for sample in samples)
            details.extend(['', f'- `{row["moving"]}` ↔ `{row["fixed"]}`',
                            f'  - 碰撞采样角（度）：{angles}；最大交集 {max(sample["overlap_mm3"] for sample in samples):.3f} mm³。'])
        if completed and not hits:
            details.extend(['', '本轮单关节采样未发现超过判据的外部干涉；暂不修改。'])
    lines.extend(details)
    if revised:
        lines.extend(['', '## 第一项修复结果：S01 / 对称 S06', '',
            '表中首碰角为全面检查输入的修复前结果。当前预览已修复这两项：原前侧下方壳体螺丝座从 Z=2.857885 移至 Z=25 mm，扩大前侧开口至 X=32.5 mm，保留前沿3.5 mm连续壳壁。仅修改 body_front、body_middle_bottom。',
            'S01/S06均检查完整 -30..+30°：黄色连接件1°步长，其他随动件5°步长，各412个零件姿态，采样间隙下界0.52036 mm。新增材料对14关节各自完整范围5°采样检查通过；没有据此声称其他旧碰撞已解决。',
            '新固定孔通路、承压环、承接材料、两件单实体及装配137叶逐面导出一致性检查通过；软件限位、舵机位置及4mm舵机连接接口未改。',
            '下一项：S03（及对称S08）护罩与髋部连接件碰撞；之后S05/S10、S11、S12、S13、S14。验收与提交状态见 UPPER_REFIT_ISSUES.md 的“当前验收与提交状态”。'])
    if flush and flush['output_sha256'] == digest:
        lines.extend(['', '## S02/S07 下架凸边修复', '',
                      '按用户外观反馈，去除两个下架外侧0.60 mm及内侧约2.47 mm剩余凸边，底板内外侧均与对应侧壁齐平。S02/S07完整±25°范围逐1°检查通过，最小随动间隙8.16398 mm。舵机底面及两侧承托面完整保留，其余135个零件逐面一致；未改限位及舵机接口。记录：LOWER_BRACKET_FLUSH_REPORT.json。'])
    (FOLDER / 'MOTION_COLLISION_REGISTER.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:25]))


if __name__ == '__main__':
    main()
