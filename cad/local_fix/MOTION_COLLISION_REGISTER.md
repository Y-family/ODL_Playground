# 舵机运动限位、碰撞与逐项修复台账

全面检查输入 STEP SHA256：`993ac313d66fab58c424a1c54d9b6e69165a0e24e090927024f7b47b1b0a0af3`。当前预览 STEP SHA256：`945755d7426bbf28b3c6a4287605ddcdbf206a2ae3e8879e663378aa50a43976`。

限位来源：`playground/open_duck_mini_zp10d/xmls/open_duck_mini_zp10d.xml` 的 joint range（弧度转度）。
正负角度按 MJCF 的关节轴及父级姿态定义；轴心取当前 STEP 实测值。旧报告曾使用相反轴向，其正负角度不得直接引用。
每次只转动一个关节，其他关节保持当前 STEP 零位；全范围 5°网格初筛，检测到的首个干涉附近追加 1°实体检查。
“首个碰撞角”表示本次采样中距零位最近的干涉角（交集体积 >0.05 mm³），不是连续运动安全限位；未发现不等于全姿态无碰撞。
同一采购舵机内部壳体/舵盘配合及 Frame 坐标参考实体不计外部碰撞；不同实例均独立编号。

|舵机|功能|仿真下限～上限|最大绝对角 / 总行程|本次测试范围|负向首个碰撞角|正向首个碰撞角|状态|
|---|---|---|---|---|---|---|---|
|S01|右髋偏航|-30°～+30°|30° / 60°|-30°～+30°|未发现|+8°|已修复，当前全范围抽样通过；左列为修复前首碰角|
|S02|右髋侧摆|-25°～+25°|25° / 50°|-25°～+25°|未发现|未发现|本轮采样未发现|
|S03|右髋俯仰|-30°～+70°|70° / 100°|-30°～+70°|未发现|+45°|有碰撞，待修复|
|S04|右膝|-90°～+90°|90° / 180°|-90°～+90°|未发现|未发现|本轮采样未发现|
|S05|右踝|-90°～+90°|90° / 180°|-90°～+90°|-61°|+41°|有碰撞，待修复|
|S06|左髋偏航|-30°～+30°|30° / 60°|-30°～+30°|-8°|未发现|已修复，当前全范围抽样通过；左列为修复前首碰角|
|S07|左髋侧摆|-25°～+25°|25° / 50°|-25°～+25°|未发现|未发现|本轮采样未发现|
|S08|左髋俯仰|-70°～+30°|70° / 100°|-70°～+30°|-45°|未发现|有碰撞，待修复|
|S09|左膝|-90°～+90°|90° / 180°|-90°～+90°|未发现|未发现|本轮采样未发现|
|S10|左踝|-90°～+90°|90° / 180°|-90°～+90°|-61°|+41°|有碰撞，待修复|
|S11|颈部俯仰|-20°～+65°|65° / 85°|-20°～+65°|未发现|+39°|有碰撞，待修复|
|S12|头部俯仰|-45°～+45°|45° / 90°|-45°～+45°|-15°|+15°|有碰撞，待修复|
|S13|头部偏航|-160°～+160°|160° / 320°|-160°～+160°|-54°|+107°|有碰撞，待修复|
|S14|头部侧摆|-30°～+30°|30° / 60°|-30°～+30°|-5°|+12°|有碰撞，待修复|

## S01 右髋偏航

仿真关节：`right_hip_yaw`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/right_roll_to_pitch_assembly <1>[16]/right_roll_to_pitch[0]` ↔ `/body_assembly <1>[14]/body_middle_bottom[1]`
  - 碰撞采样角（度）：8, 9, 10, 15, 20, 25, 30；最大交集 1367.991 mm³。

## S02 右髋侧摆

仿真关节：`right_hip_roll`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

本轮单关节采样未发现超过判据的外部干涉；暂不修改。

## S03 右髋俯仰

仿真关节：`right_hip_pitch`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/right_cache[5]` ↔ `/right_roll_to_pitch_assembly <1>[16]/right_roll_to_pitch[0]`
  - 碰撞采样角（度）：45, 50, 55, 60, 65, 70；最大交集 1598.507 mm³。

## S04 右膝

仿真关节：`right_knee`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

本轮单关节采样未发现超过判据的外部干涉；暂不修改。

## S05 右踝

仿真关节：`right_ankle`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/foot_assembly <1>[17]/foot_side[1]` ↔ `/knee_and_ankle_assembly <3>[13]/left_knee_to_ankle_right_sheet[3]`
  - 碰撞采样角（度）：-90, -85, -80, -75, -70, -65, -64, -63, -62, -61, 41, 42, 43, 44, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90；最大交集 954.108 mm³。

- `/foot_assembly <1>[17]/foot_side[1]` ↔ `/knee_and_ankle_assembly <3>[13]/leg_spacer[0]`
  - 碰撞采样角（度）：87, 88, 89, 90；最大交集 1.203 mm³。

- `/foot_assembly <1>[17]/foot_top[2]` ↔ `/knee_and_ankle_assembly <3>[13]/left_knee_to_ankle_left_sheet[1]`
  - 碰撞采样角（度）：-90, -85, -80, -75, -70, 52, 53, 54, 55, 60, 65, 70, 75, 80, 85, 90；最大交集 755.964 mm³。

## S06 左髋偏航

仿真关节：`left_hip_yaw`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/left_roll_to_pitch_assembly <1>[25]/left_roll_to_pitch[0]` ↔ `/body_assembly <1>[14]/body_middle_bottom[1]`
  - 碰撞采样角（度）：-30, -25, -20, -15, -10, -9, -8；最大交集 1367.997 mm³。

## S07 左髋侧摆

仿真关节：`left_hip_roll`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

本轮单关节采样未发现超过判据的外部干涉；暂不修改。

## S08 左髋俯仰

仿真关节：`left_hip_pitch`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/left_cache[0]` ↔ `/left_roll_to_pitch_assembly <1>[25]/left_roll_to_pitch[0]`
  - 碰撞采样角（度）：-70, -65, -60, -55, -50, -45；最大交集 1598.507 mm³。

## S09 左膝

仿真关节：`left_knee`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

本轮单关节采样未发现超过判据的外部干涉；暂不修改。

## S10 左踝

仿真关节：`left_ankle`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/foot_assembly <2>[22]/foot_side[0]` ↔ `/knee_and_ankle_assembly <4>[20]/left_knee_to_ankle_right_sheet[0]`
  - 碰撞采样角（度）：-90, -85, -80, -75, -70, -65, -64, -63, -62, -61, 41, 42, 43, 44, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90；最大交集 954.108 mm³。

- `/foot_assembly <2>[22]/foot_side[0]` ↔ `/knee_and_ankle_assembly <4>[20]/leg_spacer[2]`
  - 碰撞采样角（度）：87, 88, 89, 90；最大交集 1.203 mm³。

- `/foot_assembly <2>[22]/foot_top[2]` ↔ `/knee_and_ankle_assembly <4>[20]/left_knee_to_ankle_left_sheet[1]`
  - 碰撞采样角（度）：-90, -85, -80, -75, -70, 52, 53, 54, 55, 60, 65, 70, 75, 80, 85, 90；最大交集 755.964 mm³。

## S11 颈部俯仰

仿真关节：`neck_pitch`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/head_assembly <1>[21]/head[4]` ↔ `/body_assembly <1>[14]/18650 pack <1>[7]/holder[1]`
  - 碰撞采样角（度）：62, 63, 64, 65；最大交集 255.115 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/body_assembly <1>[14]/battery_pack_lid[4]`
  - 碰撞采样角（度）：57, 58, 59, 60, 65；最大交集 894.300 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/body_assembly <1>[14]/body_back[2]`
  - 碰撞采样角（度）：41, 42, 43, 44, 45, 50, 55, 60, 65；最大交集 3620.271 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：39, 40, 45, 50, 55, 60, 65；最大交集 1558.050 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/body_assembly <1>[14]/power_switch[6]`
  - 碰撞采样角（度）：45, 50, 55, 60；最大交集 170.969 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/body_assembly <1>[14]/usb_c_charger[3]`
  - 碰撞采样角（度）：58, 59, 60, 65；最大交集 132.076 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/trunk_assembly <1>[24]/Board[2]`
  - 碰撞采样角（度）：64, 65；最大交集 65.934 mm³。

- `/head_assembly <1>[21]/head[4]` ↔ `/trunk_assembly <1>[24]/trunk_top[0]`
  - 碰撞采样角（度）：64, 65；最大交集 26.383 mm³。

- `/head_assembly <1>[21]/head_bot_sheet[11]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：40, 45, 50, 55, 60, 65；最大交集 1444.105 mm³。

- `/head_assembly <1>[21]/head_bot_sheet[11]` ↔ `/trunk_assembly <1>[24]/Board[2]`
  - 碰撞采样角（度）：65；最大交集 57.713 mm³。

- `/head_assembly <1>[21]/head_bot_sheet[11]` ↔ `/trunk_assembly <1>[24]/trunk_top[0]`
  - 碰撞采样角（度）：65；最大交集 1.438 mm³。

- `/head_assembly <1>[21]/roll_bearing[3]` ↔ `/body_assembly <1>[14]/body_back[2]`
  - 碰撞采样角（度）：57, 58, 59, 60, 65；最大交集 281.800 mm³。

- `/head_assembly <1>[21]/roll_bearing[3]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：53, 54, 55, 60；最大交集 401.561 mm³。

- `/head_assembly <1>[21]/roll_bearing[3]` ↔ `/body_assembly <1>[14]/power_switch[6]`
  - 碰撞采样角（度）：63, 64, 65；最大交集 61.935 mm³。

- `/head_assembly <1>[21]/sg90[13]` ↔ `/body_assembly <1>[14]/body_back[2]`
  - 碰撞采样角（度）：56, 57, 58, 59, 60, 65；最大交集 550.356 mm³。

- `/head_assembly <1>[21]/sg90[13]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：54, 55, 60, 65；最大交集 904.378 mm³。

- `/head_assembly <1>[21]/sg90[22]` ↔ `/body_assembly <1>[14]/body_back[2]`
  - 碰撞采样角（度）：56, 57, 58, 59, 60, 65；最大交集 635.499 mm³。

- `/head_assembly <1>[21]/sg90[22]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：54, 55, 60, 65；最大交集 921.177 mm³。

- `/head_pitch_to_yaw[6]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：65；最大交集 53.240 mm³。

- `/neck_pitch_assembly <1>[8]/neck_left_sheet[0]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：55, 60, 65；最大交集 13.897 mm³。

- `/neck_pitch_assembly <1>[8]/neck_left_sheet[0]` ↔ `/trunk_assembly <1>[24]/S06_LEFT_HIP_YAW_ZP10D[5]/ZP10D_vendor_passive_horn[1]`
  - 碰撞采样角（度）：52, 53, 54, 55, 60, 65；最大交集 67.763 mm³。

- `/neck_pitch_assembly <1>[8]/neck_left_sheet[0]` ↔ `/trunk_assembly <1>[24]/trunk_top[0]`
  - 碰撞采样角（度）：47, 48, 49, 50, 55, 60, 65；最大交集 182.489 mm³。

- `/neck_pitch_assembly <1>[8]/neck_right_sheet[1]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：55, 60, 65；最大交集 39.725 mm³。

- `/neck_pitch_assembly <1>[8]/neck_right_sheet[1]` ↔ `/trunk_assembly <1>[24]/S01_RIGHT_HIP_YAW_ZP10D[8]/ZP10D_vendor_passive_horn[1]`
  - 碰撞采样角（度）：55, 60, 65；最大交集 74.267 mm³。

- `/neck_pitch_assembly <1>[8]/neck_right_sheet[1]` ↔ `/trunk_assembly <1>[24]/trunk_top[0]`
  - 碰撞采样角（度）：46, 47, 48, 49, 50, 55, 60, 65；最大交集 234.000 mm³。

- `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]` ↔ `/body_assembly <1>[14]/body_back[2]`
  - 碰撞采样角（度）：57, 58, 59, 60, 65；最大交集 399.638 mm³。

- `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]` ↔ `/body_assembly <1>[14]/body_middle_top[5]`
  - 碰撞采样角（度）：48, 49, 50, 55, 60, 65；最大交集 1255.638 mm³。

- `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]` ↔ `/body_assembly <1>[14]/power_switch[6]`
  - 碰撞采样角（度）：61, 62, 63, 64, 65；最大交集 296.882 mm³。

## S12 头部俯仰

仿真关节：`head_pitch`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/head_pitch_to_yaw[6]` ↔ `/neck_pitch_assembly <1>[8]/S12_HEAD_PITCH_ZP10D[2]/ZP10D_vendor_body[0]`
  - 碰撞采样角（度）：-45, -40, -35, -30, -25, -20, -15, 15, 20, 25, 30, 35, 40, 45；最大交集 115.876 mm³。

## S13 头部偏航

仿真关节：`head_yaw`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/head_assembly <1>[21]/speaker_stand[7]` ↔ `/head_pitch_to_yaw[6]`
  - 碰撞采样角（度）：-140, -135, -130, -125, -120, -115, -75, -70, -65, -60, -55, -54；最大交集 3.096 mm³。

- `/neck_yaw_assembly <1>[15]/S13_HEAD_YAW_ZP10D[0]/ZP10D_vendor_body[0]` ↔ `/head_pitch_to_yaw[6]`
  - 碰撞采样角（度）：-160, -155, -150, -145, -140, -135, -130, -125, -120, -115, -110, -109, 109, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160；最大交集 1907.606 mm³。

- `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]` ↔ `/head_pitch_to_yaw[6]`
  - 碰撞采样角（度）：-160, -155, -150, -145, -140, -135, -130, -125, -120, -115, -110, -109, -108, -107, 107, 108, 109, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160；最大交集 178.468 mm³。

## S14 头部侧摆

仿真关节：`head_roll`。后续按 S01→S14 顺序处理；修改原零件，保持软件限位。

- `/head_assembly <1>[21]/sg90[13]` ↔ `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]`
  - 碰撞采样角（度）：12, 13, 14, 15, 20, 25, 30；最大交集 894.801 mm³。

- `/head_assembly <1>[21]/sg90[22]` ↔ `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]`
  - 碰撞采样角（度）：-30, -25, -20, -15, -14, -13, -12；最大交集 894.801 mm³。

- `/head_assembly <1>[21]/speaker_bottom[14]` ↔ `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]`
  - 碰撞采样角（度）：-30, -25, -20, -15；最大交集 101.831 mm³。

- `/head_assembly <1>[21]/speaker_interface[2]` ↔ `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]`
  - 碰撞采样角（度）：-30, -25, -20, -15, -10, -9；最大交集 142.039 mm³。

- `/head_assembly <1>[21]/speaker_stand[7]` ↔ `/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]`
  - 碰撞采样角（度）：-30, -25, -20, -15, -10, -5；最大交集 263.385 mm³。

## 第一项修复结果：S01 / 对称 S06

表中首碰角为全面检查输入的修复前结果。当前预览已修复这两项：原前侧下方壳体螺丝座从 Z=2.857885 移至 Z=25 mm，扩大前侧开口至 X=32.5 mm，保留前沿3.5 mm连续壳壁。仅修改 body_front、body_middle_bottom。
S01/S06均检查完整 -30..+30°：黄色连接件1°步长，其他随动件5°步长，各412个零件姿态，采样间隙下界0.52036 mm。新增材料对14关节各自完整范围5°采样检查通过；没有据此声称其他旧碰撞已解决。
新固定孔通路、承压环、承接材料、两件单实体及装配137叶逐面导出一致性检查通过；软件限位、舵机位置及4mm舵机连接接口未改。
下一项：S03（及对称S08）护罩与髋部连接件碰撞；之后S05/S10、S11、S12、S13、S14。验收与提交状态见 UPPER_REFIT_ISSUES.md 的“当前验收与提交状态”。

## S02/S07 下架凸边修复

按用户外观反馈，去除两个下架外侧0.60 mm及内侧约2.47 mm剩余凸边，底板内外侧均与对应侧壁齐平。S02/S07完整±25°范围逐1°检查通过，最小随动间隙8.16398 mm。舵机底面及两侧承托面完整保留，其余135个零件逐面一致；未改限位及舵机接口。记录：LOWER_BRACKET_FLUSH_REPORT.json。
