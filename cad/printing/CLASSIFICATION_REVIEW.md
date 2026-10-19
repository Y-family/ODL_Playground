# 当前 ZP10D 制造分类复核（2026-10-19）

本轮弃用后来添加的人工 `PRINTED` 名单，重新使用原项目官方打印指南、实际提供的打印 STL、公开采购 BOM 与当前装配来源。网格有效性与制造分类分别核对。

当前 115 个 CAD 实例中：52 个有官方打印依据，61 个映射为成品件，2 根天线制造方式待确认。61 是 CAD 实例数，舵机壳体/舵盘及扬声器分体分别建模；不是完整实物采购清单或采购数量。

## 可追溯来源

- 原项目：https://github.com/apirrone/Open_Duck_Mini ，固定提交 `b23317a485b3cec7d8417f352478778b3475173c`。本地打印指南与该提交的远端原文逐字节一致，原指南最后相关提交 `730c14bc50e4e60c4a983f57fd739e015510cc76`（2025-04-19，adding missing stl files）。
- 官方打印指南：https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/docs/print_guide.md ，列出36种、共51件；当前用量按当前装配实例计，不盲目沿用原版数量。
- 官方采购 BOM（原项目 README 指向）：https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit ，本轮只读下载CSV快照。该在线表可变，快照日期与哈希固定如下。
- 当前正式 STEP：`cad/local_fix/open_duck_mini_upper_refit.step`，SHA256 `bb9c851dfcae554653981919bf69a950febc2730c4362e562f58a0a63f7ce740`。装配位置、层级和局部改件来源保留；本轮没有修改几何。

|本地证据快照|SHA256|
|---|---|
|cad/printing/sources/print_guide.md|`3ba00d465138d00a89661c759aaab0a3c977c2fe5c9ce849b853710e58601700`|
|cad/printing/sources/bom.csv|`aca4e86823c006efe3af2d8e17e3d95da10bc41a0d0de887c554ebb443bd84e8`|

原指南的PLA/TPU参数仅作历史来源记录；当前用户指定PETG、外壳10%/其余30%填充仍优先。Onshape/MuJoCo的visual/collision忽略配置不代表制造方式，未用它判断是否打印。

## 纠正项与几何复核

原50件名单中移出2根无制造依据的天线，补回左右眼各1件与bulb环件2件，得到52件有依据的打印实例。bulb在官方指南写1件，当前装配有两个相同环件，因此记录为2件并明确这一数量差异。
对左右眼与两个bulb，比较官方print目录STL和当前正式视觉网格：单位统一为mm、仅平移至各自质心，不缩放、不拟合旋转；每个方向500个固定种子表面样本，对照尺寸/体积。下列距离是网格间采样值，不是全表面误差证明；当前旧视觉网格弦差设置0.35 mm。此次打印输出重新按0.08 mm网格导出。

|当前实例|官方打印文件|双向各500点采样最大距离 mm|当前CAD体积 mm³|
|---|---|---|---|
|063 left_eye|print/left_eye.stl|0.043847|1425.511988|
|064 right_eye|print/right_eye.stl|0.048605|4632.955033|
|075 bulb|print/bulb.stl|0.035691|1697.181674|
|076 bulb|print/bulb.stl|0.035691|1697.181674|

|官方参考打印STL|SHA256|
|---|---|
|print/left_eye.stl|`04ec9038953f63f5ff9855338465f6f2c6bea92bd2211ee9bcd4e808074c6e0f`|
|print/right_eye.stl|`1d8ab5b5f5f153a2d73b4f195d4967ff5ba5dca7e761aa41ad0c50140e6b9e4d`|
|print/bulb.stl|`043cea34b4b37867de16f4f3e1729cabf460e633c08d45ee24c88c9e224f96f8`|

## 有官方打印依据的当前零件

“当前名称→官方参考文件”映射是明确记录的对应关系。改过的ZP10D连接件继续使用当前STEP几何，不用原版STL替换新尺寸；腿片名称增加left前缀的两种别名已显式映射。几何通过不等于实物承载/装配已验收。

|当前名称|当前数量|实例index|官方打印参考|原指南数量|
|---|---|---|---|
|left_cache|1|0|print/left_cache.stl|1|
|right_antenna_holder|1|1|print/right_antenna_holder.stl|1|
|right_cache|1|3|print/right_cache.stl|1|
|head_pitch_to_yaw|1|4|print/head_pitch_to_yaw.stl|1|
|neck_left_sheet|1|5|print/neck_left_sheet.stl|1|
|neck_right_sheet|1|6|print/neck_right_sheet.stl|1|
|left_antenna_holder|1|10|print/left_antenna_holder.stl|1|
|roll_motor_top|2|12, 88|print/roll_motor_top.stl|2|
|roll_motor_bottom|2|13, 87|print/roll_motor_bottom.stl|2|
|leg_spacer|4|17, 51, 54, 111|print/leg_spacer.stl|4|
|left_knee_to_ankle_left_sheet|4|18, 46, 53, 109|print/knee_to_ankle_left_sheet.stl|4|
|left_knee_to_ankle_right_sheet|4|22, 47, 52, 110|print/knee_to_ankle_right_sheet.stl|4|
|body_front|1|23|print/body_front.stl|1|
|body_middle_bottom|1|24|print/body_middle_bottom.stl|1|
|body_back|1|25|print/body_back.stl|1|
|battery_pack_lid|1|27|print/battery_pack_lid.stl|1|
|body_middle_top|1|28|print/body_middle_top.stl|1|
|head_yaw_to_roll|1|37|print/head_yaw_to_roll.stl|1|
|right_roll_to_pitch|1|38|print/right_roll_to_pitch.stl|1|
|foot_bottom_tpu|2|42, 84|print/foot_bottom_tpu.stl|2|
|foot_side|2|43, 83|print/foot_side.stl|2|
|foot_top|2|44, 85|print/foot_top.stl|2|
|foot_bottom_pla|2|45, 86|print/foot_bottom_pla.stl|2|
|flash_reflector_interface|1|59|print/flash_reflector_interface.stl|1|
|speaker_interface|1|60|print/speaker_interface.stl|1|
|head|1|62|print/head.stl|1|
|left_eye|1|63|print/left_eye.stl|1|
|right_eye|1|64|print/right_eye.stl|1|
|speaker_stand|1|65|print/speaker_stand.stl|1|
|flash_light_module|1|66|print/flash_light_module.stl|1|
|head_bot_sheet|1|69|print/head_bot_sheet.stl|1|
|head_roll_mount|1|70|print/head_roll_mount.stl|1|
|bulb|2|75, 76|print/bulb.stl|1|
|trunk_top|1|92|print/trunk_top.stl|1|
|trunk_bottom|1|99|print/trunk_bottom.stl|1|
|left_roll_to_pitch|1|105|print/left_roll_to_pitch.stl|1|

## 成品件映射依据

BOM映射结合当前CAD名称、装配位置/用途和数量，不是SKU尺寸/电气兼容验收。例如holder位于18650 pack，glass为眼部4个约29.6 mm的薄圆片，对应BOM的4个Eye diffusers。ZP10D壳体/舵盘来自当前卖家舵机替换组，不沿用原版STS3215型号。

|当前名称|CAD实例数|来源|
|---|---|---|
|ZP10D_vendor_body|14|当前ZP10D卖家舵机组件（舵机本体/舵盘）|
|ZP10D_vendor_passive_horn|12|当前ZP10D卖家舵机组件（舵机本体/舵盘）|
|ZP10D_vendor_active_horn|14|当前ZP10D卖家舵机组件（舵机本体/舵盘）|
|usb_c_charger|1|BOM CSV第11行：usb-c charger|
|power_switch|1|BOM CSV第10行：Small power switch|
|cell|2|BOM CSV第6行：18650 cell|
|holder|1|BOM CSV第7行：18650 cell holder|
|bms|1|BOM CSV第8行：2S BMS|
|RaspberryPiZeroW|1|BOM CSV第15行：Raspberry pi zero 2w|
|roll_bearing|3|BOM CSV第27行：Bearing|
|glass|4|BOM CSV第37行：Eye diffusers|
|sg90|2|BOM CSV第20行：9g servos|
|speaker_bottom|1|BOM CSV第33行：Speaker|
|speaker_top|1|BOM CSV第33行：Speaker|
|Cam|1|BOM CSV第39行：Raspberry pi camera|
|BNO055|1|BOM CSV第14行：IMU (BNO055)|
|Board|1|BOM CSV第17行：Servo control board|

## 待确认：两根天线

index 2、11的antenna未出现在官方打印指南或print/顶层目录，公开BOM也未给出天线本体的材料/采购项。原仓库相关issue只讨论舵机动作，不能证明天线的制造方式。没有把“不在打印清单”当作“采购件”。
两件原导出保留于unclassified/STL及unclassified/STEP，仅供审核；不在52件确认包中。需要原设计制造说明或明确的成品/打印选择，才能补齐这两项；本轮不凭细长外形猜材料。

本轮只修正打印导出分类。原动力学质量/碰撞模型与整机STEP保持，历史动力学分类字段不作为打印依据。导出器遇到新名称默认待确认，并在manifest中记录，不再自动排为采购件。

## 验证

分类回归覆盖：左右眼/bulb/腿片别名归入官方打印来源，18650 holder匹配BOM，新未知名称与天线保留待确认。52件实际STL重载与有效BRep验收通过，源STEP哈希保持；切片壁厚/支撑限制见PRINT_REPORT.md。
