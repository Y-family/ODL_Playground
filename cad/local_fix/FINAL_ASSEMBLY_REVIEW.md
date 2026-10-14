<!-- DEMONSTRATED_COLLISION_ACCEPTANCE_START -->
## 2026-10-14 用户接受本次动画展示的运动干涉

用户在查看碰撞运动3D动画后明确：“这个级别的碰撞可以不管”。本次动画展示的这一级别运动干涉记录为已接受，不再为这些已展示干涉修改当前结构或外观。对应片段为 `head_body`、`antenna_top`、`antenna_back`、`feet`；检测姿态、交集体积和动画验证证据继续保留。

该决定仅针对本次动画所展示的运动干涉，不设通用碰撞体积豁免阈值；未展示的碰撞、零位材料重叠及其它复核条目仍保留各自验证状态，不据此宣称整机无碰撞或全部验收通过。正式STEP SHA256仍为 `6c0ee39b6e657904bc7fb6935aebcdd1d92c974a0aa1325e4fb520de9a9bd63c`。
<!-- DEMONSTRATED_COLLISION_ACCEPTANCE_END -->

<!-- COLLISION_MOTION_REPLAY_START -->
## 2026-10-14 碰撞运动动画

用户要求直接在3D动画中演示。已在正式STEP旁添加 `open_duck_mini_upper_refit.step.js`，Viewer直接加载四段12秒动画：头壳/身体上壳、右天线夹持件/身体上壳、夹持件/身体后壳、双脚极限组合。正式STEP几何和SHA256保持不变。

打开 http://127.0.0.1:3245/?file=open_duck_mini_upper_refit.step ，点击底部Play；右侧Animation的Clip下拉可切换片段。第6～7.5秒停在选定零件对首次检测到正体积干涉的姿态，第9～10.5秒展示已确认的碰撞终点，其余时段缓慢运动/返回；可暂停、拖动时间和旋转视角。碰撞零件保持突出，其余实体半透明，参考坐标架隐藏。

沿所选路径，头壳/身体上壳首次确认干涉约在S11=16.2456°、S12=18.7449°、S13=−41.6553°、S14=−12.4966°；天线夹持件/身体上壳约在15.4277°、17.8011°、−39.5581°、−11.8674°。只表示指定零件对沿此路径的离散采样及局部二分结果，不表示全机首次接触或连续轨迹无碰撞证明。

本轮局部BRep路径复核四组起点零干涉、终点干涉复现；动画36个时间点、4140个零件变换与CAD计算一致，最大矩阵元素差4.44e−14，目标引用无缺失。浏览器自动化启动受限（Chromium进程SIGTRAP），未声称已完成浏览器端播放实测；Viewer页面及动画模块HTTP均可访问。完整证据见 `FINAL_ASSEMBLY_REVIEW.json` 的 `motion_demonstration`。
<!-- COLLISION_MOTION_REPLAY_END -->

# 当前正式装配整体复核（2026-10-14）

结论：当前版本仍有组合运动碰撞、零位体积重叠、运行时限位不同步及仿真模型过期问题，不能作为整机验收通过版。本轮未改变正式 STEP 的几何或外观。

正式文件：`open_duck_mini_upper_refit.step`；SHA256 `6c0ee39b6e657904bc7fb6935aebcdd1d92c974a0aa1325e4fb520de9a9bd63c`。参考构造几何不参与物理碰撞，实际检查115个实体叶节点；同一采购舵机内部零件间的重叠不作装配碰撞。

## 主要问题

| 编号 | 问题和本轮证据 | 建议 |
|---|---|---|
| R01 / P1 | 站姿下 S11=19.5°、S12=22.5°、S13=−50°、S14=−15°，头壳/身体上壳重叠395.942 mm³；同类姿态还碰头底板、天线及夹持件。 | 先建立头颈组合姿态约束；单轴±100°不能保证联动安全。 |
| R02 / P1 | 双腿极限组合存在互碰，例如左右 foot_top 重叠11685.709 mm³；还发现脚/身体、腿板/身体等组合碰撞。每个见证姿态均在各轴当前范围内。 | 建立腿部联动与双腿互避约束，再确定实际允许动作范围。 |
| R03 / P1 | 零位已有12组明显体积重叠：body_back/bms 494.508 mm³，trunk_top/trunk_bottom 238.919 mm³；电池仓盖/电芯各194.212 mm³，脚底装配也有重叠。另head/Cam约0.002333 mm³。 | 逐项核实装配建模与真实材料空间，不能把固定不动视为没有碰撞。 |
| R04 / P1 | Runtime默认/示例仍为S11 −20…65°、S13 ±135°，本项目为−20…39°、±100°；未取得已部署实机配置，实体S13限位尚未写入。 | 同步运行配置及实际校准后的硬件限位，并回读/验证；当前未对实体下发命令。 |
| R05 / P1 | MJCF来自旧covers_forward_review.step；S11/S12/S13轴线误差8.000 mm，头壳边界误差32.954 mm。机器人只有两个脚底启用接触，头部/支架自碰撞不会由MuJoCo检测。 | 从正式STEP更新运动学、网格、质量及惯量；建立保留壳体/圆环空腔的碰撞表示。 |
| R06 / P2 | 标准几何校验自交告警：right_antenna_holder、left_antenna_holder、head。实体拓扑与正体积初检通过，但不等于自交检查通过。 | 定位并修复/复核自交后再打印；未以is_valid替代完整校验。 |
| R07 / P2 | 原4 mm舵机螺丝的实际有效螺纹深度、头型承压及工具空间未完成实物核实。孔轴检查不能证明啮合或连接强度。 | 按实际卖家螺丝与螺纹深度验收，不默认换长螺丝。 |

## 本轮已完成的检查

- 14主轴单独运动：324个姿态条目，每5°并包含端点。4组踝部网格候选经BRep复核无体积交集；其余单轴抽样未发现候选碰撞。此项没有覆盖关节联动，也不是全范围0.2 mm间隙证明。
- 整机组合：939个姿态条目（头部81、单腿各243、双腿243、随机整机128及零位）；278组候选见证已处理，其中273组确认有共同材料，错误0。见证角度和零件完整路径在JSON中。
- 启动到站姿：同步关节插值21个点未发现候选碰撞；站姿配合头部半幅活动81个姿态，确认8组碰撞。尚未验证实物的不同步、速度和超调。
- 两侧天线：155个诊断姿态，包含自身±90°每5°及头部yaw/roll组合；未发现网格碰撞候选，错误0。±90°仅用于几何诊断。
- 标准CLI refs/facts/planes/positioning及validate：115物理实体；自交告警3处，partial=False。完整报告保留在JSON的standard_validation。
- S13两对已知局部接口：±100°内每1°复核，共402个配对姿态，零体积干涉，最小运动间隙0.300939 mm。仅此两对及单轴范围通过，不能覆盖头颈联动。
- 14舵机实际孔轴复核已完成：39组模式、156个孔轴均匹配（最大轴线误差小于0.000001 mm）；零件到对应连接件的最近距离不超过0.225 mm。这只检查轴线与最近距离。逐孔结果见JSON的interface_validation；没有把旧检查器失败当作孔道缺失证据，也没有以孔轴匹配替代4 mm螺丝啮合验收。

## MuJoCo数值诊断（旧模型）

实际编译MuJoCo 3.12.0，14执行器，旧模型总质量2.233770 kg。三组测试各8秒；固定基座测试仅移除free joint并保留重力，站姿保持使用自由基座。没有本地行走策略ONNX，因此未运行学习策略行走回放。

| 测试 | 最大目标跟踪误差 | 最大执行器饱和比例 | 最大软限位超出 | 基座结果 |
|---|---:|---:|---:|---|
| fixed_base_neutral | 1.019° | 0.00% | 0.0000° | 最低Z=0.2200 m；最大倾角0.000°；警告0 |
| fixed_base_sweep | 4.974° | 2.12% | 0.0565° | 最低Z=0.2200 m；最大倾角0.000°；警告0 |
| free_home_hold | 2.840° | 0.00% | 0.0000° | 最低Z=0.1500 m；最大倾角7.295°；警告0 |

这些结果只说明旧模型的数值行为。kp、摩擦、阻尼和连续扭矩未实测；堵转扭矩0.980665 N·m不代表连续额定能力。旧模型站姿保持成功不能替代当前正式装配的动力学验收。

## 证据与范围

- 数据：`FINAL_ASSEMBLY_REVIEW.json`、`FULL_MOTION_CURRENT_REPORT.json`、`FULL_MOTION_EXACT_REPORT.json`。
- 图：`FINAL_ASSEMBLY_REVIEW.png`，由当前STEP网格和实测关节轴生成，红色为碰撞配对零件。无可见几何修改，标准snapshot按inspection-only规则跳过。
- 全部重型任务串行，原生线程/worker均1，导入几何库前地址空间硬限额4,000,000,000字节；未放宽资源限额。
- 离散采样不能证明所有连续角度、所有组合或全机0.2 mm运行间隙；FCL阴性也不作为闭合体包含关系的完整证明。正碰撞用BRep交集或两个实体内的正半径球验证，球法仅报告体积下界，不伪装成交集总体积。
- 优先处理R01/R02/R04的控制约束和R05仿真同步，再逐项处理零位重叠与自交；保持当前外观要求，不擅自缩环、移舵机或删除双端支撑。

## 零位体积交集明细

| 零件A | 零件B | 交集体积 mm³ |
|---|---|---:|
| body_assembly <1> / body_back | body_assembly <1> / holder | 31.779718 |
| body_assembly <1> / body_back | body_assembly <1> / bms | 494.508000 |
| body_assembly <1> / battery_pack_lid | body_assembly <1> / cell | 194.212146 |
| body_assembly <1> / battery_pack_lid | body_assembly <1> / cell | 194.212146 |
| body_assembly <1> / battery_pack_lid | body_assembly <1> / bms | 20.300000 |
| foot_assembly <1> / foot_bottom_tpu | foot_assembly <1> / foot_side | 610.749435 |
| foot_assembly <1> / foot_side | foot_assembly <1> / foot_bottom_pla | 6.464909 |
| foot_assembly <1> / foot_top | foot_assembly <1> / foot_bottom_pla | 44.700231 |
| head_assembly <1> / head | head_assembly <1> / Cam | 0.002333 |
| foot_assembly <2> / foot_side | foot_assembly <2> / foot_bottom_tpu | 610.748728 |
| foot_assembly <2> / foot_side | foot_assembly <2> / foot_bottom_pla | 6.464909 |
| foot_assembly <2> / foot_top | foot_assembly <2> / foot_bottom_pla | 44.700231 |
| trunk_assembly <1> / trunk_top | trunk_assembly <1> / trunk_bottom | 238.919130 |

这些是当前CAD的材料体积交集；可能包含沿用的装配建模问题。必须核实实际材料、装配与必要间隙，不能直接认定为允许的固定接触。
