# Codex 项目交接（2026-11-01）

当前目标是迁移环境，已切分支 `migration/zp10d-training-20261101`，保存训练、
仿真、奖励函数、标定、参考轨迹、模型输入以及完整历史权重。
迁移操作见 `MIGRATION.md`；验证记录和归档哈希见迁移包清单。

## 首先遵守

- 读取根 `AGENTS.md`、`cad/local_fix/USER_RULES.md`。
  用户问答/进度询问不取消任务，目标未完成且能推进时不得主动停止。
- 工作区总实际内存不超过 10 GB。CAD 命令经 `run_memory_limited.py`
  限地址空间 4 GB；所有 JAX 命令加 `--gpu`，cgroup 限实际 RAM 7 GB、无 swap。
  共用串行锁，单原生线程；训练默认 MX350 CUDA、512 环境，保留评估。
- 不自行改变 CAD 几何，不新增机械零件；通常仅改原连接件尺寸。
  最小间隙默认 0.2 mm；SolidWorks 同时仅一个文档，禁止反复打开窗口耗尽内存。
- 用户要本地 MuJoCo 原生 3D 窗口，不要以浏览器渲染替代；必须区分实时训练和轨迹回放。
- 不删除训练进度/原始采集；不在未核验时声称已完成或已能稳定行走。

## CAD 状态

用户已接受 `db1f780`：两侧髋部橙色连接件各再内收 1.36 mm，腿距共缩 2.72 mm；
身体底部与背部衔接的后方矩形缺口按原壁厚平滑补齐。
同名 STEP、SLDPRT 已导出，禁止把前方开口当成后方缺口再改。
最新 STEP 哈希 `91e610bc6e8640f1fc8698a5cfdc121c2be64f48ce824873e99b85a4ce08533e`。
近期按用户边界仅查单关节/直接接口，组合运动碰撞暂不纳入。

**动力学仍是 `ae7b161`（STEP 哈希 `528d82d2a2bda0a7757f28600dc02b289c73220f0cf98c8b0d39ebacd92bf65b`）。**
已有 checkpoint 绑定这个 URDF/MJCF/参考；直接从最新 CAD 重新生成会改变模型。
使用 MIGRATION.md 的独立工作树复现，不伪造哈希或关闭同步保护。
制造、轴承、限位与碰撞逐项记录见 `cad/local_fix/UPPER_REFIT_ISSUES.md`，
不要以过往扫描替代修改后的新验证。

## 训练/奖励/控制关键位置

- `playground/open_duck_mini_zp10d/joystick.py`：PPO 环境、抬脚/完成落脚奖励、
  重心/支撑、身体摆动/漂移代价、归一化观测和阶段课程。
- `actuators.py`、`liftoff.py`、`policy.py`：辨识电机参数、延迟反馈、
  Placo 50 Hz 前馈、抬脚事件与 checkpoint 观测迁移。
- `contacts.py`：严格绑定 MuJoCo 3.12.0 的脚底接触点适配。
- `playground/common/mjx_compat.py`：WSL/Pascal 质量矩阵适配和已采用的线搜索优化。
- `runner.py`、`evaluate.py`、`supervise.py`、`training_window.py`：训练、
  完整原生评估、监督、原生窗口；`--live-training` 才是实际训练样本。
- `reference.py`、`training_model.py`、`data/`、URDF/XML/assets：规划和几何输入。
- `scripts/collect_zp10d.py`、`SERVO_IDENTIFICATION.md`：实体串口协议/采集。
  不得为迁移验证启动硬件舵机动作。

关键运行输入 `training/reference_repair/fitted_audit.json`、
`training/calibrated_actuator/assessment.json` 和 BAM 初步拟合已明确加入 Git。
全部实验/权重另存 `training-runs.tar.gz`，勿误判为已删除的无用文件。

## 实际训练状态，避免旧 README 顶部信息误导

当前没有训练进程。本轮仅归档和回归检查，没有新增训练步数。
已有抬脚不等于稳定踏步或行走。

- 历史选中 `sampled_control/drift_feedback/.../000000262144`（guidance10）：
  累计消耗 21,387,264，20 s 确定性完成 19/18 次，但有漂移和随机失稳。
- 较新完整 `phase_swing_roll_limit_long/.../000000262144`（guidance13）：
  累计 21,780,480，保存的 8 s walk 6/6 次、峰值 12.25/11.86 mm。
- 最新 `phase_swing_roll_limit_live` 到 22,042,624，保存 checkpoint 后失败，
  错误 `TypeError: 'bool' object is not subscriptable`；原生 walk 跌倒。
  该失败试验及全部证据均保留，不宣称它是最佳策略。

若继续训练，先复现已保存模型/权重，再排查最新失败及身体扭摆。
用户要求通过正确的转重和摆动腿抬升避免左右髋部一起乱扭；抬脚 10 mm 即达目标，
不追求越高越好，超过门槛持续时间短可以接受。
保持原地不动/仅卸载不能持续刷抬脚奖励的约束。
用户允许暂不把舵机峰值扭矩作为抬脚研究阻碍；倍率2仅是仿真敏感性条件。
继续训练计数与所恢复 actor 年龄分开，Brax warm start 不恢复优化器状态。

## 对话恢复

项目长会话 ID：`01a11c88-3df1-7071-8d1b-c56452520d17`。
原始会话包为私有备份，不加入 Git，不含 auth.json。
恢复 CLI 会话失败时，新建会话先读本文件、MIGRATION.md 和问题清单即可继续。
