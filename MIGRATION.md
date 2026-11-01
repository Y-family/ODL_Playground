# 项目迁移（2026-11-01）

迁移分支：`migration/zp10d-training-20261101`。项目代码、奖励函数、控制器、
舵机辨识代码、Placo 参考、URDF/MJCF、仿真网格和必要标定输入均在 Git 中。
训练历史、权重和原始 Codex 对话单独保存，恢复时需要一起复制迁移目录。

## 迁移包

本次导出目录：`/root/work/Open_Duck_Playground-migration-20261101/`。

- `Open_Duck_Playground.bundle`：迁移分支及其完整祖先历史，可直接克隆。
- `training-runs.tar.gz`：完整 `training/`、`training_gpu/`，包含被 Git 忽略的
  checkpoints、metadata、日志、评估轨迹、输入快照、原始舵机采集和辨识结果。
- `codex-context.tar.gz`：本项目原始会话、筛选后的索引/用户历史、全局规则、
  本项目关联的技能及可读交接文件。不含登录凭据、全局配置中的密钥和其他项目会话。
- `MANIFEST.json`、`SHA256SUMS`：文件数量、大小、来源及归档校验值。

`.venv`、CUDA 驱动、Windows SolidWorks/MCP、CAD 缓存、可重建打印排盘不在包内。
本次没有推送到远程仓库；bundle 可以离线传输。原始会话包含完整工具输出，
应按私有项目资料保管。

## CAD 与训练模型的边界

**最新 CAD 的动力学已同步；历史步态参考和权重仍绑定旧模型。不要绕过哈希校验。**

- 当前 CAD、MuJoCo、Placo URDF 和训练模型生成器一致，STEP SHA256：
  `91e610bc6e8640f1fc8698a5cfdc121c2be64f48ce824873e99b85a4ce08533e`。
  包括每侧内收 1.36 mm 和后部缺口修复。113 个物理零件，估算质量 1.976184 kg。
  独立同步检查、33 姿态 URDF 对照、辨识舵机加既有反馈的 4 s 站姿检查和训练模型结构对照通过。
  辨识参数下固定目标、无反馈会失稳，新旧版本都存在；不把固定目标试验记作通过。
- 历史参考/权重使用 `ae7b161` 的 STEP，SHA256：
  `528d82d2a2bda0a7757f28600dc02b289c73220f0cf98c8b0d39ebacd92bf65b`。
- `reference.source_hashes()` 已可校验最新模型；环境仍会拒绝旧参考与新模型混用。
  本轮没有重标旧轨迹哈希，也没有启动 PPO。

为保留最新 CAD，同时重放已有训练，在独立工作树使用其匹配的 CAD 输入：

```bash
# 在放置迁移包的目录执行；先校验传输是否完整。
sha256sum -c SHA256SUMS
git clone -b migration/zp10d-training-20261101 Open_Duck_Playground.bundle Open_Duck_Playground
cd Open_Duck_Playground
git worktree add --detach ../Open_Duck_Training 1ac9284
git -C ../Open_Duck_Training restore --source=ae7b161 -- cad/local_fix/open_duck_mini_upper_refit.step
cd ../Open_Duck_Training
# 替换为实际放置迁移包的绝对路径。
tar -xzf /path/to/migration/training-runs.tar.gz
```

该工作树的 STEP 会显示为已修改；它明确用于复现旧训练；`1ac9284` 保留旧动力学/参考。正式迁移分支保留最新 CAD 和已同步动力学。
要用最新 CAD 重新训练，下一步是重生成 Placo 参考/拟合力矩审计，再验证并评估 checkpoint 兼容性。

## 重建运行环境

已用环境：Linux/WSL2 x86_64、Python 3.12，JAX/JAXlib/CUDA 插件 0.8.2、
MuJoCo/MJX 3.12.0、Brax 0.14.2、Flax 0.12.2、Placo 0.9.16。
`requirements-migration-linux-py312.txt` 记录当前完整依赖版本，作为约束使用。
原环境 `uv pip check` 通过；不是目标机器安装成功的保证。

```bash
# 在上面的 Open_Duck_Training 工作树内执行；先安装 uv 和 Python 3.12。
uv venv --python 3.12
uv pip install --python .venv/bin/python -c requirements-migration-linux-py312.txt \
  'playground==0.2.0' 'jax[cuda12]==0.8.2' 'placo==0.9.16' \
  'build123d==0.11.1' osqp coacd pygame framesviewer onnxruntime
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
```

CPU 环境将 `jax[cuda12]` 换为 `jax`，执行 JAX 命令时设置 `JAX_PLATFORMS=cpu`。
不要直接复制 `.venv`，也不要无约束升级 MuJoCo：`contacts.py` 的原生脚底接触适配
明确绑定 3.12.0。GPU 驱动在目标系统安装；原机 MX350 2 GB、驱动 582.78。
原生窗口还需可用的 OpenGL/GLFW、Linux 桌面或 WSLg；无显示环境可以先做数值验证。

所有 JAX 命令（包括 CPU）使用 `scripts/cad/run_memory_limited.py --gpu`，
依赖 systemd 用户会话/cgroup v2，限任务及子进程实际 RAM 7 GB、swap 0；
CAD 模式地址空间上限 4 GB。全工作区 10 GB，计算串行。迁移后先确认
`systemd-run --user --scope` 可用；不能用普通 Python 绕过限额。

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --gpu --check
.venv/bin/python scripts/cad/run_memory_limited.py playground/open_duck_mini_zp10d/training_model.py
.venv/bin/python scripts/cad/run_memory_limited.py --gpu playground/open_duck_mini_zp10d/test_policy.py
.venv/bin/python scripts/cad/run_memory_limited.py --gpu playground/open_duck_mini_zp10d/test_lift_curriculum.py --sampled-only
```

训练场景会重新生成 `/tmp/zp10d-training/scene.xml`，含原始脚底网格、其他部位盒体和
仅地面接触；完整 CAD 原生模型仍是最终评估依据。无需复制旧 `/tmp` 或 JAX 缓存。
MJX/Brax/Playground 安装目录没有额外 Python 补丁；适配代码已在仓库中。
如需重做 BAM 辨识，另安装 Rhoban/bam（`identify_servo.py --bam-source`）；
已有标定和训练不要求重新连接实体舵机。厂商 PDF/Windows 串口环境需另行迁移。

## 训练进度与恢复

已保存迁移输入并同步最新 CAD 动力学；未启动新 PPO 训练，未修改 CAD 或奖励行为。
下列评估/续训命令用于上文 `1ac9284` 的旧模型工作树，不能直接用于最新模型。
训练尚未完成持续稳定踏步/行走验收。

- README 曾选中的 `sampled_control/drift_feedback/checkpoints/000000262144`
  是 guidance v10；其累计消耗步数 21,387,264，已有抬脚但长期漂移/随机稳定性仍有问题。
- 较新完整试验 `phase_swing_roll_limit_long/checkpoints/000000262144`
  是 guidance v13，累计 21,780,480；保存的原生 8 s walk 为 6/6 次完整抬脚，
  最高整脚净空约 12.25/11.86 mm，不能据此宣称行走验收。
- 最新 `phase_swing_roll_limit_live` 累计 22,042,624，保留 262,144 checkpoint，
  状态为失败：`TypeError: 'bool' object is not subscriptable`；保存的 walk 有跌倒。
  不把它自动提升为最佳策略。

所有路径前缀均为
`playground/open_duck_mini_zp10d/training_gpu/placo_lift_12mm/course/`。
`metadata.json` 是各试验配置/来源依据；`inputs/` 保留当时的代码与参考。

先用原生完整模型评估较新完整试验，再决定继续训练：

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --gpu \
  playground/open_duck_mini_zp10d/evaluate.py \
  playground/open_duck_mini_zp10d/training_gpu/placo_lift_12mm/course/sampled_control/phase_swing_roll_limit_long/checkpoints/000000262144 \
  --seconds 8 --output /tmp/zp10d-migration-eval
```

当前代码 guidance v13。若选择该完整试验继续研究，可用下面的显式 warm start，
创建新的输出目录，保留评估：

```bash
JAX_PLATFORMS=cuda .venv/bin/python scripts/cad/run_memory_limited.py --gpu \
  playground/open_duck_mini_zp10d/runner.py \
  --envs 512 --steps 131072 --checkpoint-every 131072 \
  --restore playground/open_duck_mini_zp10d/training_gpu/placo_lift_12mm/course/sampled_control/phase_swing_roll_limit_long/checkpoints/000000262144 \
  --prior-steps 22042624 --reward-mode optimized --lift-curriculum \
  --coordinated-lift --reference-action --reference-feedforward --sampled-reference-control \
  --drift-observation --position-feedback 1.1275290085055225 \
  --position-feedback-damping 0.07053243703737014 --torque-limit-scale 2 \
  --experimental-reference --output playground/open_duck_mini_zp10d/training_gpu/migration_resume
```

这是继续研究的命令，**本轮未执行**；最新失败原因仍需排查。
`--prior-steps` 是累计消耗，不能当成所恢复 actor 的训练年龄。
Brax 加载策略、价值网络和归一化数据，但不恢复优化器状态或运行计数。
奖励/控制/接触签名变化时，runner 会重置价值网络。当前接触签名包含训练 XML
中的绝对网格路径，因此换目录也会触发该重置；策略和归一化参数仍保留，
不能把它当作逐位连续的训练恢复。
扭矩倍率 2 是既有仿真敏感性条件，不是实物额定能力。
实时训练窗口需另加 `--live-training`，它与已保存轨迹回放不同。

## Codex 上下文

可靠的跨机器交接入口是 [CODEX_HANDOFF.md](CODEX_HANDOFF.md)、`AGENTS.md`、
`cad/local_fix/UPPER_REFIT_ISSUES.md` 和本文件。新会话先让 Codex 读取这些文件。

原始记录归档包含本项目 45 个会话文件；原机 Codex CLI 版本 0.164.0。
先在新环境安装兼容版本并独立登录，再把归档中的 `codex/` 合并到
`${CODEX_HOME:-$HOME/.codex}/`。已有同名历史/索引文件先备份并合并，避免覆盖新机历史。
`codex/skills/` 与 `agents/skills/` 分别对应原机 `~/.codex/skills` 与
`~/.agents/skills` 的项目相关技能；按包内 `CONTEXT_MANIFEST.json` 还原位置。
全局权限/MCP 配置需按目标机器重配。

```bash
cd /path/to/Open_Duck_Playground
codex resume --all
# 或明确恢复当前项目长会话，并指定新路径：
codex resume 01a11c88-3df1-7071-8d1b-c56452520d17 -C /path/to/Open_Duck_Playground
```

官方文档确认 `codex resume`、`--all`、`--cd` 和 `CODEX_HOME` 的用途；
手动导入跨版本会话并不保证索引/UI 自动恢复。本包保留原始 JSONL 和可读交接，
即使恢复入口不兼容，也可新建会话读取交接继续工作。
当前长会话有两个同 ID 的物理记录文件，均原样保存；导出截点见归档清单。
本轮导出之后的消息可在退出会话后再补拷最新 JSONL。

OpenAI Docs：
- https://developers.openai.com/codex/developer-commands?surface=cli#codex-resume
- https://developers.openai.com/codex/config-file/environment-variables
