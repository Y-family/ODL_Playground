# ZP10D 线搜索优化与对照（2026-10-18）

已采用限制摩擦临时数组范围的优化，ZP10D 后续 GPU 训练自动启用。512 环境真实 PPO 对照的预热后吞吐提高 **10.60%**，线搜索归属内核耗时下降 **18.09%**；物理参数和 FP32 精度保持。

本页保留第一轮历史对照。第二轮已采用切片更新替代系数拼接，独立 512 环境对照为 929.60→989.48 环境步/s（+6.44%）；补充独立 FP64/自动微分/边界及 2 s 物理验证，见 [`reduction/RESULTS.md`](reduction/RESULTS.md)。

## 实际 PPO 对照

MX350、同一新策略初始种子、同一模型/参考/奖励、512 环境，unroll 16、batch 512、每批 2 次更新。每版实际运行 49,152 环境步、6 轮 PPO，保存 6 个检查点。第 1、2 轮含编译，第 4 轮采集 JAX/CUPTI，稳态采用未采集的第 3、5、6 轮中位数。正式训练仍保留评估，本次吞吐诊断使用 `--no-eval`。

| 指标 | 原版 | 优化版 |
|---|---:|---:|
| 预热后 PPO 吞吐 | 843.75 环境步/s | **933.16 环境步/s** |
| 三次未采集稳态样本 | 843.75 / 837.95 / 852.18 | 932.25 / 933.16 / 936.78 |
| 8,192 环境步对应稳态计算时间 | 9.709 s | 8.779 s |
| 第四轮线搜索内核累计时间 | 4.976 s | **4.076 s** |
| 线搜索占全部内核时间 | 53.57% | 48.04% |
| 第四轮全部内核累计时间 | 9.289 s | 8.485 s |
| 第四轮内核调用数 | 191,114 | 169,555 |
| 160 个物理批次的求解器迭代总数 | 1,138 | 1,137 |
| 每批求解器平均迭代 | 7.1125 | 7.1063 |
| 策略参数最大变化 | 0.00329882 | 0.00328313 |
| 完整诊断耗时，含编译/采集/保存 | 213.83 s | 277.20 s |
| 峰值进程 RSS | 3.026 GB | 2.969 GB |

持续训练相同步数的计算时间约少 **9.58%**。首次编译更慢，因此这次只有 6 轮的完整诊断总时间增加约 63 s；不能把稳态提升解释为短任务启动也更快。

按实际求解器迭代数归一化，线搜索归属内核每轮从约 4.373 ms 降至 3.585 ms，下降约 18.02%；求解器总迭代几乎相同，收益不是降低求解上限所得。原先约 1.052 s 的 `loop_compare_multiply_fusion` 不再独立出现，部分工作合入其它融合核。全部内核累计时间下降约 8.65%。跨模块融合时间从 0.348 s 增为 0.455 s，分类仅是编译标签归属，不能当作算术操作的精确拆分；端到端结论以未采集 PPO 吞吐为准。

两版 13 个记录输入的 SHA256 一致，各版运行前后也一致。元数据确认原版方法为 `create`、优化版为 `_pyramidal_point`，两版 cgroup 均无上限触发或 OOM。源代码与参考在测试中未修改，结果不代表长期收敛或行走通过。

## 实现

`playground/common/mjx_compat.py` 的 `_pyramidal_point` 保留 MJX 原线搜索代价、一阶导数和二阶导数公式，仅将摩擦损耗分段计算限制在真正的摩擦约束行。本模型有 304 条约束，其中 16 条为摩擦损耗约束；其余行的原始二次系数直接保留，避免生成全宽摩擦临时数组。最终活动掩码和归约公式保持原样。非金字塔摩擦锥调用原实现。

ZP10D 的 GPU 环境构造时安装适配；原 Pascal 静态质量矩阵掩码修复保留。CPU 入口使用原实现。训练 `metadata.json` 记录实际线搜索方法名与适配文件 SHA256，便于辨认新旧训练。

保持 FP32、20 ms 控制周期、2 ms 物理步、每控制步 10 个子步、求解器上限 50、线搜索 10 次扫描、原容差和求导路径。模型、惯量、执行器增益/限位、奖励及 PPO 设置均未因本次加速调整。

曾测试以有界 `while_loop` 提前结束线搜索；本机同状态物理测试反而慢约 19%，该候选已移除。失败记录保留在 `validation.json` 的 `rejected_early_exit`，当前采用的仍是原固定扫描。

## 数值与物理检查

`scripts/check_mjx_linesearch.py` 在 MX350 GPU 上通过以下检查，证据为 `validation.json` 和 `validation.log`：

- 512 组随机线搜索输入，在启用/禁用摩擦损耗时，代价、一阶和二阶导数与原实现逐值完全相等。
- 站立、3/30/60 mm 落下及向下 0.3 m/s 冲击混合批次，512 环境运行 24 个控制步，即 0.48 s、每环境 240 个物理步。各分支共 122,880 个物理子步，使用相同确定性随机动作。
- 另抽样 48 个标量环境控制步，在实际求解过程中逐次核对线搜索代价和两阶导数，容差 `rtol=2e-6, atol=2e-6`。这是抽样检查，未在全部 512 环境中插入断言；JAX 0.8.2 的 checkify 无法转换该批量 MJX while。
- 数值有限，终止状态一致；对位姿、速度、接触穿透及总载荷冲量做短轨迹比较。

整段 GPU 编译可能重新排列 FP32 运算。严格逐数组诊断有差异，不能宣称完整轨迹、接触力、观测或奖励逐位相等。额外用原求解器在每个控制步前给 qpos/qvel 加正负一个 FP32 可表示步长，测量接触问题对舍入的敏感性。轨迹通过条件为显式物理界限和该参考漂移两倍中的较大者，终止状态要求完全相同；这不是高精度真值验证。

| 最大独立轨迹差异 | 优化版对原版 | 原版单 ULP 扰动参考 |
|---|---:|---:|
| 机身位置 | 0.0983 mm | 0.1117 mm |
| 关节位置 | 0.0006503 rad | 0.0006505 rad |
| 机身线速度 | 0.00324 m/s | 0.00468 m/s |
| 广义速度 | 0.04254 | 0.06093 |
| 接触穿透 | 0.0313 mm | 0.1380 mm |
| 总脚载荷冲量 | 0.10515 N·s | 0.10515 N·s |
| 终止状态 | 相同 | 相同 |

瞬时脚载荷差异在优化版和单 ULP 参考中均可达到 10.44 N；短测不能证明长期训练收敛、完整模型行走或硬件部署质量。严格逐数组诊断和实际接受界限完整保存在 JSON 中。验证进程峰值 RSS 1.580 GB。

## 复现与证据

从仓库根目录运行，GPU 任务串行并经过原有内存限制入口。验证命令：

```bash
JAX_PLATFORMS=cuda .venv/bin/python scripts/cad/run_memory_limited.py --gpu --wait \
  scripts/check_mjx_linesearch.py
```

对照分析分别使用两个尚无检查点的输出目录；原版使用 `--original-linesearch`，优化版默认启用适配：

```bash
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda .venv/bin/python scripts/cad/run_memory_limited.py --gpu --wait \
  scripts/profile_zp10d.py --original-linesearch --output /tmp/zp10d-linesearch-original
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda .venv/bin/python scripts/cad/run_memory_limited.py --gpu --wait \
  scripts/profile_zp10d.py --output /tmp/zp10d-linesearch-optimized
.venv/bin/python scripts/analyze_zp10d_profile.py /tmp/zp10d-linesearch-original
.venv/bin/python scripts/analyze_zp10d_profile.py /tmp/zp10d-linesearch-optimized
```

本轮实际使用一次性 `benchmark.py` 在同一串行锁和资源组内顺序运行两版，以检查输入哈希一致；该包装脚本现仅保留在本地和历史提交 `dea711a3`。复现第一轮应使用该提交，当前生产实现已包含第二轮切片优化。上述独立复现命令未额外重复执行；跨两个命令须保持模型、参考和源代码不变。

- `benchmark.json`：两版 PPO 计时、输入哈希与吞吐比值。
- `baseline/analysis.json`、`optimized/analysis.json`：两版独立 GPU 内核分析。各版 `profile.json` 的计时/配置已包含在 `benchmark.json`，不再重复跟踪；逐次元数据、日志、检查点和原始轨迹仅保留本地。
- `validation.json`、`validation.log`：数值、轨迹和同状态物理性能检查，包括失败候选及严格比较诊断。
- `training_scene.xml`：与本轮输入哈希一致的训练接触近似副本。
- `environment.json`：本轮 Python 与 JAX/MuJoCo/Brax/Flax 包版本。

训练接触近似包含 17 个身体包围盒和地面，省略自碰撞。当前测试结果只适用于该 ZP10D 训练场景和本机 MX350；完整 3,296-geom 模型及更成熟策略的接触负载仍需另行验证。
