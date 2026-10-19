# 当前 ZP10D 打印件

输入为已验收装配 `cad/local_fix/open_duck_mini_upper_refit.step`，按原项目官方
打印指南及 BOM 分类；来源快照在 `sources/`，逐项依据见 `CLASSIFICATION_REVIEW.md`。
不再使用动力学脚本中的人工 `PRINTED` 名单。未知项单独待确认，不自动归为采购件。
不修改 CAD 几何、不缩放；
移除 STL 精度转换后的零面积/重复三角形，细分共线接缝，并仅按源 CAD 中确实存在的
平面恢复漏掉的三角网格，不填真实开口。保留封闭内腔的反向边界面。
每个装配实例独立导出，文件前缀对应动力学清单的 index，每份打印一次；同名件也不擅自合并。

```bash
.venv/bin/python scripts/cad/run_memory_limited.py --wait \
  scripts/cad/export_zp10d_print_parts.py --step
```

输出：`STL/` 为通过几何/网格检查的毫米制 STL；`STEP/` 为相同打印坐标下的独立
精确 CAD 实体（加 `--step` 时导出）；`quarantine/` 保留未通过的候选，
不放入正式打印包。`manifest.json` 包含源文件哈希、装配路径、尺寸、坐标变换、
材料/填充率、分类来源及检查结果。待确认件的旧导出保留在 `unclassified/`，
不放入确认打印包。生成文件和预览不提交 Git；官方来源快照和分类审查文档保留。
沿用项目现有 CAD 环境；漏面恢复另需 Shapely ≥2.1 的约束 Delaunay 三角剖分。
修复回归可经同一限额启动器运行 `scripts/cad/test_print_mesh.py`。

按用户设置：PETG，外壳 10% 填充，其余 30%；外壳类别列在导出脚本 `SHELLS` 中。
层高、壁层、顶底层、支撑、填充图案尚未指定。初始摆放仅让最大平面朝下、居中并落在 Z=0，
仍须在切片器核对支撑和受力方向；不用装配姿态直接打印整机。
`foot_bottom_tpu` 是沿用的零件名：若按当前 PETG 设置制作，其抓地和柔顺性需实测。
打印重量应实际称量，不能将填充率直接换成动力学密度。

舵机测试优先使用腿部连接片、spacer、roll_motor、roll_to_pitch 和足部件，
按清单装配路径区分左右及上下段；不需要先打印头壳和装饰件。
网格通过不代表壁厚、支撑、螺纹/螺丝咬合、承载或实物装配已经验收。
实物采集流程见 `playground/open_duck_mini_zp10d/SERVO_IDENTIFICATION.md`。

2026-10-19 分类纠正：原 50 件名单漏掉左右眼和两个 `bulb` 环件（官方有打印 STL），
并在缺少依据时选入两根 `antenna`。当前有官方打印依据的 52 个实例导出 STL 和 STEP；
61 个装配实例按成品件排除，2 根天线待确认，整机共 115 个实例。
52/52 件通过有效实体及实际闭合 STL 检查，尺寸均在 180 mm 包络内。
原指南 `bulb` 写 1 件，但当前装配有 2 个相同环件，当前用量按实际装配记录，
不是将所有同名零件盲目套用原版数量。61 个成品件实例也不是完整实物采购 BOM。
`left_cache`、`right_cache` 各细分 2 个接缝顶点；`head` 按原 CAD 的一个平面
恢复网格并细分 12 个错缝顶点，接缝偏差最大 0.003931 mm。三件实际 STL、独立
STEP 重载、孔径及每件 400 点 CAD 表面距离复核通过；采样最大偏差 0.030828 mm，
小于 0.08 mm 网格弦差设置，但不是全表面误差证明。正式装配 STEP 哈希保持。

`open_duck_mini_zp10d_print_parts.zip` 含 52 STL + 52 STEP；
`leg_servo_test_parts.zip` 为其中 26 件腿部连接/足部件的测试子集。
`repaired_parts.glb` 和 `STEP/repaired_parts.step` 将修复三件分开放置以便浏览，
后者不计入打印用量。`printable_parts.glb` 为有来源依据打印件的分开展示。
整机装配预览仍使用 `cad/local_fix/open_duck_mini_upper_refit.step`。

DfAM 按每件最多 2000 点测量、比较 6 个轴向摆放；11 件的 p05 低于通用有支撑壁
1.2 mm 建议（右腿盖、头壳、颈左片、左右足侧/足顶、反光接口、左眼、两个bulb环件），
切片前需核对，未自动改薄壁。48件复用实际STL哈希一致的测量，4件新增测量。
结果不等于结构承载或实物装配验收，完整测量及支撑建议见本地 `PRINT_REPORT.md`。
