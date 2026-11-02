# Mage-VL 在昇腾 910B2 上的迁移实施与验证

[English](README.en.md) · [部署手册](docs/RUNBOOK.md) · [实施清单](docs/CHECKLIST.md) · [移植点](docs/PORTING.md) · [排障](docs/TROUBLESHOOTING.md) · [Agent 指示](prompts/DEPLOYMENT_AGENT.zh-CN.md)

本仓库将一次 Mage-VL 在 BML Notebook / Ascend 910B2 的真实迁移整理为可复用的代码、阶段门禁与验收方法。目标是让下一位用户或 Agent 知道每步为什么做、通过标准是什么、失败后查哪一层。

**工程贡献**：原模型的诊断加载适配、NPU 输入与设备约束、StreamMind Mamba1 有界 Torch 适配、门控独立数值验收、保持输入输出的媒体准备优化、RTSP 时间戳修正，以及可审计的部署流程。未训练新模型、修改权重、开发融合 NPU 内核或证明“全球首次移植”。模型能力及原创架构归 [Microsoft Mage](https://github.com/microsoft/Mage)。

## 两种验证范围

历史部署于 2026-10-27～28 在真实 910B2 上验证了图片、原生视频、StreamMind 文件回放和两种真实 RTSP 传输的录像测试流。**本仓库是 2026-11-02 参数化整理版，发布检查在本机离线完成，尚未重新在 NPU 上完整验收公开入口。** 不应将历史部署结果当作任意版本或本公开版的直接保证。

| 能力 | 历史实机结论 | 本公开版交付 |
|---|---|---|
| 图片理解 | 两种场景、中文/英文、自然 EOS、真实视觉前向、服务重启后回归 | 原验收脚本 + 可参数化 CLI |
| 原生视频 | 保留帧索引、时间戳、patch 坐标和视频输入 | 原生输入助手 + CLI |
| StreamMind 文件 | FP32 gate、64帧前缀、正例和静默、热调用同任务优化 | Torch adapter、stream_fast、独立 gate 资格脚本；需本机验收收据 |
| RTSP | 2种真实TCP录像测试源；32秒/64样本；只测得静默 | 已验证 reader 源码和实施指引；不提供成品 RTSP 模型服务 |
| API/Notebook 桥 | 历史认证回环 API 和本机桥已实际调用 | 协议说明；不分发私有连接器或通用 API 服务管理器 |

## 历史参考组合

Ascend 910B2 / 64GB级卡；aarch64；Python 3.11.15；PyTorch 2.9.0+cpu 与 torch_npu 2.9.0；现场 CANN 8.5.1；Transformers 5.7.0；OpenCV 4.11.0 / NumPy 2.2.6 / safetensors 0.8.0。主模型 696 个 BF16 参数张量、4,741,793,792 元素；独立 gate 64 个参数张量、536,743,424 元素，运行时 FP32。

这是观察到成功的版本组合，不是所有固件/驱动/CANN 镜像的兼容矩阵。`+cpu` 字样不能单独判断是否使用 NPU；须检查 torch_npu、实际参数和算子设备。平台软件配套以 [Ascend 官方资料](https://github.com/Ascend/pytorch)为准。

## 最短路径：图片，再视频

取得有权使用并审查过自定义源码的完整 Mage-VL 权重；本仓库不分发权重或样本。参考 manifest 固定的是历史制品内容，不代表当前上游下载一定相同。

```bash
python3 -m venv --system-site-packages .venv
# 先按 RUNBOOK 核对构建工具及运行依赖；不自动升级平台底层栈。
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
export MODEL_DIR=/data/models/Mage-VL
export IMAGE_A=/data/acceptance/image-a.jpg
export IMAGE_B=/data/acceptance/image-b.jpg
export VIDEO=/data/acceptance/clip.mp4
export PYTHONPATH=src

.venv/bin/python -m mage_ascend.cli doctor --device npu:0 --out runs/environment.json
.venv/bin/python -m mage_ascend.cli verify --model-dir "$MODEL_DIR" \
  --manifest examples/reference-model-manifest.json --out runs/identity.json
# 原始多场景验收 harness；先完成上面的全制品校验。
.venv/bin/python scripts/image_baseline.py --model-dir "$MODEL_DIR" --device npu:0 \
  --dtype bfloat16 --attention sdpa --images "$IMAGE_A" "$IMAGE_B" \
  --repeat 2 --output runs/image-baseline.json
.venv/bin/python -m mage_ascend.cli infer --model-dir "$MODEL_DIR" \
  --manifest examples/reference-model-manifest.json --device npu:0 \
  --mode video --input "$VIDEO" --out runs/video-001
```

每个目录/结果名必须新建，CLI 拒绝覆盖。`PYTHONPATH=src` 可在未安装入口时使用；构建工具缺失应先修复工具，不触动 Torch。图片 CLI 使用 `--mode image`；统一入口最多128新token，输出原始 token、结束原因及审计，截断不自动追加请求。

StreamMind 需要先完成 [门控数值资格](docs/RUNBOOK.md#g4-streammind-独立门控资格)，再执行：

```bash
.venv/bin/python -m mage_ascend.cli infer --model-dir "$MODEL_DIR" \
  --manifest examples/reference-model-manifest.json --device npu:0 \
  --mode stream_fast --input "$VIDEO" --gate-qualification runs/gate-qualified.json \
  --out runs/stream-001
```

该路径固定8秒窗口、每段最多16帧、最多64前缀帧、文件≤32秒、门槛0.5。它按窗口末帧决定是否生成；没有逐帧实时主动触发、持续跨请求记忆、SSE、无限流或硬实时承诺。

## 历史结果

同一30秒视频、同实例、同采样/精度/阈值/生成预算：旧 stream 本机204.591秒、服务器179.602秒；stream_fast 热调用本机38.518秒、服务器13.399秒。1500源帧、完整输入、BF16视觉特征字节、gate logits、决策、41新token及自然EOS均一致；这支持该媒体准备优化，不能外推不同镜像、吞吐或精度。

两种 RTSP 录像测试流均完成32秒媒体和64采样帧；最终PTS修正版服务器34.337秒，窗口关闭到gate 0.928～1.291秒，但四窗均静默。段内高分可能被窗口末策略遗漏；RTSP 触发后的语言生成仍未验收。完整边界、显存口径和失败案例见 [VALIDATION](docs/VALIDATION.md)。

本仓库包含 CLI、历史实现的可追溯参数化版本、独立资格工具、RTSP reader、测试与发布检查；不包含私人 Notebook 登录设施、模型服务运行状态或完整业务工作台。MIT 与第三方 Apache-2.0 边界见 [NOTICE](NOTICE)；发布步骤见 [PUBLISHING](docs/PUBLISHING.md)。
