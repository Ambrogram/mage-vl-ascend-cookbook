# 分阶段实施手册

所有命令在仓库根目录运行；先 `export PYTHONPATH=src`。每步记录输入、版本、命令、结果和失败出口。没有授权的平台/服务不要操作。已有驻留模型先复用，勿为探针再加载一份。

## G0 制品与资源

输入：用户已取得的完整权重、许可、真实样本、目标逻辑设备、空闲HBM。记录代码revision与权重manifest，两者不是同一身份。上游代码参考 `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`；历史权重没有不可变的HF revision，依内容SHA固定。

执行 README 的 verify。79项参考清单任何一项变化均停止，先审查差异；新制品建立经审查的新清单，但 loader 仍绑定8个自定义源码SHA，不能只生成新manifest绕过源码审查。模型档案内 neural codec 的 `.tar` 实为可能含pickle的Torch ZIP，frames路径不需要反序列化它们。

通过：全部校验通过、架构mage_vl、无quantization_config、足够内存且不会影响旧业务。失败：明确是源码、下载或资源问题，勿反复重下已正确的大分片。

## G1 保留平台软件配套

从可用torch_npu的Python创建 `--system-site-packages` venv。先记录 `npu-smi info`、可见设备变量、包版本、`pip check` 前后差分。独立venv只解决用户态隔离，不隔离设备、驱动、CANN或共享HBM。

参考用户态版本见 examples/requirements-userland.txt；它不是完整锁文件，不能盲目pip install全部再升级Torch。先确认 Pillow、einops、accelerate、tokenizers、Transformers 等可导入。`pip install --no-deps --no-build-isolation -e .` 只安装本cookbook，若 setuptools 或 wheel 缺失先在隔离环境准备构建工具，或直接PYTHONPATH运行。`invalid command bdist_wheel` 属于本机构建工具缺失，不是模型或NPU问题；本次发布检查在独立环境使用setuptools75.8.0/wheel0.45.1完成构建，未升级全局Python或服务器。

通过：doctor 的真实 NPU 小算子完成，实际设备明确。失败：保留错误，先查设备映射/驱动配套；torch标签+cpu不是换装CUDA的理由。逻辑npu:0不一定对应物理卡0。

## G2 单图基线

执行README中两张不同场景、每张两次的 image_baseline。使用BF16、SDPA、greedy、use_cache=True。该脚本来自真实基线，保留完整输入、源码SHA、加载诊断、视觉hooks和生成tokens。

通过：missing/unexpected/mismatched等诊断为空；无meta/off-target参数buffer；图像grid/patch_positions为整数；每次一次视觉前向；非空生成、自然EOS；人工核对两图主体和场景。EOS只证明完整结束，不证明语义准确。单token不能验证KV视觉复用。

失败：加载tuple问题查PORTING；重复视觉forward查cache；截断查stop_reason与prompt。不要重写tokenizer、RoPE或伪造EOS。参数化CLI的 `--mode image` 使用原服务Runtime，适合复现同样输入，但仍须独立语义审查。

## G3 原生视频

执行 `infer --mode video`。输入必须是有确定FPS/帧数的普通文件。公开入口固定16帧、150000像素/帧、128新token、8192输入token预算。使用原生frames处理分支，不是无时间关系的多图问答，也未验收神经codec压缩路径。

助手明确同时配置image_processor.size和video_processor预算。原生processor单独调用与combined调用的像素/坐标必须exact；这两次读取是资格审计开销，普通video路径未采用stream_fast优化。

通过：帧索引递增、真实时间戳与index/FPS一致、patch位置/grid/token计数正确、真实视觉hook、自然EOS，人工检查场景变化。失败：先查预处理预算是否实际生效、解码FPS和字段完整性，再查NPU计算。

## G4 StreamMind 独立门控资格

单图不触发mamba_ssm；不能因此宣布StreamMind可用。不要在NPU镜像里照搬CUDA扩展安装。适配器只用于固定checkpoint的Mamba1结构，不是通用Mamba替代品。

在没有其他驻留重复模型的空闲设备上，先导出真实视觉特征：

```bash
.venv/bin/python scripts/export_gate_features.py --model-dir "$MODEL_DIR" \
  --manifest examples/reference-model-manifest.json --video "$VIDEO" \
  --device npu:0 --out runs/features-001
.venv/bin/python scripts/qualify_gate.py --model-dir "$MODEL_DIR" \
  --features runs/features-001/features.safetensors --device npu:0 \
  --out runs/gate-qualified.json
```

资格fixture是64帧真实视觉特征，导出一次64帧视觉pass可能比16帧服务占用更多HBM；先核对空闲资源，不在共享繁忙设备试探OOM。fixture用于数值资格，不用于声称stream分段采样等价。

资格程序按顺序分别加载独立官方selective_scan_ref和候选bounded_scan路径，比较16/32/48/64前缀的完整gate logits、0.5决策、前缀稳定性，rtol/atol均2e-3。记录adapter、gate权重与features SHA。失败收据不能启用stream；源码/设备变化重新验收，不能编辑passed字段放行。

历史FP32 gate通过，BF16 gate出现最高0.03125前缀漂移而未通过；主视觉和语言仍保持BF16。公开入口只启用FP32 gate，不以“候选和reference同dtype一致”替代前缀测试。新增算子或数学改动必须另做独立oracle及真实硬件验证。

## G5 有界文件回放与优化

通过G4后运行README的stream_fast。输出result.json及每段gate-features/审计，固定8秒分段、最多4段、64帧前缀、阈值0.5。保存两种结果：实际静默、自然触发后的语言生成。没有触发的样本不是失败；也不能靠降低阈值做假正例。

新路径每段顺序解码，保留原选中帧、OpenCV resize/颜色、处理器和时间，准备一次并在响应时复用。没有修改权重、gate、RoPE、tokenizer或减少预算；没有新增权重展开缓存、跨请求记忆或增量Mamba状态。

通过：源帧/完整输入/视觉特征/门控/输出与冻结基线同条件比较；分开记录首次与热调用。没有基线输入或oracle不得仅凭速度验收。公开版重新参数化后需重新做NPU A/B，历史13.40倍服务器加速不是新机器承诺。

## G6 服务与远程连接

公开仓库以CLI为复现入口，不分发通用API管理器或私有BML连接器。历史服务契约与后续封装要求见 API_AND_TRANSPORT.md。先真实CLI，后驻留服务；不能使用CLI循环来冒充常驻服务速度。

如需连接Notebook，使用平台正式允许的SSH/Jupyter HTTPS-WebSocket连接方式，正常登录授权。服务器和用户电脑的127.0.0.1不同。TLS校验、唯一request ID、一次dispatch、unknown审计、进程归属与busy保护必须保持。

## G7 RTSP：单独验收

见RTSP.md。先隔离测试源CPU接收，再像素/PTS对照，再真实视觉/gate，再自然触发后的语言输出。每项分别记录；不能用文件回放正例替代RTSP正例。

## G8 交付与回退

模型制品、环境、补丁、资格、CLI结果、服务实例、连接器分别记录。停止/重启前核对归属和空闲，结果未知不重发、不删除锁或强杀旧数字PID。保留最近成功环境与失败记录。交付时注明每一能力的代码、工程、实机、语义四种状态，记录原业务回归结果。
