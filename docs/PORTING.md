# 移植点与转化依据

| 层 | 原约束/观察 | 具体转化 | 为什么这样做 / 验证 |
|---|---|---|---|
| 平台 | 上游示例面向CUDA；平台Torch带+cpu标签 | 显式import torch_npu，逻辑npu:N、线程内set_device、同步计时 | 实际NPU算子、参数/buffer布局；未全局替换.cuda() |
| 类加载 | 自定义from_pretrained override假定父类返回model；诊断返回tuple | 8个源码SHA绑定的父类诊断入口，再执行原override唯一后置路径赋值 | strict诊断、真实视觉与输出；override故障来自源码分析，未单独实机复现该故障 |
| Attention | eager在高分辨率图片上消耗较多HBM | 原生SDPA；BF16主干，cache=True | 原图A/B和峰值显存；未手写attention内核，未删mask |
| 输入 | 像素浮点与token/grid/patch位置整数混合 | 仅pixel_values转换BF16；其余保持整数 | 完整输入布局及token计数审计 |
| 视频预算 | combined processor的max_pixels参数并不约束所有frames子路径 | 同时配置共享image_processor与video_processor实例 | 双路径输入exact、实际有效像素与采样索引 |
| Mamba依赖 | mamba_ssm导入牵连CUDA扩展 | 固定Mamba v2.2.0源码的AST选择；本地factory；Torch原生conv慢路径与bounded scan | 不改磁盘checkpoint源；不向sys.modules塞伪mamba模块；仅固定Mamba1维度 |
| Gate数值 | BF16完整gate随前缀batch形状出现舍入漂移 | gate单独FP32；主视觉/语言BF16 | 独立scan/full-gate oracle与前缀决策；BF16失败保留 |
| 媒体准备 | FFV1中间编码、回读、重复处理占大多数耗时 | stream_fast当前段内存采样、单次prepare、响应复用 | 完整BGR/模型输入/特征字节/logits/token等价，而非降帧提速 |
| RTSP时钟 | H264带B帧时POS_MSEC可0,-20,0,20 | 真实CAP_PROP_PTS/FPS，减首PTS，严格递增 | 64样本像素与源PTS；不按到达时钟/帧号编造时间 |

Mamba selective scan采用FP32递推状态与原输出dtype规则，不提供训练/backward、复数/分组通用变体、CUDA融合或增量状态。AST执行仍是执行第三方代码，必须先检查SHA与许可，不能当作安全沙箱。

历史管理器另有空集合诊断`set()`的精确兼容；它是固定旧harness的报告格式适配，不能通用地吞掉加载错误。公开CLI沿用Runtime的结构化诊断，不分发此平台专属管理器补丁。

GPTQModel参与了量化可行性评估，但已验收服务使用原始非量化BF16权重；本仓库无需GPTQModel，不宣称INT4部署、融合量化内核或量化提速。神经codec分支、音频、训练、多卡也未由本案例验收。
