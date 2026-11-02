# 来源与贡献边界

- Microsoft Mage: https://github.com/microsoft/Mage/tree/76bec2bb3818863f470de7e867c2dc7f1d0bfd83/mage_vl 。此commit固定推理脚本来源，不代表权重revision。上游当前main可能继续变化。
- 权重官方入口：https://huggingface.co/microsoft/Mage-VL 。自行核对许可、revision和文件SHA。本包79项manifest来自用户已取得的历史制品；未分发模型自定义代码或权重。
- Mamba v2.2.0：https://github.com/state-spaces/mamba/tree/v2.2.0 。随包4份源码分别来自 mamba_ssm/modules/mamba_simple.py、modules/block.py、models/mixer_seq_simple.py、ops/selective_scan_interface.py；精确SHA见gate_adapter.UPSTREAM_SHA及evidence/code-provenance.json，保留Apache-2.0。
- Ascend PyTorch：https://github.com/Ascend/pytorch 。镜像匹配由平台维护方决定；本案例版本是历史实测组合。
- OpenCV 4.11 PTS时间基：https://docs.opencv.org/4.11.0/d4/d15/group__videoio__flags__base.html 。

本仓库原创贡献是部署控制、输入契约、固定结构Torch适配、独立数值检查、媒体准备和RTSP时钟工程；上游模型、Mamba结构与官方参考算法不归本仓库。没有系统性检索足以证明“首次移植”，故不作优先权/论文创新性断言。可作为工程复现、迁移案例与后续研究起点。

公开版继承的数学函数未改；model_runtime/video_inputs/stream_fast只把实验根路径改为可配置值，video_inputs移除绑定现场的旧main。新CLI/资格包装属于新代码，尚需目标NPU完整回归。来源表记录原SHA与公开SHA；私人主机/账户/进程身份不进入公开包。
