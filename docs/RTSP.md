# RTSP 分阶段接入

本仓库公开 `src/mage_ascend/rtsp_ingest.py` 为原9db0版本reader，尚未把它接成公开CLI/模型服务。其源注册表由operator提供并设0600，只允许source_id、url、allowed_host、allowed_port匹配；别提交含账户密码的源地址。

建议顺序：先在Notebook内部以MediaMTX发布有权使用的真实录像，按媒体时钟remux；CPU接收并比对源像素/PTS；再执行真实BF16视觉+FP32 gate；最后验证自然触发后的语言生成。测试结束停止归属明确的发布器并释放端口，保留模型服务。

原现场工具为MediaMTX 1.21.3 arm64和PyAV 16.1.0，独立目录安装并验证来源SHA，未替换平台依赖；这些二进制和录像不在包内。用户电脑的回环RTSP不是Notebook的回环RTSP，实际摄像头需另验路由、权限、码流与GOP。

reader按每0.5秒真实媒体PTS目标取首个到达帧；不同于文件linspace。8秒窗口×最多4，队列2，最大64采样帧；开流5秒、读取10秒；墙钟57秒是在blocking read之前检查，末次读取仍可能再消耗10秒，不是绝对57秒deadline。溢出、缺样、时序异常或超时明确失败，无静默丢帧、自动重连或编造时间。

H264 B帧流上POS_MSEC可能倒退，改用CAP_PROP_PTS/FPS（OpenCV 4.11 FFmpeg backend定义），减首个真实PTS作为相对时间。必须检查有限值、严格递增和采样完整性；后端不支持PTS时明确拒绝，不用帧序号/到达时钟fallback。[OpenCV属性文档](https://docs.opencv.org/4.11.0/d4/d15/group__videoio__flags__base.html)。

历史两个录像测试源完成64样本像素对照；最终源的服务器34.337秒处理32秒媒体，窗口结束到gate0.928～1.291秒。所有窗口末分数低于0.5，因此未实际调用RTSP语言生成；源缺失负例明确400/input_failed、0帧、无重试，服务保持ready。

段内6/14/22/30秒有较高分数，但当前只看窗口末帧。不能声称已完成逐帧主动触发，不能把峰值直接当事件真值或随意取max触发。后续需检查上游因果语义、视觉时序分组、策略及去抖，并独立验收。无限流、SSE、长时增量状态、生产摄像头均未验收；网络丢包不可由应用队列无溢出推断为0。
