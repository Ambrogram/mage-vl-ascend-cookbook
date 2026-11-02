# 避坑：按观察定位

| 症状 | 先查什么 | 下一步 | 避免 |
|---|---|---|---|
| torch版本含+cpu | torch_npu导入、npu可用、真实算子 | 记录实际device和配套 | 重装CUDA Torch |
| 模型加载tuple属性错误 | override与output_loading_info返回值 | 检查固定源码loader及后置赋值 | 关闭所有加载诊断 |
| 图像显存很高 | eager/SDPA、真实像素、cache | 同输入冷暖分测 | 降像素后宣称同任务等价 |
| 视频max_pixels无效 | image与video processor共享实例配置 | 看effective_frame_size和patch计数 | 只信顶层参数 |
| 安装mamba需要nvcc | 实际是不是StreamMind gate依赖 | 固定Torch慢路径与数值资格 | 把NPU伪装CUDA或全局假模块 |
| BF16 gate候选与官方一致仍异常 | 前缀改变时完整gate漂移 | 单独测FP32及决策稳定性 | 只比同dtype一组输出 |
| 30秒视频花205秒 | FFV1、重复decode/processor、JSON审计区间 | 对照媒体叶级计时，重用当前段prepare | 归咎NPU算力或先量化 |
| 首次比暖态慢 | 首个视觉前向、连接重建、模型加载 | 分别测，不归因未经证实的编译 | 混合冷热数据 |
| RTSP首帧超时 | 下一个IDR何时到达、读超时 | 有界延长至覆盖GOP并记录 | 反复重连/伪造首帧 |
| POS_MSEC倒退 | 后端CAP_PROP_PTS/FPS和源presentation时间 | 严格按真实PTS归一化 | 关闭时序检查或用帧号代替 |
| 四窗静默但段内高分 | 当前实现仅看窗口末帧 | 记录策略缺口，独立研究因果逐帧触发 | 直接取window max冒充主动模式 |
| REST平台失败但旧WS能调用 | 控制面会话与已有worker通道不同 | 正常人工授权刷新；检查原请求审计 | 自动抓取浏览器Cookie或全服务重启 |
| HTTP超时 | request ID、已接单/执行/结果审计 | 只读核对unknown | 自动推理重发 |

数值错误先保留完整输入和非量化基线。权重、tokenizer、模型数学没有证据指向时不修改；不通过改成功标志或阈值消灭失败。
