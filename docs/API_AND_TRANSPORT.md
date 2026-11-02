# 历史 API 与 Notebook 接入契约

本公开版提供CLI和源码组件，不附带通用HTTP守护进程/进程管理器，也不直接启动BML服务。下面是历史部署中实际验证的协议，可用于独立实现服务适配；不是克隆仓库后自动存在的端点。

历史单图：远端回环8018，本机18018；POST /infer，JSON image_base64/raw Base64、prompt、max_new_tokens；≤32MiB、≤4M像素、≤512生成token。历史视频：远端8019，本机18019；JSON video_base64、prompt、mode(video/stream/stream_fast)、num_frames≤16、max_pixels≤150000、max_new_tokens≤128。普通video历史上限60秒；stream/stream_fast≤32秒/64帧。公开CLI与服务的限制不完全相同，必须分别说明。

Bearer认证；固定回环Host；拒绝Origin；GET /health查询模型就绪、GET /status查询桥状态。单运行槽，忙时明确429。X-Request-ID为唯一32hex，发送前持久化输入SHA与参数，远端原子拒绝重复ID。结果未知先查审计，零自动推理重发。

输出普通text/raw_text、真实token及stop_reason；stream.events包含起止媒体时间、门控概率、decision及generation。silent、生成失败、未接单、运行中、unknown分开。不是OpenAI Chat Completions协议，不能只改baseurl替换JoyAI的multipart/turns协议。

RTSP独立POST /rtsp只在历史远端8019，未暴露在18019桥。只接私有白名单source_id，不接受请求提供任意URL/服务器路径。网页应走自己的后端；不要向浏览器下发模型Bearer key。

正常SSH隧道或平台已授权HTTPS/WebSocket均可承载回环API；只使用平台允许的方式，不假设Notebook存在公网SSH。不要关闭TLS校验。桥、worker、模型进程、授权是不同生命周期：连接重建不等于权重重载；历史PID不可用于恢复时杀进程。
