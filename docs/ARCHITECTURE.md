# 路径与边界

```mermaid
flowchart TD
  A[自有真实样本与已校验权重] --> B[源码SHA与平台NPU小算子]
  B --> C[BF16主模型 + 原生Mage处理器 + SDPA]
  C --> D[图片/普通视频生成与自然EOS检查]
  C --> E[文件当前8秒窗口: 保留真实16帧与时间/patch位置]
  E --> F[原生视觉特征: BF16]
  F --> G[累积至64帧可见前缀]
  G --> H[独立验收过的FP32 StreamMind gate]
  H --> I{窗口末概率至少0.5?}
  I -->|否| J[真实silence及审计]
  I -->|是| K[复用当前段准备输入进行BF16语言生成]
  K --> L[原始token/EOS/时间与语义待审]
  R[RTSP reader: 真实PTS和有界队列] -.公开版尚未接线.-> E
```

源文件顺序解码到当前窗口后作门控；没有提前解码未来窗口。gate仍重算可见前缀，不是增量Mamba状态。当前段请求内输入/视觉特征复用不是量化权重展开缓存。普通video资格路径保留两次预处理用于审计，stream_fast路径消除重复媒体准备。

参数化配置只改变输出根和模型路径；数学函数/class AST与历史文件一致，见 evidence/source-reuse-validation.json。CLI、特征导出和资格包装是公开版新增集成层，需另外做目标NPU端到端验收。NPU模型、API守护进程、远程连接、业务工作台是不同组件，本仓库只交付表明范围内的CLI与源码组件。
