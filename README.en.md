# Mage-VL on Ascend 910B2: deployment cookbook

[中文完整手册](README.md)

An independent engineering cookbook derived from a real BML Notebook deployment of
Microsoft Mage-VL: BF16 image/video inference, a Torch Mamba1 adapter for the FP32
StreamMind gate, exact-input media preparation, and bounded RTSP timestamp handling.
It is not an official Microsoft/Ascend release, a new model, or a claim of priority.

Historical hardware validation took place on 2026-10-27/28. The portable public
entry points were prepared on 2026-11-02 and tested offline; they have NOT received
a fresh end-to-end NPU acceptance run. Keep these two evidence scopes separate.

Start with docs/RUNBOOK.md and docs/CHECKLIST.md. The CLI exposes doctor, verify,
and infer (image/video/stream_fast). Use a platform-matched Torch/torch_npu/CANN
environment; installation never resolves accelerator dependencies automatically.
Full checkpoint verification and review of custom model code precede execution.
Streaming requires independent full-gate and causal-prefix qualification on your
target device. It is finite prerecorded replay with window-end decisions, not
per-frame proactive dispatch, infinite streaming, or a production RTSP service.

No weights, photos/videos, wheels, credentials, or private BML connectors are shipped.
Original integration code/documentation is MIT. Bundled unmodified Mamba v2.2.0
sources are Apache-2.0; see NOTICE and third-party license files.
