# Agent deployment guidance

Read README.md, docs/RUNBOOK.md, CHECKLIST, PORTING and VALIDATION before acting.
Treat archived evidence as historical, not current runtime state or fresh public-version acceptance.
Do not reinstall platform Torch/torch_npu/CANN or load duplicate resident models blindly.
Verify checkpoint and custom-code hashes before executing model code. Preserve integer metadata,
native prompts, RoPE, attention masks, EOS and sampling budgets. Never fake outputs or decisions.
Math/kernel changes require a separate independent Torch oracle, target-hardware validation,
exact discrete decisions and explicit arithmetic/output-rounding error. Existing native BF16
weights are not quantized caches; do not introduce expanded caches for any later quantized path.
Use FP32 gate until independent full-gate AND prefix qualification passes; image success is
not streaming acceptance. Do not lower threshold or take window maximum to force a response.
Reserve unique request/result IDs before dispatch. Unknown means reconcile read-only, never
automatic inference resend, lock deletion or killing a PID copied from an old report.
No credential/model/media redistribution. Use only authorized platform access. Keep original
services available and manage only processes whose identity/ownership you established.
Record code, offline engineering checks, real hardware integration, and semantic quality separately.
Two failures in one error family without new evidence: narrow the reproducer or change diagnostic
layer, not random version churn. Run tools/check_release.py before packaging.
