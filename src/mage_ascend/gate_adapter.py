"""Prepared experimental Mage gate adapter; no installed-package/global shim.

Only the selective scan arithmetic is new. Official Mamba/Block definitions
are AST-selected from the pinned Apache-2.0 sources alongside this file.
The checkpoint's gate source remains unchanged on disk. No incremental cache
or training/backward path is supported; callers recompute bounded prefixes.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import math
from pathlib import Path
from collections import Counter
from typing import Optional

GATE_SOURCE_SHA = "5d9a9d7525aeecc0360ffd43f891ce9d2220297df5092a293a98e2fd31e9d99d"
GATE_WEIGHT_SHA = "01938c515679c1130cff2e6a2af2e4cbc3aad10ea7ccb29229e64c2cfdbf6535"
UPSTREAM_SHA = {
    "mamba_simple.py": "a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3",
    "block.py": "801aea9125a40c13e776b26ea2895e8afbfb204845a6a28bf4b37d4d86f91f03",
    "selective_scan_interface.py": "fe606b4c7e81b47bb091cf59dc474aece1112a6ca01eb6f22309090e56030bc4",
    "mixer_seq_simple.py": "13409d7044e930ea3271e4b8ddceaf8155ec49b8e5ac299fba7bb0df6d80cb21",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _checked_source(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Source identity mismatch: {path}: {actual}")
    return Path(path).read_text(encoding="utf-8")


def _dependencies():
    import torch
    from torch import nn, Tensor
    from torch.nn import functional as F
    from einops import rearrange, repeat
    return dict(torch=torch, nn=nn, F=F, Tensor=Tensor, Optional=Optional,
                math=math, rearrange=rearrange, repeat=repeat,
                __name__=__name__)


def _definition(path, expected, name, namespace, remove_output_cast=False):
    tree = ast.parse(_checked_source(path, expected), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one definition {name}")
    node = nodes[0]
    if remove_output_cast:
        # Diagnostic oracle only: retain the official FP32 result immediately
        # before its mandatory output cast, without changing earlier arithmetic.
        matches = [n for n in node.body if isinstance(n, ast.Assign)
                   and ast.unparse(n) == "out = out.to(dtype=dtype_in)"]
        if len(matches) != 1:
            raise ValueError("Official output-cast AST changed")
        node.body.remove(matches[0])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec", dont_inherit=True), namespace)
    return namespace[name]


def official_scan(raw_output=False):
    root = Path(__file__).resolve().parent / "gate-upstream"
    return _definition(root / "selective_scan_interface.py",
                       UPSTREAM_SHA["selective_scan_interface.py"],
                       "selective_scan_ref", _dependencies(), raw_output)


def bounded_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                 delta_softplus=False, return_last_state=False, *, raw_output=False):
    """Real Mamba1 scan, FP32 state, at most 128 prefix frames, inference only.

    This intentionally supports only this checkpoint's variable B/C [B,N,T]
    and real A. It does not claim grouped/complex/general Mamba compatibility.
    The recurrent state and weights have no retained cache.
    """
    import torch
    from torch.nn import functional as F
    if torch.is_grad_enabled():
        raise RuntimeError("Experimental scan requires torch.inference_mode()")
    if u.ndim != 3 or delta.shape != u.shape or A.ndim != 2:
        raise ValueError("Unsupported scan shapes")
    batch, width, length = u.shape
    state_size = A.shape[1]
    if not (1 <= length <= 128) or width != 5120 or state_size != 16:
        raise ValueError("Only bounded real Mage Mamba1 [B,5120,T<=128], N=16 is supported")
    if B.shape != (batch, state_size, length) or C.shape != B.shape or A.is_complex():
        raise ValueError("Only real variable B/C [B,16,T] is supported")
    if u.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Only separately tested FP32/BF16 activation paths are supported")
    dtype_in = u.dtype
    uf, dt = u.float(), delta.float()
    if delta_bias is not None:
        dt = dt + delta_bias.float()[None, :, None]
    if delta_softplus:
        dt = F.softplus(dt)
    bf, cf = B.float(), C.float()
    state = A.new_zeros((batch, width, state_size))
    outputs = []
    for i in range(length):
        step = dt[:, :, i]
        transition = torch.exp(step[:, :, None] * A[None, :, :])
        drive = (step[:, :, None] * bf[:, None, :, i]) * uf[:, :, i, None]
        state = transition * state + drive
        outputs.append(torch.sum(state * cf[:, None, :, i], dim=-1))
    output = torch.stack(outputs, dim=2)
    if D is not None:
        output = output + uf * D[None, :, None]
    if z is not None:
        # The official reference applies SiLU in z's original dtype.
        output = output * F.silu(z)
    if not raw_output:
        output = output.to(dtype_in)
    return (output, state) if return_last_state else output


def _gate_namespace(model_dir, scan_fn):
    root = Path(__file__).resolve().parent / "gate-upstream"
    namespace = _dependencies()
    namespace.update(selective_scan_fn=scan_fn, causal_conv1d_fn=None,
                     causal_conv1d_update=None, selective_state_update=None,
                     RMSNorm=None, layer_norm_fn=None, rms_norm_fn=None)

    def unavailable(*args, **kwargs):
        raise RuntimeError("CUDA/fused path is not part of this experiment")

    namespace["mamba_inner_fn"] = unavailable
    mamba_cls = _definition(root / "mamba_simple.py", UPSTREAM_SHA["mamba_simple.py"], "Mamba", namespace)
    block_cls = _definition(root / "block.py", UPSTREAM_SHA["block.py"], "Block", namespace)
    # Audit the original factory even though unrelated CUDA/Mamba2/attention
    # imports are never executed. These are precisely its default Mamba1 choices.
    _checked_source(root / "mixer_seq_simple.py", UPSTREAM_SHA["mixer_seq_simple.py"])
    nn = namespace["nn"]

    def create_block(d_model, d_intermediate, layer_idx):
        if (d_model, d_intermediate, layer_idx) != (2560, 0, 0):
            raise ValueError("Unreviewed gate architecture")
        block = block_cls(2560,
                          mixer_cls=functools.partial(mamba_cls, layer_idx=0, use_fast_path=False),
                          mlp_cls=nn.Identity,
                          norm_cls=functools.partial(nn.LayerNorm, eps=1e-5),
                          fused_add_norm=False, residual_in_fp32=False)
        block.layer_idx = layer_idx
        return block

    gate_path = Path(model_dir).resolve() / "streammind_gate.py"
    source = _checked_source(gate_path, GATE_SOURCE_SHA)
    tree = ast.parse(source, filename=str(gate_path))
    replacements = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                    and n.module == "mamba_ssm.models.mixer_seq_simple"
                    and [(a.name, a.asname) for a in n.names] == [("create_block", None)]]
    if len(replacements) != 1:
        raise ValueError("Expected exact create_block import")
    tree.body.remove(replacements[0])
    namespace["create_block"] = create_block
    exec(compile(ast.fix_missing_locations(tree), str(gate_path), "exec", dont_inherit=True), namespace)
    return namespace


def load_gate(model_dir, device, dtype, scan_backend="npu_reference", gate_attention=None, *, _scan_override=None):
    """Strictly load the unchanged real gate; no monkeypatch or service access.

    Returns (gate, diagnostic). `npu_reference` is the candidate bounded scan;
    `official_reference` is independent official selective_scan_ref. The private
    scan override is reserved for the standalone numerical probe's input audit.
    Runtime consumers must wait for the probe's target-hardware pass evidence.
    """
    import torch
    from safetensors.torch import load_file
    if scan_backend not in ("npu_reference", "official_reference"):
        raise ValueError("Unknown scan backend")
    if str(device).startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
    if isinstance(dtype, str):
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Only FP32/BF16 is in the experiment")
    model_dir = Path(model_dir).resolve()
    weight_path = model_dir / "streammind_gate.safetensors"
    weight_sha = sha256(weight_path)
    if weight_sha != GATE_WEIGHT_SHA:
        raise ValueError("Gate checkpoint identity mismatch")
    scan_fn = official_scan() if scan_backend == "official_reference" else bounded_scan
    namespace = _gate_namespace(model_dir, _scan_override or scan_fn)
    # A single CPU construction is immediately moved to the selected device.
    # The probe releases each complete gate before constructing the other path.
    gate = namespace["StreamMindGate"](2560)
    state = load_file(str(weight_path), device="cpu")
    if len(state) != 64 or any(t.dtype != torch.bfloat16 for t in state.values()):
        raise ValueError("Unexpected checkpoint tensor count/dtype")
    result = gate.load_state_dict(state, strict=True)
    del state
    before_attention = gate.cls_net.cls_model.config._attn_implementation
    if gate_attention is not None:
        if gate_attention not in ("eager", "sdpa"):
            raise ValueError("Only explicitly recorded eager/sdpa selection is allowed")
        gate.cls_net.cls_model.set_attn_implementation(gate_attention)
    gate = gate.to(device=device, dtype=dtype).eval().requires_grad_(False)
    diagnostic = {
        "adapter_status": "experimental_requires_numerical_acceptance",
        "adapter_sha256": sha256(__file__), "gate_source_sha256": GATE_SOURCE_SHA,
        "gate_weight_sha256": weight_sha, "upstream_tag": "v2.2.0",
        "upstream_sha256": UPSTREAM_SHA, "scan_backend": scan_backend,
        "source_adaptation": "AST removes exactly create_block import; explicit local factory; original Mamba and Block class ASTs",
        "mamba_config": {"d_model": 2560, "d_inner": 5120, "d_state": 16, "d_conv": 4, "dt_rank": 160, "use_fast_path": False},
        "strict_loading": True, "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        "tensor_count": 64, "parameter_numel": sum(p.numel() for p in gate.parameters()),
        "parameter_dtype": dict(Counter(str(p.dtype) for p in gate.parameters())),
        "parameter_device": dict(Counter(str(p.device) for p in gate.parameters())),
        "buffer_dtype": dict(Counter(str(p.dtype) for p in gate.buffers())),
        "classifier_attention_before": before_attention,
        "classifier_attention_requested": gate_attention,
        "classifier_attention_actual": gate.cls_net.cls_model.config._attn_implementation,
        "changed_arithmetic": "selective_scan only; native Torch convolution is the official slow path",
        "incremental_cache_supported": False, "maximum_prefix_frames": 128,
        "original_weights_unchanged": True,
    }
    return gate, diagnostic
