#!/usr/bin/env python3
"""Unquantized, single-device Mage-VL image baseline. Prepared; not yet run remotely.

No model code is imported for --help. This harness does not modify/monkeypatch
model code. The loading adapter is deliberately bound to the inspected source
hash: its public from_pretrained override mishandles output_loading_info tuples.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import uuid
import warnings


EXPECTED_SOURCES = {
    "config.json": "fd4621212569e893ce2caa110431fde73f3cc1a1d92e1b8320bb4e0cfbcad34a",
    "model.safetensors.index.json": "2cad26f4364750d9b0551dff11e776837b9ef822fe29e068f68d935a8e65cc54",
    "modeling_mage_vl.py": "3fe8bb1e40fa38db4ae08da097ee1238aa4258c4d454e66fe93a44d08ab641ec",
    "configuration_mage_vl.py": "7e3da0c90726ef33da4f0362b3c71dc28650003f1cc11e061d3e99404a8bfed8",
    "processing_mage_vl.py": "c755ac93a161377cbe15f4d6dc30fae6460ef90e6d156d1bff7df82f3dd828cf",
    "video_processing_mage_vl.py": "67f28ea772c260b131983e8a90ad15948134a1baa769bfe6adbf0dd383541acb",
    "codec_video_processing_mage_vl.py": "f3bff7fe00f84a91d542e08561d5d819dd6a4b4192cf075858eb8c495f9e0b02",
    "streammind_gate.py": "5d9a9d7525aeecc0360ffd43f891ce9d2220297df5092a293a98e2fd31e9d99d",
}
CLASS_REFERENCE = "modeling_mage_vl.MageVLForConditionalGeneration"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, help="Explicit logical npu:N, cuda:N, or cpu")
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attention", required=True, choices=("eager", "sdpa"))
    parser.add_argument("--images", required=True, nargs="+", type=Path,
                        help="At least two distinct real images; semantic review remains required")
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each image with the same loaded model and parameters")
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON result path; refuses to overwrite existing evidence")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:npu|cuda):[0-9]+|cpu", args.device):
        parser.error("--device must specify npu:N, cuda:N, or cpu")
    if len(args.images) < 2:
        parser.error("--images requires at least two distinct scenes")
    if not 1 <= args.max_new_tokens <= 4096 or not 1 <= args.repeat <= 20:
        parser.error("Generation budget must be 1..4096 and repeat must be 1..20")
    if not args.prompt.strip():
        parser.error("--prompt cannot be blank")
    return args


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def tensor_description(value, torch):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device), "stride": list(value.stride())}
    if isinstance(value, dict):
        return {str(k): tensor_description(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_description(v, torch) for v in value]
    return {"type": type(value).__name__}


def module_placement(model):
    rows = {}
    for name, module in model.named_modules():
        parameters = list(module.named_parameters(recurse=False))
        buffers = list(module.named_buffers(recurse=False))
        if parameters or buffers:
            rows[name or "<root>"] = {
                "class": type(module).__name__,
                "parameters": {n: {"device": str(v.device), "dtype": str(v.dtype), "shape": list(v.shape)} for n, v in parameters},
                "buffers": {n: {"device": str(v.device), "dtype": str(v.dtype), "shape": list(v.shape)} for n, v in buffers},
            }
    return rows


def run(args, report):
    model_dir = args.model_dir.resolve(strict=True)
    report["stage"] = "asset_identity"
    report["source_sha256"] = {}
    for name, expected in EXPECTED_SOURCES.items():
        actual = sha256(model_dir / name)
        report["source_sha256"][name] = actual
        if actual != expected:
            raise ValueError(f"Inspected source identity mismatch: {name}; review before changing the allowlist")
    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config_data.get("model_type") != "mage_vl" or config_data.get("auto_map", {}).get("AutoModelForImageTextToText") != CLASS_REFERENCE:
        raise ValueError("Model is not the inspected Mage-VL architecture")
    if config_data.get("quantization_config"):
        raise ValueError("This baseline requires the original unquantized checkpoint")
    weight_index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(weight_index["weight_map"].values()))
    if any(Path(name).name != name for name in shards):
        raise ValueError("Unexpected weight shard path")
    report["weights"] = {name: {"bytes": (model_dir / name).stat().st_size} for name in shards}
    report["weight_hash_verification"] = "Not repeated by inference harness; verify the supplied manifest before running"
    image_records = []
    for path in args.images:
        path = path.resolve(strict=True)
        image_records.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    if len({item["sha256"] for item in image_records}) < 2:
        raise ValueError("At least two byte-distinct real images are required")
    report["input_files"] = image_records
    report["model_config"] = config_data
    save_report(args.output, report)

    # Keep every custom-code lookup offline and its dynamic module cache local.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_MODULES_CACHE"] = str(args.output.parent / "hf_modules")
    report["stage"] = "runtime_import"
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    backend = None
    if args.device.startswith("npu:"):
        import torch_npu
        report["torch_npu"] = {"version": torch_npu.__version__, "file": torch_npu.__file__}
        backend = torch.npu
    elif args.device.startswith("cuda:"):
        backend = torch.cuda
    if backend is not None:
        if not backend.is_available():
            raise RuntimeError(f"Requested accelerator is unavailable: {args.device}")
        backend.set_device(args.device)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    report["runtime"] = {"python_executable": sys.executable, "python": sys.version,
                         "platform": platform.platform(), "machine": platform.machine(),
                         "torch": {"version": torch.__version__, "file": torch.__file__},
                         "transformers": {"version": transformers.__version__, "file": transformers.__file__},
                         "visible_devices": {k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}}
    report["runtime"]["package_versions"] = {}
    for package in ("accelerate", "tokenizers", "safetensors", "torchvision", "tokenicer"):
        try:
            report["runtime"]["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["runtime"]["package_versions"][package] = None

    def synchronize():
        if backend is not None:
            backend.synchronize(device)

    def memory():
        if backend is None:
            return {"accelerator_memory": "not applicable"}
        values = {}
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            try:
                values[name + "_bytes"] = int(getattr(backend, name)(device))
            except Exception as exc:
                values[name + "_error"] = str(exc)
        try:
            free, total = backend.mem_get_info(device)
            values.update(device_free_bytes=int(free), device_total_bytes=int(total))
        except Exception as exc:
            values["mem_get_info_error"] = str(exc)
        return values

    def reset_peak():
        if backend is not None:
            backend.reset_peak_memory_stats(device)

    report["memory_before_load"] = memory()
    report["stage"] = "model_load"
    report["loading_adapter"] = {
        "route": "super(MageVLClass, MageVLClass).from_pretrained with output_loading_info=True",
        "reason": "Inspected public override assumes the parent returns a model, while diagnostics return (model, loading_info)",
        "evidence_kind": "source inspection; the public override failure was not separately reproduced on this runtime",
        "equivalent_postload_action": "model.model._streammind_model_path = str(model_dir)",
        "source_modified": False, "monkeypatch": False,
    }
    save_report(args.output, report)
    reset_peak()
    synchronize()
    started = time.perf_counter()
    log_path = args.output.with_suffix(".load.log")
    report["loading_log"] = str(log_path)
    with log_path.open("x", encoding="utf-8") as log_stream:
        handler = logging.StreamHandler(log_stream)
        logging.getLogger().addHandler(handler)
        try:
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(log_stream), redirect_stderr(log_stream):
                warnings.simplefilter("always")
                config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
                model_cls = get_class_from_dynamic_module(CLASS_REFERENCE, str(model_dir), local_files_only=True)
                if model_cls.__name__ != "MageVLForConditionalGeneration":
                    raise ValueError("Unexpected dynamic model class")
                report["model_load_attempts"] += 1
                save_report(args.output, report)
                model, loading_info = super(model_cls, model_cls).from_pretrained(
                    str(model_dir), config=config, local_files_only=True,
                    dtype=dtype, attn_implementation=args.attention,
                    device_map={"": args.device}, output_loading_info=True,
                )
                report["model_load_completed"] = True
                report["loading_info"] = loading_info
                model.model._streammind_model_path = str(model_dir)
                model.eval()
                processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
                report["loading_warnings"] = [str(w.message) for w in caught]
        finally:
            logging.getLogger().removeHandler(handler)
    synchronize()
    report["load_seconds"] = time.perf_counter() - started
    report["loading_info"] = loading_info
    report["memory_after_load"] = memory()
    report["module_placement"] = module_placement(model)
    bad_parameters = [name for name, value in model.named_parameters() if value.device != device]
    meta_buffers = [name for name, value in model.named_buffers() if value.device.type == "meta"]
    bad_buffers = [name for name, value in model.named_buffers() if value.device != device]
    report["placement_issues"] = {"parameters_off_target": bad_parameters, "buffers_off_target": bad_buffers, "meta_buffers": meta_buffers}
    report["effective_attention"] = {
        "top": getattr(model.config, "_attn_implementation", None),
        "text": getattr(model.model.language_model.config, "_attn_implementation", None),
        "vision": getattr(model.model.visual.config, "_attn_implementation", None),
    }
    tokenizer = processor.tokenizer
    report["processor"] = {
        "class": type(processor).__name__, "module": type(processor).__module__,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_processor_config": processor.image_processor.to_dict(),
        "tokenizer_class": type(tokenizer).__name__, "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    report["generation_config"] = model.generation_config.to_dict()
    save_report(args.output, report)
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs", "conversion_errors")):
        raise RuntimeError("Nonempty loading diagnostics require review before baseline inference")
    if bad_parameters or bad_buffers:
        raise RuntimeError("Model placement is incomplete; see placement_issues")
    if any(value != args.attention for value in report["effective_attention"].values()):
        raise RuntimeError("Effective attention differs from the explicitly requested baseline")

    visual_events = []
    def vision_pre(module, positional, keyword):
        visual_events.append({"event": "pre", "inputs": tensor_description(positional, torch),
                              "kwargs": tensor_description(keyword, torch)})
    def vision_post(module, positional, keyword, output):
        visual_events.append({"event": "post", "output": tensor_description(output, torch)})
    pre_handle = model.model.visual.register_forward_pre_hook(vision_pre, with_kwargs=True)
    post_handle = model.model.visual.register_forward_hook(vision_post, with_kwargs=True)
    try:
        for repetition in range(args.repeat):
            for image_index, item in enumerate(image_records):
                case = {"case_id": str(uuid.uuid4()), "image_index": image_index, "repetition": repetition,
                        "input": item, "started_at_utc": utc_now(), "status": "running"}
                report["cases"].append(case)
                report["stage"] = "preprocess"
                save_report(args.output, report)
                t0 = time.perf_counter()
                with Image.open(item["path"]) as source_image:
                    case["source_image"] = {"size": list(source_image.size), "mode": source_image.mode}
                    image = source_image.convert("RGB")
                if min(image.size) < 2:
                    raise ValueError("A 1-pixel test image cannot validate this image baseline")
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[rendered], images=[image], return_tensors="pt")
                image.close()
                case["preprocess_seconds"] = time.perf_counter() - t0
                case["rendered_prompt"] = rendered
                case["rendered_prompt_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "patch_positions"}
                if not required.issubset(inputs):
                    raise ValueError(f"Processor omitted expected Mage-VL fields: {sorted(required - set(inputs))}")
                case["processor_fields_before_transfer"] = tensor_description(dict(inputs), torch)
                case["input_ids"] = inputs["input_ids"].tolist()
                case["attention_mask"] = inputs["attention_mask"].tolist()
                case["image_grid_thw"] = inputs["image_grid_thw"].tolist()
                for name in ("input_ids", "image_grid_thw", "patch_positions"):
                    if inputs[name].dtype not in (torch.int32, torch.int64):
                        raise ValueError(f"Expected integer metadata in {name}")
                image_tokens = int(inputs["input_ids"].eq(config.image_token_id).sum().item())
                patches = int(inputs["image_grid_thw"].prod(dim=-1).sum().item())
                expected_tokens = patches // int(config.vision_config.spatial_merge_size) ** 2
                case["vision_layout"] = {"image_tokens": image_tokens, "patches": patches, "expected_merged_tokens": expected_tokens}
                if image_tokens != expected_tokens or inputs["pixel_values"].shape[0] != patches or list(inputs["patch_positions"].shape) != [patches, 3]:
                    raise ValueError("Image tokens, grid, pixels, and patch positions disagree")
                synchronize()
                t0 = time.perf_counter()
                inputs = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
                synchronize()
                case["host_to_device_seconds"] = time.perf_counter() - t0
                case["processor_fields_on_device"] = tensor_description(inputs, torch)
                input_length = inputs["input_ids"].shape[1]
                visual_events.clear()
                reset_peak()
                case["memory_before_generate"] = memory()
                report["stage"] = "generate"
                save_report(args.output, report)
                if backend is not None:
                    backend.set_device(device)
                synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                               do_sample=False, use_cache=True, return_dict_in_generate=True)
                synchronize()
                case["generate_seconds_synchronized"] = time.perf_counter() - t0
                case["memory_after_generate"] = memory()
                case["visual_hook_events"] = list(visual_events)
                new_tokens = generated.sequences[0, input_length:].detach().cpu().tolist()
                eos = model.generation_config.eos_token_id
                eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
                ended_in_eos = bool(new_tokens) and new_tokens[-1] in eos_ids
                case.update(new_token_ids=new_tokens, new_token_count=len(new_tokens), eos_token_ids=eos_ids,
                            last_token_id=new_tokens[-1] if new_tokens else None, ended_in_eos=ended_in_eos,
                            raw_text=tokenizer.decode(new_tokens, skip_special_tokens=False),
                            text=tokenizer.decode(new_tokens, skip_special_tokens=True),
                            stop_reason="eos_token" if ended_in_eos else "max_new_tokens" if len(new_tokens) >= args.max_new_tokens else "unknown",
                            tokens_per_second_including_prefill=len(new_tokens) / case["generate_seconds_synchronized"],
                            visual_forward_calls=sum(e["event"] == "pre" for e in visual_events),
                            visual_completed_calls=sum(e["event"] == "post" for e in visual_events),
                            decode_iterations_after_prefill=max(0, len(new_tokens) - 1),
                            cache_visual_reuse_check="no_repeated_visual_forward" if len(new_tokens) > 1 else "not_exercised_single_generated_token",
                            semantic_review="pending; compare descriptions with both real images",
                            status="inference_completed", completed_at_utc=utc_now())
                if not new_tokens or case["visual_forward_calls"] != 1 or case["visual_completed_calls"] != 1:
                    case["status"] = "failed"
                    case["cache_visual_reuse_check"] = "failed_visual_call_contract"
                    raise RuntimeError("Single-image cached generation requires exactly one completed visual forward and nonempty token output")
                save_report(args.output, report)
                del inputs, generated
    finally:
        pre_handle.remove()
        post_handle.remove()
    report.update(status="inference_completed_semantic_review_pending", stage="complete", completed_at_utc=utc_now(),
                  limitations=["Image-only unquantized baseline; streaming gate, video, quantization, and serving were not exercised",
                               "Hook and diagnostic overhead is included in synchronized generation timing",
                               "Different file hashes do not establish different semantic scenes; review the actual images and answers"])
    save_report(args.output, report)


def main():
    args = arguments()
    args.output = args.output.resolve()
    if args.output.exists() or args.output.with_suffix(".load.log").exists():
        raise SystemExit("Refusing to overwrite existing result or load log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"run_id": str(uuid.uuid4()), "created_at_utc": utc_now(), "status": "running", "cases": [],
              "script_sha256": sha256(Path(__file__).resolve()), "arguments": vars(args).copy(),
              "model_load_attempts": 0, "model_load_completed": False,
              "generation_policy": {"do_sample": False, "use_cache": True}}
    save_report(args.output, report)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", completed_at_utc=utc_now(),
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        if report["cases"] and report["cases"][-1]["status"] == "running":
            report["cases"][-1]["status"] = "failed"
        save_report(args.output, report)
        print(json.dumps({"status": "failed", "stage": report.get("stage"), "result": str(args.output), "error": str(exc)}))
        return 1
    print(json.dumps({"status": report["status"], "result": str(args.output), "cases": len(report["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
