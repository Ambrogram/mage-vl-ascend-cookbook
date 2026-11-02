#!/usr/bin/env python3
"""Independent Mage-VL native frames-video probe/baseline; no service mutation.

--help uses only stdlib. --probe-only loads config/processor on CPU, not weights.
Full mode reuses the reviewed service Runtime loader in this separate process,
with its runtime/cache/evidence root redirected to video-stream. No API listener,
gate, codec, install, retry, or model source modification is performed.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

EXPERIMENT_ROOT = Path('runs').resolve()
MODEL_DIR = Path('models/Mage-VL').resolve()
RUNTIME_SHA256 = '7cbeb8341adcb8d9f22cab0b671ceb8bb89d771357fefc2954ed0c0eefc285c2'
MIN_PIXELS = 3136


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-module', type=Path, required=True, help='Absolute path to reviewed service model_runtime.py (read only)')
    parser.add_argument('--videos', type=Path, nargs='+', required=True, help='Absolute existing real local video paths; no URLs')
    parser.add_argument('--output', type=Path, required=True, help='New result JSON inside fixed video-stream directory')
    parser.add_argument('--probe-only', action='store_true', help='Only CPU config, processor and real video decoding; no model weights/NPU operations')
    parser.add_argument('--export-gate-features', type=Path, help='New safetensors file in video-stream; one video, export once using original gate-specific vision method')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--dtype', choices=('bfloat16',), default='bfloat16')
    parser.add_argument('--attention', choices=('sdpa', 'eager'), default='sdpa')
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--num-frames', type=int, help='Fixed count; defaults to 16 unless target-fps is set')
    sampling.add_argument('--target-fps', type=float, help='Native duration-based sampling, limited by frame-cap')
    parser.add_argument('--frame-cap', type=int, default=16)
    parser.add_argument('--max-pixels', type=int, default=150000, help='Explicit per-frame budget configured in both native processor layers')
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--min-free-before-load-gib', type=float, default=20.0)
    parser.add_argument('--min-free-before-generate-gib', type=float, default=4.0)
    parser.add_argument('--memory-fraction', type=float, default=0.25,
                        help='Per-process NPU caching-allocator fraction of device total; not a total resource isolation boundary')
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--question', default='Describe the main visual events in chronological order, including camera scene changes.')
    prompts.add_argument('--questions', nargs='+', help='Several questions per video under one model load')
    args = parser.parse_args()
    if args.num_frames is None and args.target_fps is None:
        args.num_frames = 16
    if not re.fullmatch(r'npu:[0-9]+', args.device):
        parser.error('--device must be an explicit npu:N')
    if not 1 <= args.frame_cap <= 64 or (args.num_frames is not None and not 1 <= args.num_frames <= args.frame_cap):
        parser.error('frame-cap must be 1..64 and fixed frame count must fit the cap')
    if args.target_fps is not None and not 0 < args.target_fps <= 30:
        parser.error('target-fps must be >0 and <=30')
    if not MIN_PIXELS <= args.max_pixels <= 1_000_000 or not 1 <= args.max_new_tokens <= 512:
        parser.error('pixel budget must be 3136..1000000 and generation budget 1..512')
    if not 1 <= args.repeat <= 4 or not 1 <= args.max_input_tokens <= 32768 or not args.question.strip():
        parser.error('Invalid repeat, input token budget or question')
    if args.questions is not None and (len(args.questions) > 4 or any(not question.strip() for question in args.questions)):
        parser.error('Supply 1..4 nonempty questions')
    if args.min_free_before_load_gib < 0 or args.min_free_before_generate_gib < 0:
        parser.error('Memory thresholds must not be negative')
    if not 0 < args.memory_fraction <= 1:
        parser.error('memory-fraction must be >0 and <=1')
    if args.export_gate_features is not None and (args.probe_only or len(args.videos) != 1):
        parser.error('Gate feature export requires full model mode with exactly one video (multiple questions are allowed)')
    return args


def describe(value, torch):
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device), 'stride': list(value.stride())}
    if isinstance(value, dict):
        return {str(key): describe(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item, torch) for item in value]
    return {'type': type(value).__name__}


def tensor_hash(value, torch):
    if value.device.type != 'cpu':
        raise ValueError('Evidence hash must be computed before device transfer')
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def processor_configuration(processor):
    vp = processor.video_processor
    return {'image_processor_class': type(processor.image_processor).__name__,
            'image_processor_config': processor.image_processor.to_dict(),
            'video_processor_class': type(vp).__name__,
            'video_processor': {name: getattr(vp, name) for name in (
                'max_frames', 'fixed_num_frames', 'target_fps', 'patch_size', 'spatial_merge_size',
                'temporal_patch_size', 'min_pixels', 'max_pixels', 'resize_frames')},
            'shares_image_processor': vp.image_processor is processor.image_processor}


def build_video_inputs(processor, path, question, num_frames, max_pixels, target_fps=None, frame_cap=16):
    """Native path input, explicit budgets, independent CPU audit of timestamps/patches.

    Native video processor is run once for audit, then normal combined processor
    runs once. Every visual tensor must be exactly equal across these two reads;
    audit time is recorded separately and is not hidden in generation timing.
    """
    import cv2
    import torch
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Video must be a regular local file')
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('OpenCV cannot probe this video')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not (fps > 0 and total_frames > 0 and width > 0 and height > 0):
        raise ValueError('Exact sampling requires known positive FPS/frame count/size; do not accept the native 30 FPS fallback')
    evidence = {'video': {'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size,
                           'fps_opencv': fps, 'total_frames_opencv': total_frames, 'duration_seconds_opencv': total_frames / fps,
                           'width': width, 'height': height},
                'processor_before_configuration': processor_configuration(processor)}
    vp, ip = processor.video_processor, processor.image_processor
    if vp.image_processor is not ip or vp.patch_size != 16 or vp.spatial_merge_size != 2 or vp.temporal_patch_size != 1:
        raise ValueError('Unexpected native shared image/video processor geometry')
    # Public per-instance settings only. The original model/service files remain unchanged.
    # Mage __call__(max_pixels=...) applies that argument only to the codec branch.
    ip.size = {'shortest_edge': MIN_PIXELS, 'longest_edge': int(max_pixels)}
    vp.min_pixels, vp.max_pixels = MIN_PIXELS, int(max_pixels)
    vp.max_frames, vp.fixed_num_frames, vp.target_fps = int(frame_cap), num_frames, target_fps
    vp.resize_frames = True
    evidence['processor_effective_configuration'] = processor_configuration(processor)
    messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': question}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    evidence['rendered_prompt_before_video_expansion'] = rendered
    evidence['rendered_prompt_before_video_expansion_sha256'] = hashlib.sha256(rendered.encode()).hexdigest()
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.getLogger().addHandler(handler)
    try:
        started = time.perf_counter()
        audited = vp(videos=[str(path)], return_tensors='pt')
        evidence['video_processor_audit_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        inputs = processor(text=[rendered], videos=[str(path)], video_backend='frames', num_frames=num_frames,
                           max_frames=frame_cap, target_fps=target_fps, return_tensors='pt', padding=False)
        evidence['native_combined_processor_seconds'] = time.perf_counter() - started
    finally:
        logging.getLogger().removeHandler(handler)
        evidence['native_decode_log'] = log.getvalue()
    required = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'patch_positions'}
    if set(inputs) != required:
        raise ValueError(f'Unexpected native frames model fields: {sorted(inputs)}')
    audit_grid = audited['video_grid_thw']
    if list(audit_grid.shape) != [1, 3]:
        raise ValueError('Expected one video')
    count, grid_h, grid_w = (int(value) for value in audit_grid[0])
    expected_count = min(total_frames, num_frames if target_fps is None else min(max(1, int(total_frames / fps * target_fps)), frame_cap))
    if count != expected_count or not 1 <= count <= frame_cap:
        raise ValueError(f'Decoded frame count {count} differs from exact bounded sampling expectation {expected_count}')
    expected_grid = torch.tensor([[1, grid_h, grid_w]] * count, dtype=inputs['image_grid_thw'].dtype)
    comparisons = {'pixels': torch.equal(audited['pixel_values_videos'], inputs['pixel_values']),
                   'positions': torch.equal(audited['patch_positions'], inputs['patch_positions']),
                   'grid': torch.equal(expected_grid, inputs['image_grid_thw'])}
    evidence['independent_decode_audit_matches'] = comparisons
    if not all(comparisons.values()):
        raise ValueError('Native audit and combined processor decoded different visual inputs')
    pixels_per_frame, patches_per_frame = grid_h * grid_w * 16 * 16, grid_h * grid_w
    positions = inputs['patch_positions']
    if pixels_per_frame > max_pixels or grid_h % 2 or grid_w % 2:
        raise ValueError('Effective pixel budget or spatial merge alignment was violated')
    if list(positions.shape) != [count * patches_per_frame, 3] or inputs['pixel_values'].shape[0] != count * patches_per_frame:
        raise ValueError('Video patch layout mismatch')
    indices = []
    for index in range(count):
        t = positions[index * patches_per_frame:(index + 1) * patches_per_frame, 0]
        if not bool(torch.all(t == t[0])):
            raise ValueError('A sampled frame has inconsistent temporal coordinates')
        indices.append(int(t[0]))
    timestamps = list(audited['frame_timestamps'][0])
    if len(timestamps) != count or len(set(indices)) != count or indices != sorted(indices):
        raise ValueError('Frame timestamp/index count, uniqueness or chronology mismatch')
    if not all(0 <= index < total_frames and abs(sec - index / fps) <= 1.1e-6 for index, sec in zip(indices, timestamps)):
        raise ValueError('Native timestamps disagree with source frame index/FPS')
    image_tokens = int(inputs['input_ids'].eq(151655).sum())
    if image_tokens != count * patches_per_frame // 4 or bool(inputs['input_ids'].eq(151656).any()):
        raise ValueError('Video placeholder expansion does not match merged image token count')
    expanded = processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    if not all(f'<{sec:.1f} seconds>' in expanded for sec in timestamps):
        raise ValueError('Expanded prompt does not contain the native timestamp tags')
    for name in ('input_ids', 'attention_mask', 'image_grid_thw', 'patch_positions'):
        if inputs[name].dtype not in (torch.int32, torch.int64):
            raise ValueError(f'Noninteger video metadata: {name}')
    evidence.update(processor_fields_before_transfer=describe(dict(inputs), torch),
                    sampled_frame_indices=indices, frame_timestamps_seconds=timestamps,
                    temporal_coordinates_kind='original_source_frame_indices', decoded_frame_count=count,
                    effective_frame_size=[grid_w * 16, grid_h * 16], pixels_per_frame=pixels_per_frame,
                    video_grid_thw=audit_grid.tolist(), visual_tokens=image_tokens,
                    expanded_tokenizer_decoded_prompt=expanded,
                    expanded_tokenizer_decoded_prompt_sha256=hashlib.sha256(expanded.encode()).hexdigest(),
                    tensor_sha256={key: tensor_hash(value, torch) for key, value in inputs.items()},
                    full_integer_inputs={key: inputs[key].tolist() for key in ('input_ids', 'attention_mask', 'image_grid_thw', 'patch_positions')},
                    decoder_observation=('opencv_fallback' if 'falling back to OpenCV' in evidence['native_decode_log']
                                         else 'decord' if 'decord' in sys.modules else 'unknown'),
                    cpu_only_inputs=all(value.device.type == 'cpu' for value in inputs.values()))
    return dict(inputs), evidence


def export_gate_features(runtime, device_inputs, output_path, input_evidence, source_sha256):
    """Export one original gate-specific visual forward; never imports/runs the gate."""
    import torch
    from safetensors.torch import save as serialize_safetensors
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + '.json')
    if not output_path.is_absolute() or not output_path.resolve().is_relative_to(EXPERIMENT_ROOT):
        raise ValueError('Gate export must remain inside fixed video-stream directory')
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError('Gate feature evidence already exists')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = []
    def pre(module, positional, keyword):
        events.append({'event': 'pre', 'inputs': describe(positional, torch), 'kwargs': describe(keyword, torch)})
    def post(module, positional, keyword, value):
        events.append({'event': 'post', 'output': describe(value, torch)})
    visual = runtime.model.model.visual
    handles = [visual.register_forward_pre_hook(pre, with_kwargs=True), visual.register_forward_hook(post, with_kwargs=True)]
    result = {'status': 'running', 'route': 'original model.model._streammind_vision_tokens',
              'gate_itself_executed': False, 'separate_from_normal_generate': True,
              'created_at_utc': now(), 'source_sha256': source_sha256,
              'video': input_evidence['video'], 'sampled_frame_indices': input_evidence['sampled_frame_indices'],
              'frame_timestamps_seconds': input_evidence['frame_timestamps_seconds'],
              'image_grid_thw': input_evidence['full_integer_inputs']['image_grid_thw'],
              'patch_positions_sha256': input_evidence['tensor_sha256']['patch_positions'],
              'processor_effective_configuration': input_evidence['processor_effective_configuration'],
              'device': str(runtime.device), 'dtype': str(runtime.dtype), 'attention': runtime.args.attention,
              'visual_hook_events': events}
    rope = visual.video_rope
    previous_rope = (rope.inv_freq_t, rope.inv_freq_h, rope.inv_freq_w)
    try:
        runtime.synchronize()
        runtime.backend.reset_peak_memory_stats(runtime.device)
        result['memory_before'] = runtime.memory()
        started = time.perf_counter()
        with torch.inference_mode():
            features = runtime.model.model._streammind_vision_tokens(
                device_inputs['pixel_values'], device_inputs['image_grid_thw'],
                patch_positions=device_inputs['patch_positions'])
        runtime.synchronize()
        result['vision_forward_seconds_synchronized'] = time.perf_counter() - started
        result['memory_after_forward'] = runtime.memory()
        expected_shape = [1, input_evidence['decoded_frame_count'],
                          input_evidence['visual_tokens'] // input_evidence['decoded_frame_count'], 2560]
        if list(features.shape) != expected_shape or features.dtype != runtime.dtype or features.device != runtime.device:
            raise ValueError('Unexpected gate feature shape, dtype or device')
        if sum(e['event'] == 'pre' for e in events) != 1 or sum(e['event'] == 'post' for e in events) != 1:
            raise RuntimeError('Expected one gate-specific visual pre/post call')
        if not all(getattr(rope, name) is value for name, value in zip(('inv_freq_t', 'inv_freq_h', 'inv_freq_w'), previous_rope)):
            raise RuntimeError('Original method failed to restore RoPE buffer objects')
        result['rope_buffer_objects_restored'] = True
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite gate features')
        started = time.perf_counter()
        cpu_features = features.detach().cpu().contiguous()
        with output_path.open('xb') as stream:
            stream.write(serialize_safetensors({'vision_tokens': cpu_features}, metadata={
                'source': 'original Mage-VL _streammind_vision_tokens',
                'video_sha256': input_evidence['video']['sha256'], 'dtype': str(cpu_features.dtype)}))
        result.update(status='exported', feature_file=str(output_path), feature_sha256=file_hash(output_path),
                      feature_shape=list(cpu_features.shape), feature_dtype=str(cpu_features.dtype),
                      host_transfer_and_save_seconds=time.perf_counter() - started,
                      mamba_ssm_imported='mamba_ssm' in sys.modules, completed_at_utc=now())
        with metadata_path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, default=str)
            stream.write('\n')
        del features, cpu_features
        return result
    finally:
        for handle in handles:
            handle.remove()


def run_video_case(runtime, processor, path, question, settings, *, evidence=None, persist=None,
                   helpers, export_path=None, source_sha256=None):
    """One importable CPU probe or model case; writes only via caller's evidence callback.

    settings supplies num_frames, max_pixels, target_fps, frame_cap, max_input_tokens,
    max_new_tokens, probe_only, min_free_before_generate_gib. Helpers is the reviewed
    model_runtime module. The caller owns model loading/lifetime and its own locking.
    """
    import torch
    case = evidence if evidence is not None else {}
    persist = persist or (lambda: None)
    case.update(question=question, status='running', stage='processor', started_at_utc=now())
    persist()
    try:
        inputs, processor_evidence = build_video_inputs(processor, path, question, settings.num_frames,
                                                         settings.max_pixels, settings.target_fps, settings.frame_cap)
        case.update(processor_evidence)
        if inputs['input_ids'].shape[1] > settings.max_input_tokens:
            raise ValueError('Input token budget exceeded before device transfer')
        persist()
        if settings.probe_only:
            case.update(status='processor_probe_passed', stage='complete', completed_at_utc=now())
            persist()
            return case
        runtime.backend.set_device(runtime.device)
        case['stage'] = 'host_to_device'
        runtime.synchronize()
        started = time.perf_counter()
        device_inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
        device_inputs['pixel_values'] = device_inputs['pixel_values'].to(dtype=runtime.dtype)
        runtime.synchronize()
        case['host_to_device_seconds_synchronized'] = time.perf_counter() - started
        case['processor_fields_on_device'] = describe(device_inputs, torch)
        runtime.backend.reset_peak_memory_stats(runtime.device)
        case['memory_before_generate'] = runtime.memory()
        if case['memory_before_generate'].get('device_free_bytes', 0) < settings.min_free_before_generate_gib * 2**30:
            raise RuntimeError('Free memory threshold not met; no generate call made')
        events = []
        case['visual_hook_events'] = events
        def pre(module, positional, keyword):
            events.append({'event': 'pre', 'inputs': describe(positional, torch), 'kwargs': describe(keyword, torch)})
        def post(module, positional, keyword, output):
            events.append({'event': 'post', 'output': describe(output, torch)})
        pre_handle = runtime.model.model.visual.register_forward_pre_hook(pre, with_kwargs=True)
        post_handle = runtime.model.model.visual.register_forward_hook(post, with_kwargs=True)
        try:
            case['stage'] = 'generate'
            persist()
            torch.manual_seed(0)
            runtime.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = runtime.model.generate(**device_inputs, max_new_tokens=settings.max_new_tokens,
                                                   do_sample=False, use_cache=True, return_dict_in_generate=True)
            runtime.synchronize()
            case['generate_seconds_synchronized'] = time.perf_counter() - started
            case['memory_after_generate'] = runtime.memory()
            tokens = generated.sequences[0, device_inputs['input_ids'].shape[1]:].detach().cpu().tolist()
            eos = runtime.model.generation_config.eos_token_id
            case.update(helpers.generation_evidence(tokens, [eos] if isinstance(eos, int) else list(eos or []), settings.max_new_tokens, events))
            case.update(raw_text=processor.tokenizer.decode(tokens, skip_special_tokens=False),
                        text=processor.tokenizer.decode(tokens, skip_special_tokens=True),
                        semantic_review='pending human review against sampled frames and source video')
            if not case['visual_contract_valid']:
                raise RuntimeError('Expected one completed video visual forward and nonempty tokens')
        finally:
            pre_handle.remove()
            post_handle.remove()
        if export_path is not None:
            case['stage'] = 'gate_feature_export'
            persist()
            case['gate_feature_export'] = export_gate_features(runtime, device_inputs, export_path, case, source_sha256)
        case.update(status='inference_completed_semantic_review_pending', stage='complete', completed_at_utc=now())
        persist()
        return case
    except BaseException as exc:
        case.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        if runtime is not None:
            case['memory_on_failure'] = runtime.memory()
        persist()
        raise


def run_cases(runtime, processor, args, report, save_callback, helpers):
    case_specs = [(index, video, question_index, question) for index, video in enumerate(args.videos)
                  for question_index, question in enumerate(args.questions or [args.question])]
    for repetition in range(args.repeat):
        for index, video, question_index, question in case_specs:
            case = {'index': index, 'question_index': question_index, 'repetition': repetition}
            report['cases'].append(case)
            export_path = args.export_gate_features if len(report['cases']) == 1 else None
            run_video_case(runtime, processor, video, question, args, evidence=case, persist=save_callback,
                           helpers=helpers, export_path=export_path, source_sha256=report['source_sha256'])
