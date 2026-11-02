#!/usr/bin/env python3
"""Prepared image-only Mage-VL runtime. Heavy dependencies import only in load()."""
from __future__ import annotations
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
import traceback
import warnings

ROOT = Path('runs').resolve()
MODEL_DIR = Path('models/Mage-VL').resolve()
MODEL_ID = 'microsoft/Mage-VL'
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 4_000_000
CLASS_REFERENCE = 'modeling_mage_vl.MageVLForConditionalGeneration'
EXPECTED_SOURCES = {
    'config.json': 'fd4621212569e893ce2caa110431fde73f3cc1a1d92e1b8320bb4e0cfbcad34a',
    'model.safetensors.index.json': '2cad26f4364750d9b0551dff11e776837b9ef822fe29e068f68d935a8e65cc54',
    'modeling_mage_vl.py': '3fe8bb1e40fa38db4ae08da097ee1238aa4258c4d454e66fe93a44d08ab641ec',
    'configuration_mage_vl.py': '7e3da0c90726ef33da4f0362b3c71dc28650003f1cc11e061d3e99404a8bfed8',
    'processing_mage_vl.py': 'c755ac93a161377cbe15f4d6dc30fae6460ef90e6d156d1bff7df82f3dd828cf',
    'video_processing_mage_vl.py': '67f28ea772c260b131983e8a90ad15948134a1baa769bfe6adbf0dd383541acb',
    'codec_video_processing_mage_vl.py': 'f3bff7fe00f84a91d542e08561d5d819dd6a4b4192cf075858eb8c495f9e0b02',
    'streammind_gate.py': '5d9a9d7525aeecc0360ffd43f891ce9d2220297df5092a293a98e2fd31e9d99d',
}
LOADING_ADAPTER = {
    'route': 'super(MageVLClass, MageVLClass).from_pretrained(output_loading_info=True)',
    'reason': 'Inspected override assumes model return, while diagnostics return (model, loading_info)',
    'evidence_kind': 'source inspection; public override failure not separately reproduced',
    'equivalent_postload_action': 'model.model._streammind_model_path = str(model_dir)',
    'source_modified': False, 'monkeypatch': False,
}


class InputError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_description(value, torch):
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
    if isinstance(value, dict):
        return {str(k): tensor_description(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_description(v, torch) for v in value]
    return {'type': type(value).__name__}


def module_placement(model):
    rows = {}
    for name, module in model.named_modules():
        parameters = list(module.named_parameters(recurse=False))
        buffers = list(module.named_buffers(recurse=False))
        if parameters or buffers:
            rows[name or '<root>'] = {
                'class': type(module).__name__,
                'parameters': {n: {'device': str(v.device), 'dtype': str(v.dtype), 'shape': list(v.shape)} for n, v in parameters},
                'buffers': {n: {'device': str(v.device), 'dtype': str(v.dtype), 'shape': list(v.shape)} for n, v in buffers}}
    return rows


def check_image_bounds(byte_count, size):
    if not 0 < byte_count <= MAX_IMAGE_BYTES:
        raise InputError('Image must be 1..33554432 bytes')
    if min(size) < 2 or size[0] * size[1] > MAX_PIXELS:
        raise InputError('Image must be at least 2x2 and no more than 4000000 pixels')


def generation_evidence(tokens, eos_ids, max_new_tokens, events):
    pre = sum(event['event'] == 'pre' for event in events)
    post = sum(event['event'] == 'post' for event in events)
    ended = bool(tokens) and tokens[-1] in eos_ids
    valid = bool(tokens) and pre == 1 and post == 1
    return {
        'new_token_ids': tokens, 'new_token_count': len(tokens), 'eos_token_ids': eos_ids,
        'last_token_id': tokens[-1] if tokens else None, 'ended_in_eos': ended,
        'stop_reason': 'eos_token' if ended else 'max_new_tokens' if len(tokens) >= max_new_tokens else 'unknown',
        'visual_forward_calls': pre, 'visual_completed_calls': post, 'visual_contract_valid': valid,
        'decode_iterations_after_prefill': max(0, len(tokens) - 1),
        'cache_visual_reuse_check': ('failed_visual_call_contract' if not valid else
                                     'no_repeated_visual_forward' if len(tokens) > 1 else
                                     'not_exercised_single_generated_token')}


class Runtime:
    def __init__(self, args):
        self.args = args
        self.guard = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {
            'status': 'loading', 'ready': False, 'busy': False, 'current_request_id': None,
            'pid': os.getpid(), 'instance_id': args.instance_id, 'model_id': MODEL_ID,
            'model_dir': str(MODEL_DIR), 'device': args.device, 'dtype': args.dtype,
            'attention': args.attention, 'started_at_utc': utc_now(),
            'capabilities': ['single_image'], 'max_image_bytes': MAX_IMAGE_BYTES,
            'max_image_pixels': MAX_PIXELS, 'max_new_tokens_limit': 512,
            'default_max_new_tokens': 128, 'health_runs_inference': False}
        self.instance_dir = ROOT / 'runtime/instances' / args.instance_id
        self.instance_dir.mkdir(parents=True, exist_ok=False)
        self.load_report = {'status': 'loading', 'loading_adapter': LOADING_ADAPTER,
                            'model_load_attempts': 0, 'created_at_utc': utc_now()}

    def snapshot(self):
        with self.state_lock:
            return dict(self.state)

    def set_state(self, **values):
        with self.state_lock:
            self.state.update(values)

    def synchronize(self):
        self.backend.synchronize(self.device)

    def memory(self):
        result = {}
        for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
            try:
                result[name + '_bytes'] = int(getattr(self.backend, name)(self.device))
            except Exception as exc:
                result[name + '_error'] = str(exc)
        try:
            free, total = self.backend.mem_get_info(self.device)
            result.update(device_free_bytes=int(free), device_total_bytes=int(total))
        except Exception as exc:
            result['mem_get_info_error'] = str(exc)
        return result

    def load(self):
        try:
            self._load()
            self.load_report.update(status='loaded', completed_at_utc=utc_now())
            save_json(self.instance_dir / 'load.json', self.load_report)
            self.set_state(status='ready', ready=True)
        except BaseException as exc:
            self.load_report.update(status='load_failed', error={'type': type(exc).__name__, 'message': str(exc),
                                                               'traceback': traceback.format_exc()})
            self.set_state(status='load_failed', ready=False, error_type=type(exc).__name__,
                           error='Model load failed; inspect this instance load.json and load.log')
            save_json(self.instance_dir / 'load.json', self.load_report)

    def _load(self):
        report = self.load_report
        report['source_sha256'] = {}
        for name, expected in EXPECTED_SOURCES.items():
            actual = digest((MODEL_DIR / name).read_bytes())
            report['source_sha256'][name] = actual
            if actual != expected:
                raise ValueError(f'Inspected source identity mismatch: {name}')
        config_data = json.loads((MODEL_DIR / 'config.json').read_text(encoding='utf-8'))
        if config_data.get('model_type') != 'mage_vl' or config_data.get('auto_map', {}).get('AutoModelForImageTextToText') != CLASS_REFERENCE:
            raise ValueError('Model is not the inspected Mage-VL class')
        if config_data.get('quantization_config'):
            raise ValueError('Service requires the original unquantized checkpoint')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_MODULES_CACHE'] = str(ROOT / 'runtime/hf_modules')
        import torch
        import torch_npu
        import transformers
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        self.torch, self.backend = torch, torch.npu
        if not self.backend.is_available():
            raise RuntimeError('NPU is unavailable')
        self.backend.set_device(self.args.device)
        self.device, self.dtype = torch.device(self.args.device), getattr(torch, self.args.dtype)
        report['runtime'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'transformers': transformers.__version__}
        report['memory_before_load'] = self.memory()
        report['model_load_attempts'] += 1
        save_json(self.instance_dir / 'load.json', report)
        self.synchronize()
        started = time.perf_counter()
        with (self.instance_dir / 'load.log').open('x', encoding='utf-8') as stream:
            handler = logging.StreamHandler(stream)
            logging.getLogger().addHandler(handler)
            try:
                with warnings.catch_warnings(record=True) as caught, redirect_stdout(stream), redirect_stderr(stream):
                    warnings.simplefilter('always')
                    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
                    model_cls = get_class_from_dynamic_module(CLASS_REFERENCE, str(MODEL_DIR), local_files_only=True)
                    if model_cls.__name__ != 'MageVLForConditionalGeneration':
                        raise ValueError('Unexpected dynamic class')
                    model, loading_info = super(model_cls, model_cls).from_pretrained(
                        str(MODEL_DIR), config=config, local_files_only=True, dtype=self.dtype,
                        attn_implementation=self.args.attention, device_map={'': self.args.device}, output_loading_info=True)
                    report['loading_info'] = loading_info
                    model.model._streammind_model_path = str(MODEL_DIR)
                    model.eval()
                    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
                    report['loading_warnings'] = [str(w.message) for w in caught]
            finally:
                logging.getLogger().removeHandler(handler)
        self.synchronize()
        report['load_seconds_synchronized'] = time.perf_counter() - started
        report['module_placement'] = module_placement(model)
        report['placement_issues'] = {
            'parameters_off_target': [n for n, v in model.named_parameters() if v.device != self.device],
            'buffers_off_target': [n for n, v in model.named_buffers() if v.device != self.device]}
        report['effective_attention'] = {
            'top': getattr(model.config, '_attn_implementation', None),
            'text': getattr(model.model.language_model.config, '_attn_implementation', None),
            'vision': getattr(model.model.visual.config, '_attn_implementation', None)}
        report['processor'] = {'class': type(processor).__name__, 'image_processor_class': type(processor.image_processor).__name__,
                               'image_processor_config': processor.image_processor.to_dict()}
        report['generation_config'] = model.generation_config.to_dict()
        report['memory_after_load'] = self.memory()
        save_json(self.instance_dir / 'load.json', report)
        if any(loading_info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs', 'conversion_errors')):
            raise RuntimeError('Nonempty loading diagnostics; service remains unavailable')
        if any(report['placement_issues'].values()):
            raise RuntimeError('Model parameters or buffers are off the requested NPU')
        if any(v != self.args.attention for v in report['effective_attention'].values()):
            raise RuntimeError('Effective attention differs from requested attention')
        # Transformers 5.7 stores the legacy max_pixels argument in size.
        # A Mage source comment calling this class "slow" does not establish its backend.
        image_size = getattr(processor.image_processor, 'size', {})
        native_max_pixels = (image_size.get('longest_edge') if isinstance(image_size, dict)
                             else getattr(image_size, 'longest_edge', None))
        report['processor']['native_max_pixels'] = native_max_pixels
        report['processor']['implementation_mro'] = [c.__module__ + '.' + c.__name__ for c in type(processor.image_processor).__mro__]
        if type(processor.image_processor).__name__ != 'Qwen2VLImageProcessor' or native_max_pixels != MAX_PIXELS:
            raise RuntimeError('Unexpected native image processor or pixel budget')
        self.config, self.model, self.processor = config, model, processor

    def infer(self, image_bytes, prompt, max_new_tokens, evidence):
        # NPU device selection is thread-local; this is not the loading thread.
        self.backend.set_device(self.device)
        torch, processor, model = self.torch, self.processor, self.model
        from PIL import Image, UnidentifiedImageError
        evidence.update(input_sha256=digest(image_bytes), input_bytes=len(image_bytes), prompt=prompt,
                        max_new_tokens=max_new_tokens, device=str(self.device), dtype=str(self.dtype), attention=self.args.attention,
                        generation_policy={'do_sample': False, 'use_cache': True},
                        model_load_report=str(self.instance_dir / 'load.json'), semantic_review='not automatically assessed')
        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                check_image_bounds(len(image_bytes), source.size)
                if source.format not in ('JPEG', 'PNG', 'WEBP') or getattr(source, 'n_frames', 1) != 1:
                    raise InputError('Only single-frame JPEG, PNG, or WEBP is accepted')
                evidence['source_image'] = {'size': list(source.size), 'mode': source.mode, 'format': source.format}
                image = source.convert('RGB')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InputError('Cannot decode supplied image safely') from exc
        evidence['image_decode_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = processor(text=[rendered], images=[image], return_tensors='pt')
        finally:
            image.close()
        evidence['processor_seconds'] = time.perf_counter() - started
        evidence['rendered_prompt'], evidence['rendered_prompt_sha256'] = rendered, digest(rendered.encode('utf-8'))
        required = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'patch_positions'}
        if not required.issubset(inputs):
            raise RuntimeError('Processor omitted required Mage-VL fields')
        evidence['processor_fields_before_transfer'] = tensor_description(dict(inputs), torch)
        for name in ('input_ids', 'image_grid_thw', 'patch_positions'):
            if inputs[name].dtype not in (torch.int32, torch.int64):
                raise RuntimeError(f'Non-integer metadata: {name}')
        for name in ('input_ids', 'attention_mask', 'image_grid_thw'):
            evidence[name] = inputs[name].tolist()
        image_tokens = int(inputs['input_ids'].eq(self.config.image_token_id).sum().item())
        patches = int(inputs['image_grid_thw'].prod(dim=-1).sum().item())
        merged = patches // int(self.config.vision_config.spatial_merge_size) ** 2
        evidence['vision_layout'] = {'image_tokens': image_tokens, 'patches': patches, 'expected_merged_tokens': merged}
        if image_tokens != merged or inputs['pixel_values'].shape[0] != patches or list(inputs['patch_positions'].shape) != [patches, 3]:
            raise RuntimeError('Image token/grid/patch layout mismatch')
        self.synchronize()
        started = time.perf_counter()
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.dtype)
        self.synchronize()
        evidence['host_to_device_seconds_synchronized'] = time.perf_counter() - started
        evidence['processor_fields_on_device'] = tensor_description(inputs, torch)
        input_length = inputs['input_ids'].shape[1]
        events = []
        evidence['visual_hook_events'] = events
        def pre(module, positional, keyword):
            events.append({'event': 'pre', 'inputs': tensor_description(positional, torch), 'kwargs': tensor_description(keyword, torch)})
        def post(module, positional, keyword, output):
            events.append({'event': 'post', 'output': tensor_description(output, torch)})
        pre_handle = model.model.visual.register_forward_pre_hook(pre, with_kwargs=True)
        post_handle = model.model.visual.register_forward_hook(post, with_kwargs=True)
        try:
            self.backend.reset_peak_memory_stats(self.device)
            evidence['memory_before_generate'] = self.memory()
            torch.manual_seed(0)
            self.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                           use_cache=True, return_dict_in_generate=True)
            self.synchronize()
            evidence['generate_seconds_synchronized'] = time.perf_counter() - started
            evidence['memory_after_generate'] = self.memory()
            started = time.perf_counter()
            tokens = generated.sequences[0, input_length:].detach().cpu().tolist()
            eos = model.generation_config.eos_token_id
            evidence.update(generation_evidence(tokens, [eos] if isinstance(eos, int) else list(eos or []), max_new_tokens, events))
            evidence['raw_text'] = processor.tokenizer.decode(tokens, skip_special_tokens=False)
            evidence['text'] = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            evidence['output_decode_seconds'] = time.perf_counter() - started
            evidence['tokens_per_second_including_prefill'] = len(tokens) / evidence['generate_seconds_synchronized']
            if not evidence['visual_contract_valid']:
                raise RuntimeError('Expected one completed visual forward and nonempty generated tokens')
        finally:
            pre_handle.remove()
            post_handle.remove()
