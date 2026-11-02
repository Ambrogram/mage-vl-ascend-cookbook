"""Candidate: same bounded causal replay with sampled frames held in memory.

Media preparation only: original Mage processor, prompt expansion, vision,
qualified FP32 gate, generation, sampling and thresholds remain unchanged.
No FFV1 transcoding, repeated frame seeks, or repeated processor on response.
The baseline remains in ../stream_runtime.py. This module is not auto-enabled.
"""
from copy import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import time
import traceback

ROOT = Path('runs').resolve()
IMPLEMENTATION_VERSION = 'sampled-memory-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


class PrecomputedVideo:
    """Per-request bridge to native combined processor; never a global patch."""
    def __init__(self, original, bundle):
        self.fixed_num_frames = original.fixed_num_frames
        self.max_frames = original.max_frames
        self.target_fps = original.target_fps
        self.bundle = bundle
        self.calls = 0

    def __call__(self, videos, return_tensors='pt'):
        if videos != ['sampled-memory-input'] or return_tensors != 'pt' or self.calls:
            raise ValueError('Precomputed video bundle is for exactly one native processor call')
        self.calls += 1
        return self.bundle


def build_sampled_inputs(processor, frames_bgr, frame_indices, fps, source_evidence,
                         question, num_frames, max_pixels, video_module, *, timestamps_seconds=None):
    """Preserve the original file path's sampling, OpenCV resize and metadata.

frames_bgr must be precisely the selected frames, in chronological order; frame
indices are relative to this segment, as in the prior FFV1 segment input. Native
list input defaults to dense indices/seconds, so replace only that absent source
metadata with the original module's helper outputs before combined processing.
No tokenizer behavior or template code is changed or reimplemented.
"""
    import cv2
    import torch
    started = time.perf_counter()
    if not (0 < len(frames_bgr) == len(frame_indices) <= num_frames <= 16):
        raise ValueError('Sample count outside candidate bounds')
    if frame_indices != sorted(set(frame_indices)) or fps <= 0:
        raise ValueError('Source frame metadata is not ordered/unique/positive')
    vp, ip = processor.video_processor, processor.image_processor
    original = importlib.import_module(type(vp).__module__)
    if vp.image_processor is not ip or (vp.patch_size, vp.spatial_merge_size, vp.temporal_patch_size) != (16, 2, 1):
        raise ValueError('Unexpected original processor geometry')
    evidence = {'video': source_evidence,
                'processor_before_configuration': video_module.processor_configuration(processor),
                'media_preparation': IMPLEMENTATION_VERSION,
                'independent_decode_audit_per_request': False,
                'input_equivalence_requires_separate_acceptance': True}
    ip.size = {'shortest_edge': 3136, 'longest_edge': int(max_pixels)}
    vp.min_pixels, vp.max_pixels = 3136, int(max_pixels)
    vp.max_frames, vp.fixed_num_frames, vp.target_fps, vp.resize_frames = num_frames, num_frames, None, True
    evidence['processor_effective_configuration'] = video_module.processor_configuration(processor)
    rgb = []
    for frame in frames_bgr:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        rh, rw = original.smart_resize(h, w, 16, 3136, max_pixels, align_patch_size=32)
        if (rh, rw) != (h, w):
            frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if rh < h or rw < w else cv2.INTER_LINEAR)
        rgb.append(frame)
    evidence['sampled_resize_seconds'] = time.perf_counter() - started
    started = time.perf_counter()
    bundle = dict(vp(videos=[rgb], return_tensors='pt'))
    grid = bundle['video_grid_thw']
    bundle['patch_positions'] = original.build_patch_positions(
        grid, spatial_merge_size=2, frame_indices=[torch.tensor(frame_indices, dtype=torch.int64)])
    values = [index / fps for index in frame_indices] if timestamps_seconds is None else list(timestamps_seconds)
    if len(values) != len(frame_indices) or any(not math.isfinite(t) or t < 0 for t in values) or values != sorted(values):
        raise ValueError('Explicit timestamps must be finite nonnegative and chronological')
    timestamps = [original.time_str_to_seconds(original.format_timestamp(value)) for value in values]
    bundle['frame_timestamps'] = [timestamps]
    evidence['native_video_processor_seconds'] = time.perf_counter() - started
    local = copy(processor)
    local.video_processor = PrecomputedVideo(vp, bundle)
    messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': question}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    started = time.perf_counter()
    inputs = dict(local(text=[rendered], videos=['sampled-memory-input'], video_backend='frames',
                        num_frames=num_frames, max_frames=num_frames, return_tensors='pt', padding=False))
    evidence['native_combined_processor_seconds'] = time.perf_counter() - started
    if local.video_processor.calls != 1 or processor.video_processor is not vp:
        raise RuntimeError('Request-local processor bridge violated ownership')
    count, gh, gw = (int(value) for value in grid[0])
    patches = gh * gw
    if set(inputs) != {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'patch_positions'}:
        raise ValueError('Unexpected native combined input fields')
    if count != len(frame_indices) or patches * 256 > max_pixels or gh % 2 or gw % 2:
        raise ValueError('Sampling/spatial budget mismatch')
    if not torch.equal(inputs['image_grid_thw'], torch.tensor([[1, gh, gw]] * count, dtype=grid.dtype)):
        raise ValueError('Native combined grid changed')
    if not torch.equal(inputs['patch_positions'], bundle['patch_positions']) or not torch.equal(inputs['pixel_values'], bundle['pixel_values_videos']):
        raise ValueError('Native combined visual assembly changed')
    for name in ('input_ids', 'attention_mask', 'image_grid_thw', 'patch_positions'):
        if inputs[name].dtype not in (torch.int32, torch.int64):
            raise ValueError('Metadata integer dtype changed')
    image_tokens = int(inputs['input_ids'].eq(151655).sum())
    if image_tokens != count * patches // 4 or bool(inputs['input_ids'].eq(151656).any()):
        raise ValueError('Native video placeholder expansion mismatch')
    expanded = processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    if not all(f'<{sec:.1f} seconds>' in expanded for sec in timestamps):
        raise ValueError('Native timestamp expansion missing')
    evidence.update(
        rendered_prompt_before_video_expansion=rendered,
        rendered_prompt_before_video_expansion_sha256=hash_bytes(rendered.encode()),
        expanded_tokenizer_decoded_prompt=expanded,
        expanded_tokenizer_decoded_prompt_sha256=hash_bytes(expanded.encode()),
        sampled_frame_indices=frame_indices, frame_timestamps_seconds=timestamps,
        temporal_coordinates_kind='original_source_frame_indices', decoded_frame_count=count,
        effective_frame_size=[gw * 16, gh * 16], pixels_per_frame=patches * 256,
        video_grid_thw=grid.tolist(), visual_tokens=image_tokens,
        tensor_sha256={key: video_module.tensor_hash(value, torch) for key, value in inputs.items()},
        full_integer_inputs={key: inputs[key].tolist() for key in ('input_ids', 'attention_mask', 'image_grid_thw', 'patch_positions')},
        processor_fields_before_transfer=video_module.describe(inputs, torch),
        decoder_observation='opencv_sequential_source_decode_no_transcode',
        cpu_only_inputs=all(value.device.type == 'cpu' for value in inputs.values()))
    return inputs, evidence


def run_prepared_case(runtime, processor, device_inputs, input_evidence, question, settings,
                      helpers, video_module, evidence, persist):
    """Original deterministic generation contract, using this segment's inputs."""
    import torch
    case = evidence
    case.update(input_evidence)
    case.update(question=question, status='running', stage='generate', started_at_utc=now(),
                prepared_segment_inputs_reused=True, host_to_device_reused=True)
    events = []
    case['visual_hook_events'] = events
    def pre(module, positional, keyword):
        events.append({'event': 'pre', 'inputs': video_module.describe(positional, torch), 'kwargs': video_module.describe(keyword, torch)})
    def post(module, positional, keyword, output):
        events.append({'event': 'post', 'output': video_module.describe(output, torch)})
    visual = runtime.model.model.visual
    handles = [visual.register_forward_pre_hook(pre, with_kwargs=True), visual.register_forward_hook(post, with_kwargs=True)]
    try:
        runtime.backend.reset_peak_memory_stats(runtime.device)
        case['memory_before_generate'] = runtime.memory()
        if case['memory_before_generate'].get('device_free_bytes', 0) < settings.min_free_before_generate_gib * 2**30:
            raise RuntimeError('Free memory threshold not met; no generate call made')
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
                    semantic_review='Requires comparison against source frames; native errors may persist')
        if not case['visual_contract_valid']:
            raise RuntimeError('Expected one complete visual forward and nonempty generated tokens')
        case.update(status='inference_completed_semantic_review_pending', stage='complete', completed_at_utc=now())
        persist()
        return case
    except BaseException as exc:
        case.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        persist()
        raise
    finally:
        for handle in handles:
            handle.remove()


def stream_replay(runtime, video_module, helpers, video_path, question, settings,
                  evidence, persist, request_dir, *, gate=None):
    import cv2
    import torch
    from safetensors.torch import load_file
    total_started = time.perf_counter()
    if not 1 <= settings.num_frames <= 16 or not 3136 <= settings.max_pixels <= 150000:
        raise ValueError('Stream budget exceeds validated limits')
    path, folder = Path(video_path), Path(request_dir)
    if not folder.resolve().is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Stream storage outside experiment')
    runtime.backend.set_device(runtime.device)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError('Cannot open source video')
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (1 <= fps <= 120 and count > 0 and 0 < width * height <= 4_000_000 and count / fps <= 32):
            raise ValueError('Requires <=32s, known FPS, <=4M pixels')
        segment_frames = max(1, round(8 * fps))
        segment_count = math.ceil(count / segment_frames)
        raw_bytes = count * width * height * 3
        if raw_bytes > 2 * 2**30 or segment_count * settings.num_frames > 64:
            raise ValueError('Decoded BGR/prefix budget exceeded')
        evidence.update(status='running', mode='stream', implementation_version=IMPLEMENTATION_VERSION,
            scope='causal prerecorded-file replay with in-memory samples; not live RTSP',
            source={'sha256': video_module.file_hash(path), 'bytes': path.stat().st_size,
                    'fps': fps, 'frame_count': count, 'width': width, 'height': height},
            segment_seconds=8, gate_threshold=0.5, prefix_frame_cap=64, events=[],
            features_reused_across_prefixes=True, gate_prefix_recomputed=True,
            llm_history_scope='Current segment only; original upstream streaming policy',
            incoming_video_audio_used=False, gate_dtype='float32', decoded_bgr_budget_bytes=2 * 2**30,
            decoded_bgr_total_bytes=raw_bytes, visual_and_llm_dtype=str(runtime.dtype), started_at_utc=now())
        persist()
        if gate is None:
            raise RuntimeError('Fast candidate requires the already loaded qualified FP32 gate')
        evidence['gate_load'] = {'reused_existing_gate': True}
        if {p.dtype for p in gate.parameters()} != {torch.float32}:
            raise RuntimeError('FP32 gate required')
        prefix_features, boundaries, prior_logits = [], [], None
        source_index = 0
        for segment_index in range(segment_count):
            segment_started = time.perf_counter()
            start, stop = source_index, min(count, source_index + segment_frames)
            selected = torch.linspace(0, stop - start - 1, settings.num_frames).round().long().tolist() if stop - start > settings.num_frames else list(range(stop - start))
            selected_set, frames, raw_hashes = set(selected), [], []
            event = {'segment_index': segment_index, 'status': 'decoding', 'source_frame_start': start,
                'source_frame_stop_exclusive': stop, 'start_seconds': start / fps, 'end_seconds': stop / fps,
                'received_at_utc': now(), 'future_segments_decoded_before_decision': False,
                'media_preparation': IMPLEMENTATION_VERSION, 'timings': {}}
            evidence['events'].append(event)
            persist()
            started = time.perf_counter()
            decode_seconds, hash_seconds = 0.0, 0.0
            while source_index < stop:
                tick = time.perf_counter()
                ok, frame = capture.read()
                decode_seconds += time.perf_counter() - tick
                if not ok or frame.shape[:2] != (height, width):
                    raise RuntimeError('Source decode early EOF/geometry change')
                tick = time.perf_counter()
                raw_hashes.append(hash_bytes(frame.tobytes()))
                hash_seconds += time.perf_counter() - tick
                if source_index - start in selected_set:
                    frames.append(frame)
                source_index += 1
            event.update(decoded_bgr_frame_sha256=raw_hashes, source_frames_consumed_when_deciding=source_index,
                         source_decode_only_seconds=decode_seconds, source_frame_hash_seconds=hash_seconds,
                         decode_sample_and_hash_seconds=time.perf_counter() - started,
                         selected_bgr_bytes=sum(f.nbytes for f in frames), codec='original source; no intermediate transcode', status='processor')
            source_evidence = {'path': str(path), 'sha256': evidence['source']['sha256'], 'bytes': path.stat().st_size,
                'fps_opencv': fps, 'total_frames_opencv': stop - start, 'duration_seconds_opencv': (stop - start) / fps,
                'width': width, 'height': height, 'source_segment_start': start, 'source_segment_stop_exclusive': stop}
            started = time.perf_counter()
            inputs, input_evidence = build_sampled_inputs(runtime.processor, frames, selected, fps,
                source_evidence, question, settings.num_frames, settings.max_pixels, video_module)
            event['timings']['processor_and_input_evidence_seconds'] = time.perf_counter() - started
            event['video_input'] = input_evidence
            event['absolute_sampled_source_indices'] = [start + i for i in selected]
            event['absolute_sampled_times_seconds'] = [start / fps + t for t in input_evidence['frame_timestamps_seconds']]
            del frames
            if inputs['input_ids'].shape[1] > settings.max_input_tokens:
                raise ValueError('Input token budget exceeded')
            runtime.synchronize()
            started = time.perf_counter()
            device_inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
            device_inputs['pixel_values'] = device_inputs['pixel_values'].to(runtime.dtype)
            runtime.synchronize()
            event['timings']['host_to_device_seconds_synchronized'] = time.perf_counter() - started
            feature_path = folder / f'gate-features-{segment_index:03d}.safetensors'
            event['gate_visual'] = video_module.export_gate_features(runtime, device_inputs, feature_path, input_evidence, helpers.EXPECTED_SOURCES)
            started = time.perf_counter()
            loaded = load_file(str(feature_path))
            if set(loaded) != {'vision_tokens'}:
                raise RuntimeError('Unexpected feature export schema')
            prefix_features.append(loaded['vision_tokens'].to(runtime.device))
            boundaries.append(sum(int(feature.shape[1]) for feature in prefix_features))
            prefix = torch.cat(prefix_features, dim=1).float()
            runtime.synchronize()
            event['timings']['feature_reload_and_prefix_seconds_synchronized'] = time.perf_counter() - started
            started = time.perf_counter()
            with torch.inference_mode():
                logits = gate(prefix, response_positions=boundaries)[0]
            runtime.synchronize()
            event['gate_seconds_synchronized'] = time.perf_counter() - started
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError('Nonfinite gate output')
            event.update(prefix_frames=int(prefix.shape[1]), prefix_boundaries=list(boundaries),
                         prefix_logits=logits.detach().float().cpu().tolist())
            if prior_logits is not None:
                previous = logits[:prior_logits.shape[0]]
                event['previous_prefix_max_logit_drift'] = float((previous - prior_logits).abs().max().item())
                event['previous_prefix_decisions_unchanged'] = bool(torch.equal(previous.softmax(-1)[:, 1] >= 0.5, prior_logits.softmax(-1)[:, 1] >= 0.5))
                event['previous_prefix_logits_allclose_2e3'] = bool(torch.allclose(previous, prior_logits, rtol=2e-3, atol=2e-3))
                if not event['previous_prefix_decisions_unchanged'] or not event['previous_prefix_logits_allclose_2e3']:
                    raise RuntimeError('Causal prefix changed beyond accepted tolerance')
            prior_logits = logits.detach().clone()
            p = float(logits[boundaries[-1] - 1].float().softmax(-1)[1].item())
            event.update(speak_probability=p, decision='response' if p >= 0.5 else 'silence', decision_at_utc=now(), gate_memory=runtime.memory())
            persist()
            if p >= 0.5:
                event['generation'] = {'called': True}
                run_prepared_case(runtime, runtime.processor, device_inputs, input_evidence, question, settings,
                                  helpers, video_module, event['generation'], persist)
                if not event['generation'].get('ended_in_eos'):
                    raise RuntimeError('Triggered generation did not terminate naturally')
            else:
                event['generation'] = {'called': False, 'reason': 'Actual gate probability below fixed 0.5 threshold'}
            event.update(status='completed', completed_at_utc=now())
            event['timings']['segment_total_seconds'] = time.perf_counter() - segment_started
            del inputs, device_inputs, loaded, prefix, logits
            persist()
        evidence.update(status='completed', completed_at_utc=now(), responses=sum(e['decision'] == 'response' for e in evidence['events']),
                        silences=sum(e['decision'] == 'silence' for e in evidence['events']), source_frame_count_consumed=source_index,
                        final_memory=runtime.memory(), stream_total_seconds=time.perf_counter() - total_started)
        persist()
        return evidence
    finally:
        capture.release()
