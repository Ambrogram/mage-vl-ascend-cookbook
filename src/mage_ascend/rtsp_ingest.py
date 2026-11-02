"""Bounded RTSP/TCP receive path. No model imports and no automatic reconnect.

RTSP sources are selected from an operator-owned private JSON registry, never
from an inference HTTP parameter. A stopped/failed session requires a new ID.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import stat
import threading
import time
from urllib.parse import urlsplit


def load_source(registry_path, source_id):
    path = Path(registry_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Source registry must be a regular private file')
    if os.name == 'posix' and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('Source registry must not grant group/other access')
    data = json.loads(path.read_text(encoding='utf-8'))
    source = data['sources'][source_id]
    if set(source) != {'url', 'allowed_host', 'allowed_port'}:
        raise ValueError('Unexpected source registry fields')
    url = source['url']
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 32 for c in url):
        raise ValueError('Invalid source URL')
    parsed = urlsplit(url)
    if parsed.scheme != 'rtsp' or not parsed.hostname or parsed.fragment:
        raise ValueError('Only configured RTSP sources are supported in this stage')
    if parsed.hostname != source['allowed_host'] or (parsed.port or 554) != source['allowed_port']:
        raise ValueError('Source endpoint does not match operator allowlist')
    # The caller must not expose url or userinfo in request/status/error evidence.
    return {'source_id': source_id, 'url': url,
            'source_identity_sha256': hashlib.sha256(url.encode()).hexdigest()}


class RtspSegmentReader:
    """Continuously decode, retain bounded samples, fail explicitly on overload.

    Segments are [0,8), [8,16), ... relative to the first decoded media PTS.
    Sampling chooses the first received frame at/after each 0.5 s target. It
    differs from finite-file linspace and is not asserted pixel-identical to it.
    Full receive runs independently of a slow consumer. No future segment is
    included in a yielded segment. A queue overflow fails rather than silently
    deleting observations. At most 2 queued + 1 forming + 1 consumed segments
    retain 16 frames each. Every decoded frame is limited to 4 M pixels.
    """
    def __init__(self, source, *, segment_seconds=8.0, samples_per_segment=16,
                 max_segments=4, queue_segments=2, open_timeout_ms=5000,
                 read_timeout_ms=10000, capture_factory=None, cv_module=None):
        if (segment_seconds != 8.0 or samples_per_segment != 16 or
                not 1 <= max_segments <= 4 or not 1 <= queue_segments <= 2):
            raise ValueError('Unqualified RTSP sampling or buffer budget')
        if not 100 <= open_timeout_ms <= 10000 or not 100 <= read_timeout_ms <= 10000:
            raise ValueError('Timeout budget exceeded')
        self.source = source
        self.segment_seconds = segment_seconds
        self.samples_per_segment = samples_per_segment
        self.max_segments = max_segments
        self.queue = queue.Queue(maxsize=queue_segments)
        self.open_timeout_ms, self.read_timeout_ms = open_timeout_ms, read_timeout_ms
        self.capture_factory, self.cv = capture_factory, cv_module
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.thread = None
        self.error = None
        self.stats = {'source_id': source['source_id'],
                      'source_identity_sha256': source['source_identity_sha256'],
                      'transport': 'RTSP over TCP', 'status': 'created',
                      'open_timeout_ms': open_timeout_ms, 'read_timeout_ms': read_timeout_ms,
                      'queue_capacity_segments': queue_segments, 'queue_high_water': 0,
                      'decoded_frames': 0, 'sampled_frames': 0,
                      'not_sampled_frames': 0, 'queue_overflow_segments': 0,
                      'sample_targets_skipped': 0, 'media_gaps_over_1p5_frame_intervals': 0,
                      'pos_msec_nonfinite_frames': 0,
                      'backend_frame_loss': 'not measurable from decoded OpenCV frames',
                      'reconnects': 0, 'sampling': 'first received frame at/after each 0.5 s media target',
                      'timestamp_basis': 'OpenCV CAP_PROP_PTS / FPS timebase, relative to first decoded PTS', 'initial_raw_timestamp_observations': [],
                      'max_retained_sample_frames_including_consumer': (queue_segments + 2) * 16,
                      'max_source_pixels': 4_000_000, 'max_segments': max_segments,
                      'max_stream_media_seconds': segment_seconds * max_segments}
        self.stats['decoded_index_basis'] = 'zero-based received decoded frame sequence, not camera frame ID'
        self.stats['wallclock_limit_seconds'] = segment_seconds * max_segments + (open_timeout_ms + read_timeout_ms) / 1000 + 10

    def start(self):
        if self.thread is not None:
            raise RuntimeError('A reader session cannot be started twice')
        self.thread = threading.Thread(target=self._run, name='mage-rtsp-cpu-reader', daemon=True)
        self.thread.start()
        return self

    def _emit(self, segment):
        if not segment['frames_bgr']:
            raise RuntimeError('empty_media_segment')
        segment['closed_monotonic'] = time.monotonic()
        segment['source_decoded_count_when_closed'] = self.stats['decoded_frames']
        try:
            self.queue.put_nowait(segment)
        except queue.Full:
            self.stats['queue_overflow_segments'] += 1
            raise RuntimeError('consumer_backpressure_queue_full') from None
        self.stats['queue_high_water'] = max(self.stats['queue_high_water'], self.queue.qsize())

    def _run(self):
        cap = None
        started = time.monotonic()
        self.stats.update(status='opening', started_monotonic=started)
        try:
            if self.cv is None:
                import cv2
                self.cv = cv2
            cv = self.cv
            if hasattr(cv, 'setLogLevel'):
                cv.setLogLevel(0)  # URL/userinfo must not be emitted by OpenCV logger.
            # A dedicated reader process/validated server must set this before
            # opening; changing it while another capture is open is unsupported.
            options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
            if options not in (None, 'rtsp_transport;tcp'):
                raise RuntimeError('unexpected_ffmpeg_capture_options')
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            params = [cv.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
                      cv.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms]
            if hasattr(cv, 'CAP_PROP_N_THREADS'):
                params += [cv.CAP_PROP_N_THREADS, 2]
            factory = self.capture_factory or cv.VideoCapture
            cap = factory(self.source['url'], cv.CAP_FFMPEG, params)
            self.stats['open_seconds'] = time.monotonic() - started
            if not cap.isOpened():
                raise RuntimeError('rtsp_open_failed')
            fps = float(cap.get(cv.CAP_PROP_FPS))
            if not math.isfinite(fps) or not 1 <= fps <= 120:
                raise RuntimeError('unsupported_fps_metadata')
            self.stats.update(status='receiving', backend=cap.getBackendName(), fps_metadata=fps)
            first_media, previous_media, first_receive, first_pts_value = None, None, None, None
            segment_index, sample_target = 0, 0.0
            segment = self._segment(segment_index)
            while not self.stop_event.is_set():
                if time.monotonic() - started > self.stats['wallclock_limit_seconds']:
                    raise RuntimeError('receive_wallclock_limit_exceeded')
                before_read = time.monotonic()
                ok, frame = cap.read()
                received = time.monotonic()
                self.stats['last_read_seconds'] = received - before_read
                if not ok:
                    raise RuntimeError('rtsp_read_failed_or_source_ended')
                if (frame is None or len(frame.shape) != 3 or frame.shape[2] != 3 or
                        min(frame.shape[:2]) < 2 or frame.shape[0] * frame.shape[1] > 4_000_000):
                    raise RuntimeError('frame_geometry_outside_budget')
                pts_value = float(cap.get(cv.CAP_PROP_PTS))
                pos_msec = float(cap.get(cv.CAP_PROP_POS_MSEC))
                media = pts_value / fps
                if not math.isfinite(media):
                    raise RuntimeError('media_timestamp_unavailable')
                pos_msec_finite = math.isfinite(pos_msec)
                if not pos_msec_finite:
                    pos_msec = None
                    self.stats['pos_msec_nonfinite_frames'] += 1
                if len(self.stats['initial_raw_timestamp_observations']) < 8:
                    self.stats['initial_raw_timestamp_observations'].append({'pts_in_fps_timebase': pts_value, 'pos_msec': pos_msec, 'pos_msec_finite': pos_msec_finite})
                if previous_media is not None and media <= previous_media:
                    raise RuntimeError('media_timestamp_not_strictly_increasing')
                if previous_media is not None and media - previous_media > 1.5 / fps:
                    self.stats['media_gaps_over_1p5_frame_intervals'] += 1
                previous_media = media
                if first_media is None:
                    first_media, first_receive, first_pts_value = media, received, pts_value
                    self.stats.update(first_frame_seconds=received - started,
                                      first_media_timestamp_seconds=media,
                                      first_received_monotonic=received,
                                      decoded_frame_size=[int(frame.shape[1]), int(frame.shape[0])])
                relative = (pts_value - first_pts_value) / fps
                self.stats['decoded_frames'] += 1
                decoded_index = self.stats['decoded_frames'] - 1
                self.stats.update(last_media_timestamp_seconds=media,
                                  last_relative_media_seconds=relative,
                                  latest_receive_monotonic=received,
                                  receive_elapsed_since_first_seconds=received - first_receive,
                                  arrival_minus_media_seconds=(received - first_receive) - relative)
                while relative + 1e-7 >= (segment_index + 1) * self.segment_seconds:
                    segment['end_seconds'] = (segment_index + 1) * self.segment_seconds
                    self._emit(segment)
                    segment_index += 1
                    if segment_index >= self.max_segments:
                        self.stats['status'] = 'completed'
                        return
                    segment = self._segment(segment_index)
                    sample_target = segment_index * self.segment_seconds
                # Use this frame once, count elapsed targets, then resume the clock
                # schedule. Never create several samples from one received frame.
                if relative + 1e-7 >= sample_target and len(segment['frames_bgr']) < 16:
                    segment['frames_bgr'].append(frame)
                    segment['sample_decoded_indices'].append(decoded_index)
                    segment['sample_media_seconds'].append(relative)
                    segment['sample_backend_pts_in_fps_timebase'].append(pts_value)
                    segment['sample_backend_pos_msec'].append(pos_msec)
                    segment['sample_receive_monotonic'].append(received)
                    segment['sample_target_seconds'].append(sample_target)
                    segment['sample_lateness_seconds'].append(max(0.0, relative - sample_target))
                    self.stats['sampled_frames'] += 1
                    step = self.segment_seconds / self.samples_per_segment
                    missed = max(0, int(math.floor((relative + 1e-7 - sample_target) / step)))
                    self.stats['sample_targets_skipped'] += missed
                    segment['sample_targets_skipped'] += missed
                    sample_target += (missed + 1) * step
                else:
                    self.stats['not_sampled_frames'] += 1
            self.stats['status'] = 'stopped'
        except Exception as exc:
            # No raw backend message or URL in public evidence.
            safe = str(exc) if type(exc) is RuntimeError else type(exc).__name__
            self.error = safe
            self.stats.update(status='failed', error_code=safe)
        finally:
            if cap is not None:
                cap.release()
            self.stats['elapsed_seconds'] = time.monotonic() - started
            self.done_event.set()

    def _segment(self, index):
        return {'segment_index': index, 'start_seconds': index * self.segment_seconds,
                'frames_bgr': [], 'sample_decoded_indices': [], 'sample_media_seconds': [],
                'sample_receive_monotonic': [], 'sample_target_seconds': [],
                'sample_backend_pts_in_fps_timebase': [], 'sample_backend_pos_msec': [],
                'sample_lateness_seconds': [], 'sampling_clock': 'media PTS',
                'sample_targets_skipped': 0,
                'future_segment_frames_included': False}

    def segments(self):
        if self.thread is None:
            raise RuntimeError('Reader has not started')
        while not self.done_event.is_set() or not self.queue.empty():
            try:
                yield self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
        if self.error:
            raise RuntimeError(self.error)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join((self.open_timeout_ms + self.read_timeout_ms) / 1000 + 2)
            if self.thread.is_alive():
                raise RuntimeError('capture_thread_did_not_stop_within_timeout')


def segment_evidence(segment):
    result = {key: value for key, value in segment.items() if key != 'frames_bgr'}
    result['sample_count'] = len(segment['frames_bgr'])
    result['sample_bgr_sha256'] = [hashlib.sha256(frame.tobytes()).hexdigest()
                                  for frame in segment['frames_bgr']]
    result['consumer_received_monotonic'] = time.monotonic()
    result['queue_wait_seconds'] = result['consumer_received_monotonic'] - segment['closed_monotonic']
    return result
