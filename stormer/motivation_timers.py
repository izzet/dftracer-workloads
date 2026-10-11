"""Default-off, bounded application observation. No fences, collectives or RNG draws.

Host backward intervals are inclusive parents, never estimates of exposed DDP.
Fused dataset reads remain unresolved. Copies are vendored in each owning app.
"""
import atexit
import contextlib
import functools
import json
import os
from pathlib import Path
import socket
import time

ENABLED = os.environ.get('LIVEFLOW_MOTIVATION_TIMERS', '0') == '1'


class Recorder:
    def __init__(self):
        self.pid = os.getpid()
        self.limit = int(os.environ.get('LIVEFLOW_MOTIVATION_CAPTURE_STEPS', '400'))
        self.warmup = int(os.environ.get('LIVEFLOW_MOTIVATION_WARMUP_STEPS', '20'))
        if self.limit < 1 or self.warmup < 0:
            raise ValueError('invalid motivation capture bounds')
        self.rows = []
        self.rank = None
        self.world = None
        self.clock_anchor = dict(monotonic_before_ns=time.monotonic_ns(), wall_ns=time.time_ns(), monotonic_after_ns=time.monotonic_ns())
        self.global_step = 0
        self.eligible = 0
        self.current = None
        self.profile = None
        self.profile_closed = False
        self.flushed = False
        atexit.register(self.flush)

    def begin(self, epoch, local_step):
        now = time.monotonic_ns()
        self.global_step += 1
        # Keep successor even after the bounded capture has filled.
        if self.current is not None:
            self.current['next_begin_ns'] = now
            self.current['next_epoch'] = epoch
        active = self.global_step > self.warmup and self.eligible < self.limit
        self.current = None
        if not active:
            return
        self.eligible += 1
        self.current = dict(epoch=int(epoch), local_step=int(local_step),
                            global_step=self.global_step, begin_ns=now,
                            consumed_sample_ids=[], spans=[])
        self.rows.append(self.current)

    def add(self, kind, begin, end, **metadata):
        if self.current is not None:
            self.current['spans'].append(dict(span_kind=kind, begin_ns=begin,
                                              end_ns=end, **metadata))

    def flush(self):
        if os.getpid() != self.pid or self.flushed:
            return
        self.flushed = True
        if self.profile is not None and not self.profile_closed:
            self.profile.stop()
            self.profile_closed = True
        root = Path(os.environ.get('LIVEFLOW_MOTIVATION_OUTPUT_DIR', 'motivation-timers'))
        root.mkdir(parents=True, exist_ok=True)
        rank = self.rank if self.rank is not None else os.environ.get('RANK', os.environ.get('FLUX_TASK_RANK', os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK', 'unknown'))))
        stem = f'{socket.gethostname()}-rank{rank}-pid{self.pid}'
        if self.profile is not None:
            self.profile.export_chrome_trace(str(root / (stem + '-gpu.json')))
        with (root / (stem + '.jsonl')).open('w') as out:
            out.write(json.dumps(dict(schema='liveflow-motivation-v1', rank=rank,
                host=socket.gethostname(), pid=self.pid, world=self.world, clock='CLOCK_MONOTONIC_NS', clock_anchor=self.clock_anchor,
                capture_steps=self.limit, warmup_global_steps=self.warmup,
                recorded_steps=len(self.rows),
                unresolved=['fused_hdf5_storage_decode', 'async_ddp_tail_requires_gpu_trace_reduction'],
                gpu_trace=self.profile is not None)) + '\n')
            for row in self.rows:
                out.write(json.dumps(row) + '\n')


_recorder = None

def recorder():
    global _recorder
    if ENABLED and (_recorder is None or _recorder.pid != os.getpid()):
        _recorder = Recorder()
    return _recorder


def begin(epoch, step):
    if ENABLED:
        recorder().begin(epoch, step)


@contextlib.contextmanager
def span(kind, **metadata):
    if not ENABLED:
        yield
        return
    rec = recorder()
    if rec.current is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    started = time.monotonic_ns()
    # record_function carries real CPU/GPU correlation when passive capture enabled.
    if rec.profile is None:
        with contextlib.nullcontext():
            try:
                yield
            finally:
                rec.add(kind, started, time.monotonic_ns(), **metadata)
    else:
        import torch
        with torch.profiler.record_function('motivation.step%d.%s' % (rec.current['global_step'], kind)):
            try:
                yield
            finally:
                rec.add(kind, started, time.monotonic_ns(), **metadata)


def timed(kind):
    def decorate(fn):
        if not ENABLED:
            return fn
        @functools.wraps(fn)
        def call(*args, **kwargs):
            with span(kind):
                return fn(*args, **kwargs)
        return call
    return decorate


def start_gpu_capture():
    if not ENABLED or os.environ.get('LIVEFLOW_MOTIVATION_GPU_TRACE', '0') != '1':
        return
    import torch
    rec = recorder()
    if rec.profile is not None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU trace requested without CUDA/HIP device')
    # GPU activity is for a finite diagnostic only. Start before warmup and stop
    # at existing shutdown; no profiler stop/device drain inside captured steps.
    if os.environ.get("LIVEFLOW_MOTIVATION_GPU_BOUNDED_RUN", "0") != "1":
        raise RuntimeError("GPU activity requires an explicitly bounded diagnostic")
    rec.profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False,
        with_flops=False)
    rec.profile.start()


def flush():
    if ENABLED and _recorder is not None:
        _recorder.flush()


class ObservedIterator:
    def __init__(self, iterable, epoch, offset=0):
        self.iterable = iterable
        self.inner = None
        self.epoch = epoch
        self.step = offset
    def __iter__(self):
        return self
    def __next__(self):
        if self.inner is None:
            self.inner = iter(self.iterable)
        self.step += 1
        rec = recorder()
        previous = rec.current
        begin(self.epoch, self.step)
        try:
            with span('fetch', ownership='parent_unresolved'):
                value = next(self.inner)
        except StopIteration:
            if rec.current is not None:
                rec.rows.remove(rec.current)
                rec.eligible -= 1
            rec.current = None
            rec.global_step -= 1
            if previous is not None:
                previous.pop('next_begin_ns', None)
                previous.pop('next_epoch', None)
            raise
        return unwrap(value)


def observe(iterable, epoch, offset=0):
    return ObservedIterator(iterable, epoch, offset) if ENABLED else iterable


class TimedDataset:
    """Metadata travels with the ordered result, never with an arbitrary active worker."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)
    def __getitem__(self, key):
        started = time.monotonic_ns()
        _producer.timeline = []
        try:
            result = self.dataset[key]
        finally:
            timeline = _producer.timeline
            _producer.timeline = None
        ended = time.monotonic_ns()
        return result, dict(sample_key=key, host=socket.gethostname(),
                            begin_ns=started, end_ns=ended, stages=timeline, stage='fused_input_unresolved')


class TimedCollate:
    def __init__(self, collate, batched):
        self.collate = collate
        self.batched = batched
    def __call__(self, values):
        from torch.utils.data._utils.collate import default_collate, default_convert
        if self.batched:
            payload = [v[0] for v in values]
            timeline = [v[1] for v in values]
        else:
            payload, meta = values
            timeline = [meta]
        started = time.monotonic_ns()
        fn = self.collate or (default_collate if self.batched else default_convert)
        result = fn(payload)
        ended = time.monotonic_ns()
        # A dict survives pin_memory's built-in recursive traversal.
        return {'__motivation_payload__': result, '__motivation_timeline__': timeline,
                '__motivation_collate__': [started, ended]}


def loader_inputs(dataset, collate=None, batched=True):
    if not ENABLED:
        return dataset, collate
    return TimedDataset(dataset), TimedCollate(collate, batched)


def unwrap(value):
    if not ENABLED or not isinstance(value, dict) or '__motivation_payload__' not in value:
        return value
    rec = recorder()
    if rec.current is not None:
        rec.current['producer_timeline'] = value['__motivation_timeline__']
        rec.current['consumed_sample_ids'] = [m['sample_key'] for m in value['__motivation_timeline__']]
        rec.current['producer_collate_ns'] = value['__motivation_collate__']
    return value['__motivation_payload__']


def wrap_get_data(inner):
    if not ENABLED or not hasattr(inner, '_get_data') or getattr(inner, '_motivation_wrapped', False):
        return
    original = inner._get_data
    def get(*args, **kwargs):
        # Includes queue polling, IPC and unpickle. Untyped duration remains Other.
        with span('ordered_result_receive', ownership='unresolved'):
            return original(*args, **kwargs)
    inner._get_data = get
    inner._motivation_wrapped = True

# Producer observers are thread-local and travel inside the existing result envelope.
import threading
_producer = threading.local()

@contextlib.contextmanager
def producer_span(kind):
    timeline = getattr(_producer, 'timeline', None)
    if not ENABLED or timeline is None:
        yield
        return
    started = time.monotonic_ns()
    try:
        yield
    finally:
        timeline.append(dict(stage=kind, begin_ns=started, end_ns=time.monotonic_ns()))


def producer_timed(kind):
    def decorate(fn):
        if not ENABLED:
            return fn
        @functools.wraps(fn)
        def call(*a, **kw):
            with producer_span(kind):
                return fn(*a, **kw)
        return call
    return decorate


def identify(rank, world=None):
    if ENABLED:
        rec = recorder()
        rec.rank, rec.world = int(rank), world
