try:
    from dftracer.python.logger import (
        dftracer as PerfTracer,
        dft_fn as Profile,
        DFTRACER_ENABLE as PERFTRACER_ENABLE,
    )
    from dftracer.python.ai_common import AI

    dft_ai = AI(enable=PERFTRACER_ENABLE)
except Exception:
    try:
        from dftracer.logger import (
            dftracer as PerfTracer,
            dft_fn as Profile,
            ai as dft_ai,
            DFTRACER_ENABLE as PERFTRACER_ENABLE,
        )
    except Exception:
        # Compact fallback classes when dftracer is not available
        # fmt: off
        class _NoOp:
            """Universal no-op class that accepts any method call and returns self or identity function"""
            def __init__(self, *args, **kwargs): pass
            def __call__(self, fn=None, *args, **kwargs): return fn if callable(fn) else self
            def __getattr__(self, name): return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def log(self, fn=None, *args, **kwargs): return fn if callable(fn) else lambda f: f
            def log_init(self, fn=None, *args, **kwargs): return fn if callable(fn) else lambda f: f
            def log_static(self, fn=None, *args, **kwargs): return fn if callable(fn) else lambda f: f
            def iter(self, func, *args, **kwargs): return func
            def update(self, *args, **kwargs): pass
            def flush(self): pass
            def reset(self): pass
            def enable(self): pass
            def disable(self): pass
            def derive(self, name): return self
            def init(self, fn=None, *args, **kwargs): return fn if callable(fn) else lambda f: f
            def create_children(self, names): 
                for name in names: setattr(self, name, _NoOp())
            def get_instance(self): return self
            def initialize_log(self, *args, **kwargs): return self
            def enter_event(self): pass
            def exit_event(self): pass
            def log_event(self, *args, **kwargs): pass
            def finalize(self): pass
            def get_time(self): return 0
            @property
            def cat(self): return ""
            @property 
            def name(self): return ""
            @property
            def type(self): return None
            @property
            def logger(self): return self
        # fmt: on

        # Export instances so call sites like PerfTracer.initialize_log(...)
        # and decorator usage continue to work when dftracer is unavailable.
        Profile = _NoOp()
        PerfTracer = _NoOp()
        dftracer = _NoOp()
        dft_ai = _NoOp()

        PERFTRACER_ENABLE = False

__all__ = [
    "PerfTracer",
    "Profile",
    "PERFTRACER_ENABLE",
    "dft_ai",
]
