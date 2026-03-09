from typing import Any

class noop_context:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


def wrap_context(context: Any, *args, **kwargs):
    if context is None:
        return noop_context(*args, **kwargs)
    return context(*args, **kwargs)
