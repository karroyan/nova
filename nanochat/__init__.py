"""
nanochat — namespace package re-exporting from nanochat.nanochat.

This allows ``from nanochat.dataloader import ...`` to work when the
actual package lives under ``nanochat/nanochat/``.
"""

import importlib
import pkgutil
import sys
from types import ModuleType

# The real package
_REAL_PKG = "nanochat.nanochat"

# Ensure the real package is importable (PYTHONPATH should include the repo root)
_real_mod = importlib.import_module(_REAL_PKG)

# Expose the real package's __all__ at this level
__all__ = getattr(_real_mod, "__all__", [])


class _LazyReExporter(ModuleType):
    """
    A module proxy that lazily forwards attribute access to
    ``nanochat.nanochat.<submodule>``.
    """

    def __getattr__(self, name):
        # Skip dunder attributes
        if name.startswith("_"):
            raise AttributeError(name)

        # Try the real package's top-level attributes first
        try:
            return getattr(_real_mod, name)
        except AttributeError:
            pass

        # Try as a submodule: nanochat.nanochat.<name>
        submodule_name = f"{_REAL_PKG}.{name}"
        try:
            return importlib.import_module(submodule_name)
        except ImportError:
            raise AttributeError(
                f"module 'nanochat' has no attribute '{name}'. "
                f"Neither 'nanochat.nanochat.{name}' exists."
            )


# Install the proxy as sys.modules["nanochat"]
_proxy = _LazyReExporter("nanochat")
_proxy.__path__ = _real_mod.__path__
_proxy.__package__ = "nanochat"
_proxy.__file__ = __file__
_proxy.__spec__ = _real_mod.__spec__
sys.modules[__name__] = _proxy
