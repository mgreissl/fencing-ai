import sys


def _patch_hydra_for_python314():
    """
    Patch hydra's LazyCompletionHelp to be a str subclass.

    Python 3.14 argparse now calls `'%' in help_string` during add_argument(),
    which fails on non-string/non-iterable objects. Making LazyCompletionHelp
    inherit from str fixes this.
    """
    try:
        import hydra._internal.utils as _hydra_utils

        original_cls = getattr(_hydra_utils, "LazyCompletionHelp", None)
        if original_cls is not None and not issubclass(original_cls, str):

            class PatchedLazyCompletionHelp(str):
                def __new__(cls):
                    return str.__new__(cls, "Install or Uninstall shell completion")

                def __repr__(self) -> str:
                    return f"Install or Uninstall shell completion:\\n{_hydra_utils._get_completion_help()}"

            # Monkey-patch at module level
            _hydra_utils.LazyCompletionHelp = PatchedLazyCompletionHelp
    except (ImportError, AttributeError):
        pass  # Hydra not installed or structure changed


if sys.version_info >= (3, 14):
    _patch_hydra_for_python314()
