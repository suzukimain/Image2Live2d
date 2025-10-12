__version__ = "0.0.1"

# Re-export main API for easy import
from .Image2Live2d import Split_image as SplitImage  # noqa: E402,F401


def split_image_run(*args, **kwargs):  # noqa: D401
	"""Shortcut to run SplitImage.run(*args, **kwargs)."""
	return SplitImage.run(*args, **kwargs)


__all__ = [
	"SplitImage",
	"split_image_run",
	"__version__",
]
