__version__ = "0.0.1"

# Re-export SplitImage from the canonical implementation.
try:
	from .Image2Live2d import SplitImage  # type: ignore
except Exception:
	# Fallback to alias package if needed
	from Image2Live2d import SplitImage  # type: ignore
