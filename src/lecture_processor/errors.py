class LectureProcessorError(Exception):
    """Base exception for expected product errors."""


class DependencyMissingError(LectureProcessorError):
    """Raised when an optional binary or Python package is unavailable."""


class ProcessingError(LectureProcessorError):
    """Raised when a media processing step fails."""


class ProcessingStopped(LectureProcessorError):
    """Raised when the user stops processing for a single file."""
