"""Severity levels for :mod:`slsqp_jax.sqpdax.logging`.

The numeric values mirror the standard-library :mod:`logging` module so the
two can be mixed without translation; ``DISABLED`` sits above every real
level and is the default of an unconfigured logger.
"""

from __future__ import annotations

__all__ = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "DISABLED",
    "parse_level",
    "level_name",
]

DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
DISABLED = 100

_NAME_TO_LEVEL: dict[str, int] = {
    "DEBUG": DEBUG,
    "INFO": INFO,
    "WARNING": WARNING,
    "WARN": WARNING,
    "ERROR": ERROR,
    "DISABLED": DISABLED,
    "OFF": DISABLED,
}
_LEVEL_TO_NAME: dict[int, str] = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    WARNING: "WARNING",
    ERROR: "ERROR",
    DISABLED: "DISABLED",
}


def parse_level(level: int | str) -> int:
    """Normalise a level given as a name or an integer.

    Parameters
    ----------
    level
        Either a numeric level or one of ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"`` (``"WARN"``), ``"ERROR"``, ``"DISABLED"``
        (``"OFF"``); names are case-insensitive.

    Returns
    -------
    int
        Numeric level.

    Raises
    ------
    ValueError
        If ``level`` is an unknown name.
    TypeError
        If ``level`` is neither a string nor an integer.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.logging import parse_level, INFO
    >>> parse_level("info") == INFO
    True
    >>> parse_level(25)
    25
    """
    match level:
        case bool():
            raise TypeError("log level must be an int or a level name, not a bool")
        case int():
            return level
        case str() if (key := level.strip().upper()) in _NAME_TO_LEVEL:
            return _NAME_TO_LEVEL[key]
        case str():
            raise ValueError(
                f"Unknown log level {level!r}; expected one of "
                f"{sorted(_LEVEL_TO_NAME.values())}"
            )
        case _:
            raise TypeError(
                f"log level must be an int or str; got {type(level).__name__}"
            )


def level_name(level: int) -> str:
    """Return the canonical name of a numeric level.

    Parameters
    ----------
    level
        Numeric level.

    Returns
    -------
    str
        ``"DEBUG"`` / ``"INFO"`` / ... for the standard levels, otherwise
        ``"Level {level}"`` as in the standard library.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.logging import level_name, WARNING
    >>> level_name(WARNING)
    'WARNING'
    >>> level_name(35)
    'Level 35'
    """
    return _LEVEL_TO_NAME.get(level, f"Level {level}")
