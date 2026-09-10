from collections.abc import Callable
from typing import TypeVar

from sensei.data.bins import EncodedSample

_T = TypeVar("_T")


def pair_check(
    check_one: Callable[..., bool], x1: EncodedSample, x2: EncodedSample, *args: object
) -> bool:
    """
    Lift a single-point check into a pair check: both x1 and x2 must pass
    `check_one(x, *args)`. Every validity layer (type, functional-dependency,
    domain-rule, plausibility) needs exactly this composition, so it lives
    here once instead of being hand-written in each one.

    Args:
        check_one (Callable[..., bool]): Single-point check to lift.
        x1 (EncodedSample): The first point.
        x2 (EncodedSample): The second point.
        *args (object): Extra arguments forwarded to `check_one`.

    Returns:
        bool: True iff both x1 and x2 pass `check_one`.
    """

    return check_one(x1, *args) and check_one(x2, *args)


def pair_reason(
    reason_one: Callable[..., str | None],
    x1: EncodedSample,
    x2: EncodedSample,
    *args: object,
) -> str | None:
    """
    Diagnostic form of `pair_check`: the first point to fail its
    single-point check, with its reason prefixed by which point it came
    from (`"x1."`/`"x2."`), or None if both pass. x1 is always checked
    before x2, so the reason returned here is always the one that would
    have caused `pair_check(single_check, x1, x2, *args)` to return False.

    Args:
        reason_one (Callable[..., str | None]): Single-point diagnostic
            check to lift.
        x1 (EncodedSample): The first point.
        x2 (EncodedSample): The second point.
        *args (object): Extra arguments forwarded to `reason_one`.

    Returns:
        str | None: The first failure reason found, prefixed with
            `"x1."`/`"x2."`, or None if both pass.
    """

    reason: str | None = reason_one(x1, *args)
    if reason is not None:
        return f"x1.{reason}"

    reason = reason_one(x2, *args)
    if reason is not None:
        return f"x2.{reason}"

    return None
