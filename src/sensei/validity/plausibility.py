import math

from sensei.data.bins import EncodedSample, FrozenBins
from sensei.validity.pairwise import pair_check, pair_reason


class PlausibilityChecker:
    """
    Q2 checks: plausibility of points/pairs against a frozen binning of the training
    data. Rejects points/pairs with log plausibility below a threshold.
    """

    @staticmethod
    def is_plausible(x: EncodedSample, bins: FrozenBins, theta: float) -> bool:
        """
        Check if a point is plausible against the frozen bins and threshold.
        log plaus(x) >= log(theta) is a hard threshold, not a preference.
        theta<=0.0 means "no floor" -- log_plaus(x) is always finite, so every
        point passes; short-circuit rather than call `math.log(0.0)`, which
        raises `ValueError: math domain error`.

        Args:
            x (EncodedSample): The point to check.
            bins (FrozenBins): Frozen plausibility bins.
            theta (float): Plausibility threshold.

        Returns:
            bool: True iff `x` clears the `theta` floor.
        """

        if theta <= 0.0:
            return True

        return bins.log_plaus(x) >= math.log(theta)

    @staticmethod
    def is_plausible_pair(
        x1: EncodedSample, x2: EncodedSample, bins: FrozenBins, theta: float
    ) -> bool:
        """
        Both `x1` and `x2` independently satisfy `is_plausible`.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            bins (FrozenBins): Frozen plausibility bins.
            theta (float): Plausibility threshold.

        Returns:
            bool: True iff both points clear the `theta` floor.
        """

        return pair_check(PlausibilityChecker.is_plausible, x1, x2, bins, theta)

    @staticmethod
    def plausibility_reason(
        x: EncodedSample, bins: FrozenBins, theta: float
    ) -> str | None:
        """
        Diagnostic form of `is_plausible`: `x`'s log-plausibility against the
        `log(theta)` floor it failed to clear, or None if `x` is plausible.

        Args:
            x (EncodedSample): The point to check.
            bins (FrozenBins): Frozen plausibility bins.
            theta (float): Plausibility threshold.

        Returns:
            str | None: The failing log-plausibility vs. floor, or None.
        """

        if theta <= 0.0:
            return None

        log_plaus: float = bins.log_plaus(x)
        log_theta: float = math.log(theta)

        if log_plaus >= log_theta:
            return None

        return f"log_plaus={log_plaus:.4f} < log_theta={log_theta:.4f}"

    @staticmethod
    def plausibility_pair_reason(
        x1: EncodedSample, x2: EncodedSample, bins: FrozenBins, theta: float
    ) -> str | None:
        """
        Diagnostic form of `is_plausible_pair`: the first implausible
        point's reason, prefixed with which point it came from
        (`"x1."`/`"x2."`), or None if both points are plausible.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            bins (FrozenBins): Frozen plausibility bins.
            theta (float): Plausibility threshold.

        Returns:
            str | None: The first failure's reason, prefixed with
                `"x1."`/`"x2."`, or None.
        """

        return pair_reason(PlausibilityChecker.plausibility_reason, x1, x2, bins, theta)
