import math

from sensei.data.bins import FrozenBins, Point


class PlausibilityChecker:
    """
    Q2 checks: plausibility of points/pairs against a frozen binning of the training
    data. Rejects points/pairs with log plausibility below a threshold.
    """

    @staticmethod
    def is_plausible(x: Point, bins: FrozenBins, theta: float) -> bool:
        """
        Check if a point is plausible against the frozen bins and threshold.
        log plaus(x) >= log(theta) is a hard threshold, not a preference.
        """

        return bins.log_plaus(x) >= math.log(theta)

    @staticmethod
    def is_plausible_pair(x1: Point, x2: Point, bins: FrozenBins, theta: float) -> bool:
        """
        Check if a pair of points is plausible against the frozen bins and threshold.

        Args:
            x1 (Point): The first point to check.
            x2 (Point): The second point to check.
            bins (FrozenBins): The frozen bins to check against.
            theta (float): The threshold for plausibility.

        Returns:
            bool: True if both points are plausible, False otherwise.
        """

        return PlausibilityChecker.is_plausible(
            x1, bins, theta
        ) and PlausibilityChecker.is_plausible(x2, bins, theta)
