"""EuroLeague shot coordinates (``Points`` endpoint) -> court metres, basket at the origin.

Raw ``COORD_X``/``COORD_Y`` are centimetres from the basket centre, with every shot mirrored
onto one basket: x runs across the court (sidelines at about +-750), y runs from the baseline
(about -157) toward half court. Free throws carry the sentinel (-1, -1) and are not shots.
Validated on 2011-12 onward against the FIBA line (reports/week3_closeout.md, 7a); earlier
seasons fit neither the 6.75 m nor the old 6.25 m line and need care.
"""

import numpy as np
import numpy.typing as npt

from eurohoops.models.elo import FloatArray

THREE_RADIUS_M = 6.75
CORNER_THREE_M = 6.60  # |x| of the straight corner segments
CORNER_END_Y_M = 2.99 - 1.575  # corner lines end 2.99 m from the baseline; basket 1.575 m in
FREE_THROW_SENTINEL = -1
FIRST_VALIDATED_SEASON = 2011


def to_court_coords(coord_x: FloatArray, coord_y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Raw centimetres -> metres; the free-throw sentinel (-1, -1) becomes NaN."""
    sentinel = (coord_x == FREE_THROW_SENTINEL) & (coord_y == FREE_THROW_SENTINEL)
    x = np.where(sentinel, np.nan, coord_x / 100.0)
    y = np.where(sentinel, np.nan, coord_y / 100.0)
    return x, y


def beyond_three_line(
    x: FloatArray, y: FloatArray, tolerance: float = 0.0
) -> npt.NDArray[np.bool_]:
    """True where (x, y) in metres lies at or beyond the three-point line moved in by ``tolerance``.

    The line is the 6.75 m arc joined to straight corner segments at |x| = 6.60 m for
    y <= 1.415 m. A negative ``tolerance`` moves the line out instead.
    """
    corner = (np.abs(x) >= CORNER_THREE_M - tolerance) & (y <= CORNER_END_Y_M)
    arc = np.hypot(x, y) >= THREE_RADIUS_M - tolerance
    return corner | arc
