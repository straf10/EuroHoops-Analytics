"""Static M2 shot charts (F8): matplotlib PNGs in ``docs/models/m2/``, dev dependency only.

The court comes from the ``parse.shots`` constants (FIBA dimensions, basket at the origin, y
toward half court). Charts are hexbins over the shots' court coordinates: the league xPTS
surface (mean xPTS per hexagon), and actual minus expected points per shot for a team or a
player with a diverging colour scale centred at 0.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.parse import shots as court

HEX_GRID = 26
MIN_SHOTS_PER_HEX = 5
LINE_COLOUR = "#333333"
LINE_WIDTH = 1.2
HEX_EDGE = "#9a9a9a"  # so a white (as-expected) hexagon is visible against empty court
EXTENT = (-court.SIDELINE_X_M, court.SIDELINE_X_M, court.BASELINE_Y_M, 9.5)


def court_geometry() -> dict[str, float]:
    """Every dimension the drawing uses, straight from ``parse.shots``."""
    return {
        "three_radius": court.THREE_RADIUS_M,
        "corner_three_x": court.CORNER_THREE_M,
        "corner_end_y": court.CORNER_END_Y_M,
        "baseline_y": court.BASELINE_Y_M,
        "sideline_x": court.SIDELINE_X_M,
        "key_half_width": court.KEY_HALF_WIDTH_M,
        "free_throw_line_y": court.FREE_THROW_LINE_Y_M,
        "restricted_radius": court.RESTRICTED_AREA_RADIUS_M,
    }


def draw_court(ax: Any) -> None:
    from matplotlib.patches import Arc, Circle, Rectangle  # noqa: PLC0415 - dev dependency

    g = court_geometry()
    line: dict[str, Any] = {"color": LINE_COLOUR, "linewidth": LINE_WIDTH}
    ax.add_patch(Circle((0.0, 0.0), 0.225, fill=False, **line))
    ax.add_patch(
        Rectangle(
            (-g["key_half_width"], g["baseline_y"]),
            2 * g["key_half_width"],
            g["free_throw_line_y"] - g["baseline_y"],
            fill=False,
            **line,
        )
    )
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            2 * g["restricted_radius"],
            2 * g["restricted_radius"],
            theta1=0,
            theta2=180,
            **line,
        )
    )
    # three-point line: corner segments, then the arc between their ends
    theta = np.degrees(np.arccos(g["corner_three_x"] / g["three_radius"]))
    for side in (-1.0, 1.0):
        ax.plot(
            [side * g["corner_three_x"]] * 2,
            [g["baseline_y"], g["corner_end_y"]],
            color=line["color"],
            linewidth=line["linewidth"],
        )
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            2 * g["three_radius"],
            2 * g["three_radius"],
            theta1=theta,
            theta2=180 - theta,
            color=line["color"],
            linewidth=line["linewidth"],
        )
    )
    ax.plot(
        [-g["sideline_x"], g["sideline_x"]],
        [g["baseline_y"]] * 2,
        color=line["color"],
        linewidth=line["linewidth"],
    )
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _figure(title: str) -> tuple[Any, Any]:
    import matplotlib as mpl  # noqa: PLC0415 - dev dependency

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=120)
    ax.set_title(title, fontsize=11)
    return fig, ax


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    fig.clf()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)


def xpts_surface(shots: pd.DataFrame, title: str, path: Path) -> None:
    """Mean xPTS per hexagon over ``shots`` (x, y, xpts)."""
    fig, ax = _figure(title)
    hexes = ax.hexbin(
        shots["x"],
        shots["y"],
        C=shots["xpts"],
        reduce_C_function=np.mean,
        gridsize=HEX_GRID,
        extent=EXTENT,
        mincnt=MIN_SHOTS_PER_HEX,
        cmap="viridis",
        linewidths=0.3,
        edgecolors=HEX_EDGE,
    )
    draw_court(ax)
    bar = fig.colorbar(hexes, ax=ax, shrink=0.8)
    bar.set_label("expected points per shot (xPTS)")
    ax.text(
        EXTENT[0] + 0.2,
        EXTENT[3] - 0.5,
        f"{len(shots):,} shots; hexagons with ≥ {MIN_SHOTS_PER_HEX} shots",
        fontsize=8,
    )
    _save(fig, path)


def residual_chart(shots: pd.DataFrame, title: str, path: Path, limit: float = 0.6) -> None:
    """Actual minus expected points per shot per hexagon; the colour scale is centred at 0."""
    from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415 - dev dependency

    fig, ax = _figure(title)
    residual = shots["points"] - shots["xpts"]
    hexes = ax.hexbin(
        shots["x"],
        shots["y"],
        C=residual,
        reduce_C_function=np.mean,
        gridsize=HEX_GRID,
        extent=EXTENT,
        mincnt=MIN_SHOTS_PER_HEX,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
        linewidths=0.3,
        edgecolors=HEX_EDGE,
    )
    draw_court(ax)
    bar = fig.colorbar(hexes, ax=ax, shrink=0.8)
    bar.set_label("actual - expected points per shot (0 = as expected)")
    total = 100.0 * float(residual.mean())
    ax.text(
        EXTENT[0] + 0.2,
        EXTENT[3] - 0.5,
        f"{len(shots):,} shots; {total:+.1f} points per "
        f"100 shots vs xPTS; hexagons with ≥ {MIN_SHOTS_PER_HEX} shots",
        fontsize=8,
    )
    _save(fig, path)
