import json

import numpy as np
import pytest

from eurohoops.parse.shots import beyond_three_line, to_court_coords
from tests.conftest import REPO

FIXTURE = REPO / "tests/fixtures/points_E2024_1.json"


def test_centimetres_become_metres_and_free_throws_nan() -> None:
    x, y = to_court_coords(np.array([-370.0, -1.0, 0.0]), np.array([156.0, -1.0, 0.0]))
    np.testing.assert_allclose(x, [-3.70, np.nan, 0.0])
    np.testing.assert_allclose(y, [1.56, np.nan, 0.0])


@pytest.mark.parametrize(
    ("x", "y", "beyond"),
    [
        (0.0, 6.80, True),  # top of the arc
        (0.0, 6.70, False),
        (6.65, 0.50, True),  # corner, past the 6.60 m straight line
        (-6.55, 0.50, False),
        (6.40, 1.80, False),  # above the corner segment the arc decides: 6.65 m
        (6.55, 1.80, True),  # 6.79 m
        (4.80, 4.80, True),  # 6.79 m on the diagonal
        (4.70, 4.70, False),  # 6.65 m
    ],
)
def test_three_point_line_geometry(x: float, y: float, beyond: bool) -> None:
    assert beyond_three_line(np.array([x]), np.array([y]))[0] == beyond


def test_tolerance_moves_the_line_in() -> None:
    x, y = np.array([0.0, 6.52]), np.array([6.65, 0.5])
    assert not beyond_three_line(x, y).any()
    assert beyond_three_line(x, y, tolerance=0.15).all()


def test_real_game_shots_agree_with_their_labels() -> None:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["Rows"]
    shots = [r for r in rows if r["ID_ACTION"].strip() in {"2FGA", "2FGM", "3FGA", "3FGM"}]
    x, y = to_court_coords(
        np.array([r["COORD_X"] for r in shots], dtype=np.float64),
        np.array([r["COORD_Y"] for r in shots], dtype=np.float64),
    )
    threes = np.array([r["ID_ACTION"].strip().startswith("3") for r in shots])
    assert len(shots) > 100
    assert beyond_three_line(x[threes], y[threes], tolerance=0.15).all()
    assert not beyond_three_line(x[~threes], y[~threes], tolerance=-0.15).any()
