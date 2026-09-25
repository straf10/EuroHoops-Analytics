import pytest

from eurohoops.config import Grid
from eurohoops.eval.backtest import grid_edges
from eurohoops.models.elo import EloParams

GRID = Grid(k=(10.0, 20.0, 30.0), hca=(0.0, 50.0, 100.0), reversion=(0.0, 0.25, 0.5))


@pytest.mark.parametrize(
    ("best", "edges"),
    [
        (EloParams(20.0, 50.0, 0.25), []),
        (EloParams(20.0, 0.0, 0.0), []),  # a lower bound of 0 is natural, not an edge
        (EloParams(10.0, 100.0, 0.5), ["k", "hca", "reversion"]),
        (EloParams(30.0, 50.0, 0.25), ["k"]),
    ],
)
def test_grid_edges(best: EloParams, edges: list[str]) -> None:
    assert grid_edges(GRID, best) == edges
