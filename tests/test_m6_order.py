"""``scripts/checks/m6_order.py``: the verdict commit comes before the validation commit, which
comes before the test commit, and the report committed at the verdict holds no validation number.
Each case builds a throwaway git repository."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
ORDER_SCRIPT = REPO / "scripts" / "checks" / "m6_order.py"
PROGRESS = "week16-18_progress.md"
REPORT = "backtest_m6.json"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, message: str, *, report: dict[str, bool], progress: list[str]) -> str:
    (repo / "reports").mkdir(exist_ok=True)
    (repo / "reports" / REPORT).write_text(json.dumps(report), encoding="utf-8")
    (repo / "reports" / PROGRESS).write_text("\n".join(progress), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _check(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORDER_SCRIPT)], cwd=repo, capture_output=True, text=True, check=False
    )


def _history(repo: Path, *, leak_validation_at_verdict: bool = False) -> dict[str, str]:
    """verdict (tuning only) -> validation scored -> test scored."""
    _git(repo, "init", "-q")
    verdict = _commit(
        repo,
        "verdict",
        report={"validation_scored": leak_validation_at_verdict, "test_scored": False},
        progress=["# progress"],
    )
    validation = _commit(
        repo,
        "validation",
        report={"validation_scored": True, "test_scored": False},
        progress=["# progress", f"VERDICT {verdict}"],
    )
    test = _commit(
        repo,
        "test",
        report={"validation_scored": True, "test_scored": True},
        progress=["# progress", f"VERDICT {verdict}", f"VALIDATION {validation}"],
    )
    return {"verdict": verdict, "validation": validation, "test": test}


def _name_them(repo: Path, shas: dict[str, str], *, test: bool = True) -> None:
    lines = [f"VERDICT {shas['verdict']}", f"VALIDATION {shas['validation']}"]
    if test:
        lines.append(f"TEST {shas['test']}")
    _commit(
        repo,
        "name the commits",
        report={"validation_scored": True, "test_scored": True},
        progress=lines,
    )


def test_m6_order_accepts_verdict_then_validation_then_test(tmp_path: Path) -> None:
    _name_them(tmp_path, _history(tmp_path))
    done = _check(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verdict < validation < test: yes" in done.stdout


def test_m6_order_needs_the_test_commit_once_test_is_scored(tmp_path: Path) -> None:
    _name_them(tmp_path, _history(tmp_path), test=False)
    done = _check(tmp_path)
    assert done.returncode == 1
    assert "NO" in done.stdout


def test_m6_order_rejects_a_validation_number_at_the_verdict_commit(tmp_path: Path) -> None:
    _name_them(tmp_path, _history(tmp_path, leak_validation_at_verdict=True))
    assert _check(tmp_path).returncode == 1


def test_m6_order_rejects_the_commits_in_the_wrong_order(tmp_path: Path) -> None:
    shas = _history(tmp_path)
    swapped = {**shas, "verdict": shas["validation"], "validation": shas["verdict"]}
    _name_them(tmp_path, swapped)
    assert _check(tmp_path).returncode == 1


def test_m6_order_does_not_need_a_test_commit_before_test_is_scored(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    verdict = _commit(
        tmp_path,
        "verdict",
        report={"validation_scored": False, "test_scored": False},
        progress=["# progress"],
    )
    _commit(
        tmp_path,
        "validation",
        report={"validation_scored": True, "test_scored": False},
        progress=[f"VERDICT {verdict}"],
    )
    head = _git(tmp_path, "rev-parse", "HEAD")
    _commit(
        tmp_path,
        "name it",
        report={"validation_scored": True, "test_scored": False},
        progress=[f"VERDICT {verdict}", f"VALIDATION {head}"],
    )
    done = _check(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr


def test_m6_order_needs_a_verdict_and_a_validation_line(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _commit(
        tmp_path,
        "nothing named",
        report={"validation_scored": False, "test_scored": False},
        progress=["# progress"],
    )
    assert _check(tmp_path).returncode == 1
