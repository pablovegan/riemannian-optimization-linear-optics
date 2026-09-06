"""Shared convergence table for the benchmark notebooks."""

import numpy as np
import pandas as pd

def _annotate(results, problem, infid_thresh: float) -> None:
    """Attach ``fidelity`` and ``prob`` to each result in `results`.

    Multi-herald problems score with ``success_prob()``, summed over every detection
    pattern whose infidelity against a target is below `infid_thresh`. Problems without
    post-selection expose neither method; their runs always succeed, so they are scored
    with a probability of 1.
    """
    herald_prob = getattr(problem, "herald_prob", lambda: 1.0)
    for result in results:
        problem.update(result.point)
        result.fidelity = problem.fidelity()
        if hasattr(problem, "success_prob"):
            result.prob = problem.success_prob(1 - infid_thresh)
        else:
            result.prob = herald_prob()


def _build(problem):
    """Return `problem` itself, or the problem its factory builds."""
    return problem if hasattr(problem, "update") else problem()


def _per_row(results, problem):
    """Pair each table row's results with the problem that scores them."""
    if isinstance(results, dict):
        results_dict = results
    elif hasattr(results, "point"):
        results_dict = {"Results": [results]}
    else:
        results_dict = {"Results": list(results)}

    problems = (
        {name: _build(p) for name, p in problem.items()}
        if isinstance(problem, dict)
        else dict.fromkeys(results_dict, _build(problem))
    )
    return results_dict, problems


def benchmarks_table(
    results,
    problem,
    prob_thresh: float = 0.0,
    infid_thresh: float = 1e-7,
    time_format: str = ".3f",
    index_name: str = "Optimizer",
) -> pd.DataFrame:
    """Convergence table for one or several sets of optimizer runs.

    A run counts as converged when its success probability beats `prob_thresh`
    and its infidelity is below `infid_thresh`; mean time and mean iterations are
    averaged over the converged runs only.

    Args:
        results: a single result, a list of results, or a dict mapping an
            optimizer name to its list of results (one table row per key).
        problem: the qoptcraft problem that scores the runs, or a zero-argument
            factory returning one (as the parallel notebooks use). Pass a dict
            keyed like `results` when each row is scored against its own problem
            (e.g. optimizing the same target over different manifolds).
        prob_thresh: minimum success probability for a run to count as converged.
            Leave at 0 for problems without post-selection.
        infid_thresh: maximum infidelity for a run to count as converged.
        time_format: format spec for the mean-time column, e.g. ``".4f"`` for
            the sub-second timings of the permanent-based optimizers.
        index_name: header of the index column, when the rows compare something
            other than optimizers.

    Returns:
        A DataFrame indexed by row name, which notebooks display directly.
    """
    results_dict, problems = _per_row(results, problem)

    rows = []
    for name, optimizer_results in results_dict.items():
        _annotate(optimizer_results, problems[name], infid_thresh)
        converged = [
            result
            for result in optimizer_results
            if result.prob > prob_thresh and (1 - result.fidelity) < infid_thresh
        ]
        mean_time = np.mean([r.time for r in converged]) if converged else float("nan")
        mean_iters = np.mean([r.iterations for r in converged]) if converged else float("nan")
        rows.append({
            index_name: name,
            "Runs": len(optimizer_results),
            "Converged (%)": f"{100 * len(converged) / len(optimizer_results):.1f}%",
            "Mean time (s)": f"{mean_time:{time_format}}",
            "Mean iters": f"{mean_iters:.0f}",
        })
    return pd.DataFrame(rows).set_index(index_name)


def examples_table(
    results,
    problem,
    infid_thresh: float = 1e-7,
    time_format: str = ".3f",
    prob_format: str = ".3f",
    index_name: str = "Optimizer",
) -> pd.DataFrame:
    """Convergence table for examples with no best-known success probability.

    A run counts as converged when its infidelity is below `infid_thresh`; mean
    time and mean iterations are averaged over the converged runs only, and the
    best success probability found by those runs is reported as is, rather than
    scored against a reference value as `benchmarks_table` does.

    Args:
        results: a single result, a list of results, or a dict mapping an
            optimizer name to its list of results (one table row per key).
        problem: the qoptcraft problem that scores the runs, or a zero-argument
            factory returning one (as the parallel notebooks use). Pass a dict
            keyed like `results` when each row is scored against its own problem
            (e.g. optimizing the same target over different manifolds).
        infid_thresh: maximum infidelity for a run to count as converged.
        time_format: format spec for the mean-time column, e.g. ``".4f"`` for
            the sub-second timings of the permanent-based optimizers.
        prob_format: format spec for the best-probability column, which is
            reported as a percentage.
        index_name: header of the index column, when the rows compare something
            other than optimizers.

    Returns:
        A DataFrame indexed by row name, which notebooks display directly.
    """
    results_dict, problems = _per_row(results, problem)

    rows = []
    for name, optimizer_results in results_dict.items():
        _annotate(optimizer_results, problems[name], infid_thresh)
        converged = [
            result for result in optimizer_results if (1 - result.fidelity) < infid_thresh
        ]
        best_prob = f"{100 * max(r.prob for r in converged):{prob_format}}%" if converged else "nan"
        mean_time = np.mean([r.time for r in converged]) if converged else float("nan")
        mean_iters = np.mean([r.iterations for r in converged]) if converged else float("nan")
        rows.append({
            index_name: name,
            "Runs": len(optimizer_results),
            "Converged (%)": f"{100 * len(converged) / len(optimizer_results):.1f}%",
            "Best prob (%)": best_prob,
            "Mean time (s)": f"{mean_time:{time_format}}",
            "Mean iters": f"{mean_iters:.0f}",
        })
    return pd.DataFrame(rows).set_index(index_name)
