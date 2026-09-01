"""Generate the wave 2 choice design.

Wave 1 asked respondents to board the arriving bus or wait for a next bus that
always had seats. That structure cannot identify a seated baseline: if the
arriving bus already has seats, waiting is dominated, so the seated level never
appears as a real alternative. The published crowding literature values crowding
against a seated trip, so wave 1 estimates cannot be compared with it.

Wave 2 therefore shows two buses per task. Bus A is at the stop now, bus B
arrives in a stated number of minutes, and both carry a crowding level. Seated,
standing and packed all appear on both sides of the choice, which makes seated an
identifiable reference level.

The design is D-efficient under a null prior, balanced so that each wait level
appears in exactly four tasks, and split into two blocks of eight that each
retain full rank on the main effects.
"""

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "design.json"

# (packed, standing) dummies; seated is the reference level.
LEVEL = {"seated": (0, 0), "standing": (0, 1), "packed": (1, 0)}
RANK = {"seated": 0, "standing": 1, "packed": 2}
WAIT = [2, 5, 8, 12]
PEAK = [0, 1]
N_TASKS = 16
SEED = 2026


def candidates():
    """All non-dominated tasks: bus B must be less crowded than bus A."""
    out = []
    for a, b in itertools.product(LEVEL, repeat=2):
        if RANK[b] >= RANK[a]:
            continue
        for wait, peak in itertools.product(WAIT, PEAK):
            out.append({"bus_a": a, "bus_b": b, "wait": wait, "peak": peak})
    return out


def design_row(task):
    """Utility difference, bus B minus bus A. Wait is scaled for conditioning."""
    dp = LEVEL[task["bus_b"]][0] - LEVEL[task["bus_a"]][0]
    ds = LEVEL[task["bus_b"]][1] - LEVEL[task["bus_a"]][1]
    w = task["wait"] / max(WAIT)
    return [1.0, w, dp, ds, task["peak"], w * task["peak"]]


def d_criterion(X, idx):
    sign, logdet = np.linalg.slogdet(X[idx].T @ X[idx])
    return -np.inf if sign <= 0 else logdet


def search(cands, X, restarts=1500):
    """Modified Fedorov exchange, restricted to keep wait levels balanced."""
    groups = {w: [i for i, c in enumerate(cands) if c["wait"] == w] for w in WAIT}
    per_level = N_TASKS // len(WAIT)
    rng = np.random.default_rng(SEED)
    best, best_d = None, -np.inf
    for _ in range(restarts):
        idx = [
            int(i)
            for w in WAIT
            for i in rng.choice(groups[w], per_level, replace=False)
        ]
        improved = True
        while improved:
            improved = False
            for pos in range(N_TASKS):
                current = d_criterion(X, idx)
                for cand in groups[cands[idx[pos]]["wait"]]:
                    if cand in idx:
                        continue
                    trial = idx.copy()
                    trial[pos] = cand
                    if d_criterion(X, trial) > current + 1e-9:
                        idx, current, improved = trial, d_criterion(X, trial), True
        if d_criterion(X, idx) > best_d:
            best, best_d = idx, d_criterion(X, idx)
    return best, best_d


def block(tasks, X):
    """Two blocks of eight, each balanced on wait and peak and still full rank."""
    target = {w: (N_TASKS // len(WAIT)) // 2 for w in WAIT}
    best = None
    for combo in itertools.combinations(range(N_TASKS), N_TASKS // 2):
        first = list(combo)
        second = [i for i in range(N_TASKS) if i not in combo]
        if Counter(tasks[i]["wait"] for i in first) != target:
            continue
        if sum(tasks[i]["peak"] for i in first) != len(first) // 2:
            continue
        mains = [X[b][:, :5] for b in (first, second)]
        if any(np.linalg.matrix_rank(m) < 5 for m in mains):
            continue
        score = sum(np.linalg.slogdet(m.T @ m)[1] for m in mains)
        if best is None or score > best[0]:
            best = (score, first, second)
    return best[1], best[2]


def main():
    cands = candidates()
    X = np.array([design_row(c) for c in cands], float)
    idx, best_d = search(cands, X)
    tasks = [cands[i] for i in idx]
    Xd = X[idx]

    corr = np.corrcoef(Xd[:, 1:].T)
    print(f"candidate tasks: {len(cands)}")
    print(f"D-criterion (log det): {best_d:.3f}")
    print(f"rank: {np.linalg.matrix_rank(Xd)} of {Xd.shape[1]} parameters")
    print(f"max |correlation| off diagonal: {np.abs(corr - np.eye(5)).max():.3f}")
    for key in ("bus_a", "bus_b", "wait", "peak"):
        print(f"  {key}: {dict(Counter(t[key] for t in tasks))}")

    first, second = block(tasks, Xd)
    out = {}
    for name, ids in (("1", first), ("2", second)):
        ids = sorted(ids, key=lambda i: (tasks[i]["wait"], tasks[i]["peak"]))
        out[name] = [tasks[i] for i in ids]
        rank = np.linalg.matrix_rank(Xd[ids][:, :5])
        print(f"block {name}: rank {rank} of 5 main effects")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
