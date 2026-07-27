"""mojo-arrow against pyarrow.compute on the same arrays.

    pixi run bench

pyarrow's kernels are C++ with SIMD and a thread pool behind some of them.
This prints the ratio either way round; the losses are as informative as the
wins and both are in the README.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojoarrow as ma  # noqa: E402

N = 5_000_000


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def build_data():
    rng = np.random.default_rng(0)
    values = rng.normal(size=N)
    mask = rng.random(N) < 0.1
    return {
        "dense": pa.array(values),
        "nulls": pa.array(values, mask=mask),
        "ints": pa.array(rng.integers(0, 1000, size=N)),
        "keys": pa.array(rng.integers(0, 50, size=N)),
        "idx": pa.array(rng.integers(0, N, size=N // 10)),
    }


def cases(d):
    dense, nulls, ints, keys, idx = (
        d["dense"], d["nulls"], d["ints"], d["keys"], d["idx"]
    )
    mask = pc.greater(dense, 0.0)
    small = pa.array(np.random.default_rng(1).normal(size=200_000))
    yield "sum, no nulls", lambda: ma.sum(dense), lambda: pc.sum(dense)
    yield "sum, 10% nulls", lambda: ma.sum(nulls), lambda: pc.sum(nulls)
    yield "sum int64", lambda: ma.sum(ints), lambda: pc.sum(ints)
    yield "min", lambda: ma.min(dense), lambda: pc.min(dense)
    yield "variance", lambda: ma.variance(dense), lambda: pc.variance(dense)
    yield ("compare > 0",
           lambda: ma.compare(dense, ">", 0.0),
           lambda: pc.greater(dense, 0.0))
    yield "filter", lambda: ma.filter(dense, mask), lambda: dense.filter(mask)
    yield "take (500k of 5M)", lambda: ma.take(dense, idx), lambda: dense.take(idx)
    yield ("fill_null",
           lambda: ma.fill_null(nulls, 0.0),
           lambda: pc.fill_null(nulls, 0.0))
    yield ("cast int64->float64",
           lambda: ma.cast(ints, pa.float64()),
           lambda: ints.cast(pa.float64()))
    yield ("sort_indices (200k)",
           lambda: ma.sort_indices(small),
           lambda: pc.sort_indices(small))

    table = pa.table({"k": keys, "v": dense})
    yield ("group_by sum, 50 keys",
           lambda: ma.group_by_sum(keys, dense),
           lambda: table.group_by("k").aggregate([("v", "sum")]))


def main() -> None:
    data = build_data()
    print(f"{N:,} float64 elements\n")
    print(f"{'kernel':<26}{'mojo-arrow':>14}{'pyarrow':>12}{'ratio':>9}")
    print("-" * 63)
    for name, ours, theirs in cases(data):
        ours(), theirs()  # warm
        a, b = timeit(ours), timeit(theirs)
        flag = "faster" if a < b else "slower"
        print(f"{name:<26}{a * 1e3:>12.2f}ms{b * 1e3:>10.2f}ms{b / a:>8.2f}x  {flag}")


if __name__ == "__main__":
    main()
