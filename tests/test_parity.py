"""Every kernel is checked against pyarrow.compute on the same arrays.

Sliced and null-carrying inputs get their own cases: those are where a
columnar kernel goes wrong quietly, because an unsliced dense array works even
when the offset handling is broken.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import mojoarrow as ma


@pytest.fixture(scope="module")
def floats():
    rng = np.random.default_rng(0)
    values = rng.normal(size=1000)
    mask = rng.random(1000) < 0.1
    return pa.array(values, mask=mask)


@pytest.fixture(scope="module")
def dense():
    rng = np.random.default_rng(1)
    return pa.array(rng.normal(size=1000))


@pytest.fixture(scope="module")
def ints():
    rng = np.random.default_rng(2)
    values = rng.integers(-1000, 1000, size=1000)
    mask = rng.random(1000) < 0.1
    return pa.array(values, mask=mask)


def test_sum_matches(floats, dense, ints):
    assert ma.sum(dense) == pytest.approx(pc.sum(dense).as_py())
    assert ma.sum(floats) == pytest.approx(pc.sum(floats).as_py())
    assert ma.sum(ints) == pc.sum(ints).as_py()


def test_min_max_matches(floats, ints):
    assert ma.min(floats) == pytest.approx(pc.min(floats).as_py())
    assert ma.max(floats) == pytest.approx(pc.max(floats).as_py())
    assert ma.min(ints) == pc.min(ints).as_py()
    assert ma.max(ints) == pc.max(ints).as_py()


def test_mean_and_variance(floats):
    assert ma.mean(floats) == pytest.approx(pc.mean(floats).as_py())
    assert ma.variance(floats, ddof=0) == pytest.approx(
        pc.variance(floats, ddof=0).as_py()
    )
    assert ma.variance(floats, ddof=1) == pytest.approx(
        pc.variance(floats, ddof=1).as_py()
    )
    assert ma.stddev(floats, ddof=1) == pytest.approx(
        pc.stddev(floats, ddof=1).as_py()
    )


def test_count_valid(floats, dense):
    assert ma.count_valid(floats) == len(floats) - floats.null_count
    assert ma.count_valid(dense) == len(dense)


def test_all_null_aggregates_are_none():
    empty = pa.array([None, None], type=pa.float64())
    assert ma.sum(empty) is None
    assert ma.min(empty) is None
    assert ma.mean(empty) is None


def test_compare_matches(floats):
    for op, fn in ((">", pc.greater), ("<=", pc.less_equal), ("==", pc.equal)):
        ours = ma.compare(floats, op, 0.5)
        theirs = fn(floats, 0.5)
        assert ours.to_pylist() == theirs.to_pylist()


def test_filter_matches(floats):
    mask = ma.compare(floats, ">", 0.0)
    ours = ma.filter(floats, mask)
    theirs = floats.filter(mask)
    assert ours.to_pylist() == theirs.to_pylist()


def test_filter_int(ints):
    mask = pa.array([i % 3 == 0 for i in range(len(ints))])
    assert ma.filter(ints, mask).to_pylist() == ints.filter(mask).to_pylist()


def test_take_matches(floats):
    idx = pa.array([0, 5, 999, 3, 3, 100])
    assert ma.take(floats, idx).to_pylist() == floats.take(idx).to_pylist()


def test_take_out_of_range_is_null(dense):
    result = ma.take(dense, pa.array([0, 10_000, -1]))
    assert result[0].as_py() == pytest.approx(dense[0].as_py())
    assert result[1].as_py() is None
    assert result[2].as_py() is None


def test_fill_null_matches(floats):
    ours = ma.fill_null(floats, -1.0)
    theirs = pc.fill_null(floats, -1.0)
    assert ours.to_pylist() == theirs.to_pylist()


def test_cast_matches(ints, dense):
    assert ma.cast(ints, pa.float64()).to_pylist() == ints.cast(pa.float64()).to_pylist()
    ours = ma.cast(dense, pa.int64()).to_pylist()
    theirs = [int(v) for v in dense.to_pylist()]
    assert ours == theirs


def test_sort_indices_matches(floats):
    ours = ma.sort_indices(floats).to_pylist()
    values = floats.to_pylist()
    ordered = [values[i] for i in ours]
    non_null = [v for v in ordered if v is not None]
    assert non_null == sorted(non_null)
    assert ordered[len(non_null):] == [None] * (len(ordered) - len(non_null))

    descending = ma.sort_indices(floats, descending=True).to_pylist()
    ordered = [values[i] for i in descending]
    non_null = [v for v in ordered if v is not None]
    assert non_null == sorted(non_null, reverse=True)


def test_group_by_sum_matches():
    rng = np.random.default_rng(3)
    keys = pa.array(rng.integers(0, 20, size=5000))
    values = pa.array(rng.normal(size=5000))

    ours_keys, ours_sums, ours_counts = ma.group_by_sum(keys, values)
    ours = dict(zip(ours_keys.to_pylist(), ours_sums.to_pylist()))

    table = pa.table({"k": keys, "v": values})
    grouped = table.group_by("k").aggregate([("v", "sum"), ("v", "count")])
    theirs = dict(zip(grouped["k"].to_pylist(), grouped["v_sum"].to_pylist()))

    assert set(ours) == set(theirs)
    for key, total in theirs.items():
        assert ours[key] == pytest.approx(total)
    assert sum(ours_counts.to_pylist()) == len(values)


def test_group_by_keeps_null_keys():
    keys = pa.array([1, None, 1, None, 2])
    values = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _, sums, counts = ma.group_by_sum(keys, values)
    assert sum(counts.to_pylist()) == 5
    assert sum(sums.to_pylist()) == pytest.approx(15.0)


# ---------------------------------------------------------------- slices
def test_aggregates_respect_slice_offsets(floats):
    sliced = floats.slice(37, 401)
    assert ma.sum(sliced) == pytest.approx(pc.sum(sliced).as_py())
    assert ma.min(sliced) == pytest.approx(pc.min(sliced).as_py())
    assert ma.count_valid(sliced) == len(sliced) - sliced.null_count


def test_transforms_respect_slice_offsets(floats):
    sliced = floats.slice(11, 200)
    assert ma.fill_null(sliced, 0.0).to_pylist() == pc.fill_null(sliced, 0.0).to_pylist()
    mask = ma.compare(sliced, ">", 0.0)
    assert mask.to_pylist() == pc.greater(sliced, 0.0).to_pylist()
    assert ma.filter(sliced, mask).to_pylist() == sliced.filter(mask).to_pylist()


def test_cast_respects_slice_offsets(ints):
    sliced = ints.slice(5, 100)
    assert ma.cast(sliced, pa.float64()).to_pylist() == \
        sliced.cast(pa.float64()).to_pylist()


def test_zero_copy_input(dense):
    """The kernels must read pyarrow's own buffer, not a converted copy."""
    col = ma.column(dense)
    assert col.values == dense.buffers()[1].address
    assert col.bitmap == 0          # no nulls, no bitmap
    assert col.offset == 0
    sliced = dense.slice(64, 10)
    assert ma.column(sliced).values == dense.buffers()[1].address
    assert ma.column(sliced).offset == 64
