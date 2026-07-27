"""Arrow compute functions, with `pyarrow.compute`'s names and semantics.

Outputs are built with `pa.Array.from_buffers` over buffers this module
allocates, so a result is a real Arrow array that any Arrow consumer can take
without a conversion.
"""

from __future__ import annotations

import builtins

import numpy as np
import pyarrow as pa

from ._lib import Column, column, lib

_OPS = {">": 0, ">=": 1, "<": 2, "<=": 3, "==": 4, "!=": 5}


def _bitmap_buffer(n: int) -> tuple[np.ndarray, pa.Buffer]:
    """A zeroed validity bitmap, allocated from Arrow's pool.

    Arrow's allocator recycles; numpy's does not. On a 5M-element output that
    is the difference between reusing warm pages and taking a page fault per
    4KB, which was most of the cost of every transform kernel here.
    """
    buf = _alloc(((n + 7) // 8 + 63) & ~63)
    raw = np.frombuffer(buf, dtype=np.uint8)
    raw[:] = 0
    return raw, buf


def _alloc(nbytes: int) -> pa.Buffer:
    return pa.allocate_buffer(nbytes, resizable=False)


def _values_buffer(n: int, dtype) -> tuple[np.ndarray, pa.Buffer]:
    buf = _alloc(n * dtype.itemsize)
    return np.frombuffer(buf, dtype=dtype), buf


def _is_float(col: Column) -> bool:
    return pa.types.is_floating(col.array.type)


def _is_int(col: Column) -> bool:
    return pa.types.is_integer(col.array.type)


def _require(col: Column, kind: str) -> None:
    if kind == "float" and not _is_float(col):
        raise TypeError(f"expected a float64 array, got {col.array.type}")
    if kind == "int" and not _is_int(col):
        raise TypeError(f"expected an int64 array, got {col.array.type}")


# ------------------------------------------------------------ aggregation
def sum(array):
    """Sum of the valid elements. None when every element is null."""
    col = column(array)
    if col.length == col.null_count:
        return None
    if _is_float(col):
        return lib().ma_sum_f64(col.values, col.bitmap, col.offset, col.length)
    _require(col, "int")
    return lib().ma_sum_i64(col.values, col.bitmap, col.offset, col.length)


def min(array):
    col = column(array)
    if col.length == col.null_count:
        return None
    if _is_float(col):
        return lib().ma_min_f64(col.values, col.bitmap, col.offset, col.length)
    _require(col, "int")
    return lib().ma_min_i64(col.values, col.bitmap, col.offset, col.length)


def max(array):
    col = column(array)
    if col.length == col.null_count:
        return None
    if _is_float(col):
        return lib().ma_max_f64(col.values, col.bitmap, col.offset, col.length)
    _require(col, "int")
    return lib().ma_max_i64(col.values, col.bitmap, col.offset, col.length)


def count_valid(array) -> int:
    col = column(array)
    return lib().ma_count_valid(col.bitmap, col.offset, col.length)


def mean(array):
    col = column(array)
    valid = col.length - col.null_count
    if valid == 0:
        return None
    total = sum(array)
    return total / valid


def variance(array, ddof: int = 0):
    col = column(array)
    _require(col, "float")
    if col.length - col.null_count <= ddof:
        return None
    return lib().ma_variance_f64(
        col.values, col.bitmap, col.offset, col.length, ddof
    )


def stddev(array, ddof: int = 0):
    var = variance(array, ddof)
    return None if var is None else var ** 0.5


# ----------------------------------------------------------------- filter
def filter(array, mask):
    """Keep elements where `mask` is true. Nulls in the mask drop the row."""
    col = column(array)
    m = column(mask)
    if not pa.types.is_boolean(m.array.type):
        raise TypeError("mask must be a boolean array")
    if m.length != col.length:
        raise ValueError("mask and array must be the same length")

    if _is_float(col):
        values, vbuf = _values_buffer(col.length, np.dtype(np.float64))
        kernel, dtype = lib().ma_filter_f64, pa.float64()
    else:
        _require(col, "int")
        values, vbuf = _values_buffer(col.length, np.dtype(np.int64))
        kernel, dtype = lib().ma_filter_i64, pa.int64()
    _, bbuf = _bitmap_buffer(col.length)

    kept = kernel(
        col.values, col.bitmap, col.offset, col.length,
        m.values, m.bitmap, m.offset,
        vbuf.address, bbuf.address,
    )
    return pa.Array.from_buffers(dtype, kept, [bbuf, vbuf])


def take(array, indices):
    """Gather by index. An out-of-range index yields null, never a crash."""
    col = column(array)
    idx = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    if _is_float(col):
        values, vbuf = _values_buffer(len(idx), np.dtype(np.float64))
        kernel, dtype = lib().ma_take_f64, pa.float64()
    else:
        _require(col, "int")
        values, vbuf = _values_buffer(len(idx), np.dtype(np.int64))
        kernel, dtype = lib().ma_take_i64, pa.int64()
    _, bbuf = _bitmap_buffer(len(idx))
    kernel(
        col.values, col.bitmap, col.offset, col.length,
        idx.ctypes.data, len(idx), vbuf.address, bbuf.address,
    )
    return pa.Array.from_buffers(dtype, len(idx), [bbuf, vbuf])


def compare(array, op: str, scalar: float):
    """`array <op> scalar` as a boolean array. Null in, null out."""
    if op not in _OPS:
        raise ValueError(f"unknown operator {op!r}")
    col = column(array)
    _require(col, "float")
    _, values_buf = _bitmap_buffer(col.length)
    _, bits_buf = _bitmap_buffer(col.length)
    lib().ma_compare_f64(
        col.values, col.bitmap, col.offset, col.length, float(scalar), _OPS[op],
        values_buf.address, bits_buf.address,
    )
    return pa.Array.from_buffers(pa.bool_(), col.length, [bits_buf, values_buf])


def fill_null(array, value: float):
    col = column(array)
    _require(col, "float")
    _, vbuf = _values_buffer(col.length, np.dtype(np.float64))
    lib().ma_fill_null_f64(
        col.values, col.bitmap, col.offset, col.length, float(value), vbuf.address
    )
    return pa.Array.from_buffers(pa.float64(), col.length, [None, vbuf])


def cast(array, target):
    """int64 <-> float64 only, which is the pair worth a kernel."""
    col = column(array)
    target = pa.type_for_alias(target) if isinstance(target, str) else target
    if pa.types.is_floating(target) and _is_int(col):
        _, vbuf = _values_buffer(col.length, np.dtype(np.float64))
        lib().ma_cast_i64_f64(col.values, col.offset, col.length, vbuf.address)
        out_type = pa.float64()
    elif pa.types.is_integer(target) and _is_float(col):
        _, vbuf = _values_buffer(col.length, np.dtype(np.int64))
        lib().ma_cast_f64_i64(col.values, col.offset, col.length, vbuf.address)
        out_type = pa.int64()
    else:
        raise TypeError(f"unsupported cast {col.array.type} -> {target}")
    validity = None
    if col.has_nulls:
        # the new values buffer is zero-based, so the validity has to be too
        _, validity = _bitmap_buffer(col.length)
        lib().ma_copy_bitmap(col.bitmap, col.offset, col.length, validity.address)
    return pa.Array.from_buffers(out_type, col.length, [validity, vbuf])


def sort_indices(array, descending: bool = False):
    """Indices that sort the array, nulls last."""
    col = column(array)
    _require(col, "float")
    idx = np.empty(col.length, dtype=np.int64)
    lib().ma_argsort_f64(
        col.values, col.bitmap, col.offset, col.length, idx.ctypes.data,
        1 if descending else 0,
    )
    return pa.array(idx)


# --------------------------------------------------------------- group by
def group_by_sum(keys, values):
    """Sum `values` per distinct int64 key.

    Returns `(keys, sums, counts)` as Arrow arrays. Null keys form their own
    group rather than being dropped, so the counts always add up to the number
    of rows with a non-null value.
    """
    k = column(keys)
    v = column(values)
    _require(k, "int")
    _require(v, "float")
    if k.length != v.length:
        raise ValueError("keys and values must be the same length")

    # Start small and grow: most group-bys have far fewer distinct keys than
    # rows, and a table sized for the rows is both a large allocation and a
    # guaranteed cache miss on every probe.
    capacity = 1024
    while True:
        table_keys = np.empty(capacity, dtype=np.int64)
        table_slot = np.empty(capacity, dtype=np.int64)
        group_keys = np.empty(capacity, dtype=np.int64)
        sums = np.empty(capacity, dtype=np.float64)
        counts = np.empty(capacity, dtype=np.int64)
        groups = lib().ma_group_sum_fused_i64(
            k.values, k.bitmap, k.offset,
            v.values, v.bitmap, v.offset, k.length,
            table_keys.ctypes.data, table_slot.ctypes.data, capacity,
            group_keys.ctypes.data, sums.ctypes.data, counts.ctypes.data,
        )
        if groups >= 0:
            break
        capacity <<= 1
        if capacity > 1 << 31:
            raise MemoryError("group_by: too many distinct keys")
    return (
        pa.array(group_keys[:groups]),
        pa.array(sums[:groups]),
        pa.array(counts[:groups]),
    )
