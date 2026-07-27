"""C ABI for the Arrow kernels.

Buffer addresses cross as `Int`, and 0 means "absent" — an Arrow array with no
nulls carries no validity bitmap at all, and that is the fast path rather than
an edge case.
"""

from mojoarrow.bitmap import Bits, copy_bitmap, count_valid
from mojoarrow.compute import (
    F64,
    I64,
    argsort_f64,
    cast_f64_i64,
    cast_i64_f64,
    compare_f64,
    fill_null_f64,
    filter_f64,
    filter_i64,
    group_by_i64,
    group_sum_f64,
    group_sum_fused_i64,
    max_f64,
    max_i64,
    min_f64,
    min_i64,
    sum_f64,
    sum_i64,
    take_f64,
    take_i64,
    variance_f64,
)


def f64(addr: Int) -> F64:
    return F64(unsafe_from_address=addr)


def i64(addr: Int) -> I64:
    return I64(unsafe_from_address=addr)


def bits(addr: Int) -> Bits:
    return Bits(unsafe_from_address=addr)


# ---------------------------------------------------------------- bitmaps
@export("ma_count_valid")
def ma_count_valid(bitmap: Int, offset: Int, n: Int) abi("C") -> Int:
    return count_valid(bitmap, offset, n)


@export("ma_copy_bitmap")
def ma_copy_bitmap(src: Int, src_offset: Int, n: Int, dst: Int) abi("C"):
    copy_bitmap(bits(src), src_offset, n, bits(dst))


# ------------------------------------------------------------ aggregation
@export("ma_sum_f64")
def ma_sum_f64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Float64:
    return sum_f64(f64(values), bitmap, offset, n)


@export("ma_sum_i64")
def ma_sum_i64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Int64:
    return sum_i64(i64(values), bitmap, offset, n)


@export("ma_min_f64")
def ma_min_f64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Float64:
    return min_f64(f64(values), bitmap, offset, n)


@export("ma_max_f64")
def ma_max_f64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Float64:
    return max_f64(f64(values), bitmap, offset, n)


@export("ma_min_i64")
def ma_min_i64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Int64:
    return min_i64(i64(values), bitmap, offset, n)


@export("ma_max_i64")
def ma_max_i64(values: Int, bitmap: Int, offset: Int, n: Int) abi("C") -> Int64:
    return max_i64(i64(values), bitmap, offset, n)


@export("ma_variance_f64")
def ma_variance_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, ddof: Int
) abi("C") -> Float64:
    return variance_f64(f64(values), bitmap, offset, n, ddof)


# ----------------------------------------------------------------- filter
@export("ma_filter_f64")
def ma_filter_f64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    mask: Int, mask_bitmap: Int, mask_offset: Int,
    dst: Int, dst_bitmap: Int,
) abi("C") -> Int:
    return filter_f64(
        f64(values), bitmap, offset, n, bits(mask), mask_bitmap, mask_offset,
        f64(dst), bits(dst_bitmap),
    )


@export("ma_filter_i64")
def ma_filter_i64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    mask: Int, mask_bitmap: Int, mask_offset: Int,
    dst: Int, dst_bitmap: Int,
) abi("C") -> Int:
    return filter_i64(
        i64(values), bitmap, offset, n, bits(mask), mask_bitmap, mask_offset,
        i64(dst), bits(dst_bitmap),
    )


# ------------------------------------------------------------------- take
@export("ma_take_f64")
def ma_take_f64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    indices: Int, m: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    take_f64(f64(values), bitmap, offset, n, i64(indices), m, f64(dst), bits(dst_bitmap))


@export("ma_take_i64")
def ma_take_i64(
    values: Int, bitmap: Int, offset: Int, n: Int,
    indices: Int, m: Int, dst: Int, dst_bitmap: Int,
) abi("C"):
    take_i64(i64(values), bitmap, offset, n, i64(indices), m, i64(dst), bits(dst_bitmap))


# ------------------------------------------------------------- comparison
@export("ma_compare_f64")
def ma_compare_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, scalar: Float64, op: Int,
    dst: Int, dst_bitmap: Int,
) abi("C"):
    compare_f64(
        f64(values), bitmap, offset, n, scalar, op, bits(dst), bits(dst_bitmap)
    )


# ------------------------------------------------------------- transforms
@export("ma_fill_null_f64")
def ma_fill_null_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, fill: Float64, dst: Int
) abi("C"):
    fill_null_f64(f64(values), bitmap, offset, n, fill, f64(dst))


@export("ma_cast_i64_f64")
def ma_cast_i64_f64(values: Int, offset: Int, n: Int, dst: Int) abi("C"):
    cast_i64_f64(i64(values), offset, n, f64(dst))


@export("ma_cast_f64_i64")
def ma_cast_f64_i64(values: Int, offset: Int, n: Int, dst: Int) abi("C"):
    cast_f64_i64(f64(values), offset, n, i64(dst))


# ---------------------------------------------------------------- sorting
@export("ma_argsort_f64")
def ma_argsort_f64(
    values: Int, bitmap: Int, offset: Int, n: Int, indices: Int, descending: Int
) abi("C"):
    argsort_f64(f64(values), bitmap, offset, n, i64(indices), descending != 0)


# --------------------------------------------------------------- group by
@export("ma_group_by_i64")
def ma_group_by_i64(
    keys: Int, bitmap: Int, offset: Int, n: Int,
    table_keys: Int, table_slot: Int, capacity: Int,
    group_of_row: Int, group_keys: Int,
) abi("C") -> Int:
    return group_by_i64(
        i64(keys), bitmap, offset, n, i64(table_keys), i64(table_slot), capacity,
        i64(group_of_row), i64(group_keys),
    )


@export("ma_group_sum_f64")
def ma_group_sum_f64(
    group_of_row: Int, values: Int, bitmap: Int, offset: Int, n: Int,
    sums: Int, counts: Int, groups: Int,
) abi("C"):
    group_sum_f64(
        i64(group_of_row), f64(values), bitmap, offset, n, f64(sums), i64(counts),
        groups,
    )


@export("ma_group_sum_fused_i64")
def ma_group_sum_fused_i64(
    keys: Int, key_bitmap: Int, key_offset: Int,
    values: Int, value_bitmap: Int, value_offset: Int, n: Int,
    table_keys: Int, table_slot: Int, capacity: Int,
    group_keys: Int, sums: Int, counts: Int,
) abi("C") -> Int:
    return group_sum_fused_i64(
        i64(keys), key_bitmap, key_offset, f64(values), value_bitmap, value_offset,
        n, i64(table_keys), i64(table_slot), capacity, i64(group_keys), f64(sums),
        i64(counts),
    )
