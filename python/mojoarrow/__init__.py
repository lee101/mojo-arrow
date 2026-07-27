"""mojo-arrow — Apache Arrow compute kernels written in Mojo.

    import pyarrow as pa
    import mojoarrow as ma

    col = pa.array([1.0, 2.0, None, 4.0])
    ma.sum(col)              # 7.0, nulls skipped
    ma.filter(col, ma.compare(col, ">", 1.5))

Arrays cross into Mojo as four integers — validity bitmap, values buffer,
offset, length — read straight off the pyarrow array. Nothing is converted and
nothing is copied, including on a slice.
"""

from . import compute
from ._lib import Column, build, column
from .compute import (
    cast,
    compare,
    count_valid,
    fill_null,
    filter,
    group_by_sum,
    max,
    mean,
    min,
    sort_indices,
    stddev,
    sum,
    take,
    variance,
)

__version__ = "0.1.0"
__all__ = [
    "sum", "min", "max", "mean", "variance", "stddev", "count_valid",
    "filter", "take", "compare", "fill_null", "cast", "sort_indices",
    "group_by_sum",
    "column", "Column", "build", "compute",
]
