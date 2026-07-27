"""Loading the compiled kernels, and reading Arrow buffers without copying.

A pyarrow array is already exactly the layout the Mojo kernels want: a
validity bitmap, a values buffer, a length and an offset. `columns()` reads
those four numbers straight off the array — no conversion, no copy, no
`to_numpy()` — and hands the addresses across the C ABI.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import pyarrow as pa

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.path.join(ROOT, "build", "capi.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "ma_count_valid": ([I, I, I], I),
    "ma_copy_bitmap": ([I, I, I, I], None),
    "ma_sum_f64": ([I, I, I, I], F),
    "ma_sum_i64": ([I, I, I, I], I),
    "ma_min_f64": ([I, I, I, I], F),
    "ma_max_f64": ([I, I, I, I], F),
    "ma_min_i64": ([I, I, I, I], I),
    "ma_max_i64": ([I, I, I, I], I),
    "ma_variance_f64": ([I, I, I, I, I], F),
    "ma_filter_f64": ([I] * 9, I),
    "ma_filter_i64": ([I] * 9, I),
    "ma_take_f64": ([I] * 8, None),
    "ma_take_i64": ([I] * 8, None),
    "ma_compare_f64": ([I, I, I, I, F, I, I, I], None),
    "ma_fill_null_f64": ([I, I, I, I, F, I], None),
    "ma_cast_i64_f64": ([I, I, I, I], None),
    "ma_cast_f64_i64": ([I, I, I, I], None),
    "ma_argsort_f64": ([I, I, I, I, I, I], None),
    "ma_group_by_i64": ([I] * 9, I),
    "ma_group_sum_f64": ([I] * 8, None),
    "ma_group_sum_fused_i64": ([I] * 13, I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOARROW_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOARROW_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(s) for s in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


@dataclass(frozen=True)
class Column:
    """The four numbers a kernel needs: bitmap, values, offset, length.

    `bitmap` is 0 when the array has no nulls, which Arrow represents by
    omitting the buffer entirely — the kernels treat that as the fast path.
    """

    array: pa.Array
    bitmap: int
    values: int
    offset: int
    length: int
    null_count: int

    @property
    def has_nulls(self) -> bool:
        return self.bitmap != 0


def column(array) -> Column:
    """Read an Arrow array's buffers. Chunked arrays are combined first."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if not isinstance(array, pa.Array):
        array = pa.array(array)
    buffers = array.buffers()
    validity = buffers[0]
    values = buffers[1]
    # A validity buffer can be present but the array still have no nulls; in
    # that case skip it, because the branchless path is faster and correct.
    bitmap = (
        validity.address if (validity is not None and array.null_count > 0) else 0
    )
    return Column(
        array=array,
        bitmap=bitmap,
        values=values.address if values is not None else 0,
        offset=array.offset,
        length=len(array),
        null_count=array.null_count,
    )


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
