# mojo-arrow

Apache Arrow compute kernels written in [Mojo](https://www.modular.com/mojo),
callable from Python against real pyarrow arrays with no copy.

```python
import pyarrow as pa
import mojoarrow as ma

col = pa.array([1.0, 2.0, None, 4.0])
ma.sum(col)                              # 7.0 — nulls skipped
ma.filter(col, ma.compare(col, ">", 1.5))
keys, sums, counts = ma.group_by_sum(key_column, value_column)
```

An Arrow array is already the layout these kernels want: a validity bitmap, a
values buffer, an offset and a length. mojo-arrow reads those four numbers off
the pyarrow array and passes the addresses across the C ABI — no `to_numpy()`,
no conversion, and a slice costs nothing because the offset goes with it.
Results come back as real Arrow arrays built over Arrow-pool buffers.

## What is implemented

| group | kernels |
| --- | --- |
| aggregation | `sum`, `min`, `max`, `mean`, `variance`, `stddev`, `count_valid` (float64 and int64) |
| selection | `filter`, `take`, `compare` (`>`, `>=`, `<`, `<=`, `==`, `!=`) |
| transform | `fill_null`, `cast` (int64 ↔ float64) |
| ordering | `sort_indices`, ascending or descending, nulls last |
| grouping | `group_by_sum` — hash aggregation over int64 keys |

Every one is tested against `pyarrow.compute` on the same arrays, including
against **sliced** and **null-carrying** inputs, which is where a columnar
kernel goes quietly wrong: a dense unsliced array works even when the offset
handling is broken.

## Performance

Against pyarrow on 5M float64 elements. pyarrow's kernels are C++ with hand
tuned SIMD and a thread pool behind several of them; this library is
single-threaded. The machine was under load during these runs, so treat the
ratios as directional and reproduce them with `pixi run bench`.

| kernel | mojo-arrow | pyarrow | |
| --- | ---: | ---: | --- |
| `min` | 8.8 ms | 39.0 ms | **4.4x faster** |
| `variance` | 7.2 ms | 18.2 ms | **2.6x faster** |
| `cast` int64→float64 | 11.2 ms | 17.4 ms | **1.5x faster** |
| `sum`, 10% nulls | 20.4 ms | 28.2 ms | **1.4x faster** |
| `sum`, no nulls | 5.6 ms | 7.0 ms | 1.3x faster |
| `sort_indices` (200k) | 26.1 ms | 34.5 ms | 1.3x faster |
| `sum` int64 | 6.1 ms | 6.4 ms | 1.1x faster |
| `fill_null` | 20.2 ms | 18.2 ms | 0.9x |
| `take` (500k of 5M) | 22.4 ms | 13.3 ms | 0.6x slower |
| `filter` | 58.9 ms | 32.1 ms | 0.5x slower |
| `compare > 0` | 19.2 ms | 9.9 ms | 0.5x slower |
| `group_by_sum`, 50 keys | 30.3 ms | 11.3 ms | 0.4x slower |

Three optimizations mattered more than any Mojo-level tuning, and they are the
lesson of the project so far:

1. **Allocate from Arrow's pool, not numpy.** Every transform kernel writes a
   fresh output buffer, and a fresh numpy allocation takes a page fault per
   4KB on first touch. Switching the outputs to `pa.allocate_buffer` took
   `cast` from 52 ms to 16 ms without changing a line of Mojo.
2. **Write whole bytes, not bits.** `compare` set two bits per element, each a
   read-modify-write of the same byte eight times over: 36 ms. Packing a byte
   and storing it once per eight elements: 10 ms.
3. **Do not size a hash table by the row count.** Sizing the group-by table at
   2x the rows meant a 128MB allocation and a cache miss per probe for a query
   with fifty distinct keys — 225 ms. Starting at 1024 slots and returning -1
   on overload so the caller doubles: 30–50 ms. Fusing the hash and the
   accumulation into one pass removed another 80MB of traffic.

The remaining losses are all the same loss: pyarrow uses more than one core.
Threading these kernels is the next piece of work, not a rewrite.

## Install

```bash
pixi install
pixi run test     # parity against pyarrow.compute
pixi run bench
```

The shared library builds itself on first import and rebuilds when any `.mojo`
file is newer. To force it: `python -m mojoarrow._lib --force`. Outside pixi,
set `MOJOARROW_MOJO=/path/to/mojo`.

## Design

```
python/mojoarrow/   pyarrow in, pyarrow out; reads buffer addresses, no copies
        │  ctypes, one call per kernel
src/capi.mojo       @export ... abi("C") wrappers, Int addresses, 0 = absent
src/mojoarrow/      bitmap.mojo (validity), compute.mojo (the kernels)
```

- **Nothing in Mojo allocates.** Output buffers, scratch and hash tables all
  come from the caller, so lifetimes stay in Python where Arrow already
  manages them.
- **A missing validity bitmap is the fast path, not an edge case.** Arrow omits
  the buffer when an array has no nulls; the kernels branch on that once per
  call and then run without a per-element null check.
- **Offsets are respected everywhere.** Sliced arrays share their parent's
  buffers, and the offset applies to the *bit* index in the bitmap as well as
  the element index in the values.
- **An out-of-range `take` index yields null**, not a read past the end of the
  buffer.

## Not yet

String and binary layouts (offsets buffers), chunked arrays without combining,
dictionary encoding, joins, multi-column group-by, IPC reading, and threading.
`float32`/`int32` widths are absent for the same reason as `float64`/`int64`
are present: those two carry most analytic columns, and every added width is
another monomorphic export.

## License

MIT
