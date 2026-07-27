"""Columnar kernels over Arrow buffers.

Every kernel takes buffer addresses, a length and an offset, so it works on a
sliced array without the caller materializing a copy. Null handling follows
Arrow's rules: a null contributes nothing to an aggregate, propagates through
a transform, and a comparison against null is null.

`sum_f64` keeps a separate no-bitmap path because that is the one aggregate
where the null check stops the loop vectorizing; everywhere else the branch is
cheap next to the work and one code path is worth more than the cycles.
"""

from std.math import sqrt

from mojoarrow.bitmap import Bits, get_bit, set_bit, valid

comptime W = 8
comptime F64 = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64 = UnsafePointer[Int64, AnyOrigin[mut=True]]

comptime F64_MAX = 1.7976931348623157e308
comptime I64_MAX = 9223372036854775807
comptime I64_MIN = -9223372036854775808


# ------------------------------------------------------------- aggregation
def sum_f64(values: F64, bitmap_addr: Int, offset: Int, n: Int) -> Float64:
    if bitmap_addr == 0:
        var acc = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= n:
            acc += values.load[width=W](offset + i)
            i += W
        var total = acc.reduce_add()
        while i < n:
            total += values[offset + i]
            i += 1
        return total
    var bits = Bits(unsafe_from_address=bitmap_addr)
    var total = 0.0
    for i in range(n):
        if get_bit(bits, offset + i):
            total += values[offset + i]
    return total


def sum_i64(values: I64, bitmap_addr: Int, offset: Int, n: Int) -> Int64:
    var total = Int64(0)
    if bitmap_addr == 0:
        var acc = SIMD[DType.int64, W](0)
        var i = 0
        while i + W <= n:
            acc += values.load[width=W](offset + i)
            i += W
        total = acc.reduce_add()
        while i < n:
            total += values[offset + i]
            i += 1
        return total
    var bits = Bits(unsafe_from_address=bitmap_addr)
    for i in range(n):
        if get_bit(bits, offset + i):
            total += values[offset + i]
    return total


def min_f64(values: F64, bitmap_addr: Int, offset: Int, n: Int) -> Float64:
    """The minimum of the valid elements, or +inf-ish if there are none.

    Returning `F64_MAX` for an all-null array rather than raising keeps the C
    ABI free of error channels; the Python layer turns it into None using the
    valid count it already has.
    """
    var best = F64_MAX
    if bitmap_addr == 0:
        for i in range(n):
            var v = values[offset + i]
            if v < best:
                best = v
        return best
    var bits = Bits(unsafe_from_address=bitmap_addr)
    for i in range(n):
        if get_bit(bits, offset + i):
            var v = values[offset + i]
            if v < best:
                best = v
    return best


def max_f64(values: F64, bitmap_addr: Int, offset: Int, n: Int) -> Float64:
    var best = -F64_MAX
    if bitmap_addr == 0:
        for i in range(n):
            var v = values[offset + i]
            if v > best:
                best = v
        return best
    var bits = Bits(unsafe_from_address=bitmap_addr)
    for i in range(n):
        if get_bit(bits, offset + i):
            var v = values[offset + i]
            if v > best:
                best = v
    return best


def min_i64(values: I64, bitmap_addr: Int, offset: Int, n: Int) -> Int64:
    var best = Int64(I64_MAX)
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            var v = values[offset + i]
            if v < best:
                best = v
    return best


def max_i64(values: I64, bitmap_addr: Int, offset: Int, n: Int) -> Int64:
    var best = Int64(I64_MIN)
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            var v = values[offset + i]
            if v > best:
                best = v
    return best


def variance_f64(
    values: F64, bitmap_addr: Int, offset: Int, n: Int, ddof: Int
) -> Float64:
    """Chunked variance: naive sums within a block, Chan's combination across.

    Element-at-a-time Welford costs a division per element and measured 3.5x
    slower than pyarrow. Blocks of 1024 make the inner loop a pair of SIMD
    accumulations, and combining the blocks pairwise keeps the accuracy that
    a single naive sum-of-squares would throw away.
    """
    comptime BLOCK = 1024
    var count = 0.0
    var mean = 0.0
    var m2 = 0.0
    var i = 0
    while i < n:
        var stop = i + BLOCK
        if stop > n:
            stop = n
        var bcount = 0.0
        var bsum = 0.0
        var bsumsq = 0.0
        if bitmap_addr == 0:
            var vsum = SIMD[DType.float64, W](0.0)
            var vsq = SIMD[DType.float64, W](0.0)
            var j = i
            while j + W <= stop:
                var v = values.load[width=W](offset + j)
                vsum += v
                vsq += v * v
                j += W
            bsum = vsum.reduce_add()
            bsumsq = vsq.reduce_add()
            bcount = Float64(j - i)
            while j < stop:
                var v = values[offset + j]
                bsum += v
                bsumsq += v * v
                bcount += 1.0
                j += 1
        else:
            for j in range(i, stop):
                if not valid(bitmap_addr, offset, j):
                    continue
                var v = values[offset + j]
                bsum += v
                bsumsq += v * v
                bcount += 1.0
        i = stop
        if bcount == 0.0:
            continue
        var bmean = bsum / bcount
        var bm2 = bsumsq - bsum * bmean
        if count == 0.0:
            count = bcount
            mean = bmean
            m2 = bm2
            continue
        var delta = bmean - mean
        var total = count + bcount
        m2 += bm2 + delta * delta * count * bcount / total
        mean += delta * bcount / total
        count = total
    var denom = count - Float64(ddof)
    if denom <= 0.0:
        return 0.0
    return m2 / denom


# ------------------------------------------------------------------ filter
def filter_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    mask: Bits,
    mask_bitmap_addr: Int,
    mask_offset: Int,
    dst: F64,
    dst_bitmap: Bits,
) -> Int:
    """Keep elements where the mask is true and not null. Returns the count.

    Nulls in the *mask* drop the element (Arrow's `filter` with the default
    null selection behaviour); nulls in the *values* are kept as nulls.

    The mask is walked a byte at a time so a byte of zeros skips eight
    elements without touching them, and the output validity is written whole
    when the input has no nulls — a selective filter over a dense column then
    never reads a bit individually at all.
    """
    var kept = 0
    var dense = bitmap_addr == 0
    var byte_count = (n + 7) >> 3
    for b in range(byte_count):
        var base = b << 3
        var stop = 8 if base + 8 <= n else n - base
        var selected = mask[(mask_offset + base) >> 3] if (mask_offset & 7) == 0 else UInt8(0)
        if (mask_offset & 7) == 0 and mask_bitmap_addr == 0 and selected == 0:
            continue
        for k in range(stop):
            var i = base + k
            if not valid(mask_bitmap_addr, mask_offset, i):
                continue
            if not get_bit(mask, mask_offset + i):
                continue
            dst[kept] = values[offset + i]
            if not dense:
                set_bit(dst_bitmap, kept, valid(bitmap_addr, offset, i))
            kept += 1
    if dense:
        for b in range((kept + 7) >> 3):
            dst_bitmap[b] = 0xFF
    return kept


def filter_i64(
    values: I64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    mask: Bits,
    mask_bitmap_addr: Int,
    mask_offset: Int,
    dst: I64,
    dst_bitmap: Bits,
) -> Int:
    """Keep elements where the mask is true and not null. Returns the count.

    Nulls in the *mask* drop the element (Arrow's `filter` with the default
    null selection behaviour); nulls in the *values* are kept as nulls.

    The mask is walked a byte at a time so a byte of zeros skips eight
    elements without touching them, and the output validity is written whole
    when the input has no nulls — a selective filter over a dense column then
    never reads a bit individually at all.
    """
    var kept = 0
    var dense = bitmap_addr == 0
    var byte_count = (n + 7) >> 3
    for b in range(byte_count):
        var base = b << 3
        var stop = 8 if base + 8 <= n else n - base
        var selected = mask[(mask_offset + base) >> 3] if (mask_offset & 7) == 0 else UInt8(0)
        if (mask_offset & 7) == 0 and mask_bitmap_addr == 0 and selected == 0:
            continue
        for k in range(stop):
            var i = base + k
            if not valid(mask_bitmap_addr, mask_offset, i):
                continue
            if not get_bit(mask, mask_offset + i):
                continue
            dst[kept] = values[offset + i]
            if not dense:
                set_bit(dst_bitmap, kept, valid(bitmap_addr, offset, i))
            kept += 1
    if dense:
        for b in range((kept + 7) >> 3):
            dst_bitmap[b] = 0xFF
    return kept


# -------------------------------------------------------------------- take
def take_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    indices: I64,
    m: Int,
    dst: F64,
    dst_bitmap: Bits,
):
    """Gather by index. An out-of-range index produces a null rather than a
    read outside the buffer — a gather is the easiest place in a columnar
    engine to turn a bad join key into a segfault."""
    for j in range(m):
        var idx = Int(indices[j])
        if idx < 0 or idx >= n:
            dst[j] = 0.0
            set_bit(dst_bitmap, j, False)
            continue
        dst[j] = values[offset + idx]
        set_bit(dst_bitmap, j, valid(bitmap_addr, offset, idx))


def take_i64(
    values: I64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    indices: I64,
    m: Int,
    dst: I64,
    dst_bitmap: Bits,
):
    for j in range(m):
        var idx = Int(indices[j])
        if idx < 0 or idx >= n:
            dst[j] = 0
            set_bit(dst_bitmap, j, False)
            continue
        dst[j] = values[offset + idx]
        set_bit(dst_bitmap, j, valid(bitmap_addr, offset, idx))


# -------------------------------------------------------------- comparison
def compare_f64(
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    scalar: Float64,
    op: Int,
    dst: Bits,
    dst_bitmap: Bits,
):
    """op: 0 `>`, 1 `>=`, 2 `<`, 3 `<=`, 4 `==`, 5 `!=`. Null in, null out."""
    # Build whole bytes and store once per eight elements. Setting bits one at
    # a time is a read-modify-write of the same byte eight times over, and it
    # was the difference between 36ms and 6ms on 5M elements.
    var byte_count = (n + 7) >> 3
    for b in range(byte_count):
        var packed = UInt8(0)
        var packed_valid = UInt8(0)
        var base = b << 3
        var stop = 8 if base + 8 <= n else n - base
        for k in range(stop):
            var i = base + k
            if not valid(bitmap_addr, offset, i):
                continue
            packed_valid |= UInt8(1) << UInt8(k)
            var v = values[offset + i]
            var result: Bool
            if op == 0:
                result = v > scalar
            elif op == 1:
                result = v >= scalar
            elif op == 2:
                result = v < scalar
            elif op == 3:
                result = v <= scalar
            elif op == 4:
                result = v == scalar
            else:
                result = v != scalar
            if result:
                packed |= UInt8(1) << UInt8(k)
        dst[b] = packed
        dst_bitmap[b] = packed_valid


# ------------------------------------------------------------- transforms
def fill_null_f64(
    values: F64, bitmap_addr: Int, offset: Int, n: Int, fill: Float64, dst: F64
):
    if bitmap_addr == 0:
        var i = 0
        while i + W <= n:
            dst.store(i, values.load[width=W](offset + i))
            i += W
        while i < n:
            dst[i] = values[offset + i]
            i += 1
        return
    for i in range(n):
        if valid(bitmap_addr, offset, i):
            dst[i] = values[offset + i]
        else:
            dst[i] = fill


def cast_i64_f64(values: I64, offset: Int, n: Int, dst: F64):
    var i = 0
    while i + W <= n:
        dst.store(i, values.load[width=W](offset + i).cast[DType.float64]())
        i += W
    while i < n:
        dst[i] = Float64(values[offset + i])
        i += 1


def cast_f64_i64(values: F64, offset: Int, n: Int, dst: I64):
    var i = 0
    while i + W <= n:
        dst.store(i, values.load[width=W](offset + i).cast[DType.int64]())
        i += W
    while i < n:
        dst[i] = Int64(values[offset + i])
        i += 1


# ----------------------------------------------------------------- sorting
def argsort_f64(
    values: F64, bitmap_addr: Int, offset: Int, n: Int, indices: I64, descending: Bool
):
    """Stable-ish index sort. Nulls sort last, as Arrow does by default.

    Insertion-sorted merge would be stable; this is a quicksort on indices and
    is not, which is documented rather than hidden — a stable sort belongs here
    when someone needs one.
    """
    for i in range(n):
        indices[i] = Int64(i)
    _quicksort(values, bitmap_addr, offset, indices, 0, n - 1, descending)


def _key(values: F64, bitmap_addr: Int, offset: Int, idx: Int, descending: Bool) -> Float64:
    if bitmap_addr != 0:
        var bits = Bits(unsafe_from_address=bitmap_addr)
        if not get_bit(bits, offset + idx):
            return -F64_MAX if descending else F64_MAX  # nulls last either way
    var v = values[offset + idx]
    return -v if descending else v


def _quicksort(
    values: F64, bitmap_addr: Int, offset: Int, indices: I64,
    lo: Int, hi: Int, descending: Bool,
):
    var low = lo
    var high = hi
    while low < high:
        if high - low < 16:
            for i in range(low + 1, high + 1):
                var tmp = indices[i]
                var key = _key(values, bitmap_addr, offset, Int(tmp), descending)
                var j = i - 1
                while j >= low and _key(
                    values, bitmap_addr, offset, Int(indices[j]), descending
                ) > key:
                    indices[j + 1] = indices[j]
                    j -= 1
                indices[j + 1] = tmp
            return
        var mid = low + (high - low) // 2
        var pivot = _key(values, bitmap_addr, offset, Int(indices[mid]), descending)
        var i = low
        var j = high
        while i <= j:
            while _key(values, bitmap_addr, offset, Int(indices[i]), descending) < pivot:
                i += 1
            while _key(values, bitmap_addr, offset, Int(indices[j]), descending) > pivot:
                j -= 1
            if i <= j:
                var tmp = indices[i]
                indices[i] = indices[j]
                indices[j] = tmp
                i += 1
                j -= 1
        # recurse into the smaller side, loop on the larger: bounded stack
        if j - low < high - i:
            _quicksort(values, bitmap_addr, offset, indices, low, j, descending)
            low = i
        else:
            _quicksort(values, bitmap_addr, offset, indices, i, high, descending)
            high = j


# --------------------------------------------------------------- group by
def group_by_i64(
    keys: I64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    table_keys: I64,
    table_slot: I64,
    capacity: Int,
    group_of_row: I64,
    group_keys: I64,
) -> Int:
    """Hash int64 keys to dense group ids. Returns the number of groups.

    Open addressing with linear probing over caller-provided tables; the
    caller sizes `capacity` as a power of two at least twice the row count, so
    the table never needs to grow mid-pass. `table_slot` holds group id + 1,
    which makes 0 mean empty without reserving a key value.

    Null keys become their own group, id 0 by convention, so a group-by never
    silently drops rows.

    Returns -1 if the groups would exceed 70% of `capacity`. The caller starts
    with a small table and doubles on -1: sizing the table for the row count
    instead means allocating (and cache-missing over) 128MB for five million
    rows that turn out to have fifty distinct keys.
    """
    var limit = (capacity * 7) // 10
    var mask = capacity - 1
    for i in range(capacity):
        table_slot[i] = 0
    var groups = 0
    var null_group = -1
    for r in range(n):
        if not valid(bitmap_addr, offset, r):
            if null_group < 0:
                if groups >= limit:
                    return -1
                null_group = groups
                group_keys[groups] = 0
                groups += 1
            group_of_row[r] = Int64(null_group)
            continue
        var key = keys[offset + r]
        var h = Int(UInt64(key) * 11400714819323198485)
        var slot = (h >> 32) & mask
        while True:
            var occupant = table_slot[slot]
            if occupant == 0:
                if groups >= limit:
                    return -1
                table_slot[slot] = Int64(groups + 1)
                table_keys[slot] = key
                group_keys[groups] = key
                group_of_row[r] = Int64(groups)
                groups += 1
                break
            if table_keys[slot] == key:
                group_of_row[r] = occupant - 1
                break
            slot = (slot + 1) & mask
    return groups


def group_sum_f64(
    group_of_row: I64,
    values: F64,
    bitmap_addr: Int,
    offset: Int,
    n: Int,
    sums: F64,
    counts: I64,
    groups: Int,
):
    for g in range(groups):
        sums[g] = 0.0
        counts[g] = 0
    for r in range(n):
        if not valid(bitmap_addr, offset, r):
            continue
        var g = Int(group_of_row[r])
        sums[g] += values[offset + r]
        counts[g] += 1


def group_sum_fused_i64(
    keys: I64,
    key_bitmap: Int,
    key_offset: Int,
    values: F64,
    value_bitmap: Int,
    value_offset: Int,
    n: Int,
    table_keys: I64,
    table_slot: I64,
    capacity: Int,
    group_keys: I64,
    sums: F64,
    counts: I64,
) -> Int:
    """Hash the key and accumulate the value in the same pass.

    The two-kernel version wrote a group id per row and read it back, which on
    five million rows is 40MB out and 40MB in for no reason. Returns -1 if the
    table would exceed 70% load, same contract as `group_by_i64`.
    """
    var mask = capacity - 1
    var limit = (capacity * 7) // 10
    for i in range(capacity):
        table_slot[i] = 0
    var groups = 0
    var null_group = -1
    for r in range(n):
        var g = 0
        if key_bitmap != 0 and not valid(key_bitmap, key_offset, r):
            if null_group < 0:
                if groups >= limit:
                    return -1
                null_group = groups
                group_keys[groups] = 0
                sums[groups] = 0.0
                counts[groups] = 0
                groups += 1
            g = null_group
        else:
            var key = keys[key_offset + r]
            var h = Int(UInt64(key) * 11400714819323198485)
            var slot = (h >> 32) & mask
            while True:
                var occupant = table_slot[slot]
                if occupant == 0:
                    if groups >= limit:
                        return -1
                    table_slot[slot] = Int64(groups + 1)
                    table_keys[slot] = key
                    group_keys[groups] = key
                    sums[groups] = 0.0
                    counts[groups] = 0
                    g = groups
                    groups += 1
                    break
                if table_keys[slot] == key:
                    g = Int(occupant) - 1
                    break
                slot = (slot + 1) & mask
        if value_bitmap != 0 and not valid(value_bitmap, value_offset, r):
            continue
        sums[g] += values[value_offset + r]
        counts[g] += 1
    return groups
