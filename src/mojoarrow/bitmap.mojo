"""Arrow validity bitmaps.

Arrow stores nullability as a packed little-endian bitmap: bit `i` of byte
`i // 8` is 1 when element `i` is valid. A null array may omit the bitmap
entirely, which is why every kernel here takes the bitmap address as an `Int`
and treats 0 as "everything is valid" — that case is both the common one and
the fast one, and branching on it once per array beats testing per element.

Arrays also carry an offset (a slice of a longer buffer shares its parent's
memory), and the offset applies to the *bit* index, not the byte. Getting that
wrong gives a result that is right for unsliced arrays and quietly wrong for
sliced ones, which is the worst kind of wrong.
"""

comptime Bits = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def get_bit(bitmap: Bits, i: Int) -> Bool:
    return ((bitmap[i >> 3] >> UInt8(i & 7)) & 1) != 0


def set_bit(bitmap: Bits, i: Int, value: Bool):
    var byte = i >> 3
    var mask = UInt8(1) << UInt8(i & 7)
    if value:
        bitmap[byte] |= mask
    else:
        bitmap[byte] &= ~mask


def valid(bitmap_addr: Int, offset: Int, i: Int) -> Bool:
    """Whether element `i` of an array with this bitmap and offset is valid."""
    if bitmap_addr == 0:
        return True
    return get_bit(Bits(unsafe_from_address=bitmap_addr), offset + i)


def count_valid(bitmap_addr: Int, offset: Int, n: Int) -> Int:
    if bitmap_addr == 0:
        return n
    var bits = Bits(unsafe_from_address=bitmap_addr)
    var total = 0
    # The aligned middle is counted a byte at a time with popcount; only the
    # two ragged ends go bit by bit.
    var start = offset
    var stop = offset + n
    var first_byte = (start + 7) >> 3
    var last_byte = stop >> 3
    if first_byte > last_byte:
        for i in range(start, stop):
            if get_bit(bits, i):
                total += 1
        return total
    for i in range(start, first_byte << 3):
        if get_bit(bits, i):
            total += 1
    for b in range(first_byte, last_byte):
        total += Int(pop_count(bits[b]))
    for i in range(last_byte << 3, stop):
        if get_bit(bits, i):
            total += 1
    return total


def pop_count(byte: UInt8) -> UInt8:
    var v = byte
    var count = UInt8(0)
    while v != 0:
        count += v & 1
        v >>= 1
    return count


def fill_valid(bitmap: Bits, n: Int):
    var bytes = (n + 7) >> 3
    for b in range(bytes):
        bitmap[b] = 0xFF


def copy_bitmap(src: Bits, src_offset: Int, n: Int, dst: Bits):
    """Re-base a bitmap so element `src_offset` becomes element 0.

    A kernel that writes a fresh values buffer has dropped the parent's offset;
    its validity has to be re-based to match or the nulls land on the wrong
    rows — which only shows up on sliced arrays.
    """
    for i in range(n):
        set_bit(dst, i, get_bit(src, src_offset + i))
