"""Block-level KV paging simulator. CPU swap stores metadata, not tensor data."""
from __future__ import annotations
import math
from dataclasses import dataclass

BLOCK_SIZE = 16

def blocks_for(tokens: int) -> int:
    if tokens < 0:
        raise ValueError("negative token count")
    return math.ceil(tokens / BLOCK_SIZE)

class AllocError(RuntimeError):
    pass


class BitmapFramePool:
    """O(1) fixed-frame allocation with an explicit free bitmap.

    The stack identifies the next free frame; the bitmap is the authoritative
    free/occupied state and rejects duplicate or out-of-range returns.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("invalid frame capacity")
        self.capacity = capacity
        self.bitmap = bytearray([0xFF]) * ((capacity + 7) // 8)
        if capacity % 8:
            self.bitmap[-1] &= (1 << (capacity % 8)) - 1
        self._free_stack = list(range(capacity))

    def __len__(self) -> int:
        return len(self._free_stack)

    def __iter__(self):
        return iter(self._free_stack)

    def pop(self) -> int:
        if not self._free_stack:
            raise AllocError("no free GPU frames")
        frame = self._free_stack.pop()
        if not self.is_free(frame):
            raise RuntimeError("bitmap/free-stack disagreement")
        self.bitmap[frame // 8] &= ~(1 << (frame % 8))
        return frame

    def append(self, frame: int) -> None:
        if not isinstance(frame, int) or not 0 <= frame < self.capacity:
            raise ValueError("frame outside pool")
        if self.is_free(frame):
            raise ValueError("frame already free")
        self.bitmap[frame // 8] |= 1 << (frame % 8)
        self._free_stack.append(frame)

    def is_free(self, frame: int) -> bool:
        if not 0 <= frame < self.capacity:
            raise IndexError("frame outside pool")
        return bool(self.bitmap[frame // 8] & (1 << (frame % 8)))

@dataclass
class Block:
    block_id: int
    frame: int | None
    refs: int = 1
    last_access: int = 0

class PageTable:
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.entries: dict[int, int] = {}  # logical block -> stable block id
        self.tokens = 0

    def map(self, logical: int, block_id: int) -> None:
        self.entries[logical] = block_id

    def physical_of(self, logical: int) -> int | None:
        """Legacy API: returns a stable block id, not a GPU frame."""
        return self.entries.get(logical)

    def physical_blocks(self) -> list[int]:
        return list(self.entries.values())

class KVBlockAllocator:
    def __init__(self, capacity_blocks: int, cpu_capacity_blocks: int | None = None):
        if capacity_blocks <= 0 or (cpu_capacity_blocks is not None and cpu_capacity_blocks < 0):
            raise ValueError("invalid capacity")
        self.capacity = capacity_blocks
        self.cpu_capacity = cpu_capacity_blocks
        self.pool = BitmapFramePool(capacity_blocks)
        self.page_tables: dict[int, PageTable] = {}
        self.blocks: dict[int, Block] = {}
        self.cpu_blocks: set[int] = set()
        self.next_id = 0
        self.clock = 0
        self.evictions = 0
        self.page_faults = 0

    @property
    def free(self) -> int:
        return len(self.pool)

    def _touch(self, block: Block) -> None:
        self.clock += 1
        block.last_access = self.clock

    def _frame(self, exclude: set[int] | None = None) -> int:
        if self.pool:
            return self.pool.pop()
        candidates = [b for b in self.blocks.values()
                      if b.frame is not None and b.block_id not in (exclude or set())]
        if not candidates or (self.cpu_capacity is not None and len(self.cpu_blocks) >= self.cpu_capacity):
            raise AllocError("GPU and CPU block capacity exhausted")
        victim = min(candidates, key=lambda b: b.last_access)
        frame = victim.frame
        victim.frame = None
        self.cpu_blocks.add(victim.block_id)
        self.evictions += 1
        assert frame is not None
        return frame

    def _new_block(self, exclude: set[int] | None = None) -> int:
        frame = self._frame(exclude)
        block_id = self.next_id
        self.next_id += 1
        block = Block(block_id, frame)
        self._touch(block)
        self.blocks[block_id] = block
        return block_id

    def _resident(self, block_id: int) -> int:
        block = self.blocks[block_id]
        if block.frame is None:
            if not self.pool and self.cpu_capacity is not None and len(self.cpu_blocks) >= self.cpu_capacity:
                victim = min((b for b in self.blocks.values() if b.frame is not None),
                             key=lambda b: b.last_access)
                block.frame = victim.frame
                victim.frame = None
                self.cpu_blocks.remove(block_id)
                self.cpu_blocks.add(victim.block_id)
                self.evictions += 1
            else:
                block.frame = self._frame()
                self.cpu_blocks.remove(block_id)
            self.page_faults += 1
        self._touch(block)
        return block.frame

    def new_sequence(self, seq_id: int) -> None:
        if seq_id in self.page_tables:
            raise ValueError("duplicate sequence id")
        self.page_tables[seq_id] = PageTable(seq_id)

    def append_token(self, seq_id: int) -> None:
        pt = self.page_tables[seq_id]
        logical, offset = divmod(pt.tokens, BLOCK_SIZE)
        if offset == 0:
            pt.map(logical, self._new_block())
        else:
            block_id = pt.entries[logical]
            block = self.blocks[block_id]
            if block.refs > 1:
                new_id = self._new_block({block_id})
                block.refs -= 1
                pt.map(logical, new_id)
            else:
                self._resident(block_id)
        pt.tokens += 1

    def translate(self, seq_id: int, logical_idx: int) -> tuple[int, int]:
        pt = self.page_tables[seq_id]
        if logical_idx < 0 or logical_idx >= pt.tokens:
            raise IndexError("token outside sequence")
        logical, offset = divmod(logical_idx, BLOCK_SIZE)
        return self._resident(pt.entries[logical]), offset

    def share_prefix(self, src_seq_id: int, dst_seq_id: int, shared_blocks: int) -> None:
        src = self.page_tables[src_seq_id]
        if dst_seq_id in self.page_tables or not 0 <= shared_blocks <= len(src.entries):
            raise ValueError("invalid prefix fork")
        dst = PageTable(dst_seq_id)
        for logical in range(shared_blocks):
            block_id = src.entries[logical]
            self.blocks[block_id].refs += 1
            dst.map(logical, block_id)
        dst.tokens = min(src.tokens, shared_blocks * BLOCK_SIZE)
        self.page_tables[dst_seq_id] = dst

    def fork(self, src_seq_id: int, dst_seq_id: int) -> None:
        self.share_prefix(src_seq_id, dst_seq_id, len(self.page_tables[src_seq_id].entries))

    def evict(self, seq_id: int) -> None:
        resident_ids = [block_id for block_id in set(self.page_tables[seq_id].entries.values())
                        if self.blocks[block_id].frame is not None]
        if self.cpu_capacity is not None and len(self.cpu_blocks) + len(resident_ids) > self.cpu_capacity:
            raise AllocError("CPU block capacity exhausted")
        for block_id in resident_ids:
            block = self.blocks[block_id]
            self.pool.append(block.frame)
            block.frame = None
            self.cpu_blocks.add(block_id)
            self.evictions += 1

    def release(self, seq_id: int) -> None:
        pt = self.page_tables.pop(seq_id, None)
        if pt is None:
            return
        for block_id in pt.entries.values():
            block = self.blocks[block_id]
            block.refs -= 1
            if block.refs == 0:
                if block.frame is not None:
                    self.pool.append(block.frame)
                self.cpu_blocks.discard(block_id)
                del self.blocks[block_id]

    def allocate_on_demand(self, seq_id: int, actual_tokens: int) -> bool:
        """Legacy whole-request admission, preserving the old no-swap experiment."""
        if blocks_for(actual_tokens) > self.free:
            return False
        self.new_sequence(seq_id)
        for _ in range(actual_tokens):
            self.append_token(seq_id)
        return True

    def utilization(self) -> float:
        return (self.capacity - self.free) / self.capacity

    def fragmentation_ratio(self) -> float:
        """Unused token slots in resident tail blocks / all GPU token slots."""
        unused = 0
        seen = set()
        for pt in self.page_tables.values():
            if pt.tokens % BLOCK_SIZE:
                block_id = pt.entries[pt.tokens // BLOCK_SIZE]
                if block_id not in seen and self.blocks[block_id].frame is not None:
                    unused += BLOCK_SIZE - pt.tokens % BLOCK_SIZE
                    seen.add(block_id)
        return unused / (self.capacity * BLOCK_SIZE)

class NaiveContiguousAllocator:
    def __init__(self, capacity_blocks: int):
        if capacity_blocks <= 0:
            raise ValueError("invalid capacity")
        self.capacity = capacity_blocks
        self.intervals = [(0, capacity_blocks)]
        self._held: dict[int, tuple[int, int]] = {}

    @property
    def free(self) -> int:
        return sum(length for _, length in self.intervals)

    def reserve(self, seq_id: int, max_tokens: int) -> bool:
        if seq_id in self._held:
            raise ValueError("duplicate sequence id")
        need = blocks_for(max_tokens)
        for i, (start, length) in enumerate(self.intervals):
            if length >= need:
                self._held[seq_id] = (start, need)
                if length == need:
                    self.intervals.pop(i)
                else:
                    self.intervals[i] = (start + need, length - need)
                return True
        return False

    def release(self, seq_id: int) -> None:
        if seq_id not in self._held:
            return
        self.intervals.append(self._held.pop(seq_id))
        self.intervals.sort()
        merged = []
        for start, length in self.intervals:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self.intervals = merged

    def fragmentation_ratio(self) -> float:
        return 0 if not self.free else 1 - max(length for _, length in self.intervals) / self.free
