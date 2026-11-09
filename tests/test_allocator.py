import unittest

from kv_cache.allocator import AllocError, BLOCK_SIZE, KVBlockAllocator, NaiveContiguousAllocator, blocks_for


class PagingTests(unittest.TestCase):
    def test_validation_and_manual_eviction(self):
        with self.assertRaises(ValueError):
            blocks_for(-1)
        alloc = KVBlockAllocator(2, cpu_capacity_blocks=2)
        alloc.new_sequence(1)
        with self.assertRaises(ValueError):
            alloc.new_sequence(1)
        alloc.append_token(1)
        self.assertGreater(alloc.fragmentation_ratio(), 0)
        self.assertEqual(alloc.utilization(), 0.5)
        alloc.evict(1)
        self.assertEqual(alloc.free, 2)
        self.assertEqual(len(alloc.cpu_blocks), 1)
        alloc.translate(1, 0)
        self.assertEqual(alloc.page_faults, 1)
        alloc.release(1)
        self.assertEqual(alloc.free, 2)

    def test_shared_tail_fragmentation_counted_once(self):
        alloc = KVBlockAllocator(2)
        alloc.new_sequence(1)
        alloc.append_token(1)
        ratio = alloc.fragmentation_ratio()
        alloc.fork(1, 2)
        self.assertEqual(alloc.fragmentation_ratio(), ratio)

    def test_failed_manual_eviction_keeps_state(self):
        alloc = KVBlockAllocator(2, cpu_capacity_blocks=1)
        alloc.new_sequence(1)
        for _ in range(BLOCK_SIZE + 1):
            alloc.append_token(1)
        with self.assertRaises(AllocError):
            alloc.evict(1)
        self.assertEqual(alloc.free, 0)
        self.assertEqual(alloc.cpu_blocks, set())

    def test_dynamic_growth_and_translation(self):
        alloc = KVBlockAllocator(3)
        alloc.new_sequence(1)
        for _ in range(BLOCK_SIZE + 1):
            alloc.append_token(1)
        self.assertEqual(len(alloc.page_tables[1].entries), 2)
        self.assertEqual(alloc.translate(1, BLOCK_SIZE)[1], 0)
        with self.assertRaises(IndexError):
            alloc.translate(1, BLOCK_SIZE + 1)

    def test_shared_release_and_copy_on_write(self):
        alloc = KVBlockAllocator(3)
        alloc.new_sequence(1)
        alloc.append_token(1)
        alloc.fork(1, 2)
        shared = alloc.page_tables[2].entries[0]
        alloc.release(1)
        self.assertEqual(alloc.blocks[shared].refs, 1)
        self.assertEqual(alloc.free, 2)
        alloc.new_sequence(3)
        alloc.append_token(3)
        self.assertNotEqual(alloc.page_tables[3].entries[0], shared)
        alloc.fork(2, 4)
        alloc.append_token(4)
        self.assertNotEqual(alloc.page_tables[4].entries[0], shared)
        self.assertEqual(alloc.blocks[shared].refs, 1)

    def test_lru_swap_and_fault(self):
        alloc = KVBlockAllocator(1, cpu_capacity_blocks=1)
        alloc.new_sequence(1)
        alloc.append_token(1)
        alloc.new_sequence(2)
        alloc.append_token(2)
        self.assertEqual(alloc.evictions, 1)
        self.assertEqual(len(alloc.cpu_blocks), 1)
        alloc.translate(1, 0)
        self.assertEqual(alloc.page_faults, 1)
        self.assertEqual(len(alloc.cpu_blocks), 1)
        alloc.new_sequence(3)
        with self.assertRaises(AllocError):
            alloc.append_token(3)
        self.assertEqual(alloc.page_tables[3].tokens, 0)

    def test_contiguous_external_fragmentation(self):
        alloc = NaiveContiguousAllocator(6)
        for seq in range(3):
            self.assertTrue(alloc.reserve(seq, 2 * BLOCK_SIZE))
        alloc.release(0)
        alloc.release(2)
        self.assertEqual(alloc.free, 4)
        self.assertFalse(alloc.reserve(3, 3 * BLOCK_SIZE))
        self.assertGreater(alloc.fragmentation_ratio(), 0)


if __name__ == "__main__":
    unittest.main()
