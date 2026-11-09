import random
import unittest
from collections import Counter

from kv_cache.allocator import AllocError, KVBlockAllocator


def assert_invariants(test, alloc):
    references = Counter(block_id for pt in alloc.page_tables.values()
                         for block_id in pt.entries.values())
    test.assertEqual(set(references), set(alloc.blocks))
    for block_id, block in alloc.blocks.items():
        test.assertEqual(block.refs, references[block_id])
        test.assertEqual(block.frame is None, block_id in alloc.cpu_blocks)
    frames = [block.frame for block in alloc.blocks.values() if block.frame is not None]
    test.assertEqual(len(frames), len(set(frames)))
    test.assertEqual(len(alloc.pool), len(set(alloc.pool)))
    test.assertEqual(len(alloc.pool.bitmap), (alloc.capacity + 7) // 8)
    test.assertEqual({i for i in range(alloc.capacity) if alloc.pool.is_free(i)},
                     set(alloc.pool))
    test.assertEqual(set(frames) & set(alloc.pool), set())
    test.assertEqual(set(frames) | set(alloc.pool), set(range(alloc.capacity)))
    test.assertLessEqual(len(alloc.cpu_blocks), alloc.cpu_capacity)
    for pt in alloc.page_tables.values():
        test.assertEqual(set(pt.entries), set(range((pt.tokens + 15) // 16)))


class InvariantTests(unittest.TestCase):
    def test_seeded_mixed_operations(self):
        rng = random.Random(2026)
        alloc = KVBlockAllocator(3, cpu_capacity_blocks=4)
        next_id = 0
        for _ in range(500):
            seq_ids = list(alloc.page_tables)
            action = rng.choice(("new", "append", "fork", "translate", "evict", "release"))
            try:
                if action == "new" or not seq_ids:
                    alloc.new_sequence(next_id)
                    next_id += 1
                elif action == "append":
                    alloc.append_token(rng.choice(seq_ids))
                elif action == "fork":
                    alloc.fork(rng.choice(seq_ids), next_id)
                    next_id += 1
                elif action == "translate":
                    seq_id = rng.choice(seq_ids)
                    if alloc.page_tables[seq_id].tokens:
                        alloc.translate(seq_id, 0)
                elif action == "evict":
                    alloc.evict(rng.choice(seq_ids))
                else:
                    alloc.release(rng.choice(seq_ids))
            except AllocError:
                pass
            assert_invariants(self, alloc)
