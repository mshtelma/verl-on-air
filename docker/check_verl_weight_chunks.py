#!/usr/bin/env python3
"""CPU numerical regression for the installed serializer, using real torch.

Load the actual functions without starting Ray or probing GPU backends. Exercise
strided exports through byte chunking and reconstruction, including both receiver
paths and the metadata-only relay that failed in the H100 qualification.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, AsyncGenerator, Generator
import unittest

import torch

from patch_verl_weight_chunks import AFTER, BEFORE, BEFORE_SHA256, apply, source_path


async def ensure_async_iterator(values):
    if hasattr(values, "__aiter__"):
        async for value in values:
            yield value
    else:
        for value in values:
            yield value


def load_functions():
    tree = ast.parse(source_path().read_text())
    names = {"TensorMeta", "split_weight_chunks", "merge_weight_chunks"}
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    if len(body) != len(names):
        raise RuntimeError("installed serializer functions are missing")
    namespace = {"torch": torch, "dataclass": dataclass, "Any": Any,
                 "AsyncGenerator": AsyncGenerator, "Generator": Generator,
                 "ensure_async_iterator": ensure_async_iterator}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path()), "exec"), namespace)
    return namespace["split_weight_chunks"], namespace["merge_weight_chunks"]


class WeightChunkRegression(unittest.IsolatedAsyncioTestCase):
    async def test_strided_exports_round_trip_with_mixed_dtypes(self):
        split, merge = load_functions()
        tensors = [
            ("gdn.bf16", torch.arange(60, dtype=torch.bfloat16).reshape(6, 10).T),
            ("gdn.fp32", torch.arange(84, dtype=torch.float32).reshape(7, 12)[:, ::2]),
            ("small", torch.tensor([1.0, 3.0], dtype=torch.bfloat16)),
        ]
        self.assertFalse(tensors[0][1].is_contiguous())
        self.assertFalse(tensors[1][1].is_contiguous())
        received = [item async for item in merge(split(iter(tensors), 32), 32)]
        self.assertEqual([name for name, _ in received], [name for name, _ in tensors])
        for (_, expected), (_, actual) in zip(tensors, received, strict=True):
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.dtype, actual.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    async def test_metadata_relay_does_not_read_or_copy_tensor_values(self):
        split, _ = load_functions()

        class MetadataOnly:
            shape = torch.Size((10, 6))
            dtype = torch.bfloat16
            nbytes = 120

        chunks = [item async for item in split(iter([("gdn", MetadataOnly())]), 32, meta_only=True)]
        self.assertEqual([meta.chunk_offset for meta, _ in chunks], [0, 32, 64, 96])
        self.assertEqual(sum(meta.chunk_size for meta, _ in chunks), 120)
        self.assertTrue(all(buffer is None for _, buffer in chunks))

    async def test_contiguous_exports_reuse_the_original_storage(self):
        split, _ = load_functions()
        tensor = torch.arange(8, dtype=torch.float32)
        chunks = [item async for item in split(iter([("contiguous", tensor)]), 128)]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1].untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())


class PatchGuardRegression(unittest.TestCase):
    def test_only_the_pinned_source_is_accepted_and_reapplying_is_idempotent(self):
        original = source_path().read_text().replace(AFTER, BEFORE, 1)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "base.py"
            target.write_text(original)
            first = apply(target)
            self.assertTrue(first["changed"])
            self.assertEqual(first["before_sha256"], BEFORE_SHA256)
            mtime = target.stat().st_mtime_ns
            self.assertFalse(apply(target)["changed"])
            self.assertEqual(target.stat().st_mtime_ns, mtime)
            changed = target.read_text() + "\n# unrelated source change\n"
            target.write_text(changed)
            with self.assertRaises(RuntimeError):
                apply(target)
            self.assertEqual(target.read_text(), changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
