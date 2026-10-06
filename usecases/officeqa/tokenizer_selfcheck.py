#!/usr/bin/env python3
"""Qualify simultaneous cold judge-tokenizer calls on the actual AIR image.

Import only reward before the calls: importing other HF helpers first would hide
the Transformers lazy-import race this check is intended to exercise.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import reward


def main():
    workers = 8
    barrier = Barrier(workers)
    messages = [{"role": "user", "content": "Verify OfficeQA evidence: FY1940 national defense was 2,602 million dollars."}]
    assert reward._tokenizer is None, "cold tokenizer initialization is required"

    def count(_):
        barrier.wait(timeout=30)
        return reward.judge_token_count(messages)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(count, range(workers)))
    assert len(set(counts)) == 1 and counts[0] > 0, counts
    assert reward.judge_token_count(messages) == counts[0]
    print(json.dumps({"officeqa_tokenizer_selfcheck": "PASS", "cold_concurrent_calls": workers,
                      "prompt_tokens": counts[0], "tokenizer": type(reward._tokenizer).__name__}), flush=True)


if __name__ == "__main__":
    main()
