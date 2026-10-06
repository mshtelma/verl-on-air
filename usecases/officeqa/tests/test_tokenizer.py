import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import reward


def test_cold_concurrent_counts_load_once_and_serialize_shared_backend(monkeypatch):
    loads = []
    workers = 8
    barrier = Barrier(workers)

    class Tokenizer:
        busy = False

        def apply_chat_template(self, messages, **kwargs):
            assert not self.busy, "shared tokenizer backend was used concurrently"
            self.busy = True
            try:
                time.sleep(0.005)
                return list(range(17))
            finally:
                self.busy = False

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            loads.append(model_path)
            time.sleep(0.03)  # Other callers reach the uninitialized tokenizer.
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(reward, "_tokenizer", None)
    monkeypatch.setenv("JUDGE_TOKENIZER_PATH", "/fake/judge")

    def count(_):
        barrier.wait(timeout=10)
        return reward.judge_token_count([{"role": "user", "content": "Evidence"}])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(count, range(workers)))
    assert counts == [17] * workers
    assert loads == ["/fake/judge"]
