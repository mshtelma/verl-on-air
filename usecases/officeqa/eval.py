#!/usr/bin/env python3
"""Evaluate with the actual training controller, tools, reward and token budgets.

vLLM's token-in/token-out completions API supplies the server_manager interface;
no decode/re-encode fallback or independently implemented agent loop is used.
The fixed held-out subset is never used to select training difficulty.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "engine" / "serve"))
sys.path.insert(0, str(HERE.parents[1] / "engine" / "train"))
sys.path.insert(0, str(HERE.parents[1] / "engine" / "lib"))
import eval_contract as ec  # noqa: E402
import run_control  # noqa: E402

import metrics
import reward
import tool
from agent_loop import OfficeQAToolAgentLoop


class CompletionServer:
    def __init__(self, session, base_url, served, seed):
        self.session, self.base_url, self.served, self.seed = session, base_url, served, seed

    async def generate(self, *, prompt_ids, sampling_params, **kwargs):
        from verl.workers.rollout.replica import TokenOutput
        payload = {"model": self.served, "prompt": prompt_ids, "return_token_ids": True,
                   "seed": self.seed, "skip_special_tokens": False, **sampling_params}
        reply = await ec.post_json(self.session, self.base_url + "/completions", payload, retries=1)
        try:
            choice = reply["choices"][0]
            ids = choice["token_ids"]
            if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
                raise ValueError("empty/invalid token_ids")
            return TokenOutput(token_ids=ids, num_preempted=0)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ec.InfraError("infra_inference", f"token-in/token-out completion failed: {error}") from error


def loop_config():
    from omegaconf import OmegaConf
    prompt = int(os.environ.get("OQ_PROMPT_TOKENS", "2048"))
    per_turn = int(os.environ.get("OQ_MAX_GENERATION_TOKENS", "1024"))
    turns = int(os.environ.get("OQ_MAX_TURNS", "12"))
    episode = (prompt + per_turn) * turns  # same arithmetic as the fully-async launcher
    rollout = {"prompt_length": prompt, "response_length": episode - prompt, "full_determinism": False,
               "multi_turn": {"max_assistant_turns": turns, "max_user_turns": turns, "max_parallel_calls": 1,
                              "max_tool_response_length": int(os.environ.get("OQ_TOOL_MAX_CHARS", "4000")),
                              "tool_response_truncate_side": "right", "format": os.environ.get("TOOL_FORMAT", "qwen3_coder")}}
    return OmegaConf.create({"actor_rollout_ref": {"rollout": rollout}})


def make_loop(server, tokenizer, hf_type, processor=None):
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY
    from verl.utils.dataset.rl_dataset import RLHFDataset
    tools = [FUNCTION_TOOL_REGISTRY[name] for name in sorted(tool.SCHEMAS)]
    return OfficeQAToolAgentLoop(
        trainer_config=DictConfigWrap(loop_config()), server_manager=server, tokenizer=tokenizer,
        processor=processor, dataset_cls=RLHFDataset, data_config=DictConfigWrap(OmegaConf.create({})),
        hf_model_type=hf_type, tools=ToolListWrap(tools),
    )


def load_questions():
    import pyarrow.parquet as pq
    root = Path(os.environ["OQ_DATA_DIR"])
    questions, receipts = [], []
    for filename, split, limit in [
        ("pilot_train.parquet", "train_probe", int(os.environ.get("OQ_EVAL_TRAIN_LIMIT", "16"))),
        ("test.parquet", "heldout", int(os.environ.get("OQ_EVAL_HELDOUT_LIMIT", "8"))),
    ]:
        path = root / filename
        receipt = ec.file_fingerprint(path)
        rows = pq.read_table(path).to_pylist()
        receipts.append(receipt)
        for row in rows[:limit] if limit > 0 else []:
            row["eval_split"] = split
            questions.append(row)
    if not questions:
        raise ValueError("empty OfficeQA evaluation set")
    ids = [row["extra_info"]["uid"] for row in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate/overlapping evaluation questions")
    return questions, {"name": "officeqa-miles-pilot", "files": receipts,
                       "manifest": ec.file_fingerprint(root / "DATA_MANIFEST.json")}


async def main():
    import aiohttp
    from transformers import AutoConfig
    from verl.utils import hf_processor, hf_tokenizer
    started = time.time()
    out = os.environ["EVAL_OUT"]
    trace_out = os.environ["EVAL_TRACE_OUT"]
    ec.refuse_overwrite(out, trace_out)
    samples = int(os.environ.get("OQ_EVAL_SAMPLES", "4"))
    questions, dataset = load_questions()
    expected = int(os.environ.get("EVAL_EXPECT_N", str(len(questions) * samples)))
    tokenizer = hf_tokenizer(os.environ["MODEL_PATH"], trust_remote_code=True)
    processor = hf_processor(os.environ["MODEL_PATH"], trust_remote_code=True)
    if processor is not None and not getattr(processor, "chat_template", None):
        processor.chat_template = tokenizer.chat_template
    hf_type = AutoConfig.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True).model_type
    base_url = os.environ["EVAL_BASE_URL"].rstrip("/")
    served = os.environ.get("EVAL_MODEL", "eval")
    parts = ec.PartsWriter(out)
    concurrency = int(os.environ.get("EVAL_CONCURRENCY", "8"))
    semaphore = asyncio.Semaphore(concurrency)
    temperature = float(os.environ.get("EVAL_TEMPERATURE", "1.0"))
    seed = int(os.environ.get("SEED", "42"))
    policy = {"version": 1, "controller": "OfficeQAToolAgentLoop", "tokenization": "verl_continuous_token",
              "max_turns": int(os.environ.get("OQ_MAX_TURNS", "12")),
              "generation_tokens_per_turn": int(os.environ.get("OQ_MAX_GENERATION_TOKENS", "1024")),
              "prompt_tokens": int(os.environ.get("OQ_PROMPT_TOKENS", "2048")),
              "tool_max_chars": int(os.environ.get("OQ_TOOL_MAX_CHARS", "4000")), "parallel_tool_calls": 1,
              "samples_per_question": samples, "temperature": temperature, "top_p": 1.0, "top_k": -1, "seed": seed,
              "tool_format": os.environ.get("TOOL_FORMAT", "qwen3_coder"),
              "tools_sha256": hashlib.sha256(json.dumps(tool.TOOLS, sort_keys=True).encode()).hexdigest(),
              "system_prompt_sha256": hashlib.sha256(questions[0]["prompt"][0]["content"].encode()).hexdigest(),
              "reward": "binary_correct_answer_and_supported_path", "answer_tolerance": 0.0,
              "judge_model": os.environ.get("JUDGE_MODEL_PATH"), "judge_reasoning_effort": "low",
              "retrieval_backend": "bm25s" if tool.corpus.bm25s else "keyword_overlap"}
    config = loop_config()
    policy["response_tokens_per_episode"] = config.actor_rollout_ref.rollout.response_length
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("EVAL_REQ_TIMEOUT", "300")))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await ec.check_served_model(session, base_url, served)

        async def episode(index, row, sample):
            uid = row["extra_info"]["uid"]
            result = {"id": f"{uid}/sample_{sample}", "uid": uid, "sample": sample,
                      "group": f"{row['eval_split']}/{uid}", "split": row["eval_split"],
                      "difficulty": row["extra_info"]["difficulty"], "status": "scored", "reward": 0.0}
            async with semaphore:
                try:
                    if run_control.read_abort():
                        raise ec.InfraError("infra_harness", "run abort already requested")
                    server = CompletionServer(session, base_url, served, seed + index * samples + sample)
                    loop = make_loop(server, tokenizer, hf_type, processor)
                    output = await loop.run({"temperature": temperature, "top_p": 1.0, "top_k": -1},
                                            raw_prompt=row["prompt"], extra_info=row["extra_info"])
                    grade = await reward.compute_score(ground_truth=row["reward_model"]["ground_truth"],
                                                        extra_info={**row["extra_info"], **output.extra_fields})
                    if grade["infrastructure_error"]:
                        raise ec.InfraError("infra_harness", "unresolved reward; see ABORT.json and reward trace")
                    result.update(reward=grade["score"], metrics=grade, num_turns=output.num_turns,
                                  generated_tokens=sum(output.response_mask), response_tokens=len(output.response_ids),
                                  record=json.loads(output.extra_fields["officeqa_record"]),
                                  trajectory=tokenizer.decode(output.response_ids, skip_special_tokens=False))
                except ec.InfraError as error:
                    result.update(status=error.kind, error=error.detail)
                    run_control.request_abort(str(error), "usecases/officeqa/eval.py")
                except Exception as error:
                    result.update(status="infra_harness", error=f"{type(error).__name__}: {error}")
                    run_control.request_abort(result["error"], "usecases/officeqa/eval.py")
            parts.write(result["id"], result)
            print(json.dumps({"officeqa_eval": result["id"], "split": result["split"], "status": result["status"],
                              "reward": result["reward"], "turns": result.get("num_turns")}), flush=True)
            return result

        try:
            results = await asyncio.gather(*(episode(i, row, sample) for i, row in enumerate(questions) for sample in range(samples)))
        finally:
            await reward.close_sessions()
    validity = ec.verdict(results, n_loaded=len(results), n_expected=expected)
    artifact = {**ec.header(dataset=dataset, question_ids=[r["id"] for r in results], policy=policy, started_at=started),
                **validity, "summary": {split: metrics.summarize([r for r in results if r["split"] == split], samples)
                                        for split in sorted({r["split"] for r in results})},
                "train_group_variance": ec.group_variance([r for r in results if r["split"] == "train_probe"]),
                "results": [{k: v for k, v in r.items() if k not in {"record", "trajectory"}} for r in results]}
    ec.write_json_atomic(out, artifact)
    path = Path(trace_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    temporary.replace(path)
    print(json.dumps({"officeqa_eval_artifact": out, "valid": validity["valid"], "summary": artifact["summary"]}), flush=True)
    return ec.report_and_exit_code(validity)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
