# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from typing import Dict, List

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files

# SHARED_TEMP_DIR = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/aficek/synth/data/sandbox_files"

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

@nested_dataclass(kw_only=True)
class ICPCEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.json"
    input_file: str = None
    test_batch_size: int = 16  # Controls the asyncio Semaphore limit
    shared_dir: str = "/tmp"

class ICPCEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_cfg = ICPCEvaluatorConfig(_init_nested=True, **config)

        self.sandbox = None
        self.metadata = None
        self.inputdata = None
        self.precompiled_cache: Dict[str, str] = {}
        # Semaphore prevents overwhelming the sandbox with too many concurrent executions
        self.semaphore = asyncio.Semaphore(self.eval_cfg.test_batch_size)
        # Lock to prevent race conditions during initialization
        self._init_lock = asyncio.Lock()

    async def _initialize_runtime(self):
        """Lazy initialization of metadata and sandbox on the main event loop."""
        # Use lock to prevent race conditions
        async with self._init_lock:
            # Double-check after acquiring lock
            if self.sandbox is not None:
                return

            self.sandbox = LocalSandbox()

            if not os.path.exists(self.eval_cfg.test_file):
                raise FileNotFoundError(f"Metadata file {self.eval_cfg.test_file} not found.")

            def _load_data():
                # Load JSONL file where each line contains competition ID and its problems metadata
                md = {}
                with open(self.eval_cfg.test_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        competition = entry["competition"]
                        # The key is "metadata" not "problems" in the actual file format
                        problems = entry.get("metadata") or entry.get("problems", {})
                        md[competition] = problems

                idat = None
                if self.eval_cfg.input_file and os.path.exists(self.eval_cfg.input_file):
                    with open(self.eval_cfg.input_file, "r") as f:
                        idat = json.load(f)
                return md, idat

            self.metadata, self.inputdata = await asyncio.to_thread(_load_data)
            print(f"DEBUG: Loaded metadata for competitions: {list(self.metadata.keys())}")

    async def _precompile_grader(self, competition: str, problem_name: str, problem_metadata: dict) -> str:
        """Precompile grader assets on the head node/sandbox."""
        pre_dir = f"{self.eval_cfg.shared_dir}/icpc_pre_{competition}_{problem_name}_{os.getpid()}"
        os.makedirs(os.path.join(pre_dir, "graders"), exist_ok=True)

        for filepath, content in problem_metadata["grader_files"]:
            target_path = os.path.join(pre_dir, filepath)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

        for script in ["compile", "run", "user_run"]:
            path = os.path.join(pre_dir, f"{script}.sh")
            with open(path, "w", encoding="utf-8") as f:
                f.write(problem_metadata[script])
            os.chmod(path, 0o755)

        # Standard timeout for initial grader setup
        await self.sandbox.execute_code(f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)
        return pre_dir

    async def _run_test_async(self, competition: str, problem_id: str, code: str, test_input: str, test_output: str, pre_dir: str) -> dict:
        """Full test execution (Compile + Run) logic wrapped in a semaphore."""
        async with self.semaphore:
            unique_dir = f"{self.eval_cfg.shared_dir}/icpc_run_{competition}_{problem_id}_{time.time_ns()}"
            try:
                # 1. Setup local environment
                os.makedirs(os.path.join(unique_dir, "graders"), exist_ok=True)
                if pre_dir and os.path.isdir(pre_dir):
                    os.system(f"cp -rp {pre_dir}/* {unique_dir}/")
                
                with open(os.path.join(unique_dir, "graders", f"{problem_id}.cpp"), "w") as f:
                    f.write(code)
                with open(os.path.join(unique_dir, "input.txt"), "w") as f:
                    f.write(test_input)
                with open(os.path.join(unique_dir, "correct_output.txt"), "w") as f:
                    f.write(test_output)

                # 2. Compilation Step
                compile_result, _ = await self.sandbox.execute_code(
                    f"cd {unique_dir} && ./compile.sh", 
                    language="shell", 
                    timeout=60
                )

                if compile_result.get("stderr"):
                    return {
                        "compile_success": False,
                        "compile_stderr": compile_result.get("stderr"),
                        "score": 0.0
                    }

                # 3. Execution Step (with critical timeout protection)
                # We use a 30s timeout here to catch infinite loops in model rollouts
                run_result, _ = await self.sandbox.execute_code(
                    f"cd {unique_dir} && ./run.sh", 
                    language="shell", 
                    timeout=30 
                )

                run_stdout = run_result.get("stdout", "").strip()
                
                try:
                    score = float(run_stdout) if run_stdout else 0.0
                except (ValueError, TypeError):
                    score = 0.0

                return {
                    "compile_success": True,
                    "run_stdout": run_stdout,
                    "run_stderr": run_result.get("stderr", ""),
                    "score": score
                }
                
            except Exception as e:
                return {"score": 0.0, "error": str(e)}
            finally:
                if os.path.exists(unique_dir):
                    shutil.rmtree(unique_dir, ignore_errors=True)

    async def _evaluate_entry(self, entry: dict) -> dict:
        await self._initialize_runtime()

        competition = entry.get("competition")
        if not competition:
            raise ValueError("Missing 'competition' field in entry")

        pid = entry["icpc_id"]

        if competition not in self.metadata:
            available = list(self.metadata.keys())
            raise ValueError(f"Competition '{competition}' not found. Available competitions: {available}")

        competition_problems = self.metadata[competition]
        if competition_problems is None:
            raise ValueError(f"Competition '{competition}' has no problems loaded (None)")

        if pid not in competition_problems:
            available_problems = list(competition_problems.keys()) if competition_problems else []
            raise ValueError(f"Problem '{pid}' not found in competition '{competition}'. Available problems: {available_problems}")

        problem_metadata = competition_problems[pid]
        completion = self._prepare_code(entry["generation"], pid)

        cache_key = f"{competition}_{pid}"
        if cache_key not in self.precompiled_cache:
            self.precompiled_cache[cache_key] = await self._precompile_grader(competition, pid, problem_metadata)
        pre_dir = self.precompiled_cache[cache_key]

        all_test_configs = []
        for tname, t in problem_metadata["sample_tests"].items():
            all_test_configs.append((tname, t["input"], t["output"], "sample"))
        for tname, t in problem_metadata["tests"].items():
            all_test_configs.append((tname, t["input"], t["output"], "test"))

        # Concurrent execution of all tests for this specific rollout
        tasks = [
            self._run_test_async(competition, pid, completion, cfg[1], cfg[2], pre_dir)
            for cfg in all_test_configs
        ]
        results = await asyncio.gather(*tasks)

        problem_state = {"outputs": [], "sample_passed": True, "test_passed": True}
        for (tname, _, _, ttype), res in zip(all_test_configs, results):
            res["test_name"] = tname
            res["test_type"] = ttype
            problem_state["outputs"].append(res)

            # Binary scoring: if any test in a category fails, the whole category is false
            if res.get("score", 0.0) < 1.0:
                if ttype == "sample": problem_state["sample_passed"] = False
                else: problem_state["test_passed"] = False

        return {
            "name": entry.get("name", pid),
            "test_case_results": {
                "sample_score": float(problem_state["sample_passed"]),
                "score": float(problem_state["test_passed"]),
                "outputs": problem_state["outputs"],
            },
            "input_case_results": []
        }

    @staticmethod
    def summarize_test_case_results(evaluation_result: dict) -> Dict[str, float]:
        """Summarize raw test outputs into reward-oriented aggregate metrics."""
        outputs = []
        if isinstance(evaluation_result, dict):
            test_case_results = evaluation_result.get("test_case_results")
            if isinstance(test_case_results, dict):
                maybe_outputs = test_case_results.get("outputs")
                if isinstance(maybe_outputs, list):
                    outputs = maybe_outputs

        num_outputs = len(outputs)
        if num_outputs == 0:
            return {"reward": 0.0, "num_outputs": 0.0, "num_perfect": 0.0}

        score_sum = 0.0
        num_perfect = 0
        for item in outputs:
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0))
            score_sum += score
            if score >= 1.0:
                num_perfect += 1

        reward = score_sum / max(1, num_outputs)
        return {
            "reward": reward,
            "num_outputs": float(num_outputs),
            "num_perfect": float(num_perfect),
        }

    def _prepare_code(self, gen: str, pid: str) -> str:
        # Extracts the last C++ block and ensures common headers are present
        pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
        matches = re.findall(pattern, gen, re.DOTALL)
        code = matches[-1] if matches else ""
        
        if not code:
            return ""
            
        header = "#include <bits/stdc++.h>\nusing namespace std;\n"
        # Avoid double-including if the model already provided it
        if "using namespace std;" in code:
            return code
        return header + code

    async def eval_single(self, data_point: dict):
        result = await self._evaluate_entry(data_point)
        summary = self.summarize_test_case_results(result)
        competition = data_point.get("competition", "unknown_competition")
        icpc_id = data_point.get("icpc_id", data_point.get("name", "unknown_problem"))
        print(
            f"[ICPC_EVAL_DEBUG] icpc.py eval_single result for {competition}:{icpc_id} "
            f"summary={summary} test_case_results={result.get('test_case_results')}"
        )
        return result

    async def eval_full(self, input_files):
        # Implementation for batch JSONL evaluation
        pass
