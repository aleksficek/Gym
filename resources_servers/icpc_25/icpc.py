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

SHARED_TEMP_DIR = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/aficek/synth/data/sandbox_files"

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

@nested_dataclass(kw_only=True)
class ICPCEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.json"
    input_file: str = None
    test_batch_size: int = 16  # Controls the asyncio Semaphore limit

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

    async def _initialize_runtime(self):
        """Lazy initialization of metadata and sandbox on the main event loop."""
        if self.sandbox is not None:
            return

        self.sandbox = LocalSandbox()
        
        if not os.path.exists(self.eval_cfg.test_file):
            raise FileNotFoundError(f"Metadata file {self.eval_cfg.test_file} not found.")
            
        def _load_data():
            with open(self.eval_cfg.test_file, "r") as f:
                md = json.load(f)
            idat = None
            if self.eval_cfg.input_file and os.path.exists(self.eval_cfg.input_file):
                with open(self.eval_cfg.input_file, "r") as f:
                    idat = json.load(f)
            return md, idat

        self.metadata, self.inputdata = await asyncio.to_thread(_load_data)

    async def _precompile_grader(self, problem_name: str, problem_metadata: dict) -> str:
        """Precompile grader assets on the head node/sandbox."""
        pre_dir = f"{SHARED_TEMP_DIR}/icpc_pre_{problem_name}_{os.getpid()}"
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

    async def _run_test_async(self, problem_id: str, code: str, test_input: str, test_output: str, pre_dir: str) -> dict:
        """Full test execution (Compile + Run) logic wrapped in a semaphore."""
        async with self.semaphore:
            unique_dir = f"{SHARED_TEMP_DIR}/icpc_run_{problem_id}_{time.time_ns()}"
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
        
        pid = entry["icpc_id"]
        problem_metadata = self.metadata[pid]
        completion = self._prepare_code(entry["generation"], pid)

        if pid not in self.precompiled_cache:
            self.precompiled_cache[pid] = await self._precompile_grader(pid, problem_metadata)
        pre_dir = self.precompiled_cache[pid]

        all_test_configs = []
        for tname, t in problem_metadata["sample_tests"].items():
            all_test_configs.append((tname, t["input"], t["output"], "sample"))
        for tname, t in problem_metadata["tests"].items():
            all_test_configs.append((tname, t["input"], t["output"], "test"))

        # Concurrent execution of all tests for this specific rollout
        tasks = [
            self._run_test_async(pid, completion, cfg[1], cfg[2], pre_dir) 
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
        return await self._evaluate_entry(data_point)

    async def eval_full(self, input_files):
        # Implementation for batch JSONL evaluation
        pass