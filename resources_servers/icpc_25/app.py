# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import socket
import subprocess
import time
import shutil
import json
import threading
import math
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)

from icpc import ICPCEvaluator

class IcpcVerifyRequest(BaseVerifyRequest):
    icpc_id: str

# --- Custom Response Class ---
class IcpcVerifyResponse(BaseVerifyResponse):
    details: Dict[str, Any] = {}

class Icpc25ResourcesServerConfig(BaseResourcesServerConfig):
    # test_file: str = "/home/aficek/software/storage/data/icpc_25/metadata/icpc25_metadata.json"
    test_file: str = os.getenv("TEST_FILE", "/home/aficek/software/storage/data/icpc_25/metadata/icpc25_metadata.json")
    sandbox_host: str = os.getenv("NEMO_SKILLS_SANDBOX_HOST", os.getenv("RAY_HEAD_IP", "localhost"))
    sandbox_port: int = 6000
    test_batch_size: int = 4
    num_parallel_requests: int = 2
    shared_dir: str = os.getenv("SHARED_TEMP_DIR", "/tmp")

    sandbox_image: str = "docker.io/igitman/nemo-skills-sandbox:0.7.1"
    # data_volume: str = "/home/aficek/software/storage/data/icpc_25:/home/aficek/software/storage/data/icpc_25"
    data_volume: str = "/tmp:/tmp"

    # On-policy distillation settings
    enable_distillation: bool = False
    teacher_model_server: Optional[ModelServerRef] = None
    teacher_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    distillation_temperature: float = 1.0
    max_distillation_reward: float = 0.5


class Icpc25ResourcesServer(SimpleResourcesServer):
    config: Icpc25ResourcesServerConfig
    
    _evaluator: Optional[ICPCEvaluator] = PrivateAttr(default=None)

    def _is_port_open(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) == 0

    def launch_sandbox(self):
        """Launches the Docker sandbox if it's not already running."""
        host = self.config.sandbox_host
        port = self.config.sandbox_port
        
        if self._is_port_open(host, port):
            print(f"✅ Sandbox already running on {host}:{port}")
            return

        print(f"🚀 Launching Docker Sandbox on port {port}...")

        # IMPORTANT: Mapping host port 6000 to container port 8000
        cmd = [
            "docker", "run", 
            "--rm",      
            "-d",        
            "-p", f"{port}:{port}",
            "-v", self.config.data_volume,
            "--name", f"nemo_skills_sandbox_{port}", 
            self.config.sandbox_image
        ]

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            print("Waiting for sandbox to initialize...")
            for i in range(30): 
                if self._is_port_open(host, port):
                    time.sleep(1) 
                    print("✅ Sandbox connected successfully.")
                    return
                time.sleep(1)
            
            print("⚠️ WARNING: Sandbox container started, but port is not yet open.")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to launch sandbox: {e}")

    async def _compute_distillation_reward(
        self,
        body: IcpcVerifyRequest,
    ) -> float:
        """Compute on-policy distillation reward using KL divergence between student and teacher.

        Returns:
            float: Distillation reward in [0, max_distillation_reward] range.
                   Lower KL divergence = higher reward.
        """
        if not self.config.enable_distillation:
            return 0.0

        if not self.config.teacher_model_server or not self.config.teacher_responses_create_params:
            print("⚠️ Distillation enabled but teacher model not configured")
            return 0.0

        try:
            # Step 1: Prepare teacher request with same input as student, but request logprobs
            teacher_params = self.config.teacher_responses_create_params.model_copy(deep=True)
            teacher_params.input = body.responses_create_params.input

            # Request logprobs from teacher (if API supports it)
            if hasattr(teacher_params, 'logprobs'):
                teacher_params.logprobs = True
            if hasattr(teacher_params, 'top_logprobs'):
                teacher_params.top_logprobs = 20  # Get top 20 tokens for better distribution

            # Step 2: Call teacher model
            print(f"DEBUG: Calling teacher model: {self.config.teacher_model_server.name}")
            response = await self.server_client.post(
                server_name=self.config.teacher_model_server.name,
                url_path="/v1/responses",
                json=teacher_params,
            )
            teacher_response = NeMoGymResponse.model_validate(await response.json())

            # Step 3: Extract logprobs from student and teacher responses
            student_logprobs = self._extract_logprobs(body.response)
            teacher_logprobs = self._extract_logprobs(teacher_response)

            if not student_logprobs or not teacher_logprobs:
                print("⚠️ Could not extract logprobs from student or teacher response")
                return 0.0

            # Step 4: Compute KL divergence KL(teacher || student)
            kl_divergence = self._compute_kl_divergence(teacher_logprobs, student_logprobs)
            print(f"DEBUG: KL divergence: {kl_divergence:.4f}")

            # Step 5: Convert KL to reward (lower KL = higher reward)
            # Using negative exponential: reward = max_reward * exp(-kl / temperature)
            reward = self.config.max_distillation_reward * math.exp(
                -kl_divergence / self.config.distillation_temperature
            )
            reward = max(0.0, min(reward, self.config.max_distillation_reward))

            print(f"DEBUG: Distillation reward: {reward:.4f}")
            return reward

        except Exception as e:
            print(f"ERROR in distillation reward computation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _extract_logprobs(self, response: NeMoGymResponse) -> Optional[List[Dict[str, float]]]:
        """Extract token-level logprobs from a response.

        Returns:
            List of dicts mapping token to logprob, one dict per token position.
            Returns None if logprobs not available.
        """
        try:
            # Navigate to logprobs in response structure
            # Response structure: output[i].content[j] may have logprobs
            for output_item in response.output:
                if getattr(output_item, "type", None) != "message":
                    continue

                for content_item in output_item.content:
                    # Check if this content item has logprobs
                    logprobs = getattr(content_item, "logprobs", None)
                    if logprobs is not None:
                        # Parse logprobs structure (varies by API)
                        # OpenAI format: logprobs.content is a list of token objects
                        if hasattr(logprobs, "content") and logprobs.content:
                            result = []
                            for token_logprob in logprobs.content:
                                token = getattr(token_logprob, "token", None)
                                logprob = getattr(token_logprob, "logprob", None)
                                top_logprobs = getattr(token_logprob, "top_logprobs", [])

                                # Build distribution from top_logprobs
                                if top_logprobs:
                                    token_dist = {}
                                    for top_item in top_logprobs:
                                        t = getattr(top_item, "token", None)
                                        lp = getattr(top_item, "logprob", None)
                                        if t is not None and lp is not None:
                                            token_dist[t] = lp
                                    result.append(token_dist)
                                elif token is not None and logprob is not None:
                                    # Fallback: use single token
                                    result.append({token: logprob})

                            if result:
                                return result

            return None

        except Exception as e:
            print(f"ERROR extracting logprobs: {e}")
            return None

    def _compute_kl_divergence(
        self,
        teacher_logprobs: List[Dict[str, float]],
        student_logprobs: List[Dict[str, float]]
    ) -> float:
        """Compute KL divergence KL(teacher || student) from token logprobs.

        Args:
            teacher_logprobs: List of dicts mapping token to logprob (teacher distribution)
            student_logprobs: List of dicts mapping token to logprob (student distribution)

        Returns:
            Average KL divergence across token positions
        """
        if len(teacher_logprobs) == 0:
            return float('inf')

        total_kl = 0.0
        num_positions = 0

        # Compute KL for each token position
        for pos in range(min(len(teacher_logprobs), len(student_logprobs))):
            teacher_dist = teacher_logprobs[pos]
            student_dist = student_logprobs[pos]

            # Compute KL(teacher || student) = sum_k P_teacher(k) * log(P_teacher(k) / P_student(k))
            pos_kl = 0.0
            for token, teacher_logprob in teacher_dist.items():
                teacher_prob = math.exp(teacher_logprob)

                # Get student logprob for this token (use very low prob if not present)
                student_logprob = student_dist.get(token, -20.0)  # ~2e-9 probability
                student_prob = math.exp(student_logprob)

                # KL contribution: P(teacher) * log(P(teacher) / P(student))
                if teacher_prob > 1e-10:  # Avoid numerical issues
                    pos_kl += teacher_prob * (teacher_logprob - student_logprob)

            total_kl += pos_kl
            num_positions += 1

        # Return average KL per token
        return total_kl / num_positions if num_positions > 0 else float('inf')

    def setup_webserver(self) -> FastAPI:
        # --- AUTO-LAUNCH SANDBOX ---
        if os.getenv("SHARED_TEMP_DIR") is None:
            self.launch_sandbox()
        # ---------------------------

        app = super().setup_webserver()

        # FIX FOR 404: Explicitly register the verify route
        app.add_api_route("/verify", self.verify, methods=["POST"])

        os.environ["NEMO_SKILLS_SANDBOX_HOST"] = self.config.sandbox_host
        os.environ["NEMO_SKILLS_SANDBOX_PORT"] = str(self.config.sandbox_port)

        print(f"Initializing ICPC Evaluator with metadata: {self.config.test_file}")

        self._evaluator = ICPCEvaluator(
            config={
                "test_file": self.config.test_file,
                "input_file": None, 
                "test_batch_size": self.config.test_batch_size,
                "shared_dir": self.config.shared_dir,
            },
            num_parallel_requests=self.config.num_parallel_requests,
        )

        return app

    async def verify(self, body: IcpcVerifyRequest) -> IcpcVerifyResponse:
        print("DEBUG: Verify request received.")

        if not self._evaluator:
            raise RuntimeError("Evaluator not initialized.")

        if getattr(body.response, "output", None):
            for out in body.response.output:
                if getattr(out, "type", None) == "message" and getattr(out, "role", None) == "assistant":
                    content = getattr(out, "content", None) or []
                    # content items are like NeMoGymResponseOutputText(type='output_text', text='...')
                    parts = []
                    for c in content:
                        t = getattr(c, "text", None)
                        if t:
                            parts.append(t)
                    generation = "\n".join(parts).strip()
                    print(f"DEBUG: Extracted Generation: {generation!r}")
                    if generation:
                        break
        
        params = body.responses_create_params
        prompt = getattr(params, "input", None)
        
        if not prompt and hasattr(params, "messages") and params.messages:
             prompt = next((m.content for m in reversed(params.messages) if m.role == "user"), "")
        
        if prompt is None: 
            prompt = ""

        problem_id = body.icpc_id

        sample = {
            "name": problem_id, # FIX FOR KeyError: 'name'
            "icpc_id": problem_id,
            "generation": generation,
        }

        reward = 0.0
        evaluation_result = {}
        distillation_reward = 0.0

        try:
            evaluation_result = await self._evaluator.eval_single(sample)

            outputs = []
            # Check if evaluation_result is valid before accessing it
            if evaluation_result and isinstance(evaluation_result, dict):
                test_results = evaluation_result.get("test_case_results")
                if test_results and isinstance(test_results, dict):
                    outputs = test_results.get("outputs", [])

            if outputs:
                reward = sum(o.get("score", 0.0) for o in outputs) / len(outputs)
            else:
                reward = 0.0

        except Exception as e:
            print(f"CRITICAL ERROR in evaluation: {e}")
            evaluation_result = {"error": str(e)}
            reward = 0.0

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"CRITICAL ERROR in evaluation: {e}")
            reward = 0.0
            evaluation_result = {"error": str(e)}

        # On-policy distillation: Use teacher model when tests fail (reward = 0)
        if reward == 0.0 and self.config.enable_distillation:
            print("🎓 DEBUG: Tests failed (reward=0), computing distillation reward from teacher model")
            distillation_reward = await self._compute_distillation_reward(body)
            reward = distillation_reward
            evaluation_result["distillation_reward"] = distillation_reward
            evaluation_result["reward_source"] = "distillation"
            print(f"🎓 DEBUG: Using distillation reward: {reward:.4f}")
        else:
            evaluation_result["reward_source"] = "tests"
            if self.config.enable_distillation:
                print(f"✅ DEBUG: Tests passed (reward={reward:.4f}), skipping distillation")

        print(f"DEBUG: Extracted Problem ID: {problem_id}")
        print(f"DEBUG: Generation: {generation}")
        print(f"DEBUG: Evaluation Result: {evaluation_result}")
        print(f"DEBUG: NEMO_SKILLS_SANDBOX_HOST {os.getenv('NEMO_SKILLS_SANDBOX_HOST')}")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                hb_url = f"http://{os.getenv('NEMO_SKILLS_SANDBOX_HOST')}:{os.getenv('NEMO_SKILLS_SANDBOX_PORT')}/health"
                hb_resp = await client.get(hb_url, timeout=2.0)
                print(f"DEBUG: Sandbox health check: {hb_resp.status_code}")
        except Exception as e:
            print(f"CRITICAL: Worker at {socket.gethostname()} cannot reach sandbox at {os.getenv('NEMO_SKILLS_SANDBOX_HOST')}: {e}")

            
        return IcpcVerifyResponse(
            **body.model_dump(), 
            reward=reward, 
            details=evaluation_result
        )

if __name__ == "__main__":
    Icpc25ResourcesServer.run_webserver()