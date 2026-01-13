# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import socket
import subprocess
import time
import shutil
import threading
from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)

from icpc import ICPCEvaluator

# --- Custom Response Class ---
class IcpcVerifyResponse(BaseVerifyResponse):
    details: Dict[str, Any] = {}

class Icpc25ResourcesServerConfig(BaseResourcesServerConfig):
    test_file: str = "/home/aficek/software/storage/data/icpc_25/metadata/icpc25_metadata.json"
    sandbox_host: str = "localhost"
    sandbox_port: int = 6000
    test_batch_size: int = 4
    num_parallel_requests: int = 2
    
    sandbox_image: str = "docker.io/igitman/nemo-skills-sandbox:0.7.1"
    data_volume: str = "/home/aficek/software/storage/data/icpc_25:/home/aficek/software/storage/data/icpc_25"


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

    def setup_webserver(self) -> FastAPI:
        # --- AUTO-LAUNCH SANDBOX ---
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
            },
            num_parallel_requests=self.config.num_parallel_requests,
        )

        return app

    async def verify(self, body: BaseVerifyRequest) -> IcpcVerifyResponse:
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
                    if generation:
                        break
        
        params = body.responses_create_params
        prompt = getattr(params, "input", None)
        
        if not prompt and hasattr(params, "messages") and params.messages:
             prompt = next((m.content for m in reversed(params.messages) if m.role == "user"), "")
        
        if prompt is None: 
            prompt = ""

        problem_id = "buggyrover" 
        print(f"DEBUG: Extracted Problem ID: {problem_id}")

        sample = {
            "name": problem_id, # FIX FOR KeyError: 'name'
            "icpc_id": problem_id,
            "generation": generation,
        }

        reward = 0.0
        evaluation_result = {}

        try:
            evaluation_result = await self._evaluator.eval_single(sample)

            outputs = evaluation_result.get("test_case_results", {}).get("outputs", [])
            if outputs:
                reward = sum(o.get("score", 0.0) for o in outputs) / len(outputs)

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

        return IcpcVerifyResponse(
            **body.model_dump(), 
            reward=reward, 
            details=evaluation_result
        )

if __name__ == "__main__":
    Icpc25ResourcesServer.run_webserver()