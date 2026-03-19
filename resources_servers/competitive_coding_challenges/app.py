# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

import ccc_eval as ccc_eval_module
from ccc_eval import CCCEvaluator


class CompetitiveCodingChallengesVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    competition_id: Optional[str] = None
    problem_id: str
    subtask: Optional[str] = None
    name: Optional[str] = None
    subtask_score: Optional[float] = None


class CompetitiveCodingChallengesVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    competition_id: Optional[str] = None
    problem_id: str
    subtask: Optional[str] = None
    name: Optional[str] = None
    subtask_score: Optional[float] = None
    details: dict[str, Any] = Field(default_factory=dict)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for output in body.response.output or []:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts).strip()


class CompetitiveCodingChallengesResourcesServerConfig(BaseResourcesServerConfig):
    # These fields are populated from the NeMo-Gym config tree, so they can be
    # overridden from `run_grpo_nemo_gym.py` via Hydra CLI args instead of being
    # hard-coded in the server implementation.
    test_file: str = "data/test_metadata.jsonl"
    test_batch_size: int = 16
    num_parallel_requests: int = 10
    time_scale: float = 1.0
    shared_dir: str = "/tmp"


class CompetitiveCodingChallengesResourcesServer(SimpleResourcesServer):
    config: CompetitiveCodingChallengesResourcesServerConfig

    _evaluator: Optional[CCCEvaluator] = PrivateAttr(default=None)

    def _subtask_max_score(self, body: CompetitiveCodingChallengesVerifyRequest) -> Optional[float]:
        if body.subtask_score is not None:
            return float(body.subtask_score)
        if not self._evaluator or not body.subtask:
            return None
        try:
            problem_meta = self._evaluator.get_problem_metadata(body.problem_id, body.competition_id)
        except Exception:
            return None
        subtask_meta = (problem_meta.get("subtasks") or {}).get(body.subtask, {})
        if not subtask_meta:
            return None
        if subtask_meta.get("aggregation") == "sum_tests":
            return float(len(subtask_meta.get("test_names", [])))
        score = subtask_meta.get("score")
        return float(score) if score is not None else None

    def _compute_reward(
        self,
        body: CompetitiveCodingChallengesVerifyRequest,
        evaluation_result: dict[str, Any],
    ) -> float:
        test_case_results = evaluation_result.get("test_case_results") or {}
        if body.subtask and body.subtask in test_case_results:
            subtask_result = test_case_results[body.subtask]
            score = float(subtask_result.get("score", 0.0) or 0.0)
            max_score = self._subtask_max_score(body)
            return 1.0 if (max_score and score >= max_score) or (max_score is None and score > 0.0) else 0.0

        total_score = sum(float(result.get("score", 0.0) or 0.0) for result in test_case_results.values())
        problem_meta = {}
        if self._evaluator:
            try:
                problem_meta = self._evaluator.get_problem_metadata(body.problem_id, body.competition_id)
            except Exception:
                problem_meta = {}

        max_total = 0.0
        for subtask_meta in (problem_meta.get("subtasks") or {}).values():
            if subtask_meta.get("aggregation") == "sum_tests":
                max_total += float(len(subtask_meta.get("test_names", [])))
            else:
                max_total += float(subtask_meta.get("score", 0.0) or 0.0)

        if max_total and total_score >= max_total:
            return 1.0
        if max_total == 0.0 and all(float(result.get("score", 0.0) or 0.0) > 0.0 for result in test_case_results.values()):
            return 1.0
        return 0.0

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        print(
            "Initializing CompetitiveCodingChallenges evaluator with config: "
            f"{self.config.model_dump_json(indent=2)}"
        )

        self._evaluator = CCCEvaluator(
            config={
                "test_file": self.config.test_file,
                "test_batch_size": self.config.test_batch_size,
                "time_scale": self.config.time_scale,
                "shared_dir": self.config.shared_dir,
            },
            num_parallel_requests=self.config.num_parallel_requests,
        )
        # The copied evaluator's module-level helpers still reference `self`.
        ccc_eval_module.self = self._evaluator

        return app

    async def verify(
        self, body: CompetitiveCodingChallengesVerifyRequest
    ) -> CompetitiveCodingChallengesVerifyResponse:
        if not self._evaluator:
            raise RuntimeError("Evaluator not initialized.")

        payload = body.model_dump()
        payload["name"] = payload.get("name") or body.problem_id

        reward = 0.0
        details: dict[str, Any] = {}
        try:
            details = await self._evaluator.eval_single(
                {
                    "competition_id": body.competition_id,
                    "name": payload["name"],
                    "problem_id": body.problem_id,
                    "subtask": body.subtask,
                    "generation": _extract_last_assistant_text(body),
                }
            )
            reward = self._compute_reward(body, details)
        except Exception as e:
            details = {"error": str(e)}

        return CompetitiveCodingChallengesVerifyResponse(**payload, reward=reward, details=details)


if __name__ == "__main__":
    CompetitiveCodingChallengesResourcesServer.run_webserver()
