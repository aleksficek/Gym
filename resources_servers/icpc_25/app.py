# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import math
import os
import socket
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

from icpc import ICPCEvaluator


class IcpcVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    icpc_id: str
    competition: str


class IcpcVerifyResponse(BaseVerifyResponse):
    details: Dict[str, Any] = Field(default_factory=dict)
    distillation_loss: float = 0.0
    execution_loss: float = 1.0
    combined_loss: float = 1.0
    shaped_reward: float = 0.0


class Icpc25ResourcesServerConfig(BaseResourcesServerConfig):
    test_file: str = os.getenv("TEST_FILE", "/home/aficek/software/storage/data/icpc/icpc_all_metadata.jsonl")
    sandbox_host: str = os.getenv("NEMO_SKILLS_SANDBOX_HOST", os.getenv("RAY_HEAD_IP", "localhost"))
    sandbox_port: int = 6000
    test_batch_size: int = 4
    num_parallel_requests: int = 2
    shared_dir: str = os.getenv("SHARED_TEMP_DIR", "/tmp")

    # Distillation controls.
    # Keep these Hydra-friendly so cluster training can set them from a single GRPO yaml.
    distillation_enabled: bool = False
    distillation_weight: float = 0.2
    distillation_floor_logprob: float = -30.0
    distillation_compare_max_positions: int = 256
    distillation_top_logprobs: int = 20
    distillation_max_tokens: int = 256
    distillation_temperature: float = 0.0
    distillation_top_p: float = 1.0
    distillation_timeout_seconds: float = 60.0

    # Usually wired to ${policy_base_url} and ${policy_model_name} in config yaml.
    # base_url may be a list of DP endpoints in multi-node training.
    distillation_base_url: Optional[Union[str, List[str]]] = None
    distillation_model: Optional[str] = None

    # teacher_mode: "same_model" reuses policy endpoint/model; "different_model" uses explicit overrides.
    distillation_teacher_mode: str = "same_model"
    distillation_teacher_base_url: Optional[Union[str, List[str]]] = None
    distillation_teacher_model: Optional[str] = None

    # Optional same-model teacher bias; expects request extra field (default: teacher_reference).
    distillation_teacher_reference_field: str = "teacher_reference"
    distillation_teacher_bias_template: str = (
        "You are the teacher policy for on-policy distillation. "
        "Use this reference as a strong guide for the correct solution style and logic:\n"
        "{reference}"
    )


class Icpc25ResourcesServer(SimpleResourcesServer):
    config: Icpc25ResourcesServerConfig

    _evaluator: Optional[ICPCEvaluator] = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:

        app = super().setup_webserver()
        app.add_api_route("/verify", self.verify, methods=["POST"])

        os.environ["NEMO_SKILLS_SANDBOX_HOST"] = self.config.sandbox_host
        os.environ["NEMO_SKILLS_SANDBOX_PORT"] = str(self.config.sandbox_port)

        print(f"Initializing ICPC evaluator with metadata: {self.config.test_file}")
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

    @staticmethod
    def _normalize_openai_base_url(base_url: str) -> str:
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            return trimmed
        return f"{trimmed}/v1"

    @staticmethod
    def _pick_base_url(
        base_url_or_urls: Optional[Union[str, List[str]]],
        routing_key: str,
        fallback: str = "http://localhost:8000/v1",
    ) -> str:
        if isinstance(base_url_or_urls, str):
            return base_url_or_urls

        urls: List[str] = []
        if isinstance(base_url_or_urls, list):
            urls = [u for u in base_url_or_urls if isinstance(u, str) and u.strip()]
        elif base_url_or_urls is not None and hasattr(base_url_or_urls, "__iter__"):
            urls = [u for u in base_url_or_urls if isinstance(u, str) and u.strip()]

        if not urls:
            return fallback
        if len(urls) == 1:
            return urls[0]

        digest = hashlib.sha256(routing_key.encode("utf-8", errors="replace")).hexdigest()
        index = int(digest[:8], 16) % len(urls)
        return urls[index]

    @staticmethod
    def _extract_generation(body: IcpcVerifyRequest) -> str:
        generation = ""
        for out in getattr(body.response, "output", []) or []:
            if getattr(out, "type", None) != "message" or getattr(out, "role", None) != "assistant":
                continue
            parts = []
            for content in getattr(out, "content", None) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
            generation = "\n".join(parts).strip()
            if generation:
                break
        return generation

    @staticmethod
    def _input_to_chat_messages(raw_input: Any) -> List[Dict[str, str]]:
        if raw_input is None:
            return []
        if isinstance(raw_input, str):
            return [{"role": "user", "content": raw_input}]

        messages: List[Dict[str, str]] = []
        if not isinstance(raw_input, list):
            return messages

        for item in raw_input:
            if hasattr(item, "model_dump"):
                item = item.model_dump(mode="json")
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue

            role = item.get("role")
            if role not in {"user", "assistant", "system", "developer"}:
                continue
            if role == "developer":
                role = "system"

            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        if part_text:
                            text_parts.append(part_text)
                text = "".join(text_parts)
            else:
                text = ""

            messages.append({"role": role, "content": text})
        return messages

    def _teacher_reference(self, body: IcpcVerifyRequest) -> Optional[str]:
        extras = body.model_extra or {}
        candidate_keys = [
            self.config.distillation_teacher_reference_field,
            "teacher_reference",
            "teacher_solution",
            "reference_solution",
            "ground_truth_solution",
        ]
        for key in candidate_keys:
            value = extras.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    async def _chat_completion_with_logprobs(
        self,
        *,
        base_url: str,
        model: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        url = f"{self._normalize_openai_base_url(base_url)}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.distillation_max_tokens,
            "temperature": self.config.distillation_temperature,
            "top_p": self.config.distillation_top_p,
            "logprobs": True,
            "top_logprobs": self.config.distillation_top_logprobs,
            "stream": False,
            # Important for stable token identity in vLLM.
            "return_tokens_as_token_ids": True,
        }

        timeout = httpx.Timeout(self.config.distillation_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        choice = ((data.get("choices") or [{}])[0]) if isinstance(data, dict) else {}
        message = choice.get("message") or {}
        logprobs = (choice.get("logprobs") or {}).get("content") or []
        return {
            "text": message.get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "token_logprobs": logprobs if isinstance(logprobs, list) else [],
        }

    @staticmethod
    def _logprob_map(token_entry: Dict[str, Any]) -> Dict[str, float]:
        token_to_lp: Dict[str, float] = {}
        token = token_entry.get("token")
        logprob = token_entry.get("logprob")
        if token is not None and logprob is not None:
            token_to_lp[str(token)] = float(logprob)

        for alt in token_entry.get("top_logprobs") or []:
            if not isinstance(alt, dict):
                continue
            alt_token = alt.get("token")
            alt_logprob = alt.get("logprob")
            if alt_token is None or alt_logprob is None:
                continue
            alt_token = str(alt_token)
            alt_logprob = float(alt_logprob)
            if alt_token not in token_to_lp or alt_logprob > token_to_lp[alt_token]:
                token_to_lp[alt_token] = alt_logprob
        return token_to_lp

    @staticmethod
    def _probs_from_logprobs(logprob_map: Dict[str, float]) -> Dict[str, float]:
        if not logprob_map:
            return {}
        max_lp = max(logprob_map.values())
        exp_probs = {token: math.exp(lp - max_lp) for token, lp in logprob_map.items()}
        z = sum(exp_probs.values())
        if z <= 0:
            return {}
        return {token: value / z for token, value in exp_probs.items()}

    def _kl_teacher_to_student(self, teacher_lp: Dict[str, float], student_lp: Dict[str, float]) -> float:
        floor = self.config.distillation_floor_logprob
        vocab = set(teacher_lp) | set(student_lp)
        if not vocab:
            return 0.0

        teacher_aug = {token: teacher_lp.get(token, floor) for token in vocab}
        student_aug = {token: student_lp.get(token, floor) for token in vocab}
        teacher_probs = self._probs_from_logprobs(teacher_aug)
        student_probs = self._probs_from_logprobs(student_aug)
        eps = 1e-12

        kl = 0.0
        for token in vocab:
            t_prob = teacher_probs.get(token, 0.0)
            s_prob = max(student_probs.get(token, 0.0), eps)
            if t_prob > 0:
                kl += t_prob * (math.log(max(t_prob, eps)) - math.log(s_prob))
        return max(0.0, kl)

    async def _compute_distillation(self, body: IcpcVerifyRequest) -> Dict[str, Any]:
        if not self.config.distillation_enabled:
            return {"enabled": False, "status": "disabled", "distillation_loss": 0.0}

        prompt_input = getattr(body.responses_create_params, "input", None)
        prompt_messages = self._input_to_chat_messages(prompt_input)
        if not prompt_messages:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "no_prompt_messages",
            }

        routing_key = f"{body.competition}:{body.icpc_id}"
        policy_base_urls = (
            self.config.distillation_base_url
            or os.getenv("policy_base_url")
            or os.getenv("POLICY_BASE_URL")
        )
        student_base_url = self._pick_base_url(policy_base_urls, routing_key)
        student_model = (
            self.config.distillation_model
            or os.getenv("policy_model_name")
            or os.getenv("POLICY_MODEL_NAME")
            or "model"
        )

        if self.config.distillation_teacher_mode == "different_model":
            teacher_base_url = self._pick_base_url(
                self.config.distillation_teacher_base_url or policy_base_urls,
                f"{routing_key}:teacher",
                fallback=student_base_url,
            )
            teacher_model = self.config.distillation_teacher_model or student_model
        else:
            teacher_base_url = student_base_url
            teacher_model = self.config.distillation_teacher_model or student_model

        teacher_messages = deepcopy(prompt_messages)
        teacher_reference = self._teacher_reference(body)
        if teacher_reference:
            try:
                bias_text = self.config.distillation_teacher_bias_template.format(reference=teacher_reference)
            except Exception:
                bias_text = (
                    "You are the teacher policy for on-policy distillation. "
                    f"Use this reference as strong guidance:\n{teacher_reference}"
                )
            teacher_messages = [{"role": "system", "content": bias_text}] + teacher_messages

        student_task = self._chat_completion_with_logprobs(
            base_url=student_base_url,
            model=student_model,
            messages=prompt_messages,
        )
        teacher_task = self._chat_completion_with_logprobs(
            base_url=teacher_base_url,
            model=teacher_model,
            messages=teacher_messages,
        )
        student_result, teacher_result = await asyncio.gather(student_task, teacher_task)

        student_tokens = student_result.get("token_logprobs") or []
        teacher_tokens = teacher_result.get("token_logprobs") or []
        aligned = min(
            len(student_tokens),
            len(teacher_tokens),
            self.config.distillation_compare_max_positions,
        )
        if aligned <= 0:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "no_aligned_token_logprobs",
                "student_num_tokens": len(student_tokens),
                "teacher_num_tokens": len(teacher_tokens),
            }

        kls: List[float] = []
        teacher_ce_on_student: List[float] = []
        token_agreement = 0
        floor = self.config.distillation_floor_logprob

        for idx in range(aligned):
            student_entry = student_tokens[idx]
            teacher_entry = teacher_tokens[idx]
            if not isinstance(student_entry, dict) or not isinstance(teacher_entry, dict):
                continue

            student_lp = self._logprob_map(student_entry)
            teacher_lp = self._logprob_map(teacher_entry)
            if not student_lp or not teacher_lp:
                continue

            kls.append(self._kl_teacher_to_student(teacher_lp, student_lp))

            student_token = str(student_entry.get("token"))
            teacher_ce_on_student.append(-teacher_lp.get(student_token, floor))
            if str(student_entry.get("token")) == str(teacher_entry.get("token")):
                token_agreement += 1

        if not kls:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "empty_kl_set",
                "student_num_tokens": len(student_tokens),
                "teacher_num_tokens": len(teacher_tokens),
            }

        distillation_loss = float(sum(kls) / len(kls))
        ce_loss = float(sum(teacher_ce_on_student) / len(teacher_ce_on_student)) if teacher_ce_on_student else 0.0
        token_agreement_ratio = float(token_agreement / aligned)

        return {
            "enabled": True,
            "status": "ok",
            "teacher_mode": self.config.distillation_teacher_mode,
            "student_base_url": self._normalize_openai_base_url(student_base_url),
            "student_model": student_model,
            "teacher_base_url": self._normalize_openai_base_url(teacher_base_url),
            "teacher_model": teacher_model,
            "teacher_reference_used": bool(teacher_reference),
            "distillation_loss": distillation_loss,
            "teacher_cross_entropy_on_student": ce_loss,
            "token_agreement_ratio": token_agreement_ratio,
            "aligned_positions": aligned,
            "student_num_tokens": len(student_tokens),
            "teacher_num_tokens": len(teacher_tokens),
        }

    async def verify(self, body: IcpcVerifyRequest) -> IcpcVerifyResponse:
        if not self._evaluator:
            raise RuntimeError("Evaluator not initialized.")

        generation = self._extract_generation(body)
        sample = {
            "name": body.icpc_id,
            "icpc_id": body.icpc_id,
            "competition": body.competition,
            "generation": generation,
        }

        eval_task = self._evaluator.eval_single(sample)
        distill_task = self._compute_distillation(body)
        eval_result_raw, distill_result_raw = await asyncio.gather(
            eval_task,
            distill_task,
            return_exceptions=True,
        )

        if isinstance(eval_result_raw, Exception):
            print(f"CRITICAL ERROR in evaluation: {eval_result_raw}")
            evaluation_result: Dict[str, Any] = {"error": str(eval_result_raw)}
            reward = 0.0
            score_summary = {"num_outputs": 0, "num_perfect": 0}
        else:
            evaluation_result = eval_result_raw if isinstance(eval_result_raw, dict) else {}
            score_summary = ICPCEvaluator.summarize_test_case_results(evaluation_result)
            reward = float(score_summary["reward"])

        if isinstance(distill_result_raw, Exception):
            print(f"CRITICAL ERROR in distillation: {distill_result_raw}")
            distillation_result: Dict[str, Any] = {
                "enabled": self.config.distillation_enabled,
                "status": "error",
                "distillation_loss": 0.0,
                "error": str(distill_result_raw),
            }
        else:
            distillation_result = distill_result_raw if isinstance(distill_result_raw, dict) else {}

        distillation_loss = float(distillation_result.get("distillation_loss", 0.0))
        execution_loss = float(max(0.0, 1.0 - reward))
        if self.config.distillation_enabled and distillation_result.get("status") == "ok":
            combined_loss = execution_loss + self.config.distillation_weight * distillation_loss
            shaped_reward = max(
                0.0,
                min(1.0, reward + self.config.distillation_weight * math.exp(-distillation_loss)),
            )
        else:
            combined_loss = execution_loss
            shaped_reward = reward

        details = dict(evaluation_result)
        details["score_summary"] = score_summary
        details["distillation"] = distillation_result
        details["losses"] = {
            "execution_loss": execution_loss,
            "distillation_loss": distillation_loss,
            "combined_loss": combined_loss,
            "distillation_weight": self.config.distillation_weight,
            "shaped_reward": shaped_reward,
        }

        try:
            hb_url = (
                f"http://{os.getenv('NEMO_SKILLS_SANDBOX_HOST')}:"
                f"{os.getenv('NEMO_SKILLS_SANDBOX_PORT')}/health"
            )
            async with httpx.AsyncClient(timeout=2.0) as client:
                hb_resp = await client.get(hb_url)
                print(f"Sandbox health check: {hb_resp.status_code}")
        except Exception as exc:
            print(
                f"Worker at {socket.gethostname()} cannot reach sandbox at "
                f"{os.getenv('NEMO_SKILLS_SANDBOX_HOST')}: {exc}"
            )

        return IcpcVerifyResponse(
            **body.model_dump(),
            reward=reward,
            details=details,
            distillation_loss=distillation_loss,
            execution_loss=execution_loss,
            combined_loss=combined_loss,
            shaped_reward=shaped_reward,
        )


if __name__ == "__main__":
    Icpc25ResourcesServer.run_webserver()
