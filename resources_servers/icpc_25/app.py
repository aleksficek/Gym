# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import math
import os
import socket
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

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
    distillation_temperature: float = 1.0
    distillation_top_p: float = 1.0
    distillation_timeout_seconds: float = 60.0
    distillation_reward_mode: str = "reward_shaping"
    distillation_token_weight_scale: float = 0.5
    distillation_token_weight_min: float = 0.25
    distillation_token_weight_max: float = 1.75
    distillation_reward_temperature: float = 10.0

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
        "Problem:\n"
        "{problem}\n\n"
        "Here is a reference solution:\n"
        "{reference}\n\n"
        "After understanding the reference solution, please try to solve this problem\n"
        "using your own approach below:\n"
        "Answer:\n"
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
    def _extract_student_generation_and_token_info(
        body: IcpcVerifyRequest,
    ) -> Tuple[str, List[str], List[float]]:
        generation = ""
        token_keys: List[str] = []
        token_logprobs: List[float] = []

        for out in getattr(body.response, "output", []) or []:
            if hasattr(out, "model_dump"):
                out_dict = out.model_dump(mode="json")
            elif isinstance(out, dict):
                out_dict = out
            else:
                continue

            if out_dict.get("type") != "message" or out_dict.get("role") != "assistant":
                continue

            content = out_dict.get("content") or []
            text_parts: List[str] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
            generation = "".join(text_parts)

            raw_token_ids = out_dict.get("generation_token_ids") or []
            raw_logprobs = out_dict.get("generation_log_probs") or []
            if isinstance(raw_token_ids, list):
                for token_id in raw_token_ids:
                    try:
                        token_keys.append(f"token_id:{int(token_id)}")
                    except Exception:
                        token_keys.append(str(token_id))
            if isinstance(raw_logprobs, list):
                for lp in raw_logprobs:
                    try:
                        token_logprobs.append(float(lp))
                    except Exception:
                        token_logprobs.append(0.0)

            # Fallback for non-training payloads where token-id fields are absent but
            # response content carries per-token logprobs.
            if not token_keys and isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_logprobs = part.get("logprobs")
                    if not isinstance(part_logprobs, list):
                        continue
                    for token_entry in part_logprobs:
                        if not isinstance(token_entry, dict):
                            continue
                        token = token_entry.get("token")
                        if token is None:
                            continue
                        token_keys.append(Icpc25ResourcesServer._normalize_token_key(token))
                        try:
                            token_logprobs.append(float(token_entry.get("logprob", 0.0)))
                        except Exception:
                            token_logprobs.append(0.0)
            break

        return generation, token_keys, token_logprobs

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

    @staticmethod
    def _messages_to_problem_text(messages: List[Dict[str, str]]) -> str:
        chunks: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            text = msg.get("content")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n\n".join(chunks).strip()

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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._normalize_openai_base_url(base_url)}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.distillation_max_tokens if max_tokens is None else max_tokens,
            "temperature": (
                self.config.distillation_temperature if temperature is None else temperature
            ),
            "top_p": self.config.distillation_top_p if top_p is None else top_p,
            "logprobs": True,
            "top_logprobs": self.config.distillation_top_logprobs,
            "stream": False,
            # Important for stable token identity in vLLM.
            "return_tokens_as_token_ids": True,
        }
        if extra_payload:
            payload.update(extra_payload)

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
            "raw": data,
        }

    @staticmethod
    def _normalize_token_key(token: Any) -> str:
        token_str = str(token)
        if token_str.startswith("token_id:"):
            return token_str
        if token_str.isdigit():
            return f"token_id:{token_str}"
        return token_str

    @staticmethod
    def _logprob_map(token_entry: Dict[str, Any]) -> Dict[str, float]:
        token_to_lp: Dict[str, float] = {}
        token = token_entry.get("token")
        logprob = token_entry.get("logprob")
        if token is not None and logprob is not None:
            token_to_lp[Icpc25ResourcesServer._normalize_token_key(token)] = float(logprob)

        for alt in token_entry.get("top_logprobs") or []:
            if not isinstance(alt, dict):
                continue
            alt_token = alt.get("token")
            alt_logprob = alt.get("logprob")
            if alt_token is None or alt_logprob is None:
                continue
            alt_token = Icpc25ResourcesServer._normalize_token_key(alt_token)
            alt_logprob = float(alt_logprob)
            if alt_token not in token_to_lp or alt_logprob > token_to_lp[alt_token]:
                token_to_lp[alt_token] = alt_logprob
        return token_to_lp

    @staticmethod
    def _prompt_logprob_entries(raw: Any) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return entries

        for item in raw:
            if not isinstance(item, dict):
                continue

            if "token" in item and "logprob" in item:
                token = Icpc25ResourcesServer._normalize_token_key(item.get("token"))
                top_logprobs = item.get("top_logprobs") or []
                entries.append(
                    {
                        "token": token,
                        "logprob": float(item.get("logprob", 0.0)),
                        "top_logprobs": top_logprobs if isinstance(top_logprobs, list) else [],
                    }
                )
                continue

            best_token: Optional[str] = None
            best_logprob = -float("inf")
            top_logprobs: List[Dict[str, Any]] = []
            for maybe_token, maybe_value in item.items():
                token = maybe_token
                logprob: Optional[float] = None
                if isinstance(maybe_value, dict):
                    token = maybe_value.get("token", maybe_token)
                    maybe_logprob = maybe_value.get("logprob")
                    if maybe_logprob is not None:
                        logprob = float(maybe_logprob)
                elif isinstance(maybe_value, (float, int)):
                    logprob = float(maybe_value)

                if token is None or logprob is None:
                    continue
                token_key = Icpc25ResourcesServer._normalize_token_key(token)
                top_logprobs.append({"token": token_key, "logprob": logprob})
                if logprob > best_logprob:
                    best_logprob = logprob
                    best_token = token_key

            if best_token is not None:
                entries.append(
                    {
                        "token": best_token,
                        "logprob": best_logprob,
                        "top_logprobs": top_logprobs,
                    }
                )
        return entries

    @staticmethod
    def _extract_prompt_logprob_entries(raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(raw_response, dict):
            return []

        choice = (raw_response.get("choices") or [{}])[0]
        if not isinstance(choice, dict):
            choice = {}
        choice_logprobs = choice.get("logprobs") or {}
        if not isinstance(choice_logprobs, dict):
            choice_logprobs = {}

        candidates = [
            raw_response.get("prompt_logprobs"),
            choice.get("prompt_logprobs"),
            choice_logprobs.get("prompt_logprobs"),
            choice_logprobs.get("prompt"),
        ]
        for candidate in candidates:
            entries = Icpc25ResourcesServer._prompt_logprob_entries(candidate)
            if entries:
                return entries
        return []

    @staticmethod
    def _align_prompt_entries_to_student_tokens(
        prompt_entries: List[Dict[str, Any]],
        student_token_keys: List[str],
        max_positions: int,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        num_student = len(student_token_keys)
        if num_student <= 0:
            return 0, []
        if len(prompt_entries) <= 0:
            return 0, []

        prompt_token_keys = [
            Icpc25ResourcesServer._normalize_token_key(entry.get("token"))
            for entry in prompt_entries
        ]

        max_window = min(len(prompt_token_keys), num_student, max_positions)
        if max_window <= 0:
            return 0, []

        # Prefer suffix alignment: prompt logprobs may be left-truncated by context
        # length, so matching the latest student tokens is most robust.
        for window in range(max_window, 0, -1):
            student_start = num_student - window
            student_slice = student_token_keys[student_start : student_start + window]
            for prompt_start in range(len(prompt_token_keys) - window, -1, -1):
                if prompt_token_keys[prompt_start : prompt_start + window] == student_slice:
                    return student_start, prompt_entries[prompt_start : prompt_start + window]

        # Fallback to prefix alignment when suffix alignment is unavailable.
        for window in range(max_window, 0, -1):
            student_slice = student_token_keys[:window]
            if prompt_token_keys[:window] == student_slice:
                return 0, prompt_entries[:window]

        return 0, []

    def _agreement_weight(self, teacher_lp_map: Dict[str, float], student_token_key: str) -> float:
        if not teacher_lp_map:
            return 1.0

        floor = self.config.distillation_floor_logprob
        student_lp = teacher_lp_map.get(student_token_key, floor)
        best_lp = max(teacher_lp_map.values())
        score = math.exp(min(0.0, student_lp - best_lp))
        weight = 1.0 + self.config.distillation_token_weight_scale * (2.0 * score - 1.0)
        return float(
            min(
                self.config.distillation_token_weight_max,
                max(self.config.distillation_token_weight_min, weight),
            )
        )

    async def _compute_distillation(self, body: IcpcVerifyRequest) -> Dict[str, Any]:
        if not self.config.distillation_enabled:
            return {"enabled": False, "status": "disabled", "distillation_loss": 0.0}

        extras = body.model_extra or {}
        ground_truth_solution = extras.get("ground_truth_solution")
        if isinstance(ground_truth_solution, str) and ground_truth_solution.strip():
            print(
                "[ICPC_DISTILL_DEBUG] ground_truth_solution (interpreted from request extras):\n"
                f"{ground_truth_solution.strip()}"
            )

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
            teacher_problem = self._messages_to_problem_text(prompt_messages)
            try:
                bias_text = self.config.distillation_teacher_bias_template.format(
                    problem=teacher_problem,
                    reference=teacher_reference,
                )
            except Exception:
                bias_text = (
                    f"Problem:\n{teacher_problem}\n\n"
                    "Here is a reference solution:\n"
                    f"{teacher_reference}\n\n"
                    "After understanding the reference solution, please try to solve this problem\n"
                    "using your own approach below:\n"
                    "Answer:\n"
                )
            teacher_messages = [{"role": "user", "content": bias_text}]

        student_generation, student_token_keys, student_token_logprobs = (
            self._extract_student_generation_and_token_info(body)
        )

        if not student_token_keys:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "no_student_tokens_in_rollout",
            }

        teacher_prompt_messages = deepcopy(teacher_messages) + [
            {"role": "assistant", "content": student_generation}
        ]
        responses_create_params = getattr(body, "responses_create_params", None)
        teacher_temperature = getattr(responses_create_params, "temperature", None)
        teacher_top_p = getattr(responses_create_params, "top_p", None)
        if teacher_temperature is None:
            teacher_temperature = self.config.distillation_temperature
        if teacher_top_p is None:
            teacher_top_p = self.config.distillation_top_p
        teacher_prompt_result = await self._chat_completion_with_logprobs(
            base_url=teacher_base_url,
            model=teacher_model,
            messages=teacher_prompt_messages,
            max_tokens=1,
            temperature=teacher_temperature,
            top_p=teacher_top_p,
            extra_payload={
                "prompt_logprobs": self.config.distillation_top_logprobs,
                "return_tokens_as_token_ids": True,
            },
        )
        prompt_entries = self._extract_prompt_logprob_entries(teacher_prompt_result.get("raw") or {})
        student_alignment_start, teacher_entries = self._align_prompt_entries_to_student_tokens(
            prompt_entries,
            student_token_keys,
            self.config.distillation_compare_max_positions,
        )
        teacher_scoring_mode = "on_policy_prompt_logprobs"

        if not teacher_entries:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "teacher_prompt_logprobs_unavailable",
                "student_num_tokens": len(student_token_keys),
                "teacher_prompt_entries": len(prompt_entries),
                "student_alignment_start": student_alignment_start,
            }

        max_positions = min(
            len(teacher_entries),
            self.config.distillation_compare_max_positions,
        )
        if max_positions <= 0:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "no_aligned_teacher_entries",
                "student_num_tokens": len(student_token_keys),
                "teacher_num_tokens": len(teacher_entries),
                "teacher_scoring_mode": teacher_scoring_mode,
                "student_alignment_start": student_alignment_start,
            }

        teacher_cross_entropy_on_student: List[float] = []
        teacher_student_logprob_gap: List[float] = []
        teacher_minus_student_logprob: List[float] = []
        teacher_logprob_on_student: List[float] = []
        student_logprob_on_student: List[float] = []
        token_advantage_weights: List[float] = [1.0] * max(0, int(student_alignment_start))
        token_agreement = 0
        floor = self.config.distillation_floor_logprob
        floor_hit_count = 0
        num_used = 0

        for idx in range(max_positions):
            student_idx = student_alignment_start + idx
            if student_idx >= len(student_token_keys):
                break
            teacher_entry = teacher_entries[idx]
            if not isinstance(teacher_entry, dict):
                token_advantage_weights.append(1.0)
                continue

            teacher_lp_map = self._logprob_map(teacher_entry)
            if not teacher_lp_map:
                token_advantage_weights.append(1.0)
                continue

            student_token_key = student_token_keys[student_idx]
            if student_token_key not in teacher_lp_map:
                floor_hit_count += 1
            teacher_lp_on_student = teacher_lp_map.get(student_token_key, floor)
            teacher_cross_entropy_on_student.append(-teacher_lp_on_student)
            teacher_logprob_on_student.append(teacher_lp_on_student)
            token_advantage_weights.append(self._agreement_weight(teacher_lp_map, student_token_key))
            if student_idx < len(student_token_logprobs):
                student_lp = float(student_token_logprobs[student_idx])
                student_logprob_on_student.append(student_lp)
                teacher_student_logprob_gap.append(
                    abs(teacher_lp_on_student - student_lp)
                )
                teacher_minus_student_logprob.append(teacher_lp_on_student - student_lp)

            teacher_best_token = max(teacher_lp_map.items(), key=lambda kv: kv[1])[0]
            if teacher_best_token == student_token_key:
                token_agreement += 1
            num_used += 1

        if num_used <= 0:
            return {
                "enabled": True,
                "status": "skipped",
                "distillation_loss": 0.0,
                "reason": "empty_teacher_scores",
                "student_num_tokens": len(student_token_keys),
                "teacher_num_tokens": len(teacher_entries),
                "teacher_scoring_mode": teacher_scoring_mode,
            }

        distillation_loss = float(sum(teacher_cross_entropy_on_student) / num_used)
        token_agreement_ratio = float(token_agreement / num_used)
        teacher_disagreement_ratio = float(1.0 - token_agreement_ratio)
        mean_teacher_logprob_on_student = (
            float(sum(teacher_logprob_on_student) / len(teacher_logprob_on_student))
            if teacher_logprob_on_student
            else 0.0
        )
        mean_student_logprob_on_student = (
            float(sum(student_logprob_on_student) / len(student_logprob_on_student))
            if student_logprob_on_student
            else 0.0
        )
        mean_teacher_student_logprob_gap = (
            float(sum(teacher_student_logprob_gap) / len(teacher_student_logprob_gap))
            if teacher_student_logprob_gap
            else 0.0
        )
        mean_teacher_minus_student_logprob = (
            float(sum(teacher_minus_student_logprob) / len(teacher_minus_student_logprob))
            if teacher_minus_student_logprob
            else 0.0
        )
        floor_hit_ratio = float(floor_hit_count / num_used)
        print(
            f"[ICPC_DISTILL_DEBUG] {body.competition}:{body.icpc_id} "
            f"loss={distillation_loss:.4f} agree={token_agreement_ratio:.4f} "
            f"disagree={teacher_disagreement_ratio:.4f} "
            f"mean_teacher_lp_on_student={mean_teacher_logprob_on_student:.4f} "
            f"mean_student_lp={mean_student_logprob_on_student:.4f} "
            f"abs_gap={mean_teacher_student_logprob_gap:.4f} "
            f"signed_gap={mean_teacher_minus_student_logprob:.4f} "
            f"floor_hit_ratio={floor_hit_ratio:.4f} "
            f"aligned={num_used}/{len(student_token_keys)}"
        )

        return {
            "enabled": True,
            "status": "ok",
            "teacher_mode": self.config.distillation_teacher_mode,
            "teacher_scoring_mode": teacher_scoring_mode,
            "student_base_url": self._normalize_openai_base_url(student_base_url),
            "student_model": student_model,
            "teacher_base_url": self._normalize_openai_base_url(teacher_base_url),
            "teacher_model": teacher_model,
            "teacher_reference_used": bool(teacher_reference),
            "distillation_loss": distillation_loss,
            "teacher_cross_entropy_on_student": distillation_loss,
            "teacher_student_logprob_gap": mean_teacher_student_logprob_gap,
            "teacher_minus_student_logprob": mean_teacher_minus_student_logprob,
            "mean_teacher_logprob_on_student": mean_teacher_logprob_on_student,
            "mean_student_logprob_on_student": mean_student_logprob_on_student,
            "token_agreement_ratio": token_agreement_ratio,
            "teacher_disagreement_ratio": teacher_disagreement_ratio,
            "token_advantage_weights": token_advantage_weights,
            "teacher_floor_hit_count": float(floor_hit_count),
            "teacher_floor_hit_ratio": floor_hit_ratio,
            "reward_mode": self.config.distillation_reward_mode,
            "aligned_positions": num_used,
            "student_alignment_start": student_alignment_start,
            "student_num_tokens": len(student_token_keys),
            "teacher_num_tokens": len(teacher_entries),
            "teacher_prompt_entries": len(prompt_entries),
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
            print(
                f"[ICPC_EVAL_DEBUG] icpc.py raw evaluation result for "
                f"{body.competition}:{body.icpc_id}: {evaluation_result}"
            )
            print(
                f"[ICPC_EVAL_DEBUG] icpc.py score summary for "
                f"{body.competition}:{body.icpc_id}: {score_summary}, reward={reward}"
            )

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

        raw_reward = reward
        distillation_loss = float(distillation_result.get("distillation_loss", 0.0))
        execution_loss = float(max(0.0, 1.0 - raw_reward))
        reward_for_training = raw_reward
        shaped_reward = raw_reward
        distillation_signal = 0.0
        distillation_bonus = 0.0
        if self.config.distillation_enabled and distillation_result.get("status") == "ok":
            combined_loss = execution_loss + self.config.distillation_weight * distillation_loss
            token_agreement_ratio = distillation_result.get("token_agreement_ratio", 0.0)
            try:
                agreement_signal = float(token_agreement_ratio)
            except Exception:
                agreement_signal = 0.0
            agreement_signal = min(1.0, max(0.0, agreement_signal))
            temperature = max(1e-6, float(self.config.distillation_reward_temperature))
            ce_signal = math.exp(-max(0.0, distillation_loss) / temperature)
            distillation_signal = max(agreement_signal, ce_signal)
            distillation_bonus = self.config.distillation_weight * distillation_signal
            shaped_reward = max(
                0.0,
                min(1.0, raw_reward + distillation_bonus),
            )
            if self.config.distillation_reward_mode == "reward_shaping":
                reward_for_training = shaped_reward
            elif self.config.distillation_reward_mode == "token_weighted_grpo":
                reward_for_training = raw_reward
            else:
                reward_for_training = shaped_reward
                distillation_result["reward_mode_warning"] = (
                    f"unknown_reward_mode:{self.config.distillation_reward_mode}. "
                    "Falling back to reward_shaping."
                )
        else:
            combined_loss = execution_loss
            shaped_reward = raw_reward

        details = dict(evaluation_result)
        details["score_summary"] = score_summary
        details["distillation"] = distillation_result
        details["losses"] = {
            "execution_loss": execution_loss,
            "distillation_loss": distillation_loss,
            "combined_loss": combined_loss,
            "distillation_weight": self.config.distillation_weight,
            "reward_mode": self.config.distillation_reward_mode,
            "raw_reward": raw_reward,
            "reward_for_training": reward_for_training,
            "shaped_reward": shaped_reward,
            "distillation_signal": distillation_signal,
            "distillation_bonus": distillation_bonus,
            "distillation_reward_temperature": self.config.distillation_reward_temperature,
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
            reward=reward_for_training,
            details=details,
            distillation_loss=distillation_loss,
            execution_loss=execution_loss,
            combined_loss=combined_loss,
            shaped_reward=shaped_reward,
        )


if __name__ == "__main__":
    Icpc25ResourcesServer.run_webserver()
