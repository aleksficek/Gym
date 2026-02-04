#!/usr/bin/env python3
"""
Test script for ICPC on-policy distillation feature.

This script helps verify that:
1. Teacher model is correctly configured and accessible
2. Logprobs are being extracted properly
3. KL divergence computation works
4. Distillation reward is computed when tests fail
"""

import asyncio
import json
from typing import Dict, Any


def create_test_request(icpc_id: str = "buggyrover") -> Dict[str, Any]:
    """Create a minimal test request for the ICPC server."""
    return {
        "icpc_id": icpc_id,
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "Write a simple C++ program that prints 'Hello World'"
                }
            ]
        },
        "response": {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "```cpp\n#include <iostream>\nint main() {\n    std::cout << \"Wrong!\" << std::endl;\n    return 0;\n}\n```"
                        }
                    ]
                }
            ]
        }
    }


async def test_distillation_config():
    """Test that distillation configuration is properly set up."""
    print("=" * 60)
    print("Testing Distillation Configuration")
    print("=" * 60)

    # This would need to be adapted to actually import and test the server
    # For now, just print guidance
    print("\n✓ To enable distillation, ensure your config has:")
    print("""
    icpc_25_resources_server:
      resources_servers:
        icpc_25:
          enable_distillation: true
          teacher_model_server:
            type: responses_api_models
            name: teacher_model_name  # Must match your teacher model config
          teacher_responses_create_params:
            input: []
            logprobs: true  # CRITICAL - must request logprobs!
            top_logprobs: 20
          distillation_temperature: 1.0
          max_distillation_reward: 0.5
    """)

    print("\n✓ Ensure your student model also requests logprobs:")
    print("""
    In your openai_model config, add:
      responses_create_params:
        logprobs: true
        top_logprobs: 20
    """)


def test_logprobs_extraction():
    """Test logprobs extraction logic."""
    print("\n" + "=" * 60)
    print("Testing Logprobs Extraction")
    print("=" * 60)

    # Example response structure with logprobs
    mock_response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "Hello",
                                    "logprob": -0.5,
                                    "top_logprobs": [
                                        {"token": "Hello", "logprob": -0.5},
                                        {"token": "Hi", "logprob": -1.2},
                                        {"token": "Hey", "logprob": -2.0}
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    print("\n✓ Expected response structure with logprobs:")
    print(json.dumps(mock_response, indent=2))

    print("\n✓ If logprobs are missing, check:")
    print("  1. Model server supports logprobs (OpenAI API compatible)")
    print("  2. logprobs: true is set in responses_create_params")
    print("  3. Model backend supports returning logprobs")


def test_kl_computation():
    """Test KL divergence computation."""
    print("\n" + "=" * 60)
    print("Testing KL Divergence Computation")
    print("=" * 60)

    import math

    # Example: Teacher and student have similar distributions
    teacher_logprobs = [
        {"token_a": -0.1, "token_b": -2.0, "token_c": -3.0}
    ]
    student_logprobs = [
        {"token_a": -0.2, "token_b": -1.8, "token_c": -3.2}
    ]

    # Compute KL(teacher || student)
    kl = 0.0
    for token, teacher_lp in teacher_logprobs[0].items():
        teacher_prob = math.exp(teacher_lp)
        student_lp = student_logprobs[0].get(token, -20.0)
        if teacher_prob > 1e-10:
            kl += teacher_prob * (teacher_lp - student_lp)

    print(f"\n✓ Example KL divergence: {kl:.4f}")
    print(f"  Teacher probs: {[math.exp(lp) for lp in teacher_logprobs[0].values()]}")
    print(f"  Student probs: {[math.exp(lp) for lp in student_logprobs[0].values()]}")

    # Convert to reward
    temperature = 1.0
    max_reward = 0.5
    reward = max_reward * math.exp(-kl / temperature)
    print(f"\n✓ Distillation reward: {reward:.4f}")
    print(f"  (Lower KL = higher reward)")


def print_debugging_tips():
    """Print debugging tips for the distillation feature."""
    print("\n" + "=" * 60)
    print("Debugging Tips")
    print("=" * 60)

    print("""
1. Check server logs for these debug messages:
   - "DEBUG: Tests failed (reward=0), computing distillation reward from teacher model"
   - "DEBUG: Calling teacher model: <teacher_name>"
   - "DEBUG: KL divergence: X.XXXX"
   - "DEBUG: Distillation reward: X.XXXX"

2. Common issues:
   - "Could not extract logprobs": Model doesn't support logprobs or not requested
   - "Distillation enabled but teacher model not configured": Missing teacher config
   - KL divergence is inf: No overlapping tokens in distributions

3. Test distillation independently:
   - Set max_distillation_reward to a small value (e.g., 0.1) initially
   - Monitor that reward_source="distillation" appears in results when tests fail
   - Verify teacher model is being called (check teacher model server logs)

4. Verify test failure triggers distillation:
   - Use a prompt that generates wrong code (to ensure reward=0)
   - Check that distillation_reward appears in evaluation_result
   - Confirm reward_source is "distillation" not "tests"

5. Performance considerations:
   - Each failed test case makes a teacher API call (adds latency)
   - Consider caching teacher responses for identical inputs
   - Monitor teacher model load if many test failures occur
    """)


def main():
    """Run all tests and print guidance."""
    print("\n" + "=" * 60)
    print("ICPC On-Policy Distillation Test Suite")
    print("=" * 60)

    asyncio.run(test_distillation_config())
    test_logprobs_extraction()
    test_kl_computation()
    print_debugging_tips()

    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. Update your icpc_25.yaml config to enable distillation
2. Ensure teacher model is running and accessible
3. Run your test command:

   ng_run "+config_paths=[...]" \\
     +simple_agent.responses_api_agents.simple_agent.resources_server.name=icpc_25_resources_server

4. In another terminal, collect rollouts:

   ng_collect_rollouts +agent_name=simple_agent \\
     +input_jsonl_fpath=example.jsonl \\
     +output_jsonl_fpath=example_rollouts.jsonl

5. Check the output JSONL for:
   - "reward_source": "distillation" (when tests fail)
   - "distillation_reward": <value>
   - Non-zero rewards even for incorrect solutions

6. Monitor logs for DEBUG messages about KL divergence and teacher calls
    """)


if __name__ == "__main__":
    main()
