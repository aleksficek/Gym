#!/bin/bash
# Quick test script for local ICPC server with distillation

echo "=========================================="
echo "ICPC Resources Server - Local Test"
echo "=========================================="
echo ""

# Check if distillation test script exists
if [ -f "test_distillation.py" ]; then
    echo "Running distillation test suite..."
    python test_distillation.py
    echo ""
fi

echo "=========================================="
echo "To test with your actual setup:"
echo "=========================================="
echo ""

echo "1. REQUIRED: Update configs/icpc_25.yaml to enable distillation"
echo "   Set enable_distillation: true and configure teacher_model_server"
echo ""

echo "2. Start the server (Terminal 1):"
echo ""
echo "config_paths="/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_agents/simple_agent/configs/simple_agent.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_models/openai_model/configs/openai_model.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25/configs/icpc_25.yaml"
echo ""
echo "ng_run \"+config_paths=[\$config_paths]\" \\"
echo "  +simple_agent.responses_api_agents.simple_agent.resources_server.name=icpc_25_resources_server"
echo ""

echo "3. Collect rollouts (Terminal 2):"
echo ""
echo "ng_collect_rollouts +agent_name=simple_agent \\"
echo "    +input_jsonl_fpath=/home/aficek/software/scripts/gym_dev/example.jsonl \\"
echo "    +output_jsonl_fpath=/home/aficek/software/scripts/gym_dev/example_rollouts.jsonl \\"
echo "    +limit=null \\"
echo "    +num_repeats=1 \\"
echo "    +num_samples_in_parallel=null"
echo ""

echo "4. Watch for these debug messages in server logs:"
echo "   🎓 DEBUG: Tests failed (reward=0), computing distillation reward from teacher model"
echo "   DEBUG: Calling teacher model: <teacher_name>"
echo "   DEBUG: KL divergence: X.XXXX"
echo "   DEBUG: Distillation reward: X.XXXX"
echo ""

echo "5. Check output JSONL for distillation_reward and reward_source fields"
echo ""

echo "=========================================="
echo "Key Configuration Requirements:"
echo "=========================================="
echo ""
echo "✓ enable_distillation: true in icpc_25.yaml"
echo "✓ teacher_model_server configured and running"
echo "✓ logprobs: true in BOTH student and teacher configs"
echo "✓ top_logprobs: 20 for better distribution estimation"
echo ""
echo "See DISTILLATION_SETUP.md for detailed instructions"
echo ""
