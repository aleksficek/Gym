#!/bin/bash
# Automated validation script for on-policy distillation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_INPUT="/tmp/test_distillation_input.jsonl"
TEST_OUTPUT="/tmp/test_distillation_output.jsonl"

echo "========================================"
echo "On-Policy Distillation Validation"
echo "========================================"
echo ""

# Check if distillation is enabled
if ! grep -q "enable_distillation: true" "$SCRIPT_DIR/configs/icpc_25.yaml"; then
    echo "❌ ERROR: enable_distillation is not set to true in configs/icpc_25.yaml"
    echo ""
    echo "To enable distillation for testing:"
    echo "1. Edit configs/icpc_25.yaml"
    echo "2. Set enable_distillation: true"
    echo "3. Uncomment teacher_model_server section"
    echo "4. Set teacher_model_server.name to 'policy_model' for testing"
    echo ""
    exit 1
fi

echo "✓ Distillation is enabled in config"
echo ""

# Check if teacher_model_server is configured
if grep -q "# teacher_model_server:" "$SCRIPT_DIR/configs/icpc_25.yaml"; then
    echo "❌ ERROR: teacher_model_server is commented out in config"
    echo ""
    echo "Uncomment the teacher_model_server section in configs/icpc_25.yaml"
    echo ""
    exit 1
fi

echo "✓ teacher_model_server is configured"
echo ""

# Create test input that will likely fail tests
echo "Creating test input..."
cat > "$TEST_INPUT" << 'EOF'
{"icpc_id": "buggyrover", "responses_create_params": {"input": [{"role": "user", "content": "Write a simple C++ program that prints Hello World"}]}}
EOF

echo "✓ Test input created at $TEST_INPUT"
echo ""

# Check if server is running
echo "Checking if NeMo Gym server is running..."
echo "⚠️  This script expects the server to be already running"
echo "   If not running, start it in another terminal with:"
echo ""
echo "   config_paths=\"/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_agents/simple_agent/configs/simple_agent.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_models/openai_model/configs/openai_model.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25/configs/icpc_25.yaml\""
echo "   ng_run \"+config_paths=[\$config_paths]\" +simple_agent.responses_api_agents.simple_agent.resources_server.name=icpc_25_resources_server"
echo ""
read -p "Press Enter when server is ready, or Ctrl+C to exit..."

echo ""
echo "Running test rollout..."
echo ""

# Run rollout
if ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath="$TEST_INPUT" \
    +output_jsonl_fpath="$TEST_OUTPUT" \
    +limit=1 \
    +num_repeats=1 \
    +num_samples_in_parallel=1 2>&1 | tee /tmp/distillation_test.log; then
    echo ""
    echo "✓ Rollout completed"
else
    echo ""
    echo "❌ Rollout failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Validation Results"
echo "========================================"
echo ""

# Check if output file exists
if [ ! -f "$TEST_OUTPUT" ]; then
    echo "❌ Output file not created: $TEST_OUTPUT"
    exit 1
fi

echo "✓ Output file created"
echo ""

# Parse output with jq
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not installed, showing raw output:"
    cat "$TEST_OUTPUT"
    echo ""
    echo "Install jq for better output parsing: apt-get install jq"
else
    echo "Parsing output..."
    echo ""

    # Extract relevant fields
    REWARD=$(jq -r '.reward // "N/A"' "$TEST_OUTPUT")
    REWARD_SOURCE=$(jq -r '.details.reward_source // "N/A"' "$TEST_OUTPUT")
    DISTILL_REWARD=$(jq -r '.details.distillation_reward // "N/A"' "$TEST_OUTPUT")
    TEST_SCORE=$(jq -r '.details.test_case_results.score // "N/A"' "$TEST_OUTPUT")

    echo "Final Reward: $REWARD"
    echo "Reward Source: $REWARD_SOURCE"
    echo "Distillation Reward: $DISTILL_REWARD"
    echo "Test Score: $TEST_SCORE"
    echo ""

    # Validate results
    VALIDATION_PASSED=true

    if [ "$REWARD_SOURCE" = "distillation" ]; then
        echo "✓ Distillation was triggered (reward_source=distillation)"

        if [ "$DISTILL_REWARD" != "N/A" ] && [ "$DISTILL_REWARD" != "null" ]; then
            echo "✓ Distillation reward was computed: $DISTILL_REWARD"

            # Check if reward is reasonable (should be > 0 for self-teacher)
            if command -v bc &> /dev/null; then
                if (( $(echo "$DISTILL_REWARD > 0" | bc -l) )); then
                    echo "✓ Distillation reward is > 0"

                    # For self-teacher, should be close to max (0.5)
                    TEACHER_NAME=$(grep -A 2 "teacher_model_server:" "$SCRIPT_DIR/configs/icpc_25.yaml" | grep "name:" | awk '{print $2}')
                    if [ "$TEACHER_NAME" = "policy_model" ]; then
                        if (( $(echo "$DISTILL_REWARD > 0.45" | bc -l) )); then
                            echo "✓ Self-teacher reward is near maximum ($DISTILL_REWARD ≈ 0.5)"
                        else
                            echo "⚠️  Self-teacher reward is lower than expected: $DISTILL_REWARD (expected ≈0.5)"
                            echo "   This might indicate logprobs extraction issues"
                        fi
                    fi
                else
                    echo "❌ Distillation reward is 0 or negative: $DISTILL_REWARD"
                    VALIDATION_PASSED=false
                fi
            fi
        else
            echo "❌ Distillation reward field is missing or null"
            VALIDATION_PASSED=false
        fi

        if [ "$TEST_SCORE" = "0" ] || [ "$TEST_SCORE" = "0.0" ]; then
            echo "✓ Tests failed as expected (score=0), triggering distillation"
        else
            echo "⚠️  Tests did not fail (score=$TEST_SCORE), but distillation was used"
        fi
    elif [ "$REWARD_SOURCE" = "tests" ]; then
        echo "⚠️  Tests passed (reward_source=tests), distillation was not needed"
        echo "   This is OK, but doesn't validate distillation functionality"
        echo "   Try with a harder problem or wrong prompt to force test failure"
    else
        echo "❌ Unexpected reward_source: $REWARD_SOURCE"
        VALIDATION_PASSED=false
    fi

    echo ""
    echo "Full output:"
    jq '.' "$TEST_OUTPUT"
fi

echo ""
echo "========================================"
echo "Server Log Analysis"
echo "========================================"
echo ""

# Check for debug messages in log
if [ -f /tmp/distillation_test.log ]; then
    echo "Checking for distillation debug messages..."
    echo ""

    if grep -q "🎓 DEBUG: Tests failed" /tmp/distillation_test.log; then
        echo "✓ Found: Tests failed message"
    else
        echo "⚠️  Not found: Tests failed message"
    fi

    if grep -q "DEBUG: Calling teacher model" /tmp/distillation_test.log; then
        echo "✓ Found: Calling teacher model message"
    else
        echo "⚠️  Not found: Calling teacher model message"
    fi

    if grep -q "DEBUG: KL divergence" /tmp/distillation_test.log; then
        KL_LINE=$(grep "DEBUG: KL divergence" /tmp/distillation_test.log | tail -1)
        echo "✓ Found: $KL_LINE"
    else
        echo "⚠️  Not found: KL divergence message"
    fi

    if grep -q "DEBUG: Distillation reward" /tmp/distillation_test.log; then
        REWARD_LINE=$(grep "DEBUG: Distillation reward" /tmp/distillation_test.log | tail -1)
        echo "✓ Found: $REWARD_LINE"
    else
        echo "⚠️  Not found: Distillation reward message"
    fi

    # Check for errors
    if grep -q "ERROR in distillation" /tmp/distillation_test.log; then
        echo ""
        echo "❌ ERRORS found in distillation:"
        grep "ERROR in distillation" /tmp/distillation_test.log
        VALIDATION_PASSED=false
    fi

    if grep -q "Could not extract logprobs" /tmp/distillation_test.log; then
        echo ""
        echo "❌ Logprobs extraction failed:"
        grep "Could not extract logprobs" /tmp/distillation_test.log
        VALIDATION_PASSED=false
    fi
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo ""

if [ "$VALIDATION_PASSED" = true ]; then
    echo "✅ VALIDATION PASSED"
    echo ""
    echo "On-policy distillation is working correctly!"
    echo ""
    echo "Next steps:"
    echo "- Configure a real teacher model (not policy_model)"
    echo "- Test with different distillation_temperature values"
    echo "- Monitor KL divergence and rewards during training"
else
    echo "❌ VALIDATION FAILED"
    echo ""
    echo "Check the errors above and:"
    echo "- Ensure logprobs are enabled in both student and teacher configs"
    echo "- Verify teacher model is accessible"
    echo "- Check server logs for detailed error messages"
    echo ""
    echo "See DISTILLATION_SETUP.md for troubleshooting"
fi

echo ""
echo "Test artifacts:"
echo "- Input: $TEST_INPUT"
echo "- Output: $TEST_OUTPUT"
echo "- Logs: /tmp/distillation_test.log"
echo ""
