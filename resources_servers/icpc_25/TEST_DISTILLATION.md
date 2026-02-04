# Testing On-Policy Distillation

## Quick Test: Use Policy Model as Teacher

The easiest way to test distillation is to use the policy model as its own teacher.

### Step 1: Update Config to Enable Distillation

Edit `configs/icpc_25.yaml`:

```yaml
icpc_25_resources_server:
  resources_servers:
    icpc_25:
      entrypoint: app.py
      domain: coding
      # Enable distillation for testing
      enable_distillation: true  # CHANGED from false
      teacher_model_server:
        type: responses_api_models
        name: policy_model  # Use same model as teacher for testing
      teacher_responses_create_params:
        input: []
        logprobs: true
        top_logprobs: 20
      distillation_temperature: 1.0
      max_distillation_reward: 0.5
```

### Step 2: Ensure Policy Model Has Logprobs Enabled

Check your `openai_model.yaml` or policy model config has:

```yaml
policy_model:
  responses_api_models:
    policy_model:
      # ... other settings ...
      responses_create_params:
        logprobs: true  # ADD THIS
        top_logprobs: 20  # ADD THIS
        # ... other params ...
```

### Step 3: Create a Test Input That Will Fail

Create a test file with a problem that will likely produce wrong code:

```bash
cat > /tmp/test_distillation.jsonl << 'EOF'
{"icpc_id": "buggyrover", "responses_create_params": {"input": [{"role": "user", "content": "Write a C++ program that outputs 'Hello World'"}]}}
EOF
```

This simple prompt will likely produce code that doesn't match the actual ICPC test cases, forcing distillation to trigger.

### Step 4: Run the Server

```bash
config_paths="/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_agents/simple_agent/configs/simple_agent.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_models/openai_model/configs/openai_model.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25/configs/icpc_25.yaml"

ng_run "+config_paths=[$config_paths]" \
  +simple_agent.responses_api_agents.simple_agent.resources_server.name=icpc_25_resources_server
```

### Step 5: Collect Rollouts (in another terminal)

```bash
ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=/tmp/test_distillation.jsonl \
    +output_jsonl_fpath=/tmp/test_distillation_output.jsonl \
    +limit=1 \
    +num_repeats=1
```

### Step 6: Check Server Logs

Look for these messages in the server terminal:

✅ **Expected when distillation triggers:**
```
🎓 DEBUG: Tests failed (reward=0), computing distillation reward from teacher model
DEBUG: Calling teacher model: policy_model
DEBUG: KL divergence: 0.0000  # Should be ~0 since using same model
DEBUG: Distillation reward: 0.5000  # Should be max_distillation_reward
🎓 DEBUG: Using distillation reward: 0.5000
```

✅ **Expected when tests pass:**
```
✅ DEBUG: Tests passed (reward=0.5000), skipping distillation
```

❌ **Error messages to watch for:**
```
⚠️ Could not extract logprobs from student or teacher response
⚠️ Distillation enabled but teacher model not configured
ERROR in distillation reward computation: ...
```

### Step 7: Verify Output

Check the output JSONL:

```bash
cat /tmp/test_distillation_output.jsonl | jq '.details'
```

Look for:
```json
{
  "reward_source": "distillation",  // Should be "distillation" when tests fail
  "distillation_reward": 0.5,       // Should be present
  "test_case_results": {
    "score": 0.0                     // Tests failed
  }
}
```

### Step 8: Verify KL Divergence is ~0 (Self-Teacher)

When using the policy model as its own teacher, KL divergence should be **very close to 0** (typically < 0.01) because the distributions are identical. This gives you the max distillation reward.

If KL is not near 0, there may be:
- Non-deterministic sampling (temperature > 0)
- Logprobs not being extracted correctly
- Different states between student and teacher calls

## Test with Different Teacher Model

Once the self-teacher test works, try with a real teacher:

### Step 1: Add Teacher Model Config

Create or use an existing teacher model config, e.g., `teacher_openai_model.yaml`:

```yaml
teacher_model:
  responses_api_models:
    teacher_model:
      entrypoint: ../../responses_api_models/openai_model/app.py
      api_base: "http://your-teacher-host:port/v1"
      model_name: "your-teacher-model"
      responses_create_params:
        logprobs: true
        top_logprobs: 20
        temperature: 0.0  # Deterministic for consistency
        max_tokens: 2048
```

### Step 2: Update ICPC Config

```yaml
teacher_model_server:
  type: responses_api_models
  name: teacher_model  # Changed from policy_model
```

### Step 3: Add Teacher Config to Paths

```bash
config_paths="...,teacher_openai_model.yaml,/path/to/icpc_25.yaml"
```

### Step 4: Run Tests

Now KL divergence should be **non-zero** (typically 0.1 - 5.0) depending on how different the models are.

Expected behavior:
- Better student (closer to teacher) → Lower KL → Higher reward
- Worse student (far from teacher) → Higher KL → Lower reward

## Validation Checklist

- [ ] Server starts without errors when distillation enabled
- [ ] When tests fail (reward=0), distillation triggers
- [ ] "🎓 DEBUG: Tests failed" message appears
- [ ] "DEBUG: Calling teacher model" message appears
- [ ] "DEBUG: KL divergence" message shows a reasonable value
- [ ] "DEBUG: Distillation reward" message shows reward > 0
- [ ] Output JSONL has `reward_source: "distillation"`
- [ ] Output JSONL has `distillation_reward` field
- [ ] When tests pass (reward>0), distillation is skipped
- [ ] Self-teacher test gives KL ≈ 0 and reward = max_distillation_reward
- [ ] Real teacher test gives KL > 0 and reward < max_distillation_reward

## Expected Reward Values

| Scenario | KL Divergence | Distillation Reward (max=0.5, temp=1.0) |
|----------|---------------|------------------------------------------|
| Self-teacher | ~0.0 | ~0.50 (max) |
| Very similar to teacher | 0.1 - 0.5 | 0.40 - 0.48 |
| Somewhat similar | 0.5 - 2.0 | 0.15 - 0.40 |
| Very different | 2.0 - 5.0 | 0.01 - 0.15 |
| Completely different | > 5.0 | < 0.01 |

## Troubleshooting

### KL Divergence is Always 0 (Even with Different Teacher)

**Possible causes:**
1. Logprobs not being extracted (both use same fallback)
2. Both models returning empty/identical responses
3. Teacher model is actually the same as student

**Debug:**
- Add print statements in `_extract_logprobs()` to verify logprobs structure
- Check that teacher and student responses are actually different

### KL Divergence is inf

**Possible causes:**
1. No overlapping tokens in distributions
2. Logprobs extraction failing for one model
3. Empty distributions

**Debug:**
- Check that both models generate actual text
- Verify top_logprobs contains multiple tokens
- Print the extracted distributions to see what's being compared

### Distillation Never Triggers

**Possible causes:**
1. Tests are passing (reward > 0)
2. enable_distillation is false
3. Config not being loaded properly

**Debug:**
- Use a simpler prompt that definitely produces wrong code
- Check that enable_distillation: true in loaded config
- Look for "✅ DEBUG: Tests passed" messages

### No Logprobs in Response

**Possible causes:**
1. Model server doesn't support logprobs
2. logprobs: true not in config
3. Backend model doesn't support logprobs

**Debug:**
- Test model API directly: `curl -X POST ... -d '{"logprobs": true}'`
- Check if backend (vLLM, TensorRT-LLM, etc.) supports logprobs
- Try different model server implementation

## Performance Considerations

**Impact of distillation on latency:**
- Each failed test case makes an additional teacher API call
- For batch evaluation, this can add significant time
- Consider:
  - Only enabling during training, not evaluation
  - Using a smaller/faster teacher model
  - Caching teacher responses for identical inputs

**Monitoring:**
- Track % of rollouts using distillation vs tests
- Monitor teacher API latency
- Watch for teacher API rate limits/errors
