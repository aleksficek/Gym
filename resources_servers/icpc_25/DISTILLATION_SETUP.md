# On-Policy Distillation Setup Guide

## Quick Start Checklist

### 1. Configure Teacher Model Server

Create or update a teacher model config (e.g., `teacher_openai_model.yaml`):

```yaml
teacher_model:
  responses_api_models:
    teacher_model:
      entrypoint: path/to/openai_model/app.py
      api_base: "http://your-teacher-model-host:port/v1"
      api_key: "your-api-key"
      model_name: "your-teacher-model-name"
      responses_create_params:
        logprobs: true  # CRITICAL - must enable logprobs!
        top_logprobs: 20
        temperature: 0.7
        max_tokens: 2048
```

### 2. Enable Distillation in ICPC Config

Update `configs/icpc_25.yaml`:

**IMPORTANT**: NeMo Gym validates all server references even when not used. You MUST uncomment the teacher_model_server config when enabling distillation.

```yaml
icpc_25_resources_server:
  resources_servers:
    icpc_25:
      entrypoint: app.py
      domain: coding
      # Enable distillation
      enable_distillation: true  # Changed from false
      teacher_model_server:  # UNCOMMENT this section
        type: responses_api_models
        name: teacher_model  # Must match teacher config name
      teacher_responses_create_params:  # UNCOMMENT this section
        input: []  # Populated at runtime
        logprobs: true  # CRITICAL!
        top_logprobs: 20
      distillation_temperature: 1.0
      max_distillation_reward: 0.5
```

**Why comment out when disabled?** NeMo Gym validates ModelServerRef fields even when `enable_distillation: false`. To avoid validation errors, keep teacher_model_server commented out until you're ready to use distillation.

### 3. Ensure Student Model Requests Logprobs

Update your policy model config (e.g., `openai_model.yaml`):

```yaml
policy_model:
  responses_api_models:
    policy_model:
      # ... other settings ...
      responses_create_params:
        logprobs: true  # Enable for distillation
        top_logprobs: 20
        # ... other params ...
```

### 4. Run the Server

```bash
# Define config paths including ALL three configs
config_paths="/path/to/simple_agent.yaml,/path/to/openai_model.yaml,/path/to/teacher_openai_model.yaml,/path/to/icpc_25.yaml"

# Run with distillation enabled
ng_run "+config_paths=[$config_paths]" \
  +simple_agent.responses_api_agents.simple_agent.resources_server.name=icpc_25_resources_server
```

### 5. Test with Rollouts

```bash
# Collect rollouts (in separate terminal)
ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=/path/to/example.jsonl \
    +output_jsonl_fpath=/path/to/example_rollouts.jsonl \
    +limit=null \
    +num_repeats=1 \
    +num_samples_in_parallel=null
```

### 6. Verify Distillation is Working

Check the server logs for these messages:

```
🎓 DEBUG: Tests failed (reward=0), computing distillation reward from teacher model
DEBUG: Calling teacher model: teacher_model
DEBUG: KL divergence: X.XXXX
DEBUG: Distillation reward: X.XXXX
🎓 DEBUG: Using distillation reward: X.XXXX
```

Check the output JSONL for:

```json
{
  "reward": 0.3,
  "details": {
    "reward_source": "distillation",
    "distillation_reward": 0.3,
    "test_case_results": { "score": 0.0 }
  }
}
```

## Troubleshooting

### "Could not find type='responses_api_models' name='teacher_model'"

**Error Message**:
```
AssertionError: Could not find type='responses_api_models' name='teacher_model' in the list of available servers
```

**Cause**: NeMo Gym validates all ModelServerRef fields in config, even when `enable_distillation: false`

**Solutions**:
1. **Keep distillation disabled** (recommended for initial testing):
   - Comment out `teacher_model_server` and `teacher_responses_create_params` in config
   - Keep `enable_distillation: false`

2. **Enable distillation**:
   - Uncomment `teacher_model_server` and `teacher_responses_create_params`
   - Add teacher model config to `config_paths`
   - Set `enable_distillation: true`

3. **Test without teacher** (for debugging):
   - Change `teacher_model_server.name` to `policy_model` (uses student as its own teacher)
   - Set `enable_distillation: true`
   - This lets you test the distillation code path without a separate teacher

### "Could not extract logprobs from student or teacher response"

**Cause**: Logprobs not available in API response

**Solutions**:
1. Ensure `logprobs: true` in both student and teacher configs
2. Verify model server supports logprobs (OpenAI API compatible)
3. Check that the model backend itself supports logprobs
4. Try requesting `top_logprobs: 20` instead of just `logprobs: true`

### "Distillation enabled but teacher model not configured"

**Cause**: Teacher model server reference is missing or incorrect

**Solutions**:
1. Verify `teacher_model_server.name` matches your teacher model config
2. Ensure teacher model config is included in `config_paths`
3. Check that teacher model server is running

### KL divergence is very high (> 10)

**Cause**: Student and teacher have very different distributions

**Solutions**:
1. This is normal early in training when student is poor
2. Consider increasing `distillation_temperature` (e.g., 2.0 or 5.0) to make rewards less sensitive
3. Verify both models are using the same tokenizer

### Distillation never triggers

**Cause**: Tests are passing (reward > 0) or distillation disabled

**Solutions**:
1. Use harder problems or intentionally wrong prompts to force test failures
2. Verify `enable_distillation: true` in config
3. Check logs for "✅ DEBUG: Tests passed" messages

### Teacher API calls are slow

**Cause**: Teacher inference adds latency to verification

**Solutions**:
1. Only use distillation during training, not evaluation
2. Consider batching teacher calls (requires code modification)
3. Use a faster/smaller teacher model
4. Cache teacher responses for repeated inputs (requires code modification)

## Configuration Parameters Explained

### `enable_distillation` (bool)
- **Default**: `false`
- **Purpose**: Toggle distillation on/off
- **When to use**: Enable during training to provide learning signal from failed solutions

### `teacher_model_server` (ModelServerRef)
- **Required**: Yes (when distillation enabled)
- **Purpose**: Reference to the teacher model server
- **Format**: `{type: responses_api_models, name: teacher_model}`

### `teacher_responses_create_params` (dict)
- **Required**: Yes (when distillation enabled)
- **Purpose**: Base parameters for teacher API calls
- **Critical fields**: `logprobs: true`, `top_logprobs: 20`

### `distillation_temperature` (float)
- **Default**: `1.0`
- **Range**: `0.1` - `10.0`
- **Purpose**: Controls sensitivity of KL→reward conversion
- **Effect**:
  - Lower (0.5): More sensitive to differences, harder to get high rewards
  - Higher (5.0): Less sensitive, easier to get rewards even with higher KL

### `max_distillation_reward` (float)
- **Default**: `0.5`
- **Range**: `0.0` - `1.0`
- **Purpose**: Maximum reward achievable from distillation
- **Rationale**: Keep < 1.0 to ensure test-based rewards (correct solutions) are always preferred

## Example Reward Behavior

| Scenario | Test Score | Distillation | Final Reward | Reward Source |
|----------|-----------|--------------|--------------|---------------|
| Perfect solution | 1.0 | (not computed) | 1.0 | tests |
| Partial pass | 0.5 | (not computed) | 0.5 | tests |
| Wrong but similar to teacher | 0.0 | 0.45 | 0.45 | distillation |
| Wrong and different from teacher | 0.0 | 0.05 | 0.05 | distillation |
| Completely wrong | 0.0 | 0.01 | 0.01 | distillation |

## Testing Distillation Independently

Run the test script:

```bash
python test_distillation.py
```

This will print:
- Configuration examples
- Logprobs structure examples
- KL divergence calculation examples
- Debugging tips
