# Quick Start: Test Distillation in 5 Minutes

## Step 1: Enable Distillation (Self-Teacher Test)

Edit `configs/icpc_25.yaml` - change these lines:

```yaml
enable_distillation: true  # CHANGE from false
teacher_model_server:      # UNCOMMENT these 3 lines
  type: responses_api_models
  name: policy_model       # Use policy_model as teacher for testing
teacher_responses_create_params:  # UNCOMMENT these 4 lines
  input: []
  logprobs: true
  top_logprobs: 20
```

**Quick edit commands:**
```bash
cd /home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25

# Backup original config
cp configs/icpc_25.yaml configs/icpc_25.yaml.backup

# Edit manually or use this sed (be careful!)
sed -i 's/enable_distillation: false/enable_distillation: true/' configs/icpc_25.yaml
sed -i 's/# teacher_model_server:/teacher_model_server:/' configs/icpc_25.yaml
sed -i 's/#   type: responses_api_models/  type: responses_api_models/' configs/icpc_25.yaml
sed -i 's/#   name: teacher_model/  name: policy_model/' configs/icpc_25.yaml
sed -i 's/# teacher_responses_create_params:/teacher_responses_create_params:/' configs/icpc_25.yaml
sed -i 's/#   input: \[\]/  input: []/' configs/icpc_25.yaml

# Add logprobs lines after input: []
cat >> /tmp/icpc_patch << 'EOF'
        logprobs: true
        top_logprobs: 20
EOF

# NOTE: Manual editing is safer - the above sed commands might need adjustment
```

## Step 2: Ensure Policy Model Has Logprobs

Edit your policy model config (e.g., `openai_model.yaml`):

```yaml
responses_create_params:
  logprobs: true      # ADD THIS
  top_logprobs: 20    # ADD THIS
```

## Step 3: Start Server (Terminal 1)

```bash
cd /home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25

config_paths="/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_agents/simple_agent/configs/simple_agent.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/responses_api_models/openai_model/configs/openai_model.yaml,/home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25/configs/icpc_25.yaml"

ng_run "+config_paths=[$config_paths]" \
  +simple_agent.responses_api_agents.simple_agent.resources_server.name=icpc_25_resources_server
```

**Watch for these messages at startup:**
- `✅ Sandbox already running` or `✅ Sandbox connected successfully`
- NO errors about missing teacher_model

## Step 4: Run Validation (Terminal 2)

```bash
cd /home/aficek/software/synth/RL/3rdparty/Gym-workspace/Gym/resources_servers/icpc_25

./validate_distillation.sh
```

This will:
1. Check config is correct
2. Create test input
3. Run a rollout
4. Validate distillation triggered
5. Check KL divergence and rewards

## Expected Output

**In validation script:**
```
✓ Distillation is enabled in config
✓ teacher_model_server is configured
✓ Test input created
✓ Rollout completed
✓ Distillation was triggered (reward_source=distillation)
✓ Distillation reward was computed: 0.5
✓ Self-teacher reward is near maximum (0.5 ≈ 0.5)
✓ Tests failed as expected (score=0), triggering distillation

✅ VALIDATION PASSED
```

**In server logs (Terminal 1):**
```
DEBUG: Extracted Problem ID: buggyrover
DEBUG: Generation: ...
🎓 DEBUG: Tests failed (reward=0), computing distillation reward from teacher model
DEBUG: Calling teacher model: policy_model
DEBUG: KL divergence: 0.0001
DEBUG: Distillation reward: 0.5000
🎓 DEBUG: Using distillation reward: 0.5000
```

## Interpretation

### Self-Teacher (policy_model as teacher):
- **KL divergence should be ≈ 0** (model agrees with itself)
- **Distillation reward should be ≈ 0.5** (max_distillation_reward)
- This validates the distillation code path works

### Real Teacher (different model):
- **KL divergence will be > 0** (models differ)
- **Distillation reward will be < 0.5** (proportional to similarity)
- Better student = lower KL = higher reward

## Troubleshooting

### Validation fails with "teacher_model_server is commented out"
→ You didn't uncomment the teacher_model_server section in config

### Validation fails with "enable_distillation: false"
→ You didn't change enable_distillation to true

### Server fails with "Could not find name='policy_model'"
→ Your policy model config name doesn't match "policy_model"
→ Check what your actual model is called in openai_model.yaml

### "Could not extract logprobs from student or teacher response"
→ Add `logprobs: true` to your policy model config
→ Ensure your model server/backend supports logprobs

### KL divergence is not near 0 for self-teacher
→ Check if temperature > 0 (causes non-determinism)
→ Verify logprobs are actually being extracted
→ Look for "Could not extract logprobs" in logs

## What's Next?

Once validation passes:

1. **Test with real teacher**: Configure a different model as teacher
2. **Tune parameters**: Adjust `distillation_temperature` and `max_distillation_reward`
3. **Run training**: Use distillation during GRPO training
4. **Monitor metrics**: Track KL divergence and reward distributions

See [TEST_DISTILLATION.md](TEST_DISTILLATION.md) for advanced testing scenarios.
