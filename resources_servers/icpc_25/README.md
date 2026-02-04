# ICPC 25 Resources Server

NeMo Gym resources server for ICPC competitive programming problems with code evaluation.

## Description

This resources server evaluates C++ solutions for ICPC problems by:
1. Compiling student-generated code in a sandboxed environment
2. Running test cases (both sample and hidden tests)
3. Computing rewards based on test pass rates
4. **NEW**: Optionally using on-policy distillation from a teacher model when tests fail

Data links: TBD

## On-Policy Distillation

When enabled, the server provides an alternative learning signal using a teacher model when test-based evaluation fails (reward = 0). This helps the student model learn even from incorrect solutions.

### How It Works

1. **Test-Based Reward (Primary)**: Solutions are evaluated against test cases. If tests pass (reward > 0), this reward is used directly.

2. **Distillation Reward (Fallback)**: When tests fail (reward = 0), the teacher model generates a response for the same input. The distillation reward is computed using KL divergence between the teacher and student token distributions:
   - Lower KL divergence (student closer to teacher) = Higher reward
   - Reward formula: `reward = max_distillation_reward * exp(-KL / temperature)`

3. **Benefit**: Provides dense reward signal even when solutions are incorrect, guiding the student toward the teacher's behavior.

### Configuration

Enable distillation in your config YAML:

```yaml
icpc_25_resources_server:
  resources_servers:
    icpc_25:
      enable_distillation: true
      teacher_model_server:
        type: responses_api_models
        name: teacher_model  # Your teacher model reference
      teacher_responses_create_params:
        input: []
        logprobs: true  # Required for KL divergence
        top_logprobs: 20  # Number of top tokens to consider
      distillation_temperature: 1.0  # Temperature for KL->reward conversion
      max_distillation_reward: 0.5  # Maximum reward from distillation (0-1)
```

### Requirements

- Teacher model must support logprobs in API responses
- Student model should also request logprobs in initial request
- Teacher model should be hosted as a separate responses_api_model

### Parameters

- `enable_distillation` (bool): Enable/disable distillation feature
- `teacher_model_server` (ModelServerRef): Reference to teacher model server
- `teacher_responses_create_params`: Base parameters for teacher API calls
- `distillation_temperature` (float): Controls sensitivity of KL->reward conversion (higher = more forgiving)
- `max_distillation_reward` (float): Maximum reward from distillation, typically < 1.0 to ensure test-based rewards are preferred

## Licensing information

Code: Apache 2.0
Data: TBD

## Dependencies

- nemo_gym: Apache 2.0
- nemo_skills: For sandbox code execution
- Docker: For sandboxed compilation and execution
