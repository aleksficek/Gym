# Description

Competitive coding resource server.

## Runtime config

Define server-specific runtime knobs on the resource server config, then read
them from `self.config` in `app.py`. For this server the relevant config path is:

`env.nemo_gym.competitive_coding_challenges_resources_server.resources_servers.competitive_coding_challenges`

Example override from NeMo RL:

```bash
uv run python examples/nemo_gym/run_grpo_nemo_gym.py --config path/to/grpo.yaml \
  env.nemo_gym.config_paths='[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/competitive_coding_challenges/configs/competitive_coding_challenges.yaml]' \
  env.nemo_gym.competitive_coding_challenges_resources_server.resources_servers.competitive_coding_challenges.test_file=/abs/path/to/test_metadata.json \
  env.nemo_gym.competitive_coding_challenges_resources_server.resources_servers.competitive_coding_challenges.test_batch_size=32 \
  env.nemo_gym.competitive_coding_challenges_resources_server.resources_servers.competitive_coding_challenges.time_scale=2.0
```

Inside the resource server, access the values as:

```python
self.config.test_file
self.config.test_batch_size
self.config.num_parallel_requests
self.config.time_scale
```

This is cleaner than pulling values from environment variables inside `app.py`
because the server config stays part of the same Hydra config tree as the rest
of the NeMo RL run.

# Licensing information

Code: ?
Data: ?

Dependencies

- nemo_gym: Apache 2.0
