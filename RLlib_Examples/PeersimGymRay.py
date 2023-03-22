import ray
import ray.rllib.agents.ppo as ppo  # Proximal Policy Optimization
import os
import shutil

ray.shutdown()
ray.init(ignore_reinit_error=True)
# print("Dashboard URL: http://{}".format(ray.get_webui_url()))

# Setup to allow for checkpoints for the weights
# Delete any pre-existing files
CHECKPOINT_ROOT = "tmp/ppo/peersim"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)  # Clears any checkpoints in the directory
ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

SELECT_ENV = "peersim_gym/PeersimEnv-v0"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)
policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())
N_ITER = 30
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
    result = agent.train()
    file_name = agent.save(CHECKPOINT_ROOT)

    print(s.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        file_name
    ))
