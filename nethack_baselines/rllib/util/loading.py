import copy
import os

import cloudpickle
import nethack_baselines.rllib.models  # noqa: F401
import ray
from nethack_baselines.rllib.envs import RLLibNLEEnv
from ray.rllib.agents import a3c, dqn, impala, ppo
from ray.tune.utils import merge_dicts

NAME_TO_TRAINER: dict = {
    "impala": (impala, impala.ImpalaTrainer),
    "a2c": (a3c, a3c.A2CTrainer),
    "dqn": (dqn, dqn.DQNTrainer),
    "ppo": (ppo, ppo.PPOTrainer),
}


def load_agent(checkpoint_location, model_class_name):
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(checkpoint_location)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    else:
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or its parent directory line!"
        )

    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config`
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)

    # Make sure we have evaluation workers.
    if not config.get("evaluation_num_workers"):
        config["evaluation_num_workers"] = config.get("num_workers", 0)

    config["num_workers"] = 0

    ray.init(num_cpus=1, num_gpus=1, include_dashboard=False)

    # Create the Trainer from config.
    _, trainer_cls = NAME_TO_TRAINER[model_class_name]
    trainer_cls = trainer_cls.with_updates(default_config=config)
    agent = trainer_cls(env=RLLibNLEEnv, config=config)

    # Load state from checkpoint, if provided.
    agent.restore(checkpoint_location)
    return agent
