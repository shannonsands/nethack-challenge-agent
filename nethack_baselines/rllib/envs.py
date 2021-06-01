import threading
from collections import OrderedDict
from typing import Tuple, Union

import gym
import nle  # noqa: F401
import numpy as np
from nle.env import tasks

ENVS = dict(
    # NLE tasks
    staircase=tasks.NetHackStaircase,
    score=tasks.NetHackScore,
    pet=tasks.NetHackStaircasePet,
    oracle=tasks.NetHackOracle,
    gold=tasks.NetHackGold,
    eat=tasks.NetHackEat,
    scout=tasks.NetHackScout,
    challenge=tasks.NetHackChallenge
)


def create_env(flags, env_id=0, lock=threading.Lock()):
    # commenting out these options for now because they use too much disk space
    # archivefile = "nethack.%i.%%(pid)i.%%(time)s.zip" % env_id
    # if flags.single_ttyrec and env_id != 0:
    #     archivefile = None

    # logdir = os.path.join(flags.savedir, "archives")

    with lock:
        env_class = ENVS[flags.env]
        kwargs = dict(
            savedir=flags.savedir,
            archivefile=None,
            character=flags.character,
            max_episode_steps=flags.max_num_steps,
            observation_keys=(
                "glyphs",
                "chars",
                "colors",
                "specials",
                "blstats",
                "message",
                "tty_chars",
                "tty_colors",
                "tty_cursor",
                "inv_glyphs",
                "inv_strs",
                "inv_letters",
                "inv_oclasses",
                # "screen_descriptions",
            ),
            penalty_step=flags.penalty_step,
            penalty_time=flags.penalty_time,
            penalty_mode=flags.fn_penalty_step,
        )
        if flags.env in ("staircase", "pet", "oracle"):
            kwargs.update(reward_win=flags.reward_win, reward_lose=flags.reward_lose)
        elif env_id == 0:  # print warning once
            # Removed because it's too noisy:
            # print("Ignoring flags.reward_win and flags.reward_lose")
            pass
        if flags.state_counter != "none":
            kwargs.update(state_counter=flags.state_counter)
        env = env_class(**kwargs)
        if flags.seedspath is not None and len(flags.seedspath) > 0:
            raise NotImplementedError("seedspath > 0 not implemented yet.")

        return env


class RLLibNLEEnv(gym.Env):
    def __init__(self, env_config: dict) -> None:
        self.gym_env = create_env(env_config["flags"])
        # We sort the observation keys so we can create the OrderedDict output
        # in a consistent order
        self._observation_keys = sorted(self.gym_env.observation_space.spaces.keys())

    @property
    def action_space(self) -> gym.Space:
        return self.gym_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self.gym_env.observation_space

    def reset(self) -> dict:
        return self._process_obs(self.gym_env.reset())

    def _process_obs(self, obs: dict) -> dict:
        return OrderedDict({key: obs[key] for key in self._observation_keys})

    def step(
        self, action: Union[int, np.int64]
    ) -> Tuple[dict, Union[np.number, int], Union[np.bool_, bool], dict]:
        obs, reward, done, info = self.gym_env.step(action)
        return self._process_obs(obs), reward, done, info

    def render(self):
        return self.gym_env.render()

    def close(self):
        return self.gym_env.close()
