#!/usr/bin/env python
# This file is an example entrypoint for your submission, using rllib-trained models

import argparse

import aicrowd_gym
import cloudpickle
import nle
import numpy as np
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from nle_baselines.rllib.util import load_agent

parser = argparse.ArgumentParser(description="RLlib Model Rollout Evaluation")
parser.add_argument(
    "--checkpoint-location", type=str, required=True, help="The full path to the model checkpoint"
)
parser.add_argument("--print-steps", action="store_true", default=False, help="Whether to print each step")
parser.add_argument("--batch-size", type=int, required=True, help="How many envs to evaluate on in parallel")
parser.add_argument(
    "--num-assessments", type=int, required=False, default=200, help="How many rollouts to evaluate"
)
parser.add_argument(
    "--model-class-name",
    type=str,
    choices=["dqn", "a2c", "ppo", "impala"],
    required=True,
    help="The Rllib trainer class used for the model",
)


class EnvBatch:
    def __init__(self, batch_size, preprocessor):
        self.batch_size = batch_size
        self.envs = [aicrowd_gym.make("NetHackChallenge-v0") for _ in range(self.batch_size)]
        self.preprocessor = preprocessor

    def _batch(self, obs):
        return [self.preprocessor.transform(ob) for ob in obs]

    def step(self, actions: np.ndarray):
        state_list, reward_list, done_list, stats = [], [], [], []
        for env, a in zip(self.envs, actions):
            s, r, d, ss = env.step(a)
            if d:
                s = env.reset()
            state_list.append(s)
            reward_list.append(r)
            done_list.append(d)
            stats.append(ss)
        rewards = np.array(reward_list)
        done = np.array(done_list)
        states = self._batch(state_list)
        return states, rewards, done, stats

    def reset(self):
        states = [e.reset() for e in self.envs]
        return self._batch(states)


def main(args):
    print("Loading agent")
    agent = load_agent(args.checkpoint_location, args.model_class_name)
    policy = agent.get_policy(DEFAULT_POLICY_ID)
    worker = agent.workers.local_worker()
    batch_size = args.batch_size

    print("Creating environment")
    env = EnvBatch(batch_size, worker.preprocessors[DEFAULT_POLICY_ID])

    obs = env.reset()
    episode_count, episode_lens, rewards = 0, np.zeros(batch_size), np.zeros(batch_size)
    action, reward = [0] * batch_size, [0] * batch_size
    state = [
        np.stack(init_states)
        for init_states in (zip(*[policy.get_initial_state() for _ in range(batch_size)]))
    ]
    print("Starting evaluation")
    while episode_count < args.num_assessments:
        actions, state, _ = policy.compute_actions(
            obs, prev_action_batch=action, prev_reward_batch=reward, state_batches=state
        )
        episode_lens += 1
        obs, reward, done, info = env.step(actions)
        rewards += reward
        if args.print_steps:
            print(f"Step done: {action}, {reward}, {done}, {info}")

        for ep_done_idx_l in np.argwhere(done):
            ep_done_idx = ep_done_idx_l[0]
            new_init_state = policy.get_initial_state()
            for i, inner_state in enumerate(state):
                inner_state[ep_done_idx] = new_init_state[i]
            print(
                f"Episode complete: {info[ep_done_idx]}, {episode_count},"
                f"{episode_lens[ep_done_idx]}, {rewards[ep_done_idx]}"
            )
            episode_lens[ep_done_idx] = 0
            rewards[ep_done_idx] = 0.0
            episode_count += 1
            actions[ep_done_idx] = 0
            reward[ep_done_idx] = 0


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
