import numpy as np
from nethack_baselines.rllib.util import load_agent
from nethack_baselines.torchbeast.models import load_model
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from agents.batched_agent import BatchedAgent

# By default, choose from impala, ppo, a2c, dqn.
ALGO_CLASS_NAME = "impala"
CHECKPOINT_LOCATION = ""
# e.g.
# CHECKPOINT_LOCATION = "outputs/2021-06-08/15-04-39/ray_results/IMPALA_2021-06-08_15-04-43/IMPALA_RLlibNLE-v0_79638_00000_0_2021-06-08_15-04-43/checkpoint_000001/checkpoint-1"


class RLlibAgent(BatchedAgent):
    """
    A BatchedAgent using the TorchBeast Model
    """

    def __init__(self, num_envs, num_actions):
        super().__init__(num_envs, num_actions)
        if CHECKPOINT_LOCATION == "":
            raise ValueError(
                "You need to specify a CHECKPOINT_LOCATION for your model, otherwise submission won't work"
            )
        agent = load_agent(CHECKPOINT_LOCATION, ALGO_CLASS_NAME)
        self.policy = agent.get_policy(DEFAULT_POLICY_ID)
        self.preprocessor = agent.workers.local_worker().preprocessors[DEFAULT_POLICY_ID]
        self.state = [
            np.stack(init_states)
            for init_states in (zip(*[self.policy.get_initial_state() for _ in range(self.num_envs)]))
        ]
        self.previous_actions = [0] * self.num_envs

    def batch_inputs(self, observations):
        return [self.preprocessor.transform(observation) for observation in observations]

    def batched_step(self, observations, rewards, dones, infos):
        """
        Perform a batched step on lists of environment outputs.

        RLlib policies:
            * Take the observation, previous action and reward, and LSTM state as input
            * return outputs as a tuple of actions, state, and action information
        """
        observations = self.batch_inputs(observations)

        actions, state, _ = self.policy.compute_actions(
            observations,
            prev_action_batch=self.previous_actions,
            prev_reward_batch=rewards,
            state_batches=self.state,
        )

        for ep_done_idx_l in np.argwhere(dones):
            ep_done_idx = ep_done_idx_l[0]
            new_init_state = self.policy.get_initial_state()
            for i, inner_state in enumerate(self.state):
                inner_state[ep_done_idx] = new_init_state[i]
            self.previous_actions[ep_done_idx] = 0

        self.state = state

        self.previous_actions = actions

        return actions
