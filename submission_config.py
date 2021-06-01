from agents.custom_agent import CustomAgent
from agents.torchbeast_agent import TorchBeastAgent
from agents.rllib_batched_agent import RLlibAgent

from envs.wrappers import addtimelimitwrapper_fn

################################################
#         Import your own agent code           #
#      Set Submision_Agent to your agent       #
#    Set NUM_PARALLEL_ENVIRONMENTS as needed   #
#  Set submission_env_make_fn to your wrappers #
#        Test with local_evaluation.py         #
################################################


class SubmissionConfig:
    ## Add your own agent class
    AGENT = CustomAgent
    # AGENT = TorchBeastAgent
    # AGENT = RLlibAgent
    # AGENT = RandomAgent


    ## Change the NUM_ENVIRONMENTS as you need
    ## for example reduce it if your GPU doesn't fit
    ## Increasing above 32 is not advisable for the Nethack Challenge 2021
    NUM_ENVIRONMENTS = 32


    ## Add a function that creates your nethack env
    ## Mainly this is to add wrappers
    ## Add your wrappers to envs/wrappers.py and change the name here
    ## IMPORTANT: Don't "call" the function, only provide the name
    MAKE_ENV_FN = addtimelimitwrapper_fn


class TestEvaluationConfig:
    # Change this to locally check a different number of rollouts
    # The AIcrowd submission evaluator will not use this
    # It is only for your local evaluation
    NUM_EPISODES = 512
