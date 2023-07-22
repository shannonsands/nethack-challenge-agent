![Nethack Banner](https://aicrowd-production.s3.eu-central-1.amazonaws.com/misc/neurips-2021-nethack-challenge-media/nethack_final_link+preview_starter_kit.jpg)

# **[NeurIPS 2021 - The NetHack Challenge](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge)** - Starter Kit


This repository is the Nethack Challenge **Starter kit**! It contains:
* **Instructions** for setting up your codebase to make submissions easy.
* **Baselines** for quickly getting started training your agent.
* **Notebooks** for introducing you to NetHack and the NLE.
* **Documentation** for how to submit your model to the leaderboard.

Quick Links:

* [The NetHack Challenge - Competition Page](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge)
* [The NetHack Challenge - Discord Server](https://discord.gg/zkFWQmSWBA)
* [The NetHack Challenge - Starter Kit](https://gitlab.aicrowd.com/nethack/neurips-2021-the-nethack-challenge)
* [IMPORTANT - Accept the rules before you submit](https://www.aicrowd.com/challenges/neurips-2021-nethack-challenge/challenge_rules)

## Quick Start

With Docker and x1 GPU
```bash
# 1. CLONE THE REPO AND DOWNLOAD BASELINE MODELS
git clone https://github.com/shannonsands/nethack-challenge-agent.git \
    && cd nethack-challenge-agent \
    && git lfs install \
    && git lfs pull  

# 2. START THE DOCKER IMAGE
docker run -it -v $(pwd):/home/aicrowd --gpus 'all' fairnle/challenge:dev 

# 3. TEST AN EXISTING SUBMISSION 
python test_submission.py      # Tests ./saved_models/pretrained_0.5B

# 3. TRAIN YOUR OWN
python nethack_baselines/torchbeast/polyhydra.py batch_size=16 
```

To Troubleshoot see [here](#setting-up-details-docker).


# Table of Contents
1. [Intro to Nethack and the Nethack Challenge](#intro-to-nethack-and-the-nethack-challenge)
2. [Setting up your codebase](#setting-up-your-codebase)
3. [Baselines](#baselines)
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)

# Intro to Nethack and the Nethack Challenge

Your goal is to produce the best possible agent for navigating the depths
of Nethack dungeons and emerging with the Amulet in hand! 
You can approach this task however you please, but a good starting point 
would be [**this notebook**](./notebooks/NetHackTutorial.ipynb) which provides
an overview of  (1) the many dynamics at play in the game and   (2) the 
observation and action space with which your agent will interact. 

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2021-nethack-challenge).
2. **Clone** this repo and start developing your solution.
3. **Train** your models on NLE, and ensure run.sh will generate rollouts.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the NLE environment for a fixed 
number of rollouts to compute and report the metrics on the leaderboard
of the competition.

![](https://images.aicrowd.com/raw_images/challenges/banner_file/423/5f69010437d25bf569c4.jpg)

# Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

### How does submission work?

The submission entrypoint is a bash script `run.sh`, that runs in an environment defined by `Dockerfile`. When this script is 
called, aicrowd will expect you to generate all your rollouts in the 
allotted time, using `aicrowd_gym` in place of regular `gym`.  This means 
that AIcrowd can make sure everyone is running the same environment, 
and can keep score!

### What languages can I use?

Since the entrypoint is a bash script `run.sh`, you can call any arbitrary
code from this script.  However, to get you started, the environment is 
set up to generate rollouts in Python. 

The repo gives you a template placeholder to load your model 
(`agents/your_agent.py`), and a config to choose which agent to load 
(`submission_config.py`). You can then test a submission, adding all of 
AIcrowd’s timeouts on the environment, with `python test_submission.py`

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the [RUNTIME.md](/docs/RUNTIME.md) file.

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json                  # Submission meta information - add tags for tracks here
├── apt.txt                       # Packages to be installed inside submission environment
├── requirements.txt              # Python packages to be installed with pip
├── rollout.py                    # This will run rollouts on a batched agent
├── test_submission.py            # Run this on your machine to get an estimated score
├── run.sh                        # Submission entrypoint
├── utilities                     # Helper scripts for setting up and submission 
│   └── submit.sh                 # script for easy submission of your code
├── envs                          # Operations on the env like batching and wrappers
│   ├── batched_env.py            # Batching for multiple envs
│   └── wrappers.py   	          # Add wrappers to your env here
├── agents                        # Baseline agents for submission
│   ├── batched_agent.py          # Abstraction reference batched agents
│   ├── random_batched_agent.py	  # Batched agent that returns random actions
│   ├── rllib_batched_agent.py	  # Batched agent that runs with the rllib baseline
│   └── torchbeast_agent.py       # Batched agent that runs with the torchbeast baseline
├── nethack_baselines             # Baseline agents for submission
│    ├── other_examples  	
│    │   └── random_rollouts.py   # Barebones random agent with no batching
│    ├── rllib	                  # Baseline agent trained with rllib
│    └── torchbeast               # Baseline agent trained with IMPALA on Pytorch
└── notebooks                 
    └── NetHackTutorial.ipynb     # Tutorial on the Nethack Learning Environment

```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** See [How do I actually make a submission?](#how-do-i-actually-make-a-submission) below for more details.


### How can I get going with an existing baseline?

The best current baseline is the torchbeast baseline. Follow the instructions 
[here](/nethack_baselines/torchbeast/) to install and start training 
the model (there are even some suggestions for improvements).

To then submit your saved model, simply set the `AGENT` in 
`submission config` to be `TorchBeastAgent`, and modify the 
`agent/torchbeast_agent.py` to point to your saved directory.

You can now test your saved model with `python test_submission.py`

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, just adapt
`YourAgent` in `agents/your_agent.py` to load your model and take a `batched_step`.

Then just set your `AGENT` in `submission_config.py` to be this class 
and you are ready to test with `python test_submission.py`

### How do I actually make a submission?

First you need to fill in you `aicrowd.json`, to give AIcrowd some info so you can be scored.
The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "neurips-2021-the-nethack-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "gpu": true
}
```

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and 
pushing to AIcrowd's GitLab. The rest is done for you!

More details are available [here](/docs/SUBMISSION.md).

### Are there any hardware or time constraints?

Your submission will need to complete 128 rollouts in 30 minutes. We will
run 4 of these in parallel, and a total of 512 episodes will be used for
evaluation. The episode will timeout and terminate if any action is
left hanging for 300 seconds, or 10,000 steps are taken without 
advancing the in game clock. 

The machine where the submission will run will have following specifications:
* 1 NVIDIA T4 GPU
* 4 vCPUs
* 16 GB RAM


## Setting Up Details [No Docker]

1. **Add your SSH key** to AIcrowd GitLab

    You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:nethack/neurips-2021-the-nethack-challenge.git
    ```
    
3. **Verify you have dependencies** for the Nethack Learning Environment

    NLE requires `python>=3.5`, `cmake>=3.14` to be installed and available both when building the
    package, and at runtime.
    
    On **MacOS**, one can use `Homebrew` as follows:
    
    ``` bash
    brew install cmake
    ```
    
    On a plain **Ubuntu 18.04** distribution, `cmake` and other dependencies
    can be installed by doing:
    
    ```bash
    # Python and most build deps
    sudo apt-get install -y build-essential autoconf libtool pkg-config \
        python3-dev python3-pip python3-numpy git flex bison libbz2-dev
    
    # recent cmake version
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    sudo apt-get update && apt-get --allow-unauthenticated install -y \
        cmake \
        kitware-archive-keyring
    ```

4. **Install** competition specific dependencies!

    We advise using a conda environment for this:
    ```bash
    # Optional: Create a conda env
    conda create -n nle_challenge python=3.8 'cmake>=3.15'
    conda activate nle_challenge
    pip install -r requirements.txt
    ```
    If `pip install` fails with errors when installing NLE, please see installation requirements at https://github.com/facebookresearch/nle.

5. **Run rollouts** with a random agent with `python test_submission.py`.

    Find more details on the [original nethack repository](https://github.com/facebookresearch/nle)

## Setting Up Details [Docker]

With Docker, setting up is very simple! Simply pull a preexisting image from the fair nle repo.

```
docker pull fairnle/challenge:dev 
```
This image is based of Ubuntu 18.04, with CUDA 10.2 and cudnn 7, and is the Docker image corresponding to the `nhc-dev` target in the `Dockerfile`. You can run it as follows:

**Without GPUS**
```
docker run -it -v $(pwd):/home/aicrowd fairnle/challenge:dev
```
**With GPUS**
```
docker run -it -v $(pwd):/home/aicrowd --gpus 'all' fairnle/challenge:dev
```

*NB* On Linux, this `--gpus` argument requires you to install `nvidia-container-toolkit`, which on Ubuntu is available with `apt install`.

This will take you into an image, with your current working directory mounted as a volume. At submission time, `nhc-submit` target will be built by AIcrowd, which copies all the files into the image, instead of simply mounting them.

If you wish to wish to build your own dev environment from the Dockerfile, you can do this with:

```
docker build --target nhc-dev  -t your-image-name .
```


# Baselines

Although we are looking to supply this repository with more baselines throughout the first month of the competition, this repository comes with a strong IMPALA-based baseline in the directory `./nethack_baselines/torchbeast`.

The [README](/nethack_baselines/torchbeast/READMEmd) has more info about the baselines, including to install and start training the model (there are even some suggestions for improvements).

The TorchBeast baseline comes with two sets of weights - the same model trained to 250 million steps, and 500 million steps. 

To download these weights, run `git lfs pull`, and check `saved_models`. 

The TorchBeast agent can then be selected by setting `AGENT=TorchBeastAgent` in the `submission_config.py`, and the weights can be changed by changing the `MODEL_DIR` in `agents/torchbeast_agent.py`. 

More information on git lfs can be found on [SUBMISSION.md](/docs/SUBMISSION.md). 


# How to Test and Debug Locally

The best way to test your model is to run your submission locally.

You can do this naively by simply running  `python rollout.py` or you can simulate the extra timeout wrappers that AIcrowd will implement by using `python test_submission.py`. 

# How to Submit

More information on submissions can be found at our [SUBMISSION.md](/docs/SUBMISSION.md).

## Contributors

- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)
- [Shivam Khandelwal](https://www.aicrowd.com/participants/shivam)
- [Eric Hambro](https://www.aicrowd.com/participants/eric_hammy)
- [Danielle Rothermel](https://www.aicrowd.com/participants/danielle_rothermel)
- [Jyotish Poonganam](https://www.aicrowd.com/participants/jyotish)


# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge
- 🗣️ Discussion Forum: https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/leaderboards

**Best of Luck** 🎉 🎉
