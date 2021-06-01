# RLlib NetHackChallenge Benchmark

This is a baseline model for the NetHack Challenge based on
[RLlib](https://github.com/ray-project/ray#rllib-quick-start).

It comes with all the code you need to train, run and submit a model, and you
can choose from a variety of algorithms implemented in RLlib.

We provide default configuration and hyperparameters for 4 algorithms:
* IMPALA
* DQN
* PPO
* A2C

You're not restricted to using these algorithms - others could be added with
minimal effort in `train.py` and `util/loading.py`.

This implementation runs many simultaneous environments with dynamic batching.

## Installation

To get this running, you'll want to create a virtual environment (probably with
conda)

```bash
conda create -n nle-competition python=3.8
conda activate nle-competition
```

Then you'll want to install the requirements at the root of this repository,
both from the `requirements.txt` and the `setup.py`:

```bash
pip install -r requirements.txt
pip install . -e
```

This will install the repository as a python package in editable mode, meaning
any changes you make to the code will be recognised.

## Running The Baseline

Once installed, from the root of the repository run:

```bash
python nethack_baselines/rllib/train.py
```

This will train the default algorithm (IMPALA) with default hyperparameters.
You can choose a different algorithm as follows:

```bash
python nethack_baselines/rllib/train.py algo=ppo
```

You can also control other hyperparameters on the command line:

```bash
python nethack_baselines/rllib/train.py algo=ppo num_sgd_iter=5 total_steps=1000000
```

An important configuration is the number of cpus and gpus that are available,
which can be set with `num_gpus` and `num_cpus` - the higher these numbers
(especially cpus) the faster training will be.

This configuration can also be changed the adjusting `config.yaml`

The output of training will be in a directory `outputs` at the root of the
repository, with each run having a date and time-based folder.

## Making a submission

Once the training is complete, model checkpoints will be available in
`outputs/<data>/<time>/ray_results/...`. At the end of training the script will
print out the file-path which needs to be used to specify the agent. This
file-path should be used in `agents/rllib_batched_agent.py` as
`CHECKPOINT_LOCATION`, and you should also set `ALGO_CLASS_NAME` to the
algorithm you used (impala, ppo, etc..). **If you don't change these values the
submission won't use your new model**.

Next, make sure to **add the model checkpoints to the git repository**, for example:

```bash
git add -f outputs/2021-06-08/15-04-39/ray_results/
```

Finally, commit all you changes (including the added checkpoint and changed `agents/rllib_batched_agent.py`),
tag the submission and push the branch and tag to AIcrowd's GitLab.

## Repo Structure

```
nle_baselines/rllib
├── models.py         #  <- Models HERE
├── util/           
├── config.yaml       #  <- Flags HERE
├── train.py          #  <- Training Loop HERE
├── env.py            #  <- Training Envionment HERE
```

The structure is simple, compartmentalising the environment setup, training
loop and models in to different files. You can tweak any of these separately,
and add parameters to the flags (which are passed around).

## About the Model

This model (`BaseNet`) we provide is simple and all in `models.py`.

* It encodes the dungeon into a fixed-size representation (`GlyphEncoder`)
* It encodes the topline message into a fixed-size representation (`MessageEncoder`)
* It encodes the bottom line statistics (eg armour class, health) into a fixed-size representation (`BLStatsEncoder`)
* It concatenates all these outputs into a fixed size, runs this through a fully connected layer
* If using an LSTM (which is controlled by RLlib), then this output is passed through and LSTM,
  and then fully connect layers for various policy ouputs (such as value function and action distribution)

As you can see there is a lot of data to play with in this game, and plenty to try, both in modelling and in the learning algorithms used.

## Improvement Ideas

*Here are some ideas we haven't tried yet, but might be easy places to start. Happy tinkering!*


### Model Improvements (`model.py`)

* The model is currently not using the terminal observations (`tty_chars`, `tty_colors`, `tty_cursor`), so it has no idea about menus - could this we make use of this somehow?
* The bottom-line stats are very informative, but very simply encoded in `BLStatsEncoder` - is there a better way to do this?
* The `GlyphEncoder` builds a embedding for the glyphs, and then takes a crop of these centered around the player icon coordinates (`@`). Should the crop be reusing these the same embedding matrix? 
* The current model constrains the vast action space to a smaller subset of actions. Is it too constrained? Or not constrained enough?

###  Environment Improvements (`envs.py`)

* Opening menus (such as when spellcasting) do not advance the in game timer. However, models can also get stuck
  in menus as you have to learn what buttons to press to close the menu. Can changing the penalty for not advancing
  the in-game timer improve the result? 
* The NetHackChallenge assesses the score on random character assignments. Might it be easier to learn on just a few of these at the beginning of training? 

### Algorithm/Optimisation Improvements (`train.py`)

* Which algorithm from RLlib works best? Which hyperparameters are the ones we expect to perform well?

## How to add an algorithm

If you wanted to use an algorithm from RLlib which we don't provide a default
configuration for, here's some pointers to what's necessary:
* Add the algorithm to `NAME_TO_TRAINER` in
  `nethack_baselines/rllib/util/loading.py`, so that it can be loaded correctly.
* Add a configuration key to `config.yaml` with the algorithm's name (e.g.
  `sac`), and under that key specify the configuration that's specific to that
  algorithm (e.g. `initial_alpha: 0.5`)

Once that's done, you should be able to use the new algorithm by running

```bash
python nle_baselines/rllib/train.py algo=sac
```
