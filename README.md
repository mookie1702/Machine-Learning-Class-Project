# Single-agent problem Solving Based on Q-learning

## Introduction

This Project was my machine learning class’s homework.

The project was about that in a fixed-size map, you had to control a agent to escape from the adversary. 

The agent should finish the missions with these conditions as follow:

* Arrive the checkpoint.
* Avoid colliding with the borders and the landmarks.
* Avoid being caught by the adversary. (The pursue strategy of adversary has been given.)

I used the Q-learning algorithm to do with the project and completed it with all conditions done, even though its effect was not good.

If you want to deal with the problem like this well, I think you have to use some Actor-Critic like algorithm, such as A3C, DDPG, MADDPG, PPO and so on.

## Environment

Ubuntu 20.04 with anaconda.

Dependencies: python 3.6 or later, gym 0.10.5, numpy, scipy, pandas.

This project’s environment is based on the [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs).

TA modified the `simple_tag.py` to add the borders, landmarks, and a checkpoints. 

## Installation

Dependencies Installation：

```bash
pip install gym==0.10.5
pip install numpy
pip install scipy
pip install pandas
```

Environment Installation:

```
cd [this_project]
pip install -e .
```

## Usage

Training

```bash
cd training
python training.py
```

Display

```bash
cd display
python display.py
```



# Maintainers

[@mookie1702](https://github.com/mookie1702)

# Licenses

[MIT](https://opensource.org/licenses/MIT)

