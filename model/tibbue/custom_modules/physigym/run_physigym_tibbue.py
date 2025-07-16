#####
# title: run_physigym_tibbue.py
#
# language: python3
#
# date: 2024-spring
# license: bsb-3-clause
# author: Alexandre Bertin, Elmar Bucher
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# run:
#   1. cd path/to/PhysiCell
#   2. python3 custom_modules/physigym/physigym/envs/run_physigym_tibbue.py
#
# description:
#   python script to run a single episode from the physigym tibbue model.
#####


# library
from extending import physicell
import gymnasium
import numpy as np
import physigym
from random import randrange

# load PhysiCell Gymnasium environment
# %matplotlib
# env = gymnasium.make("physigym/ModelPhysiCellEnv-v0", settingxml="config/PhysiCell_settings.xml", cell_type_cmap="turbo", figsize=(8,6), render_mode="human", render_fps=10, verbose=True, **kwargs)
env = gymnasium.make("physigym/ModelPhysiCellEnv-v0")

# reset the environment
r_reward = 0.0
o_observation, d_info = env.reset()

# time step loop
b_episode_over = False
while not b_episode_over:

    # policy according to o_observation
    #d_action = {"drug_1": np.array([0.5], dtype=np.float16)}
    d_action = {"drug_1": np.array([0.3], dtype=np.float16)}
    print("observation:", sorted(o_observation))
    print("reward:", r_reward)
    print("info:", sorted(d_info))
    print("action:", sorted(d_action))

    # action
    o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)
    b_episode_over = b_terminated or b_truncated
    print("episode_over:", b_episode_over)

# drop the environment
env.close()
