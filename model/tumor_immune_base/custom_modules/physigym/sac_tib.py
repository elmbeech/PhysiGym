#####
# title: model/tumor_immune_base/custom_modules/physigym/sac_tib.py
#
# language: python3
# main libraries: gymnasium, physigym, torch
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     sac implementation for tumor immune base model
#####


#### IMPORT LIBRARIES ####
# Standard Python Libraries
import os
import random
import time
import argparse

# Gymnasium PhysiCell bridge module
import physigym

# Gymnasium
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

# Torch ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from tensordict import TensorDict

import wandb
import tyro

import pandas as pd

import matplotlib.pyplot as plt


################################
# Class PhysiCellModel Wrapper #
################################


class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = [
            "drug_1",
        ],
        weight: float = 0.8,
    ):
        """
        Args:
            env (gym.Env): The environment to wrap.
            list_variable_name (list[str]): List of variable names corresponding to actions in the original env.
            weight (float): Weight corresponding how much weight is added to the reward term related to cancer cells.
        """
        super().__init__(env)

        # Check that all variable names are strings
        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
                )

        self.list_variable_name = list_variable_name
        low = np.array(
            [
                env.action_space[variable_name].low[0]
                for variable_name in list_variable_name
            ]
        )
        high = np.array(
            [
                env.action_space[variable_name].high[0]
                for variable_name in list_variable_name
            ]
        )
        self._action_space = Box(low=low, high=high, dtype=np.float64)

        self.weight = weight

    @property
    def action_space(self):
        """Returns the flattened action space for the wrapper."""
        return self._action_space

    def step(self, action: np.ndarray):
        """
        Steps through the environment using the flattened action.

        Args:
            action (np.ndarray): The flattened action array.

        Returns:
            Tuple: Observation, reward, terminated, truncated, info.
        """
        # dictionnary action
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }
        # take a step in the environment
        o_observation, r_cancer_cells, b_terminated, b_truncated, info = self.env.step(
            d_action
        )

        r_drugs = np.mean(action)
        # add information into the info dictionnary
        info["action"] = d_action
        info["reward_drugs"] = r_drugs
        info["reward_cancer_cells"] = r_cancer_cells

        r_reward = -(1 - self.weight) * r_drugs + self.weight * r_cancer_cells

        return o_observation, r_reward, b_terminated, b_truncated, info


#########################
# Class Neural Networks #
#########################


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.Mish()

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.activation = nn.Mish()

    def forward(self, x):
        x = self.activation(self.conv(x))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, env):
        super().__init__()

        obs_shape = env.observation_space.shape
        self.is_image = len(obs_shape) == 3  # Check if input is an image (C, H, W)

        if self.is_image:
            # CNN feature extractor
            layers = [
                PixelPreprocess(),
                ImpalaBlock(obs_shape[0], 16),
                ImpalaBlock(16, 32),
                ImpalaBlock(32, 32),
                nn.Flatten(),
            ]
            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            # Directly flatten vector input
            self.feature_extractor = nn.Identity()
            self.feature_size = np.prod(obs_shape)

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        return x


class QNetwork(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self, env):
        super().__init__()
        self.feature_extractor = FeatureExtractor(env)

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc3 = nn.LazyLinear(1)
        self.mish = nn.Mish()

    def forward(self, x, a):
        x = self.feature_extractor(x)  # Extract features
        x = torch.cat([x, a], dim=1)  # Concatenate state and action

        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """Policy network (Actor)"""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, env):
        super().__init__()
        self.feature_extractor = FeatureExtractor(env)
        action_dim = np.prod(env.action_space.shape)

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc_mean = nn.LazyLinear(action_dim)
        self.fc_logstd = nn.LazyLinear(action_dim)
        self.relu = nn.ReLU()
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


#### Replay Buffers ####
#
####


class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.
    Stores states, actions, rewards, next states, and done flags.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        buffer_size,
        batch_size,
        state_type=np.float32,
    ):
        """
        Initializes the replay buffer.

        Parameters:
        - state_dim tuple(int): Dimensionality of the state space.
        - action_dim tuple(int): Dimensionality of the action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - state_type (numpy dtype, optional): Data type of the state representation (default: np.float32).
        """
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
        self.next_state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
        self.action = np.empty((self.buffer_size, *action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the current number of stored experiences in the buffer.
        """
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - state (np.ndarray): Current state.
        - action (np.ndarray): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state after taking the action.
        - done (bool): Whether the episode has ended.
        """
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
        - TensorDict containing sampled states, actions, rewards, next states, and done flags.
        """
        batch_size = self.batch_size

        # Ensure there are enough samples in the buffer
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        state = torch.as_tensor(self.state[sample_index]).float()
        next_state = torch.as_tensor(self.next_state[sample_index]).float()
        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        # Tensordict sampled experiences
        sample = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample


#### Algorithm Logic ####
#
# description:
#   The code is mainly inspired from:
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
####


def run(
    s_settingxml="config/PhysiCell_settings.xml",
    r_max_time_episode=10000.0,  # xpath
    i_thread=8,  # xpath
    i_seed=int(1),
    s_observation_mode="img_mc_substrates",
    s_render_mode=None,
    s_name="sac",
    b_wandb=True,
    i_total_step_learn=int(4e5),
):
    d_arg_run = {
        # basics
        "name": s_name,  # str: the name of this experiment
        # hardware
        "cuda": True,  # bool: should torch check for gpu (nvidia, amd mroc) accelerator?
        # tracking
        "wandb_track": b_wandb,  # bool: track with wandb, if false locallt tensorboard
        # random seed
        "seed": i_seed,  # int or none: seed of the experiment
        # steps
        "total_timesteps": i_total_step_learn,  # int: the total number of steps
    }
    # wandb
    d_arg_wandb = {
        "entity": "corporate-manu-sureli",  # str: the wandb s entity name
        "project": "SAC_IMAGE_TIB",  # str: the wandb s project name
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }

    # physigym
    d_arg_physigym_model = {
        "id": "physigym/ModelPhysiCellEnv-v0",  # str: the id of the gymnasium environmenit
        "settingxml": s_settingxml,
        "cell_type_cmap": {
            "tumor": "yellow",
            "cell_1": "green",
            "cell_2": "navy",
        },  # viridis
        "figsize": (6, 6),
        "observation_mode": s_observation_mode,  # str: scalars , img_rgb , img_mc
        "render_mode": s_render_mode,  # human, rgb_array
        "verbose": False,
        "img_rgb_grid_size_x": 64,  # pixel size
        "img_rgb_grid_size_y": 64,  # pixel size
        "img_mc_grid_size_x": 64,  # pixel size
        "img_mc_grid_size_y": 64,  # pixel size
        "normalization_factor": 512,  # normalization factor
    }
    d_arg_physigym_wrapper = {
        "list_variable_name": ["drug_1"],  # list of str: of action varaible names
        "weight": 0.8,  # float: weight for the reduction of tumor
    }

    # rl algorithm
    d_arg_rl = {
        # algoritm neural network I
        "buffer_size": int(3e5),  # int: the replay memory buffer size
        "batch_size": 128,  # int: the batch size of sample from the replay memory
        "learning_starts": 25e3,  # float: timestep to start learning
        "policy_frequency": 2,  # int: the frequency of training policy (delayed)
        "target_network_frequency": 1,  # int: the frequency of updates for the target nerworks (Denis Yarats" implementation delays this by 2.)
        # algorithm neural network II
        "autotune": True,  # bool: automatic tuning the the entropy coefficient.
        "alpha": 0.05,  # float: set manual entropy regularization coefficient.
        "tau": 0.005,  # float: target smoothing coefficient (default" : 0.005)
        "q_lr": 3e-4,  # float: the learning rate of the Q network network optimizer
        "policy_lr": 3e-4,  # float: the learning rate of the policy network optimizer
        # algorithm neural network III
        "gamma": 0.99,  # float: the discount factor gamma (how much learning)
    }

    # all in one
    d_arg = {}
    d_arg.update(d_arg_run)
    d_arg.update(d_arg_wandb)
    d_arg.update(d_arg_physigym_model)
    d_arg.update(d_arg_physigym_wrapper)
    d_arg.update(d_arg_rl)

    # initialize tracking
    s_run = f"{d_arg['name']}_seed_{d_arg['seed']}_observationtype_{d_arg['observation_mode']}_weight_{d_arg['weight']}_time_{int(time.time())}"
    if d_arg["wandb_track"]:
        print("tracking: wandb ...")
        run = wandb.init(name=s_run, config=d_arg, **d_arg_wandb)
        s_dir_run = os.path.join(
            run.dir, s_run
        )  # run.dir wandb/run-20250612_123456-abcdef
    else:
        print("tracking tensorboard ...")
        s_dir_run = os.path.join("tensorboard", s_run)
    s_dir_data = os.path.join(s_dir_run, "data")

    # initialize tensorbord recording
    writer = tensorboard.SummaryWriter(s_dir_run)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [f"|{s_key}|{s_value}|" for s_key, s_value in sorted(d_arg.items())]
            )
        ),
    )

    # initialize csv recording
    ld_data = []

    # set random seed
    random.seed(d_arg["seed"])
    np.random.seed(d_arg["seed"])
    if d_arg["seed"] is None:
        torch.seed()
        torch.backends.cudnn.deterministic = False
    else:
        torch.manual_seed(d_arg["seed"])
        torch.backends.cudnn.deterministic = True

    # initialize physigym environment
    env = gym.make(**d_arg_physigym_model)
    env = PhysiCellModelWrapper(env=env, **d_arg_physigym_wrapper)
    # manipulate setting xml
    env.get_wrapper_attr("x_root").xpath("//overall/max_time")[0].text = str(
        r_max_time_episode
    )
    env.get_wrapper_attr("x_root").xpath("//parallel/omp_num_threads")[0].text = str(
        i_thread
    )

    # initialize neural networks
    o_device = torch.device(
        "cuda" if torch.cuda.is_available() and d_arg["cuda"] else "cpu"
    )  # cpu or gpu
    actor = Actor(env).to(o_device)
    qf1 = QNetwork(env).to(o_device)
    qf2 = QNetwork(env).to(o_device)
    qf1_target = QNetwork(env).to(o_device)
    qf2_target = QNetwork(env).to(o_device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["q_lr"]
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=d_arg["policy_lr"])
    # set neural network entropy alpha by automatic tuning or manual
    if d_arg["autotune"]:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(o_device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=o_device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=d_arg["q_lr"])
    else:
        alpha = d_arg["alpha"]

    # initilize the reply buffer
    rb = ReplayBuffer(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.shape,
        device=o_device,
        buffer_size=d_arg["buffer_size"],
        batch_size=d_arg["batch_size"],
        state_type=env.observation_space.dtype,
    )

    while env.unwrapped.step_env < d_arg["total_timesteps"]:
        s_dir_data_episode = os.path.join(
            s_dir_data, f"episode{str(env.unwrapped.episode).zfill(8)}"
        )
        os.makedirs(s_dir_data_episode, exist_ok=True)
        # manipulate setting xml before reset
        # bue can be used for track or not track stuff, e.g. every 1024 episode
        # env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text = f"output/episode{str(i_episode).zfill(8)}"
        # manipulate setting xml before reset to record full physicell run every 1024 episode.
        if env.unwrapped.episode % 100 == 0:
            env.get_wrapper_attr("x_root").xpath("//save/folder")[
                0
            ].text = s_dir_data_episode
            env.get_wrapper_attr("x_root").xpath("//save/full_data/enable")[
                0
            ].text = "true"
            env.get_wrapper_attr("x_root").xpath("//save/SVG/enable")[0].text = "true"
        else:
            env.get_wrapper_attr("x_root").xpath("//save/folder")[
                0
            ].text = os.path.join(s_dir_data, "devnull")
            env.get_wrapper_attr("x_root").xpath("//save/full_data/enable")[
                0
            ].text = "false"
            env.get_wrapper_attr("x_root").xpath("//save/SVG/enable")[0].text = "false"
        # reset gymnasium env
        r_cumulative_return = 0
        r_discounted_cumulative_return = 0
        o_observation, d_info = env.reset(seed=d_arg["seed"])

        # time step loop
        b_episode_over = False
        while not b_episode_over:
            # sample the action space or learn
            if env.unwrapped.step_env <= d_arg["learning_starts"]:
                a_action = np.array(env.action_space.sample(), dtype=np.float16)
            else:
                x = torch.Tensor(o_observation).to(o_device).unsqueeze(0)
                actions, _, _ = actor.get_action(x)
                a_action = actions.detach().squeeze(0).cpu().numpy()

            # physigym step
            o_observation_next, r_reward, b_terminated, b_truncated, d_info = env.step(
                a_action
            )
            r_cumulative_return += r_reward
            r_discounted_cumulative_return += r_reward * d_arg["gamma"] ** (
                env.unwrapped.step_episode
            )
            b_episode_over = b_terminated or b_truncated

            # record to replay buffer
            rb.add(
                state=o_observation,
                action=a_action,
                next_state=o_observation_next,
                reward=r_reward,
                done=b_episode_over,
            )

            # for debuging the replay buffer
            if env.unwrapped.step_env == d_arg["batch_size"] * (1.05):
                data = rb.sample()
                with torch.no_grad():
                    next_state_actions, _, _ = actor.get_action(data["next_state"])
                    qf1(data["next_state"], next_state_actions)
                    qf2(data["next_state"], next_state_actions)
                    qf1_target(data["next_state"], next_state_actions)
                    qf2_target(data["next_state"], next_state_actions)
                del data, next_state_actions

            # learning
            if env.unwrapped.step_env > d_arg["learning_starts"]:
                data = rb.sample()
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(
                        data["next_state"]
                    )
                    qf1_next_target = qf1_target(data["next_state"], next_state_actions)
                    qf2_next_target = qf2_target(data["next_state"], next_state_actions)
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                    )
                    next_q_value = data["reward"].flatten() + (
                        1 - data["done"].flatten()
                    ) * d_arg["gamma"] * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data["state"], data["action"]).view(-1)
                qf2_a_values = qf2(data["state"], data["action"]).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the target networks
                if env.unwrapped.step_env % d_arg["target_network_frequency"] == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            d_arg["tau"] * param.data
                            + (1 - d_arg["tau"]) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            d_arg["tau"] * param.data
                            + (1 - d_arg["tau"]) * target_param.data
                        )

                # update the policy
                if (
                    env.unwrapped.step_env % d_arg["policy_frequency"] == 0
                ):  # TD 3 Delayed update support
                    # compensate for the delay by doing "actor_update_interval" instead of 1
                    for _ in range(d_arg["policy_frequency"]):
                        pi, log_pi, _ = actor.get_action(data["state"])

                        qf1_pi = qf1(data["state"], pi)
                        qf2_pi = qf2(data["state"], pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        # entropy autotune
                        if d_arg["autotune"]:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data["state"])

                            alpha_loss = (
                                -log_alpha.exp() * (log_pi + target_entropy)
                            ).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()

                            alpha = log_alpha.exp().item()

                    # record policy update to tensoboard
                    losses = {
                        "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                        "losses/qf1_values": qf1_a_values.mean().item(),
                        "losses/qf2_values": qf2_a_values.mean().item(),
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/qf2_loss": qf2_loss.item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/actor_loss": actor_loss.item(),
                    }

                    if d_arg["wandb_track"]:
                        run.log(losses)
                    else:
                        for tag, value in losses.items():
                            writer.add_scalar(tag, value, env.unwrapped.step_episode)

                    # record policy update to csv
                    # pass

            # handle observation
            o_observation = o_observation_next

            # recording step to tensorboard
            scalars = {
                "env/drug_1": a_action[0],
                "env/reward_value": r_reward,
                "env/number_tumor": d_info["number_tumor"],
                "env/number_cell_1": d_info["number_cell_1"],
                "env/number_cell_2": d_info["number_cell_2"],
                "env/reward_cancer_cells": d_info["reward_cancer_cells"],
                "env/reward_drugs": d_info["reward_drugs"],
            }
            if d_arg["wandb_track"]:
                run.log(scalars)
            else:
                for tag, value in scalars.items():
                    writer.add_scalar(tag, value, env.unwrapped.step_env)

            # record step to csv
            d_data = {
                "step": env.unwrapped.step_episode,
                "reward": r_reward,
                "cumulative_return": r_cumulative_return,
                "discounted_cumulative_return": r_discounted_cumulative_return,
                "drug_1": a_action[0],
                "number_tumor": d_info["number_tumor"],
                "number_cell_1": d_info["number_cell_1"],
                "number_cell_2": d_info["number_cell_2"],
            }
            ld_data.append(d_data)

        # recording episode to tensorbord
        scalars = {
            "charts/cumulative_return": r_cumulative_return,
            "charts/episodic_length": env.unwrapped.step_episode,
            "charts/discounted_cumulative_return": r_discounted_cumulative_return,
        }
        if d_arg["wandb_track"]:
            run.log(scalars)
        else:
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, env.unwrapped.step_env)

        # recording episode to csv
        df = pd.DataFrame(ld_data)
        df.to_csv(os.path.join(s_dir_data_episode, "data.csv"), index=False)
        ld_data = []

    # finish
    env.close()
    writer.close()


########
# Main #
########

if __name__ == "__main__":
    print("run physigym learing ...")

    # argv
    parser = argparse.ArgumentParser(
        prog="run physigym episodes",
        description="script to run physigym episodes.",
    )
    # settingxml file
    parser.add_argument(
        "settingxml",
        # type = str,
        nargs="?",
        default="config/PhysiCell_settings.xml",
        help="path/to/settings.xml file.",
    )
    # max_time
    parser.add_argument(
        "--max_time_episode",
        type=float,
        nargs="?",
        default=11580.0,
        help="set overall max_time in min in the settings.xml file.",
    )
    # thread
    parser.add_argument(
        "--thread",
        type=int,
        nargs="?",
        default=8,
        help="set parallel omp_num_threads in the settings.xml file.",
    )
    # seed
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=1,
        help="set options random_seed in the settings.xml file and python.",
    )
    # observation_mode
    parser.add_argument(
        "--observation_mode",
        # type = str,
        nargs="?",
        default="img_mc",
        help="observation mode scalars, img_rgb, img_mc or img_mc_substrates",
    )
    # render_mode
    parser.add_argument(
        "--render_mode",
        # type = str,
        nargs="?",
        default="none",
        help="render mode None, rgb_array, or human. observation mode scalars needs either render mode rgb_array or human.",
    )
    # name
    parser.add_argument(
        "--name",
        # type = str,
        nargs="?",
        default="sac",
        help="experiment name.",
    )
    # wandb tracking
    parser.add_argument(
        "--wandb",
        type=bool,
        nargs="?",
        default=True,
        help="tracking online with wandb? false with track locally with tensorboard.",
    )
    # total timesteps
    parser.add_argument(
        "--total_step_learn",
        type=int,
        nargs="?",
        default=int(2e5),
        help="set total time steps for the learing process to take.",
    )

    # parse arguments
    args = parser.parse_args()
    print(args)

    # processing
    run(
        s_settingxml=args.settingxml,
        r_max_time_episode=float(args.max_time_episode),
        i_thread=args.thread,
        i_seed=args.seed,
        s_observation_mode=args.observation_mode,
        s_render_mode=None if args.render_mode.lower() == "none" else args.render_mode,
        s_name=args.name,
        b_wandb=args.wandb,
        i_total_step_learn=int(args.total_step_learn),
    )
