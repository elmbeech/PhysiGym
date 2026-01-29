# run_physigym_tip_sac_async_mbpo.py
import argparse
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
import wandb

from tqdm import tqdm

# Your project imports
from vectorized_tip import vec_envs
from rb_tip import ReplayBuffer, ReplayBufferLatentSpace
import queue
from nn_tip import Encoder

# --------------------------------------------------------------
# ---------- Encoder, Actor, Q-network, Dynamics ----------------
# --------------------------------------------------------------
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, d_arg_env, encoder_feature_size):
        super().__init__()
        self.encoder_feature_size = encoder_feature_size
        self.action_dim = np.prod(d_arg_env["action_space_shape"])
        self.fc = nn.Sequential(
            nn.Linear(self.encoder_feature_size, 128),
            nn.Mish(),
            nn.Linear(128, self.action_dim),
            nn.Tanh(),
        )

    def get_action(self, z):
        mean = self.fc(z)
        log_std = torch.zeros_like(mean)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_pi = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_pi, dist


class QNetwork(nn.Module):
    def __init__(self, d_arg_env, encoder_feature_size):
        super().__init__()
        self.encoder_feature_size = encoder_feature_size
        self.fc = nn.Sequential(
            nn.Linear(
                self.encoder_feature_size + np.prod(d_arg_env["action_space_shape"]),
                128,
            ),
            nn.Mish(),
            nn.Linear(128, 1),
        )

    def forward(self, z, action):
        q = self.fc(torch.cat([z, action], dim=-1))
        return q


class DynamicsModel(nn.Module):
    """
    Simple probabilistic dynamics model for MBPO:
    Input: state_feat + action
    Output: next_state_feat, reward
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Mish(),
        )
        self.delta_state = nn.Linear(hidden_dim, latent_dim)
        self.reward = nn.Linear(hidden_dim, 1)

    def forward(self, state_feat, action):
        x = torch.cat([state_feat, action], dim=-1)
        h = self.model(x)
        next_state_feat = state_feat + self.delta_state(h)
        reward = self.reward(h)
        return next_state_feat, reward


# --------------------------------------------------------------
# Actor process (unchanged)
# --------------------------------------------------------------
def actor_process(
    actor_queue,
    sample_queue,
    stats_queue,
    d_arg,
    stop_event,
    env_info_queue,
    encoder_queue,
):
    envs = vec_envs(d_arg)
    begin_time = time.time()
    obs = envs.reset()

    d_arg_env = {
        "action_space_shape": envs.action_space.shape,
        "observation_space_shape": envs.observation_space.shape,
        "observation_mode": d_arg["model"]["observation_mode"],
        "node_feature_dim": envs.get_attr("node_dim")[0],
        "x_min": envs.get_attr("x_min")[0],
        "x_max": envs.get_attr("x_max")[0],
        "y_min": envs.get_attr("y_min")[0],
        "y_max": envs.get_attr("y_max")[0],
        "action_space_high": envs.action_space.high,
        "action_space_low": envs.action_space.low,
        "observation_space_dtype": envs.observation_space.dtype,
        "is_graph": "graph" in d_arg["model"]["observation_mode"],
    }
    env_info_queue.put(d_arg_env)

    encoder_local = Encoder(d_arg_env)
    actor_local = Actor(d_arg_env).cpu()
    actor_local.eval()
    num_envs = envs.num_envs
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    local_step = 0

    while not stop_event.is_set():
        # fetch new policy
        try:
            while True:
                new_params = actor_queue.get_nowait()
                actor_local.load_state_dict(new_params)
                new_params = encoder_queue.get_nowait()
                encoder_local.load_state_dict(new_params)
        except queue.Empty:
            pass

        if local_step <= d_arg["rl"]["learning_starts"]:
            actions = np.array(
                [envs.action_space.sample() for _ in range(num_envs)], dtype=np.float32
            )
        else:
            with torch.no_grad():
                x = torch.from_numpy(obs).cpu()
                z = encoder_local(x).cpu()
                actions_tensor, _, _ = actor_local.get_action(z)
                actions = actions_tensor.cpu().numpy()

        next_obs, rewards, dones, infos = envs.step(actions)

        # Restart if all dead
        if all(info.get("disabled", False) for info in infos):
            envs.close()
            envs = vec_envs(d_arg)
            obs = envs.reset()
            num_envs = envs.num_envs
            episode_returns = np.zeros(num_envs, dtype=np.float64)

        episode_returns += rewards.astype(np.float64)
        local_step += num_envs - len(envs.dead_envs)

        batch_samples = []
        for i in range(num_envs):
            if i in envs.dead_envs:
                continue
            info = infos[i]
            done = dones[i]
            o = obs[i].copy() if isinstance(obs[i], np.ndarray) else obs[i]
            no = (
                next_obs[i].copy()
                if isinstance(next_obs[i], np.ndarray)
                else next_obs[i]
            )

            if done:
                try:
                    stats_queue.put_nowait(
                        {
                            "episode_return": float(episode_returns[i]),
                            "episode_length": int(info["step_episode"]),
                            "step": int(local_step),
                            "timestamp": time.time() - begin_time,
                        }
                    )
                except queue.Full:
                    pass
                episode_returns[i] = 0.0

            batch_samples.append((o, actions[i], float(rewards[i]), no, bool(dones[i])))

        if batch_samples:
            try:
                sample_queue.put_nowait(batch_samples)
            except queue.Full:
                pass

        obs = next_obs
    try:
        envs.close()
    except Exception:
        pass


# --------------------------------------------------------------
# Main async SAC with MBPO
# --------------------------------------------------------------
def run_async_sac(d_arg):
    device = torch.device(
        "cuda" if d_arg["simulation"]["cuda"] and torch.cuda.is_available() else "cpu"
    )
    seed = d_arg["simulation"]["seed"] or 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    actor_queue = mp.Queue(maxsize=5)
    encoder_queue = mp.Queue(maxsize=5)
    sample_queue = mp.Queue(maxsize=1000)
    stats_queue = mp.Queue(maxsize=100)
    env_info_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    actor_proc = mp.Process(
        target=actor_process,
        args=(
            actor_queue,
            sample_queue,
            stats_queue,
            d_arg,
            stop_event,
            env_info_queue,
            encoder_queue,
        ),
    )
    actor_proc.start()
    d_arg_env = env_info_queue.get()
    d_arg["env"] = d_arg_env

    # --- Replay buffer ---
    rb = ReplayBuffer(
        state_dim=d_arg_env["observation_space_shape"],
        action_dim=d_arg_env["action_space_shape"],
        device=device,
        buffer_size=d_arg["rl"]["buffer_size"],
        batch_size=d_arg["rl"]["batch_size"],
        state_type=d_arg_env["observation_space_dtype"],
        is_graph=d_arg_env["is_graph"],
    )

    # --- Networks ---
    encoder = Encoder(
        d_arg_env, out_channels=d_arg["architecture"]["encoder"]["out_channels"]
    ).to(device)
    actor = Actor(d_arg_env).to(device)
    qf1 = QNetwork(d_arg_env).to(device)
    qf2 = QNetwork(d_arg_env).to(device)
    dynamics = DynamicsModel(
        state_dim=encoder.feature_size,
        action_dim=np.prod(d_arg_env["action_space_shape"]),
    ).to(device)
    dynamics_optimizer = optim.Adam(dynamics.parameters(), lr=3e-4)

    # Dummy forward to initialize networks
    rb_model = ReplayBufferLatentSpace(
        latent_dim=encoder.feature_size,
        action_dim=d_arg_env["action_space_shape"],
        device=device,
        buffer_size=int(d_arg["rl"]["buffer_size"] * 0.1),
        batch_size=int(d_arg["rl"]["batch_size"] * 0.5),
    )

    dummy_state = torch.zeros(
        (1, *d_arg_env["observation_space_shape"]),
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        actions_tensor, _, _ = actor.get_action(encoder(dummy_state))
        _ = qf1(dummy_state, actions_tensor)
        _ = qf2(dummy_state, actions_tensor)

    qf1_target = deepcopy(qf1)
    qf2_target = deepcopy(qf2)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=d_arg["rl"]["q_lr"]
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=d_arg["rl"]["policy_lr"])

    # Alpha entropy
    if d_arg["rl"]["autotune"]:
        target_entropy = -float(np.prod(d_arg_env["action_space_shape"]))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=d_arg["rl"]["q_lr"])
        alpha = log_alpha.exp().item()
    else:
        alpha = float(d_arg["rl"]["alpha"])

    # Send initial policy
    try:
        actor_queue.put_nowait(
            {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        )
    except queue.Full:
        actor_queue.put({k: v.detach().cpu() for k, v in actor.state_dict().items()})

    # Logging
    output_dir = d_arg["model"]["output_dir"]
    writer = SummaryWriter(log_dir=output_dir)
    if d_arg["simulation"]["wandb_track"]:
        wandb.init(
            project=d_arg["wandb"]["project"] if "wandb" in d_arg else "SAC_ASYNC_TIP",
            name=Path(output_dir).name,
            config=d_arg,
        )

    tau = d_arg["rl"]["tau"]

    print("Starting training loop...")
    try:
        total = d_arg["rl"]["total_timesteps"]
        pbar = tqdm(total=total)
        drained = 0
        grad_steps = 0

        while drained < total:
            pbar.update(drained - pbar.n)
            local_batch = []

            while True:
                try:
                    item = sample_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, list):
                    local_batch.extend(item)
                else:
                    local_batch.append(item)

            if local_batch:
                rb.add_batch(local_batch)
                drained += len(local_batch)

            # --- Train dynamics model MBPO ---
            if len(rb) > d_arg["rl"]["batch_size"]:
                batch_real = rb.sample()
                state_feat = encoder(batch_real["state"])
                next_state_feat = encoder(batch_real["next_state"])
                action = batch_real["action"]
                reward = batch_real["reward"]

                pred_next, pred_r = dynamics(state_feat, action)
                loss_dyn = F.mse_loss(pred_next, next_state_feat) + F.mse_loss(
                    pred_r.squeeze(), reward
                )
                dynamics_optimizer.zero_grad()
                loss_dyn.backward()
                dynamics_optimizer.step()

                # Model-based rollouts
                rollout_horizon = 5
                mb_state_feat = state_feat.clone().detach()

                for t in range(rollout_horizon):
                    mb_action, _, _ = actor.get_action(mb_state_feat)
                    mb_next_feat, mb_r = dynamics(mb_state_feat, mb_action)
                    mb_done = torch.zeros(len(mb_state_feat), device=device)

                    # Add directly to replay buffer
                    rb_model.add_batch(
                        list(zip(mb_state_feat, mb_action, mb_r, mb_next_feat, mb_done))
                    )

                    mb_state_feat = (
                        mb_next_feat.detach()
                    )  # detach to avoid backprop through MBPO

            # --- Log stats ---
            while not stats_queue.empty():
                try:
                    stat = stats_queue.get_nowait()
                except queue.Empty:
                    break
                log_dict = {
                    "charts/return": stat["episode_return"],
                    "charts/length": stat["episode_length"],
                    "charts/step": stat["step"],
                    "charts/timestamp": stat["timestamp"],
                    "charts/grad_steps": grad_steps,
                    "charts/samples_drained": drained,
                }
                for tag, value in log_dict.items():
                    writer.add_scalar(tag, value, drained)
                if d_arg["simulation"]["wandb_track"]:
                    wandb.log(log_dict, step=drained)

            # --- Skip training if not enough samples ---
            if drained < max(d_arg["rl"]["learning_starts"], d_arg["rl"]["batch_size"]):
                time.sleep(0.1)
                continue

            # --- SAC updates (unchanged, uses encoder features) ---
            for _ in range(d_arg["rl"]["num_loops"]):
                batch_real = rb.sample()
                batch_model = rb_model.sample()  # latent features

                # Concatenate (real observations + decoded latent if needed)
                z = torch.cat(
                    [encoder(batch_real["state"]), batch_model["state"]], dim=0
                )
                next_z = torch.cat(
                    [encoder(batch_real["next_state"]), batch_model["next_state"]],
                    dim=0,
                )
                action = torch.cat([batch_real["action"], batch_model["action"]], dim=0)
                reward = torch.cat([batch_real["reward"], batch_model["reward"]], dim=0)
                done = torch.cat([batch_real["done"], batch_model["done"]], dim=0)

                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(next_z)
                    q1_next = qf1_target(next_z, next_actions)
                    q2_next = qf2_target(next_z, next_actions)
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    next_q = (
                        reward.flatten()
                        + (1 - done.flatten())
                        * d_arg["rl"]["gamma"]
                        * min_q_next.squeeze()
                    )

                q1 = qf1(z, action).view(-1)
                q2 = qf2(z, action).view(-1)
                qf_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()
                grad_steps += 1

                if grad_steps % d_arg["rl"]["policy_frequency"] == 0:
                    actions_pi, log_pi, _ = actor.get_action(z=z)
                    q1_pi = qf1(z, actions_pi)
                    q2_pi = qf2(z, actions_pi)
                    actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if d_arg["rl"]["autotune"]:
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy).detach()
                        ).mean()
                        alpha_optim.zero_grad()
                        alpha_loss.backward()
                        alpha_optim.step()
                        alpha = log_alpha.exp().item()

                # Soft update targets
                if grad_steps % d_arg["rl"]["target_network_frequency"] == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1.0 - tau) * target_param.data
                        )

            # Send updated policy
            if grad_steps % 64 == 0:
                try:
                    actor_queue.put_nowait(
                        {k: v.detach().cpu() for k, v in actor.state_dict().items()}
                    )
                    encoder_queue.put_nowait(
                        {
                            k: v.detach().cpu()
                            for k, v in encoder_queue.state_dict().items()
                        }
                    )
                except queue.Full:
                    pass

    except KeyboardInterrupt:
        print("Interrupted by user — shutting down.")
    finally:
        stop_event.set()
        actor_proc.join(timeout=5.0)
        if actor_proc.is_alive():
            actor_proc.terminate()
            actor_proc.join(timeout=1.0)
        writer.close()
        if d_arg["simulation"]["wandb_track"]:
            wandb.finish()


# --------------------------------------------------------------
# Entry point (unchanged)
# --------------------------------------------------------------
if __name__ == "__main__":
    import argparse, time

    print("Starting asynchronous SAC MBPO for PhysiGym...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--settingxml", default="config/PhysiCell_settings.xml")
    parser.add_argument("--settingcells", default="config/cells.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="true")
    parser.add_argument("--observation_mode", default="img_mc_cells")
    parser.add_argument("--max_time_episode", type=float, default=12900.0)
    parser.add_argument("--learning_starts", type=int, default=1e4)
    parser.add_argument("--total_timesteps", type=int, default=int(2e5))
    parser.add_argument("--rl_threads", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=9)
    parser.add_argument("--buffer_size", type=int, default=int(2e5))
    parser.add_argument("--batch_size_multiplier", type=int, default=64)
    parser.add_argument("--name", default="async_sac_mbpo_tip")
    parser.add_argument("--wandb", default="true")
    parser.add_argument("--entity", default="corporate-manu-sureli")
    parser.add_argument("--tumor", type=int, default=512)
    parser.add_argument("--cell_1", type=int, default=128)
    parser.add_argument("--cell_2_fraction", type=float, default=None)
    parser.add_argument("--s_frequency_save_data", type=int, default=1)
    args = parser.parse_args()

    # Build d_arg
    d_arg_simulation = {
        "name": args.name,
        "cuda": args.gpu.lower().startswith("t"),
        "wandb_track": args.wandb.lower().startswith("t"),
        "seed": args.seed,
        "max_time": args.max_time_episode,
    }
    d_arg_wandb = {
        "entity": args.entity,
        "project": "SAC_ASYNC_TIP",
        "sync_tensorboard": True,
        "monitor_gym": True,
        "save_code": True,
    }
    d_arg_physigym_model = {
        "id": "physigym/ModelPhysiCellEnv-v0",
        "settingxml": args.settingxml,
        "settingcells": args.settingcells,
        "cell_type_cmap": {
            "tumor": "yellow",
            "cell_1": "green",
            "cell_2": "navy",
            "other_tissue": "red",
        },
        "figsize": (6, 6),
        "observation_mode": args.observation_mode,
        "render_mode": None,
        "verbose": False,
        "img_rgb_grid_size_x": 64,
        "img_rgb_grid_size_y": 64,
        "img_mc_grid_size_x": 64,
        "img_mc_grid_size_y": 64,
        "normalization_factor": args.tumor,
    }
    d_arg_physigym_wrapper = {
        "list_variable_name": ["drug_1"],
        "w_cell": 0.7,
        "w_increase": 0.2,
        "w_amount": 0.1,
        "frequency_save_data": args.s_frequency_save_data,
    }
    d_arg_rl = {
        "total_timesteps": args.total_timesteps,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size_multiplier * args.num_envs,
        "learning_starts": args.learning_starts,
        "policy_frequency": 2,
        "target_network_frequency": 1,
        "autotune": True,
        "alpha": 0.05,
        "tau": 0.005,
        "q_lr": 3e-4,
        "policy_lr": 3e-4,
        "gamma": 0.99,
        "num_loops": 3,
    }
    d_arg_vect = {"num_envs": args.num_envs, "rl_threads": args.rl_threads}
    params = {
        "tumor": {
            "correlation_length": 45,
            "threshold": 0.55,
            "number_cells": args.tumor,
        },
        "cell_1": {
            "correlation_length": 45,
            "threshold": 0.55,
            "number_cells": args.cell_1,
        },
    }
    d_arg_generation = {
        "params": params,
        "cell_2_fraction": [args.cell_2_fraction]
        if args.cell_2_fraction is not None
        else [0.0, 0.25, 0.5, 0.75, 1.0],
        "seed": d_arg_simulation["seed"],
    }
    d_arg = {
        "simulation": d_arg_simulation,
        "vectorization": d_arg_vect,
        "wandb": d_arg_wandb,
        "rl": d_arg_rl,
        "wrapper": d_arg_physigym_wrapper,
        "model": d_arg_physigym_model,
        "architecture": {"encoder": {"out_channels": 128}},
        "generation": d_arg_generation,
    }
    d_arg["model"]["output_dir"] = (
        f"data/{d_arg['simulation']['name']}_{d_arg['simulation']['seed']}_{d_arg['model']['observation_mode']}_{int(time.time())}"
    )

    run_async_sac(d_arg=d_arg)
