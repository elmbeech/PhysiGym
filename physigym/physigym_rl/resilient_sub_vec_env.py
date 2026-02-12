import multiprocessing as mp
import numpy as np
from typing import Set

from stable_baselines3.common.vec_env.subproc_vec_env import (
    SubprocVecEnv,
    _stack_obs,
)


class ResilientSubprocVecEnv(SubprocVecEnv):
    """
    SubprocVecEnv variant that permanently disables crashing environments
    instead of restarting them (PhysiCell-safe).
    """

    def __init__(self, env_fns, start_method="spawn"):
        assert start_method == "spawn", "PhysiCell requires spawn"

        self.env_fns = env_fns
        self.dead_envs: Set[int] = set()

        super().__init__(env_fns, start_method=start_method)

        # Make mutable
        self.remotes = list(self.remotes)
        self.processes = list(self.processes)

    # ------------------------------------------------------------------
    # Crash handling
    # ------------------------------------------------------------------

    def _disable_env(self, i: int):
        if i in self.dead_envs:
            return

        print(f"[ResilientVecEnv] Disabling env {i}")

        self.dead_envs.add(i)

        try:
            if self.processes[i].is_alive():
                self.processes[i].terminate()
        except Exception:
            pass

        try:
            self.remotes[i].close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_async(self, actions):
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            if i in self.dead_envs:
                continue
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = []

        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                obs = self.observation_space.sample()
                reward = 0.0
                done = True

                info = {
                    "crashed": True,
                    "disabled": True,
                    "terminal_observation": obs,
                    "step_episode": -1,
                }

                results.append((obs, reward, done, info, info))
                continue

            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError):
                self._disable_env(i)

                obs = self.observation_space.sample()
                reward = 0.0
                done = True

                info = {
                    "crashed": True,
                    "disabled": True,
                    "step_episode": -1,
                }

                results.append((obs, reward, done, info, info))

        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)

        return (
            _stack_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                continue
            remote.send(("reset", (self._seeds[i], self._options[i])))

        results = []
        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                obs = self.observation_space.sample()
                reset_info = {
                    "crashed": True,
                    "disabled": True,
                    "step_episode": -1,
                }
                results.append((obs, reset_info))
                continue

            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError):
                self._disable_env(i)

                obs = self.observation_space.sample()
                reset_info = {
                    "crashed": True,
                    "disabled": True,
                    "step_episode": -1,
                }
                results.append((obs, reset_info))

        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()

        return _stack_obs(obs, self.observation_space)
