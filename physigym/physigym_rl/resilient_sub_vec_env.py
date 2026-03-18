import multiprocessing as mp
import numpy as np
from typing import Set
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _stack_obs
from gymnasium import spaces


# ------------------------------------------------------------------
# Dummy env used to replace crashed ones safely
# ------------------------------------------------------------------
class DummyEnv:
    """Safe placeholder for crashed environments."""

    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.action_space = None

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, True, {"crashed": True, "disabled": True, "step_episode": -1}

    def reset(self, seed=None, options=None):
        obs = self.observation_space.sample()
        return obs, {"crashed": True, "disabled": True, "step_episode": -1}


# ------------------------------------------------------------------
# Resilient VecEnv
# ------------------------------------------------------------------
class ResilientSubprocVecEnv(SubprocVecEnv):
    """
    SubprocVecEnv variant that permanently disables crashing environments
    instead of restarting them (PhysiCell-safe).
    """

    def __init__(self, env_fns, start_method="spawn"):
        assert start_method == "spawn", "PhysiCell requires spawn"

        self.env_fns = env_fns
        self.dead_envs: Set[int] = set()
        self._dummy_envs = {}

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

        # Create dummy env for safe stepping
        self._dummy_envs[i] = DummyEnv(self.env_fns[i]().observation_space)

    # ------------------------------------------------------------------
    # Safe set_attr: skips dead envs
    # ------------------------------------------------------------------
    def set_attr(self, attr_name, value, indices=None):
        indices = indices if indices is not None else range(self.num_envs)
        for i in indices:
            if i in self.dead_envs:
                continue
            super().set_attr(attr_name, value, [i])

    # ------------------------------------------------------------------
    # Safe env_method: skips dead envs
    # ------------------------------------------------------------------
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = indices if indices is not None else range(self.num_envs)
        safe_indices = [i for i in indices if i not in self.dead_envs]
        if not safe_indices:
            return []
        return super().env_method(
            method_name, *method_args, indices=safe_indices, **method_kwargs
        )

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
                obs, reward, done, info = self._dummy_envs[i].step(None)
                results.append((obs, reward, done, info, info))
                continue

            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError, OSError):
                self._disable_env(i)
                obs, reward, done, info = self._dummy_envs[i].step(None)
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
            try:
                remote.send(("reset", (self._seeds[i], self._options[i])))
            except (OSError, EOFError, BrokenPipeError):
                self._disable_env(i)

        results = []
        for i, remote in enumerate(self.remotes):
            if i in self.dead_envs:
                obs, reset_info = self._dummy_envs[i].reset()
                results.append((obs, reset_info))
                continue

            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError, OSError):
                self._disable_env(i)
                obs, reset_info = self._dummy_envs[i].reset()
                results.append((obs, reset_info))

        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()

        return _stack_obs(obs, self.observation_space)
