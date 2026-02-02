import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from network_field import create_csv
import os
import pandas as pd
import shutil


# ============================================================
# Wrapper: PhysiCellModelWrapper
# ============================================================
class PhysiCellModelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        list_variable_name: list[str] = ["drug_1"],
        w_cell=0.7,
        w_increase=0.2,
        w_amount=0.1,
        frequency_save_data=64,
    ):
        """
        Wraps a PhysiCell environment to use a flat continuous Box action space.
        Reward = weighted sum between drug penalty and cancer cell signal.
        """
        super().__init__(env)

        for variable_name in list_variable_name:
            if not isinstance(variable_name, str):
                raise ValueError(
                    f"Expected variable_name to be str, got {type(variable_name).__name__}"
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
        dtype = env.action_space[list_variable_name[0]].dtype

        self._action_space = Box(low=low, high=high, dtype=dtype)
        logits = np.array([w_cell, w_amount, w_increase])
        if np.sum(logits) == 1:
            self.w_cell, self.w_amount, self.w_increase = w_cell, w_amount, w_increase
        else:
            weights = np.exp(logits)
            weights /= np.sum(weights)
            self.w_cell, self.w_amount, self.w_increase = weights

        self.cell_positions_folder = (
            self.env.get_wrapper_attr("x_root")
            .xpath("//initial_conditions/cell_positions/folder")[0]
            .text
        )
        self.cell_name_file = (
            self.env.get_wrapper_attr("x_root")
            .xpath("//initial_conditions/cell_positions/filename")[0]
            .text
        )
        self.csv_path_init = os.path.join(
            self.cell_positions_folder, self.cell_name_file
        )
        self.generation_cfg = None
        self.output_dir = (
            self.env.get_wrapper_attr("x_root").xpath("//save/folder")[0].text
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.frequency_save_data = frequency_save_data
        self.list_data = []
        self.seed = int(
            self.env.get_wrapper_attr("x_root").xpath("//random_seed")[0].text
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_mode(self):
        return self.env.unwrapped.observation_mode

    def save_data(self):
        if self.frequency_save_data is not None and self.env.unwrapped.episode != -1:
            self.output_dir_episode = os.path.join(
                self.output_dir, f"episode{str(self.env.unwrapped.episode).zfill(8)}"
            )
            os.makedirs(self.output_dir_episode, exist_ok=True)
            episode = self.env.unwrapped.episode
            df = pd.DataFrame(self.list_data)
            if "mean_drugs" in df.columns:
                df["cumulative_mean_drugs"] = df["mean_drugs"].cumsum()
            df["cumulative_reward"] = df["reward"].cumsum()
            df.to_csv(os.path.join(self.output_dir_episode, "data.csv"), index=False)
            dst_path = os.path.join(
                self.output_dir_episode,
                os.path.basename(self.generation_cfg["csv_path"]),
            )
            shutil.copy(self.generation_cfg["csv_path"], dst_path)
            self.list_data = []
            self.output_dir_episode = os.path.join(
                self.output_dir, f"episode{str(episode).zfill(8)}"
            )
        else:
            None

    def step(self, action: np.ndarray):
        d_action = {
            variable_name: np.array([value])
            for variable_name, value in zip(self.list_variable_name, action)
        }

        obs, r_cancer_cells, terminated, truncated, info = self.env.step(d_action)

        drug_prev = self.info["prev_mean_drugs"]
        drug_t = np.mean(action)
        info["action"] = d_action
        drug_increase = max(0.0, drug_t - drug_prev)
        self.info["prev_mean_drugs"] = drug_t
        info["step_episode"] = self.env.unwrapped.step_episode

        reward = (
            self.w_cell * r_cancer_cells
            - self.w_amount * drug_t
            - self.w_increase * drug_increase
        )

        if self.frequency_save_data is not None:
            data = {
                "step": self.env.unwrapped.step_episode,
                "reward": reward,
                "mean_drugs": drug_t,
                "number_tumor": info["number_tumor"],
                "number_cell_1": info["number_cell_1"],
                "number_cell_2": info["number_cell_2"],
            }

            self.list_data.append(data)

        return obs, reward, terminated, truncated, info

    def process_update_generation_cfg(self, generation_cfg=None):
        if self.generation_cfg is not None:
            self.generation_cfg["seed"] += self.env.unwrapped.episode
            create_csv(**self.generation_cfg)
        else:
            if generation_cfg is not None and self.generation_cfg is None:
                self.generation_cfg = generation_cfg.copy()

                # complete spatial bounds from env
                self.generation_cfg["x_min"] = self.env.unwrapped.x_min * 0.9
                self.generation_cfg["y_min"] = self.env.unwrapped.y_min * 0.9
                self.generation_cfg["x_max"] = self.env.unwrapped.x_max * 0.9
                self.generation_cfg["y_max"] = self.env.unwrapped.y_max * 0.9

                # ensure seed exists
                self.generation_cfg.setdefault("seed", self.seed)
        self.generation_cfg["csv_path"] = self.csv_path_init

    def reset(self, seed=None, options=None, generation_cfg=None, **kwargs):
        if options is None:
            options = {}

        if seed is not None:
            self.seed = seed

        self.process_update_generation_cfg(generation_cfg)

        self.save_data()

        self.info = {"prev_mean_drugs": 0}

        # ---- IMPORTANT: forward seed, do not invent one ----
        return self.env.reset(seed=seed, options=options)
