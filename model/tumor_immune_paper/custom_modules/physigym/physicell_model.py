#####
# title: physigym/envs/physicell_model.py
#
# language: python3
# library: gymnasium v1.0.0a1
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     model specific implementation of the custom_modules/extending module
#     comaptible Gymnasium environment.
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
#####


# library
from extending import physicell
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from physigym.envs.physicell_core import CorePhysiCellEnv
import skimage as ski
from tysserand import tysserand as ty
from sklearn.cluster import KMeans

FIBO = np.array([1, 2, 3, 5, 7, 13, 21, 34, 55])
LENGTH_FIBO = len(FIBO)


# ---- compute number of clusters ----
def _compute_fibo(total_cells: int) -> int:
    idx = min(LENGTH_FIBO - 1, int(np.log(total_cells)))
    return int(FIBO[idx])


# function
class ModelPhysiCellEnv(CorePhysiCellEnv):
    """
    input:
        physigym.CorePhysiCellEnv

    output:
        physigym.ModelPhysiCellEnv

    run:
        import gymnasium
        import physigym

        env = gymnasium.make("physigym/ModelPhysiCellEnv")

        o_observation, info = env.reset()
        o_observation, r_reward, b_terminated, b_truncated, info = env.step(action={})
        env.close()

    description:
        this is the model physigym environment class, built on top of the
        physigym.CorePhysiCellEnv class, which is built on top of the
        gymnasium.Env class.

        fresh from the PhysiGym repo this is only a template class!
        you will have to edit this class, to specify the model specific
        reinforcement learning environment.
    """

    def __init__(
        self,
        settingxml="config/PhysiCell_settings.xml",
        cell_type_cmap="turbo",
        figsize=(6, 6),  # inch
        render_mode=None,
        render_fps=10,
        verbose=True,
        # **kwargs
        observation_mode="mean_scalars_cells_substrates",
        img_rgb_grid_size_y=64,  # pixel
        img_rgb_grid_size_x=64,  # pixel
        img_mc_grid_size_x=64,  # pixel
        img_mc_grid_size_y=64,  # pixel
        normalization_factor=512,
    ):
        self.observation_mode = observation_mode
        if "_scalars" in observation_mode:
            name = observation_mode.split("_scalars")[0]
            self.observation_mode = observation_mode[
                observation_mode.index("scalars") :
            ]
            self.reducers = {
                "mean": np.mean,
                "median": np.median,
                "min": np.min,
                "max": np.max,
                "std": np.std,
                "sum": np.sum,
                "p10": lambda x: np.percentile(x, 10),
                "p90": lambda x: np.percentile(x, 90),
                "iqr": lambda x: np.percentile(x, 75) - np.percentile(x, 25),
                "mad": lambda x: np.median(np.abs(x - np.median(x))),
            }
            if name not in self.reducers:
                raise ValueError(f"Error: unknown reducers: {name}")
        if "img" in observation_mode:
            self.observation_mode = (
                observation_mode + f"_{img_mc_grid_size_x}_{img_mc_grid_size_y}"
            )

        if self.observation_mode not in [
            "scalars_cells",
            "scalars_substrates",
            "scalars_cells_substrates",
            f"img_mc_cells_{img_mc_grid_size_x}_{img_mc_grid_size_y}",
            f"img_mc_substrates_{img_mc_grid_size_x}_{img_mc_grid_size_y}",
            f"img_mc_cells_substrates_{img_mc_grid_size_x}_{img_mc_grid_size_y}",
            "graph_delaunay",
            "graph_knn",
            "transformer_nodes",
            "transformer_nodes_2",
        ]:
            raise ValueError(
                f"Error: unknown observation type: {self.observation_mode}"
            )

        self.max_nodes = 2000  #  choose based on your env
        self.max_edges = 7500  #  number of Delaunay edges worst case
        self.node_dim = 1
        self.edge_dim = 1
        self.k = 3  # number of connections k (knn)
        self.max_clusters = FIBO[-1]
        self.features = 9
        self.features_2 = 16

        # call super class init
        super().__init__(
            settingxml=settingxml,
            cell_type_cmap=cell_type_cmap,
            figsize=figsize,
            render_mode=render_mode,
            render_fps=render_fps,
            verbose=verbose,
            # **kwargs
            observation_mode=observation_mode,
            img_rgb_grid_size_x=img_rgb_grid_size_x,
            img_rgb_grid_size_y=img_rgb_grid_size_y,
            img_mc_grid_size_x=img_mc_grid_size_x,
            img_mc_grid_size_y=img_mc_grid_size_y,
            normalization_factor=normalization_factor,
        )
        self.lambda_dt = float(
            self.x_root.xpath("//user_parameters/growth_rate")[0].text
        ) * float(self.x_root.xpath("//user_parameters/dt_gym")[0].text)

    def get_action_space(self):
        """
        input:

        output:
            d_action_space: dictionary composition space
                the dictionary keys have to match the parameter,
                custom variable, or custom vector label.
                the value has to be defined as gymnasium space object.
                + https://gymnasium.farama.org/main/api/spaces/
        run:
            internal function, user defined.

        description:
            dictionary structure built out of gymnasium.spaces elements.
            this struct has to specify type and range for each
            action parameter, action custom variable, and action custom vector.
        """

        # model dependent action_space processing logic goes here!
        d_action_space = spaces.Dict(
            {
                "drug_1": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        # output
        return d_action_space

    def get_observation_space(self):
        """
        input:

        output:
            o_observation_space structure.
                the struct have to be built out of self.kwargs['img_mc_grid_size_y']mnasium.spaces elements.
                there are no other limits.
                + https://self.kwargs['img_mc_grid_size_y']mnasium.farama.org/main/api/spaces/

        run:
            internal function, user defined.

        description:
            data structure built out of self.kwargs['img_mc_grid_size_y']mnasium.spaces elements.
            this struct has to specify type and range
            for each observed variable.
        """
        observation_mode = self.observation_mode
        self.kwargs["img_mc_grid_size_x"] = self.kwargs["img_mc_grid_size_x"]
        self.kwargs["img_mc_grid_size_y"] = self.kwargs["img_mc_grid_size_y"]
        # model dependent observation_space processing logic goes here!
        if self.observation_mode == "scalars_cells":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count,),
                dtype=np.float32,
            )

        elif self.observation_mode == "scalars_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.substrate_count,),
                dtype=np.float32,
            )

        elif self.observation_mode in "scalars_cells_substrates":
            o_observation_space = spaces.Box(
                low=-(2**8),
                high=2**8,
                shape=(self.cell_type_count + self.substrate_count,),
                dtype=np.float32,
            )

        elif observation_mode in [
            f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
            f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}",
        ]:
            # Define the Box space for the multichannel image
            self.ratio_img_mc_size_y = self.height / self.kwargs["img_mc_grid_size_y"]
            self.ratio_img_mc_size_x = self.width / self.kwargs["img_mc_grid_size_x"]
            if (
                observation_mode
                == f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
            ):
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count,
                        self.kwargs["img_mc_grid_size_x"],
                        self.kwargs["img_mc_grid_size_y"],
                    ),
                    dtype=np.uint8,
                )
            elif (
                observation_mode
                == f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
            ):
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.substrate_count,
                        self.kwargs["img_mc_grid_size_x"],
                        self.kwargs["img_mc_grid_size_y"],
                    ),
                    dtype=np.uint8,
                )
            else:
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count + self.substrate_count,
                        self.kwargs["img_mc_grid_size_x"],
                        self.kwargs["img_mc_grid_size_y"],
                    ),
                    dtype=np.uint8,
                )
        elif observation_mode in ["graph_delaunay", "graph_knn"]:
            o_observation_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_nodes, self.node_dim),
                        dtype=np.float32,
                    ),
                    "edge_index": spaces.Box(
                        low=0,
                        high=self.max_nodes,
                        shape=(2, self.max_edges),
                        dtype=np.int32,
                    ),
                    "edge_attr": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_edges, self.edge_dim),
                        dtype=np.float32,
                    ),
                    "node_mask": spaces.Box(
                        low=0, high=1, shape=(self.max_nodes,), dtype=np.float32
                    ),
                    "edge_mask": spaces.Box(
                        low=0, high=1, shape=(self.max_edges,), dtype=np.float32
                    ),
                }
            )
        elif observation_mode == "transformer_nodes":
            o_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.max_clusters, self.features),
                dtype=np.float32,
            )
        elif observation_mode == "transformer_nodes_2":
            o_observation_space = spaces.Box(
                low=-(2**2),
                high=2**2,
                shape=(self.max_clusters, self.features_2),
                dtype=np.float32,
            )

        else:
            raise ValueError(
                f"unknown observation type: {self.kwargs['observation_mode']}"
            )

        # output
        return o_observation_space

    def get_cells_scalars(self):
        a_norm_cell_count = np.zeros((self.cell_type_count,), dtype=np.float32)
        for s_cell_type, i_id in self.cell_type_to_id.items():
            a_norm_cell_count[i_id] = (
                self.df_alive.loc[
                    (self.df_alive.type == s_cell_type),
                    :,
                ].shape[0]
                / self.kwargs["normalization_factor"]
                - 1
            )
        return a_norm_cell_count

    def get_substrates_scalars(self):
        a_substrate = np.zeros(self.substrate_count, dtype=np.float32)

        for i, s_subs in enumerate(self.substrate_unique):
            microenv = np.asarray(physicell.get_microenv(s_subs))
            values = microenv[:, -1]  # substrate column
            a_substrate[i] = self.reducers(values)

        return a_substrate

    def get_img_cells(self):
        # get cell_type indices
        cell_type_indices = self.df_alive["type"].map(self.cell_type_to_id).to_numpy()

        # discretize
        x_bin = (
            (self.df_alive["x"] - self.x_min)
            / (self.x_max - self.x_min)
            * (self.kwargs["img_mc_grid_size_x"] - 1)
        ).astype(int)
        y_bin = (
            (self.df_alive["y"] - self.y_min)
            / (self.y_max - self.y_min)
            * (self.kwargs["img_mc_grid_size_y"] - 1)
        ).astype(int)

        # clip in case of rounding issues
        x_bin = np.clip(x_bin, 0, self.kwargs["img_mc_grid_size_x"] - 1)
        y_bin = np.clip(y_bin, 0, self.kwargs["img_mc_grid_size_y"] - 1)

        # get numpy array
        image = np.zeros(
            shape=(
                self.cell_type_count,
                self.kwargs["img_mc_grid_size_x"],
                self.kwargs["img_mc_grid_size_y"],
            ),
            dtype=np.float32,
        )
        np.add.at(
            image,
            (cell_type_indices, x_bin, y_bin),
            1,
        )

        return ski.util.img_as_ubyte(
            image / (self.ratio_img_mc_size_x * self.ratio_img_mc_size_y)
        )

    def get_img_substrates(self):
        self.df_subs = None
        for s_subs in self.substrate_unique:
            df_subs = pd.DataFrame(
                physicell.get_microenv(s_subs), columns=["x", "y", "z", s_subs]
            )
            if self.df_subs is None:
                self.df_subs = df_subs
            else:
                self.df_subs = pd.merge(self.df_subs, df_subs, on=["x", "y", "z"])
        # discretize
        self.df_subs["x_bin"] = (
            (
                (self.df_subs["x"] - self.x_min)
                / (self.x_max - self.x_min)
                * (self.kwargs["img_mc_grid_size_x"] - 1)
            )
            .astype(int)
            .clip(0, self.kwargs["img_mc_grid_size_x"] - 1)
        )
        self.df_subs["y_bin"] = (
            (
                (self.df_subs["y"] - self.y_min)
                / (self.y_max - self.y_min)
                * (self.kwargs["img_mc_grid_size_y"] - 1)
            )
            .astype(int)
            .clip(0, self.kwargs["img_mc_grid_size_y"] - 1)
        )

        grouped = self.df_subs.groupby(["x_bin", "y_bin"])[self.substrate_unique].max()

        # initialize image
        image = np.zeros(
            (
                len(self.substrate_unique),
                self.kwargs["img_mc_grid_size_x"],
                self.kwargs["img_mc_grid_size_y"],
            ),
            dtype=np.float32,
        )

        # fill image
        for i, subs in enumerate(self.substrate_unique):
            for (x_bin, y_bin), value in grouped[subs].items():
                image[i, x_bin, y_bin] = value
        min_vals = image.min(axis=(1, 2), keepdims=True)
        max_vals = image.max(axis=(1, 2), keepdims=True)
        scales = np.where((max_vals - min_vals) > 0, max_vals - min_vals, 1)
        return ski.util.img_as_ubyte(((image - min_vals) / scales))

    def get_observation(self):
        """
        input:

        output:
            o_observation: object compatible with the defined
                observation space struct.

        run:
            internal function, user defined.

        description:
            data for the observation object for example be retrieved by:
            + physicell.get_parameter("my_parameter")
            + physicell.get_variable("my_variable")
            + physicell.get_vector("my_vector")
            however, there are no limits.
        """
        # model dependent observation processing logic goes here!

        # get cell data frame
        self.df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "type"]
        )
        self.df_alive = self.df_cell[self.df_cell["dead"] < 0.1]

        # update tumor cell count
        self.c_prev = self.c_t
        self.c_t = self.df_alive.loc[(self.df_alive.type == "tumor"), :].shape[0]
        if self.c_prev is None:
            self.c_prev = self.c_t
        self.nb_tumor = self.c_t

        # update cell_1 cell count
        self.nb_cell_1 = self.df_alive.loc[(self.df_alive.type == "cell_1"), :].shape[0]

        # update cell_2 cell count
        self.nb_cell_2 = self.df_alive.loc[(self.df_alive.type == "cell_2"), :].shape[0]

        # observe the environemnt
        if self.observation_mode == "scalars_cells":
            o_observation = self.get_cells_scalars()
        elif self.observation_mode == "scalars_substrates":
            o_observation = self.get_substrates_scalars()
        elif self.observation_mode == "scalars_cells_substrates":
            o_observation = np.concatenate(
                [self.get_cells_scalars(), self.get_substrates_scalars()]
            )
        elif (
            self.observation_mode
            == f"img_mc_cells_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = self.get_img_cells()
        elif (
            self.observation_mode
            == f"img_mc_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = self.get_img_substrates()
        elif (
            self.observation_mode
            == f"img_mc_cells_substrates_{self.kwargs['img_mc_grid_size_x']}_{self.kwargs['img_mc_grid_size_y']}"
        ):
            o_observation = np.concatenate(
                [self.get_img_cells(), self.get_img_substrates()]
            )

        elif self.observation_mode in ["graph_delaunay", "graph_knn"]:
            self.df_alive.set_index("ID", inplace=True)
            coords = self.df_alive[["x", "y"]].values

            # Raw graph (variable size)
            pairs = (
                ty.build_delaunay(coords)
                if self.observation_mode == "graph_delaunay"
                else ty.build_knn(coords, k=self.k)
            )  # shape = (E, 2)
            distances = ty.distance_neighbors(coords, pairs)  # shape = (E,)

            # Raw node features
            node_features = (
                self.df_alive["type"]
                .map(self.cell_type_to_id)
                .to_numpy(dtype=np.float32)
                / self.cell_type_count
            )[:, None]  # shape = (N, 1)

            # Raw edge attributes
            edge_attr = (distances / max(self.width, self.height, self.depth)).astype(
                np.float32
            )
            edge_attr = edge_attr[:, None]  # shape = (E, 1)

            N = node_features.shape[0]
            E = pairs.shape[0]

            # --- Pad nodes ---
            padded_nodes = np.zeros((self.max_nodes, self.node_dim), dtype=np.float32)
            padded_nodes[:N] = node_features

            node_mask = np.zeros(self.max_nodes, dtype=np.float32)
            node_mask[:N] = 1.0

            # --- Pad edges ---
            padded_edge_index = np.zeros((2, self.max_edges), dtype=np.int32)
            padded_edge_index[:, :E] = pairs.T

            padded_edge_attr = np.zeros(
                (self.max_edges, self.edge_dim), dtype=np.float32
            )
            padded_edge_attr[:E] = edge_attr

            edge_mask = np.zeros(self.max_edges, dtype=np.float32)
            edge_mask[:E] = 1.0
            o_observation = {
                "node_features": padded_nodes,
                "edge_index": padded_edge_index,
                "edge_attr": padded_edge_attr,
                "node_mask": node_mask,
                "edge_mask": edge_mask,
            }
        elif self.observation_mode == "transformer_nodes":
            df = self.df_alive.set_index("ID", drop=True)

            n_clusters = _compute_fibo(len(df))
            o_observation = np.zeros(
                (self.max_clusters, self.features), dtype=np.float32
            )
            df["x"] = (df["x"] - self.x_min) / (self.x_max - self.x_min)
            df["y"] = (df["y"] - self.y_min) / (self.y_max - self.y_min)
            # ---- clustering ----
            coords = df[["x", "y"]].to_numpy(np.float32)

            kmeans = KMeans(
                n_clusters=n_clusters,
                algorithm="elkan",
                n_init=1,
                random_state=42,
            )
            labels = kmeans.fit_predict(coords)

            # ---- encode types as integers ----
            type_to_idx = {t: i for i, t in enumerate(self.cell_type_unique)}
            type_idx = df["type"].map(type_to_idx).to_numpy()
            n_types = len(self.cell_type_unique)
            total_len = len(df)

            counts = np.zeros((n_clusters, n_types), dtype=np.float32)

            np.add.at(counts, (labels, type_idx), 1)

            cluster_sizes = counts.sum(axis=1, keepdims=True)
            cluster_sizes_flat = cluster_sizes[:, 0]
            non_empty = cluster_sizes_flat > 0

            type_props = np.divide(
                counts,
                cluster_sizes,
                out=np.zeros_like(counts),
                where=cluster_sizes > 0,
            )

            # ============================================================
            # 2️⃣ SPATIAL STATS → shape (K, 5)
            # ============================================================
            x = coords[:, 0]
            y = coords[:, 1]

            x_sum = np.bincount(labels, weights=x, minlength=n_clusters)
            y_sum = np.bincount(labels, weights=y, minlength=n_clusters)

            x2_sum = np.bincount(labels, weights=x * x, minlength=n_clusters)
            y2_sum = np.bincount(labels, weights=y * y, minlength=n_clusters)

            x_mean = np.zeros(n_clusters, dtype=np.float32)
            y_mean = np.zeros(n_clusters, dtype=np.float32)
            x_std = np.zeros(n_clusters, dtype=np.float32)
            y_std = np.zeros(n_clusters, dtype=np.float32)

            # Means
            x_mean[non_empty] = x_sum[non_empty] / cluster_sizes_flat[non_empty]
            y_mean[non_empty] = y_sum[non_empty] / cluster_sizes_flat[non_empty]

            # Variances (numerically stable)
            x_var = np.zeros(n_clusters, dtype=np.float32)
            y_var = np.zeros(n_clusters, dtype=np.float32)

            x_var[non_empty] = (
                x2_sum[non_empty] / cluster_sizes_flat[non_empty]
                - x_mean[non_empty] ** 2
            )
            y_var[non_empty] = (
                y2_sum[non_empty] / cluster_sizes_flat[non_empty]
                - y_mean[non_empty] ** 2
            )

            # Clamp to avoid negative epsilonself
            x_var = np.maximum(x_var, 0.0)
            y_var = np.maximum(y_var, 0.0)

            x_std = np.sqrt(x_var)
            y_std = np.sqrt(y_var)

            cluster_frac = cluster_sizes[:, 0] / total_len

            stats = np.stack(
                [x_mean, y_mean, x_std, y_std, cluster_frac],
                axis=1,
            )

            # ============================================================
            # 3️⃣ FINAL OBSERVATION → (K, 8)
            # ============================================================
            data = np.concatenate([stats, type_props], axis=1).astype(np.float32)

            K = data.shape[0]

            o_observation[:K, :] = data

        elif self.observation_mode == "transformer_nodes_2":
            df = self.df_alive.set_index("ID", drop=True)

            n_clusters = _compute_fibo(len(df))
            o_observation = np.zeros(
                (self.max_clusters, self.features_2), dtype=np.float32
            )

            # Normalize coordinates
            df["x"] = (df["x"] - self.x_min) / (self.x_max - self.x_min)
            df["y"] = (df["y"] - self.y_min) / (self.y_max - self.y_min)

            # ---- clustering ----
            coords = df[["x", "y"]].to_numpy(np.float32)
            kmeans = KMeans(
                n_clusters=n_clusters,
                algorithm="elkan",
                n_init=1,
                random_state=42,
            )
            labels = kmeans.fit_predict(coords)

            # ---- encode types as integers ----
            type_to_idx = {t: i for i, t in enumerate(self.cell_type_unique)}
            type_idx = df["type"].map(type_to_idx).to_numpy()
            n_types = len(self.cell_type_unique)
            total_len = len(df)

            counts = np.zeros((n_clusters, n_types), dtype=np.float32)
            np.add.at(counts, (labels, type_idx), 1)

            cluster_sizes = counts.sum(axis=1, keepdims=True)
            cluster_sizes_flat = cluster_sizes[:, 0]
            non_empty = cluster_sizes_flat > 0

            type_props = np.divide(
                counts,
                cluster_sizes,
                out=np.zeros_like(counts),
                where=cluster_sizes > 0,
            )

            # ============================================================
            # 2️⃣ SPATIAL STATS → shape (K, new_dim)
            # ============================================================
            x = coords[:, 0]
            y = coords[:, 1]

            # Basic stats
            x_sum = np.bincount(labels, weights=x, minlength=n_clusters)
            y_sum = np.bincount(labels, weights=y, minlength=n_clusters)
            x2_sum = np.bincount(labels, weights=x * x, minlength=n_clusters)
            y2_sum = np.bincount(labels, weights=y * y, minlength=n_clusters)
            x_min = np.zeros(n_clusters, dtype=np.float32)
            x_max = np.zeros(n_clusters, dtype=np.float32)
            y_min = np.zeros(n_clusters, dtype=np.float32)
            y_max = np.zeros(n_clusters, dtype=np.float32)

            for i in range(n_clusters):
                if cluster_sizes_flat[i] > 0:
                    mask = labels == i
                    x_min[i], x_max[i] = x[mask].min(), x[mask].max()
                    y_min[i], y_max[i] = y[mask].min(), y[mask].max()

            x_mean = np.zeros(n_clusters, dtype=np.float32)
            y_mean = np.zeros(n_clusters, dtype=np.float32)
            x_std = np.zeros(n_clusters, dtype=np.float32)
            y_std = np.zeros(n_clusters, dtype=np.float32)

            # Means
            x_mean[non_empty] = x_sum[non_empty] / cluster_sizes_flat[non_empty]
            y_mean[non_empty] = y_sum[non_empty] / cluster_sizes_flat[non_empty]

            # Variances (numerically stable)
            x_var = np.zeros(n_clusters, dtype=np.float32)
            y_var = np.zeros(n_clusters, dtype=np.float32)
            x_var[non_empty] = (
                x2_sum[non_empty] / cluster_sizes_flat[non_empty]
                - x_mean[non_empty] ** 2
            )
            y_var[non_empty] = (
                y2_sum[non_empty] / cluster_sizes_flat[non_empty]
                - y_mean[non_empty] ** 2
            )
            x_var = np.maximum(x_var, 0.0)
            y_var = np.maximum(y_var, 0.0)
            x_std = np.sqrt(x_var)
            y_std = np.sqrt(y_var)

            # Derived features
            x_range = x_max - x_min
            y_range = y_max - y_min
            cluster_frac = cluster_sizes[:, 0] / total_len

            # Type entropy
            entropy = -np.sum(type_props * np.log(type_props + 1e-6), axis=1)

            # ============================================================
            # 3️⃣ FINAL OBSERVATION → enriched token features
            # ============================================================
            # Stack features: [mean, std, min, max, range, cluster_frac, entropy, type_props]
            extra_features = np.stack(
                [
                    x_mean,
                    y_mean,
                    x_std,
                    y_std,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    x_range,
                    y_range,
                    cluster_frac,
                    entropy,
                ],
                axis=1,
            )
            data = np.concatenate([extra_features, type_props], axis=1).astype(
                np.float32
            )

            K = data.shape[0]
            o_observation[:K, : data.shape[1]] = data

        else:
            raise ValueError(
                f"unknown observation type: {self.kwargs['observation_mode']}"
            )

        # output
        return o_observation

    def get_info(self):
        """
        input:

        output:
            info: dictionary

        run:
            internal function, user defined.

        description:
            function to provide additional information important for
            controlling the action of the policy. for example,
            if we do reinforcement learning on a jump and run game,
            the number of hearts (lives left) from our character.
        """
        # model dependent info processing logic goes here!
        info = {
            "df_cell": self.df_cell,
            "number_tumor": self.nb_tumor,
            "number_cell_1": self.nb_cell_1,
            "number_cell_2": self.nb_cell_2,
        }

        # output
        return info

    def get_terminated(self):
        """
        input:

        output:
            b_terminated: bool

        run:
            internal function, user defined.

        description:
            function to determine if the episode is terminated.
            for example, if we do reinforcement learning on a
            jump and run game, if our character died.
            please notice, that this ending is different form
            truncated (the episode reached the max time limit).
        """
        # model dependent terminated processing logic goes here!
        return True if (self.c_t == 0) else False  # or (self.c_t > 1536)

    def get_reset_values(self):
        """
        input:

        output:

        run:
            internal function, user defined.

        description:
            function to reset model specific self.variables. e.g.:
            self.my_variable = None
        """
        self.c_t = None
        self.c_prev = None

    def get_reward(self):
        """
        input:

        output:
            r_reward: float between or equal to 0.0 and 1.0.
                there are no other limits to the algorithm implementation enforced.
                however, the algorithm is usually based on data retrieved
                by the get_observation function (o_observation, info),
                and possibly by the render function (a_img).

        run:
            internal function, user defined.

        description:
            cost function.
        """

        expected_growth = self.c_prev * (np.exp(self.lambda_dt) - 1.0)
        expected_growth = max(expected_growth, 1e-8)

        r_tumor = (self.c_prev - self.c_t) / expected_growth
        return np.clip(r_tumor, -1, 1)

    def get_img(self):
        """
        input:

        output:
            self.fig.savefig
                instance attached matplotlib figure.

        run:
            internal function, user defined.

        description:
            template code to generate a matplotlib figure from the data.
            for example from:
            + physicell.get_microenv("my_substrate")
            + physicell.get_cell()
            + physicell.get_variable("my_variable")
            however, there are no limits.
        """
        # model dependent img processing logic goes here!
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("equal")
        ax.axis("off")

        ##################
        # substrate data #
        ##################

        # debris
        df_conc = pd.DataFrame(
            physicell.get_microenv("debris"), columns=["x", "y", "z", "debris"]
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
        df_mesh = df_conc.pivot(index="y", columns="x", values="debris")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Reds",
            alpha=1 / 3,
        )

        # pro-tumoral factor
        df_conc = pd.DataFrame(
            physicell.get_microenv("pro-tumoral factor"),
            columns=["x", "y", "z", "pro-tumoral factor"],
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
        df_mesh = df_conc.pivot(index="y", columns="x", values="pro-tumoral factor")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            alpha=1 / 3,
        )

        # anti-tumoral factor
        df_conc = pd.DataFrame(
            physicell.get_microenv("anti-tumoral factor"),
            columns=["x", "y", "z", "anti-tumoral factor"],
        )
        df_conc = df_conc.loc[df_conc.z == 0.0, :]
        df_mesh = df_conc.pivot(index="y", columns="x", values="anti-tumoral factor")
        ax.contourf(
            df_mesh.columns,
            df_mesh.index,
            df_mesh.values,
            vmin=0.0,
            vmax=1.0,
            cmap="Greens",
            alpha=1 / 3,
        )

        ######################
        # substrate colorbar #
        ######################

        # self.fig.colorbar(
        #    mappable=cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="Reds"),
        #    label="my_substrate",
        #    ax=ax,
        # )

        #############
        # cell data #
        #############

        df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "cell_type"]
        )
        df_cell = df_cell.loc[(df_cell.dead < 0.1), :]
        df_cell["color"] = None
        for s_cell_type, s_color in self.cell_type_to_color.items():
            df_cell.loc[(df_cell.cell_type == s_cell_type), "color"] = s_color
        # df_variable = pd.DataFrame(physicell.get_variable("my_variable"), columns=["my_variable"])
        # df_cell = pd.merge(df_cell, df_variable, left_index=True, right_index=True, how="left")
        df_cell = df_cell.loc[df_cell.z == 0.0, :]
        df_cell.plot(
            kind="scatter",
            x="x",
            y="y",
            c="color",
            xlim=[self.x_min, self.x_max],
            ylim=[self.y_min, self.y_max],
            #    vmin=0.0, vmax=1.0, cmap="viridis",
            #    grid=True,
            #    title=f"dt_self.kwargs['img_mc_grid_size_y']m env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} [cell]",
            ax=ax,
        )

        ################
        # save to file #
        ################

        plt.tight_layout()
        s_path = self.x_root.xpath("//save/folder")[0].text + "/render_mode_human/"
        os.makedirs(s_path, exist_ok=True)
        self.fig.savefig(
            f"{s_path}timeseries_step{str(self.step_env).zfill(3)}.jpeg",
            facecolor="white",
        )
