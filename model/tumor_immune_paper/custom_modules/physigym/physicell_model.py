#####
# title: pysigym/envs/physicell_model.py
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
        observation_mode="scalars_cells_substrates",
        img_rgb_grid_size_y=64,  # pixel
        img_rgb_grid_size_x=64,  # pixel
        img_mc_grid_size_x=64,  # pixel
        img_mc_grid_size_y=64,  # pixel
        normalization_factor=512,
    ):
        # check observation mode
        if observation_mode not in [
            "scalars_cells",
            "scalars_substrates",
            "scalars_cells_substrates",
            "img_mc_cells",
            "img_mc_substrates",
            "img_mc_cells_substrates",
            "graph_delaunay",
            "graph_knn",
        ]:
            raise ValueError(f"Error: unknown observation type: {observation_mode}")

        self.observation_mode = observation_mode
        self.max_nodes = 2000  #  choose based on your env
        self.max_edges = 7500  #  number of Delaunay edges worst case
        self.node_dim = 1
        self.edge_dim = 1
        self.k = 3  # number of connections k (knn)
        self.max_clusters = 144
        self.features = 8
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
                the struct have to be built out of gymnasium.spaces elements.
                there are no other limits.
                + https://gymnasium.farama.org/main/api/spaces/

        run:
            internal function, user defined.

        description:
            data structure built out of gymnasium.spaces elements.
            this struct has to specify type and range
            for each observed variable.
        """
        observation_mode = self.observation_mode
        gy = self.kwargs["img_mc_grid_size_y"]
        gx = self.kwargs["img_mc_grid_size_x"]
        # model dependent observation_space processing logic goes here!
        if observation_mode == "scalars_cells":
            o_observation_space = spaces.Box(
                low=0,
                high=2**12,
                shape=(self.cell_type_count,),
                dtype=np.float32,
            )

        elif observation_mode == "scalars_substrates":
            o_observation_space = spaces.Box(
                low=0,
                high=2**12,
                shape=(self.substrate_count,),
                dtype=np.float32,
            )

        elif observation_mode == "scalars_cells_substrates":
            o_observation_space = spaces.Box(
                low=0,
                high=2**12,
                shape=(self.cell_type_count + self.substrate_count,),
                dtype=np.float32,
            )

        elif observation_mode in [
            "img_mc_cells",
            "img_mc_substrates",
            "img_mc_cells_substrates",
        ]:
            # Define the Box space for the multichannel image
            self.ratio_img_mc_size_y = self.height / gy
            self.ratio_img_mc_size_x = self.width / gx
            if observation_mode == "img_mc_cells":
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count,
                        gx,
                        gy,
                    ),
                    dtype=np.uint8,
                )
            elif observation_mode == "img_mc_substrates":
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.substrate_count,
                        gx,
                        gy,
                    ),
                    dtype=np.uint8,
                )
            else:
                o_observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cell_type_count + self.substrate_count,
                        gx,
                        gy,
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
            # shape = (E, 1)
        elif observation_mode == "transformer_nodes":
            o_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.max_clusters, self.features),
                dtype=np.float32,
            )

        else:
            raise ValueError(
                f"unknown observation type: {self.kwargs['observation_mode']}"
            )

        # output
        return o_observation_space

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
        mode = self.kwargs["observation_mode"]
        norm = self.kwargs["normalization_factor"]
        gx = self.kwargs["img_mc_grid_size_x"]
        gy = self.kwargs["img_mc_grid_size_y"]
        # get cell data frame
        self.df_cell = pd.DataFrame(
            physicell.get_cell(), columns=["ID", "x", "y", "z", "dead", "type"]
        )
        df_alive = self.df_cell[self.df_cell["dead"] < 0.1]

        # update tumor cell count
        self.c_prev = self.c_t
        self.c_t = df_alive.loc[(df_alive.type == "tumor"), :].shape[0]
        if self.c_prev is None:
            self.c_prev = self.c_t
        self.nb_tumor = self.c_t

        # update cell_1 cell count
        self.nb_cell_1 = df_alive.loc[(df_alive.type == "cell_1"), :].shape[0]

        # update cell_2 cell count
        self.nb_cell_2 = df_alive.loc[(df_alive.type == "cell_2"), :].shape[0]

        def get_normalized_cell_counts():
            counts = np.zeros(self.cell_type_count, dtype=np.float32)
            for cell_type, idx in self.cell_type_to_id.items():
                counts[idx] = (df_alive.type == cell_type).sum() / norm - 1
            return counts

        def get_max_substrates():
            max_vals = np.zeros(self.substrate_count, dtype=np.float32)
            for i, subs in enumerate(self.substrate_unique):
                microenv = physicell.get_microenv(subs)
                max_vals[i] = microenv[:, -1].max()  # last column = substrate value
            return max_vals

        def discretize_xy(x, y):
            x_bin = ((x - self.x_min) / (self.x_max - self.x_min) * (gx - 1)).astype(
                int
            )
            y_bin = ((y - self.y_min) / (self.y_max - self.y_min) * (gy - 1)).astype(
                int
            )
            return (
                np.clip(x_bin, 0, gx - 1),
                np.clip(y_bin, 0, gy - 1),
            )

        def build_cell_image():
            cell_type_idx = df_alive["type"].map(self.cell_type_to_id).to_numpy()
            x_bin, y_bin = discretize_xy(
                df_alive["x"].to_numpy(), df_alive["y"].to_numpy()
            )

            img = np.zeros(
                (self.cell_type_count, gx, gy),
                dtype=np.float32,
            )
            np.add.at(img, (cell_type_idx, x_bin, y_bin), 1)

            norm = self.ratio_img_mc_size_x * self.ratio_img_mc_size_y
            return ski.util.img_as_ubyte(img / norm)

        def build_substrate_image():
            # merge all substrates once
            dfs = []
            for subs in self.substrate_unique:
                dfs.append(
                    pd.DataFrame(
                        physicell.get_microenv(subs),
                        columns=["x", "y", "z", subs],
                    )
                )

            df = dfs[0]
            for d in dfs[1:]:
                df = df.merge(d, on=["x", "y", "z"])

            x_bin, y_bin = discretize_xy(df["x"].to_numpy(), df["y"].to_numpy())
            df["x_bin"] = x_bin
            df["y_bin"] = y_bin

            grouped = df.groupby(["x_bin", "y_bin"])[self.substrate_unique].max()

            img = np.zeros(
                (len(self.substrate_unique), gx, gy),
                dtype=np.float32,
            )

            for i, subs in enumerate(self.substrate_unique):
                for (xb, yb), val in grouped[subs].items():
                    img[i, xb, yb] = val

            min_v = img.min(axis=(1, 2), keepdims=True)
            max_v = img.max(axis=(1, 2), keepdims=True)
            scale = np.where(max_v > min_v, max_v - min_v, 1)

            return ski.util.img_as_ubyte((img - min_v) / scale)

        # observe the environemnt
        if mode == "scalars_cells":
            o_observation = get_normalized_cell_counts()

        elif mode == "scalars_substrates":
            o_observation = get_max_substrates()

        elif mode == "scalars_cells_substrates":
            o_observation = np.concatenate(
                [
                    get_normalized_cell_counts(),
                    get_max_substrates(),
                ]
            )

        elif mode in {
            "img_mc_cells",
            "img_mc_substrates",
            "img_mc_cells_substrates",
        }:
            if "cells" in mode:
                img_mc_cells = build_cell_image()
                if mode == "img_mc_cells":
                    o_observation = img_mc_cells

            elif "substrates" in mode:
                img_mc_substrates = build_substrate_image()
                if mode == "img_mc_substrates":
                    o_observation = img_mc_substrates

            elif mode == "img_mc_cells_substrates":
                o_observation = np.concatenate(
                    [img_mc_cells, img_mc_substrates],
                    axis=0,
                )

        elif mode in ["graph_delaunay", "graph_knn"]:
            df_alive.set_index("ID", inplace=True)
            coords = df_alive[["x", "y"]].values

            # Raw graph (variable size)
            pairs = (
                ty.build_delaunay(coords)
                if mode == "graph_delaunay"
                else ty.build_knn(coords, k=self.k)
            )  # shape = (E, 2)
            distances = ty.distance_neighbors(coords, pairs)  # shape = (E,)

            # Raw node features
            node_features = (
                df_alive["type"].map(self.cell_type_to_id).to_numpy(dtype=np.float32)
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
        elif mode == "transformer_nodes":
            df = df_alive.set_index("ID", drop=True)

            # ---- compute number of clusters ----
            def _compute_fibo(total_cells: int) -> int:
                fibo = np.array([1, 2, 3, 5, 7, 13, 21, 34, 55, 89, 144])
                idx = min(len(fibo) - 1, int(np.log(total_cells)))
                return int(fibo[idx])

            n_clusters = _compute_fibo(len(df))
            data = np.zeros((self.max_clusters, self.features), dtype=np.float32)
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

            x_mean = x_sum / cluster_sizes[:, 0]
            y_mean = y_sum / cluster_sizes[:, 0]

            x2_sum = np.bincount(labels, weights=x * x, minlength=n_clusters)
            y2_sum = np.bincount(labels, weights=y * y, minlength=n_clusters)

            x_std = np.sqrt(x2_sum / cluster_sizes[:, 0] - x_mean**2)
            y_std = np.sqrt(y2_sum / cluster_sizes[:, 0] - y_mean**2)

            cluster_frac = cluster_sizes[:, 0] / total_len

            stats = np.stack(
                [x_mean, y_mean, x_std, y_std, cluster_frac],
                axis=1,
            )

            # ============================================================
            # 3️⃣ FINAL OBSERVATION → (K, 8)
            # ============================================================
            o_observation = np.concatenate([stats, type_props], axis=1).astype(
                np.float32
            )

            K = o_observation.shape[0]

            # ---- pad with zeros to (8, 145) ----
            data[:K, :] = o_observation

        else:
            raise ValueError(f"unknown observation type: {mode}")

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
            #    title=f"dt_gym env step {str(self.step_env).zfill(4)} episode {str(self.episode).zfill(3)} episode step {str(self.step_episode).zfill(3)} : {df_cell.shape[0]} [cell]",
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
