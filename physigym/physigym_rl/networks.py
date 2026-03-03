import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
###########################
# Classes Neural Networks #
###########################


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


class HadamaxBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First block
        self.conv1a = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv1b = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.activation = nn.Mish()

    def forward(self, x):
        # First block
        x1 = self.activation(F.layer_norm(self.conv1a(x), self.conv1a(x).shape[1:]))
        x2 = self.activation(F.layer_norm(self.conv1b(x), self.conv1b(x).shape[1:]))
        x = self.pool(x1 * x2)
        return x


class GraphFeatureExtractor(nn.Module):
    def __init__(self, in_channels=-1, out_channels=32, heads=4, **kwargs):
        super().__init__()
        self.gat1 = GATConv(in_channels=in_channels, out_channels=4, heads=heads)
        self.gat2 = GATConv(4 * heads, out_channels, heads=1)
        self.activation = nn.Mish()

    def forward(self, data):
        x = self.activation(self.gat1(data.x, data.edge_index, data.edge_attr))
        x = self.activation(self.gat2(x, data.edge_index, data.edge_attr))
        return global_mean_pool(x, data.batch)


class FeatureExtractor(nn.Module):
    """Handles image-based, vector-based and graph-based state inputs dynamically."""

    def __init__(self, cfg, neural_architecture_image="impala"):
        super().__init__()

        obs_shape = cfg["observation_space_shape"]
        self.is_graph = True if "graph" in cfg["observation_mode"] else False
        self.is_tranformer_node = (
            True if "transformer" in cfg["observation_mode"] else False
        )
        self.is_image = (len(obs_shape) == 3) and not self.is_tranformer_node

        if self.is_graph:
            # Assume node features have fixed dimension
            node_feature_dim = cfg["node_feature_dim"]
            self.feature_extractor = GraphFeatureExtractor(
                in_channels=node_feature_dim  # ✅ Correct parameter
            )
            self.feature_size = 128

        elif self.is_image:
            if neural_architecture_image == "impala":
                layers = [
                    PixelPreprocess(),
                    ImpalaBlock(obs_shape[0], 16),
                    ImpalaBlock(16, 32),
                    ImpalaBlock(32, 32),
                    nn.Flatten(),
                ]
            elif neural_architecture_image == "hadamax":
                layers = [
                    PixelPreprocess(),
                    HadamaxBlock(obs_shape[0], 16),
                    HadamaxBlock(16, 32),
                    HadamaxBlock(32, 32),
                    nn.Flatten(),
                ]
            else:
                raise ValueError(
                    f"Error: unknown neural architecture: {neural_architecture_image}"
                )

            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            self.feature_extractor = nn.Identity()
            self.feature_size = int(np.prod(obs_shape))

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image or self.is_tranformer_node:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        elif self.is_graph:
            x = self.feature_extractor(x)
        return x


class QNetwork(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(256)
        self.fc3 = nn.LazyLinear(1)
        self.mish = nn.Mish()

    def forward(self, z, a):
        x = torch.cat([z, a], dim=1)  # Concatenate state and action

        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """Policy network (Actor)"""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, cfg):
        super().__init__()
        action_dim = np.prod(cfg["action_space_shape"])

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
                (cfg["action_space_high"] - cfg["action_space_low"]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (cfg["action_space_high"] + cfg["action_space_low"]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, z):
        mean, log_std = self.forward(z)
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
