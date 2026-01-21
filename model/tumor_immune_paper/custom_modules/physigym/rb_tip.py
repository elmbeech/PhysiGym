from collections import deque
import random
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from tensordict import TensorDict


def np2torch_dtype(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.uint8:
        return torch.uint8
    else:
        raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")


class ReplayBuffer:
    """
    Replay buffer supporting:
    - array-based states (NumPy)
    - torch tensor states (PyTorch)
    - graph-based states (PyG Data objects)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        buffer_size,
        batch_size,
        state_type=np.float32,
        is_graph=False,
        use_torch_tensors=False,  # New flag
    ):
        self.device = device
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.is_graph = is_graph
        self.use_torch_tensors = use_torch_tensors

        self.buffer_index = 0
        self.full = False

        if self.is_graph:
            # For variable-size graphs, use a deque
            self.buffer = deque(maxlen=self.buffer_size)
        else:
            if self.use_torch_tensors:
                # Preallocate torch tensors on the device
                torch_dtype = np2torch_dtype(state_type)
                self.state = torch.empty(
                    (self.buffer_size, *state_dim), dtype=torch_dtype, device=device
                )
                self.next_state = torch.empty(
                    (self.buffer_size, *state_dim), dtype=torch_dtype, device=device
                )
                self.action = torch.empty(
                    (self.buffer_size, *action_dim), dtype=torch.float32, device=device
                )
                self.reward = torch.empty(
                    (self.buffer_size, 1), dtype=torch.float32, device=device
                )
                self.done = torch.empty(
                    (self.buffer_size, 1), dtype=torch.uint8, device=device
                )
            else:
                # Use NumPy arrays
                self.state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
                self.next_state = np.empty(
                    (self.buffer_size, *state_dim), dtype=state_type
                )
                self.action = np.empty(
                    (self.buffer_size, *action_dim), dtype=np.float32
                )
                self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
                self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

    def __len__(self):
        if self.is_graph:
            return len(self.buffer)
        else:
            return self.buffer_size if self.full else self.buffer_index

    def add_batch(self, batch):
        """
        batch: list of (state, action, reward, next_state, done)
        """
        for transition in batch:
            self.add(*transition)

    def add(self, state, action, reward, next_state, done):
        if not self.is_graph:
            if self.use_torch_tensors:
                # Expect torch tensors; move to correct device
                self.state[self.buffer_index].copy_(state.to(self.device))
                self.action[self.buffer_index].copy_(action.to(self.device))
                self.reward[self.buffer_index].copy_(reward.to(self.device))
                self.next_state[self.buffer_index].copy_(next_state.to(self.device))
                self.done[self.buffer_index].copy_(done.to(self.device))
            else:
                # Expect numpy arrays
                self.state[self.buffer_index] = state
                self.action[self.buffer_index] = action
                self.reward[self.buffer_index] = reward
                self.next_state[self.buffer_index] = next_state
                self.done[self.buffer_index] = done

            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.full = self.full or self.buffer_index == 0
        else:
            state_graph = self._dict_reduced(state)
            next_state_graph = self._dict_reduced(next_state)
            self.buffer.append((state_graph, action, reward, next_state_graph, done))

    def _dict_reduced(self, obs):
        """
        Convert padded dict observation into variable-size graph
        """
        node_mask = obs["node_mask"] > 0.5
        edge_mask = obs["edge_mask"] > 0.5

        nodes = obs["node_features"][node_mask]
        edge_index = obs["edge_index"][:, edge_mask]
        edges = obs["edge_attr"][edge_mask]

        return {"nodes": nodes, "edge_links": edge_index, "edges": edges}

    def sample(self):
        if self.is_graph:
            batch = random.sample(self.buffer, self.batch_size)
            _state, action, reward, _next_state, done = zip(*batch)

            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            reward = torch.tensor(
                reward, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            done = torch.tensor(done, dtype=torch.uint8, device=self.device)

            state = [
                Data(
                    x=torch.tensor(s["nodes"], dtype=torch.float, device=self.device),
                    edge_index=torch.tensor(
                        s["edge_links"], dtype=torch.long, device=self.device
                    ),
                    edge_attr=torch.tensor(
                        s["edges"], dtype=torch.float, device=self.device
                    ),
                )
                for s in _state
            ]

            next_state = [
                Data(
                    x=torch.tensor(s["nodes"], dtype=torch.float, device=self.device),
                    edge_index=torch.tensor(
                        s["edge_links"], dtype=torch.long, device=self.device
                    ),
                    edge_attr=torch.tensor(
                        s["edges"], dtype=torch.float, device=self.device
                    ),
                )
                for s in _next_state
            ]

            return {
                "state": Batch.from_data_list(state),
                "action": action,
                "reward": reward,
                "done": done,
                "next_state": Batch.from_data_list(next_state),
            }
        else:
            idx = np.random.randint(
                0, self.buffer_size if self.full else self.buffer_index, self.batch_size
            )

            if self.use_torch_tensors:
                # Already tensors on device
                state = self.state[idx]
                next_state = self.next_state[idx]
                action = self.action[idx]
                reward = self.reward[idx]
                done = self.done[idx]
            else:
                # Convert NumPy arrays to tensors
                state = torch.as_tensor(self.state[idx], device=self.device).float()
                next_state = torch.as_tensor(
                    self.next_state[idx], device=self.device
                ).float()
                action = torch.as_tensor(self.action[idx], device=self.device)
                reward = torch.as_tensor(self.reward[idx], device=self.device)
                done = torch.as_tensor(self.done[idx], device=self.device)

            return TensorDict(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                },
                batch_size=self.batch_size,
                device=self.device,
            )


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rb = ReplayBuffer(
            state_dim=(
                144,
                8,
            ),
            action_dim=(1,),
            device=torch.device(device),
            buffer_size=2e5,
            batch_size=64,
            use_torch_tensors=True,
        )

        # Suppose your state/action/reward/next_state/done are already torch tensors
        local_batch = [
            (
                torch.randn((144, 8), device=device),
                torch.randn(1, device=device),
                torch.tensor(1.0, device=device),
                torch.randn(
                    (
                        144,
                        8,
                    ),
                    device=device,
                ),
                torch.tensor(0, device=device, dtype=torch.uint8),
            )
            for _ in range(128)
        ]
        rb.add_batch(local_batch)
        sampled = rb.sample()
        print(sampled["state"].shape)
    else:
        device = "cpu"
    rb = ReplayBuffer(
        state_dim=(10,),
        action_dim=(4,),
        device=torch.device(device),
        buffer_size=1000,
        batch_size=32,
        use_torch_tensors=True,
    )

    # Suppose your state/action/reward/next_state/done are already torch tensors
    local_batch = [
        (
            torch.randn(10, device=device),
            torch.randn(4, device=device),
            torch.tensor(1.0, device=device),
            torch.randn(10, device=device),
            torch.tensor(0, device=device, dtype=torch.uint8),
        )
        for _ in range(32)
    ]

    rb.add_batch(local_batch)

    sampled = rb.sample()
    print(sampled["state"].shape)  # (32, 10)
