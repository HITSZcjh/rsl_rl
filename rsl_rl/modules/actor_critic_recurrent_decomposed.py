# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, Memory
from rsl_rl.utils import unpad_trajectories


class DecoupledActorCriticRecurrent(nn.Module):
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        def get_dim(group_name):
            d = 0
            # 如果 group_name 不存在，说明用户没配，需要报错或者给空
            if group_name not in obs_groups:
                raise KeyError(f"Missing observation group: {group_name}. Required: policy_shared, policy_x, policy_y, policy_z, policy_yaw")
            for name in obs_groups[group_name]:
                d += obs[name].shape[-1]
            return d
        
        dim_shared = get_dim('policy_shared')
        dim_x = get_dim('policy_x')
        dim_y = get_dim('policy_y')
        dim_z = get_dim('policy_z')
        dim_yaw = get_dim('policy_yaw')
        dim_critic = get_dim('critic')

        self.memory_a = Memory(dim_shared, rnn_hidden_dim, rnn_num_layers, rnn_type)
        print(f"Shared Actor RNN: {self.memory_a}")

        self.actor_x = MLP(dim_x + rnn_hidden_dim, 1, actor_hidden_dims, activation)
        self.actor_y = MLP(dim_y + rnn_hidden_dim, 1, actor_hidden_dims, activation)
        self.actor_z = MLP(dim_z + rnn_hidden_dim, 1, actor_hidden_dims, activation)
        self.actor_yaw = MLP(dim_yaw + rnn_hidden_dim, 1, actor_hidden_dims, activation)

        mix_mat = torch.tensor([[-0.5, 0.5, 0.5, -0.5],
                                [0.5, -0.5, 0.5, -0.5],
                                [0.5, 0.5, -0.5, -0.5],
                                [-0.5, -0.5, -0.5, -0.5]], 
                                dtype=torch.float32)
        self.register_buffer("mixing_matrix", mix_mat)


        # Actor observation normalization
        self.norm_shared = EmpiricalNormalization(dim_shared)
        self.norm_x = EmpiricalNormalization(dim_x)
        self.norm_y = EmpiricalNormalization(dim_y)
        self.norm_z = EmpiricalNormalization(dim_z)
        self.norm_yaw = EmpiricalNormalization(dim_yaw)

        # Critic
        self.memory_c = Memory(dim_critic, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)
        self.norm_critic = EmpiricalNormalization(dim_critic)
        print(f"Critic RNN: {self.memory_c}")
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def _get_obs_by_group(self, obs: TensorDict, group_name: str) -> torch.Tensor:
        """辅助函数：从 TensorDict 提取并拼接指定组的观测"""
        parts = [obs[name] for name in self.obs_groups[group_name]]
        return torch.cat(parts, dim=-1)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        # --- 1. Shared RNN 前向传播 ---
        obs_shared = self._get_obs_by_group(obs, 'policy_shared')
        obs_shared = self.norm_shared(obs_shared)
        # Shared Latent: (Batch, rnn_hidden_dim)
        shared_latent = self.memory_a(obs_shared, masks, hidden_state).squeeze(0)

        # --- 2. 四轴独立控制计算 (Parallel MLPs) ---
        # 准备各个轴的观测并归一化
        batch_mode = masks is not None
        if batch_mode:
            o_x = unpad_trajectories(self._get_obs_by_group(obs, 'policy_x'), masks)
            o_x = self.norm_x(o_x)
            o_y = unpad_trajectories(self._get_obs_by_group(obs, 'policy_y'), masks)
            o_y = self.norm_y(o_y)
            o_z = unpad_trajectories(self._get_obs_by_group(obs, 'policy_z'), masks)
            o_z = self.norm_z(o_z)
            o_yaw = unpad_trajectories(self._get_obs_by_group(obs, 'policy_yaw'), masks)
            o_yaw = self.norm_yaw(o_yaw)
        else:
            o_x = self.norm_x(self._get_obs_by_group(obs, 'policy_x'))
            o_y = self.norm_y(self._get_obs_by_group(obs, 'policy_y'))
            o_z = self.norm_z(self._get_obs_by_group(obs, 'policy_z'))
            o_yaw = self.norm_yaw(self._get_obs_by_group(obs, 'policy_yaw'))

        # 拼接 Latent: 输入 = [Axis Obs, Shared Latent]
        in_x = torch.cat([o_x, shared_latent], dim=-1)
        in_y = torch.cat([o_y, shared_latent], dim=-1)
        in_z = torch.cat([o_z, shared_latent], dim=-1)
        in_yaw = torch.cat([o_yaw, shared_latent], dim=-1)

        # 计算虚拟控制量 u (Batch, 1)
        u_x = self.actor_x(in_x)
        u_y = self.actor_y(in_y)
        u_z = self.actor_z(in_z)
        u_yaw = self.actor_yaw(in_yaw)

        # --- 3. 混合与控制分配 (Control Allocation) ---
        virtual_ctrl = torch.cat([u_x, u_y, u_yaw ,u_z], dim=-1)
        
        # 应用分配矩阵
        # 公式: Actions = Virtual @ Matrix.T
        # Matrix shape: (4, 4), Virtual shape: (Batch, 4) -> Result: (Batch, 4)
        mean_actions = torch.matmul(virtual_ctrl, self.mixing_matrix.t())

        # --- 4. 构建分布并采样 ---
        # 这里的 std 是针对最终 4 个电机的噪声
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean_actions)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean_actions)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = Normal(mean_actions, std)
        
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        # --- 1. Shared RNN 前向传播 ---
        obs_shared = self._get_obs_by_group(obs, 'policy_shared')
        obs_shared = self.norm_shared(obs_shared)

        # Shared Latent: (Batch, rnn_hidden_dim)
        shared_latent = self.memory_a(obs_shared).squeeze(0)

        # --- 2. 四轴独立控制计算 (Parallel MLPs) ---
        # 准备各个轴的观测并归一化
        o_x = self.norm_x(self._get_obs_by_group(obs, 'policy_x'))
        o_y = self.norm_y(self._get_obs_by_group(obs, 'policy_y'))
        o_z = self.norm_z(self._get_obs_by_group(obs, 'policy_z'))
        o_yaw = self.norm_yaw(self._get_obs_by_group(obs, 'policy_yaw'))

        # 拼接 Latent: 输入 = [Axis Obs, Shared Latent]
        in_x = torch.cat([o_x, shared_latent], dim=-1)
        in_y = torch.cat([o_y, shared_latent], dim=-1)
        in_z = torch.cat([o_z, shared_latent], dim=-1)
        in_yaw = torch.cat([o_yaw, shared_latent], dim=-1)

        # 计算虚拟控制量 u (Batch, 1)
        u_x = self.actor_x(in_x)
        u_y = self.actor_y(in_y)
        u_z = self.actor_z(in_z)
        u_yaw = self.actor_yaw(in_yaw)

        # --- 3. 混合与控制分配 (Control Allocation) ---
        # 拼接成虚拟控制向量 (Batch, 4) -> [u_x, u_y, u_z, u_yaw]
        virtual_ctrl = torch.cat([u_x, u_y, u_yaw, u_z], dim=-1)

        # 应用分配矩阵
        # 公式: Actions = Virtual @ Matrix.T
        # Matrix shape: (4, 4), Virtual shape: (Batch, 4) -> Result: (Batch, 4)
        mean_actions = torch.matmul(virtual_ctrl, self.mixing_matrix.t())

        return mean_actions

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs_c = self._get_obs_by_group(obs, 'critic')
        obs_c = self.norm_critic(obs_c)
        out_mem = self.memory_c(obs_c, masks, hidden_state).squeeze(0)
        return self.critic(out_mem)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        obs_shared = self._get_obs_by_group(obs, 'policy_shared')
        self.norm_shared.update(obs_shared)

        obs_x = self._get_obs_by_group(obs, 'policy_x')
        self.norm_x.update(obs_x)

        obs_y = self._get_obs_by_group(obs, 'policy_y')
        self.norm_y.update(obs_y)

        obs_z = self._get_obs_by_group(obs, 'policy_z')
        self.norm_z.update(obs_z)

        obs_yaw = self._get_obs_by_group(obs, 'policy_yaw')
        self.norm_yaw.update(obs_yaw)

        obs_critic = self._get_obs_by_group(obs, 'critic')
        self.norm_critic.update(obs_critic)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
