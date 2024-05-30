from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np
import wandb

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/')
sys.path.append('../../isaacgym/python/isaacgym')#"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
sys.path.append('../../../isaacgym/python/isaacgym')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm_

import time
import re
import os

import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import wandb
import pyrallis
from utilities import fill_buffer, fix_batch, get_scaled_quaternion, quat_diff_rad, pose_world_to_robot_base, axis_angle_to_quaternion, fix_peg_pos_state, fix_peg_pos_quat_state
from CORL.algorithms.offline.iql import TwinQ, ValueFunction

class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 1024,
        n_hidden: int = 5,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        layers = []
        #Add input layer
        layers.append(nn.Linear(state_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ELU())
        
        #Add hidden layers
        for i in range(1, self.n_hidden):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ELU())
        
        # Add output layer
        layers.append(nn.Linear(self.hidden_dim, act_dim))
        layers.append(nn.Tanh())
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)
                 
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor: #state is peg and hole position
        x = obs.clone()

        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        for i in range(1, self.n_hidden):
          x_res = x.clone()
          x = self.net[(i*3)+0](x)
          x = self.net[(i*3)+1](x)
          x = self.net[(i*3)+2](x)
          x += x_res
        
        x = self.net[self.n_hidden*3](x)
        x = self.net[self.n_hidden*3 + 1](x) #here we are doing just tanh
        
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )
        
class DeterministicFourierPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 1024,
        n_hidden: int = 5,
        mapping_size: int = 256,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.mapping_size = mapping_size
        layers = []
        #Add input layer
        layers.append(nn.Linear(self.mapping_size*2, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ELU())
        
        #Add hidden layers
        for i in range(1, self.n_hidden):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ELU())
        
        # Add output layer
        layers.append(nn.Linear(self.hidden_dim, act_dim))
        layers.append(nn.Tanh())
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)
        self.max_action = max_action
        
        #self.B = np.random.normal(rand_key, (self.mapping_size, 2))
        self.B = torch.randn((self.mapping_size, state_dim))
        scale=10.0 #1.0, 100.0 other options there
        self.B = self.B * scale

    #def input_mapping(x):
    #    x_proj = (2.*np.pi*x) @ self.B.T #This is v=[batch, state14] @ BT=[state14, mapping] => vBT=[batch,mapping]
    #    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
    
    def input_mapping(self, x):
        x_proj = (2.*torch.pi*x) @ self.B.T #This is v=[batch, state14] @ BT=[state14, mapping] => vBT=[batch,mapping]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1) #[batch, mapping*2]

    def forward(self, obs: torch.Tensor) -> torch.Tensor: #state is peg and hole position
        x = obs.clone()
        x = self.input_mapping(x)
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        for i in range(1, self.n_hidden):
          x_res = x.clone()
          x = self.net[(i*3)+0](x)
          x = self.net[(i*3)+1](x)
          x = self.net[(i*3)+2](x)
          x += x_res
        
        x = self.net[self.n_hidden*3](x)
        x = self.net[self.n_hidden*3 + 1](x) #here we are doing just tanh
        
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )
        
class Qf(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 1024,
        n_hidden: int = 5,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        layers = []
        #Add input layer
        layers.append(nn.Linear(state_dim+act_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ELU())
        
        #Add hidden layers
        for i in range(1, self.n_hidden):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ELU())
        
        # Add output layer
        layers.append(nn.Linear(self.hidden_dim, 1))
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor: #state is peg and hole position
        x = obs.clone()

        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        for i in range(1, self.n_hidden):
          x_res = x.clone()
          x = self.net[(i*3)+0](x)
          x = self.net[(i*3)+1](x)
          x = self.net[(i*3)+2](x)
          x += x_res
        
        x = self.net[self.n_hidden*3](x)
        
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )
        
def iterative_train(batch, q_network, actor, q_optimizer, actor_optimizer):
    (
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    ) = batch
    #Update Qf
    q_inp = torch.cat([observations, actions], 1)
    #q_output = q_network(q_inp)
    #q_loss = F.mse_loss(q_output, rewards)
    #q_optimizer.zero_grad()
    #q_loss.backward()
    #q_optimizer.step()
    #Update Actor
    #q_output = q_network(q_inp)
    #exp_adv = torch.exp(0.03 * q_output.detach()).clamp(max=100.0) #Here the actor doesn't take a direct gradient, it is scaling how hard to BC this value!
    exp_adv = torch.exp(3.0 * rewards).clamp(max=100.0)
    #print(exp_adv)
    policy_out = actor(observations)
    pi_quat, pi_pos = get_scaled_quaternion(policy_out)
    action_quat, action_pos = get_scaled_quaternion(actions)
    actor_loss_pos = F.mse_loss(pi_pos, action_pos)
    actor_loss_rot = quat_diff_rad(pi_quat, action_quat).mean()
    bc_losses = 1e5 * actor_loss_pos + 1e3 * actor_loss_rot
    #policy_loss = torch.mean(exp_adv * bc_losses) #Scaling bc_losses differently means less of this averaging stuff we see
    actor_optimizer.zero_grad()
    bc_losses.backward() #-------------------------Baseline for just BC with this
    actor_optimizer.step()
    
    return bc_losses.mean(), 1.0#q_output.mean()
    
def iterative_train_fourier_BC(observations, actions, actor, actor_optimizer):
    policy_out = actor(observations)
    
    pi_quat, pi_pos = get_scaled_quaternion(policy_out, 0.01)
    action_quat, action_pos = axis_angle_to_quaternion(actions) #actions are staying as -0.01 to 0.01 scale so don't scale them down
    actor_loss_pos = F.mse_loss(pi_pos, action_pos)
    actor_loss_rot = quat_diff_rad(pi_quat, action_quat).mean()
    
    bc_losses = 1e5 * actor_loss_pos + 1e3 * actor_loss_rot
    
    actor_optimizer.zero_grad()
    bc_losses.backward()
    actor_optimizer.step()
    
    return bc_losses.mean()
    
def state_sensitivity_fixing(state):
    #([0.13, 0.0, 1.065, 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0]), 0.001 changes need to matter, in range 0.005 of 0.13 and 1.065, I will do -0.13 and -1.065
    #Where it is true it will place the corresponding value from the true tensor so just all 1's then when false from the false tensor so all 0's so will just place 1 if true and 0 if false
    #grip_loc_success = torch.where(
    #  state[:, 0] > 0.135 and state[:, 0] < 0.145 and state[:, 2] > 1.06 and state[:, 2] < 1.07,
    #  torch.full((state.shape[0],), 1.0, device=self.device),
    #  state)
    #print((state[:, 0] > 0.135 and state[:, 0] < 0.145 and state[:, 2] > 1.06 and state[:, 2] < 1.07).shape())
    state[:, 0] -= 0.13
    state[:, 0] *= 10.0
    state[:, 2] -= 1.065
    state[:, 2] *= 10.0
    return state

    
import matplotlib.pyplot as plt
def test_individual_states(observations, actions, actor, actor_optimizer, t):

    #remove anything too far off the straight up rotation
    state_template = torch.tensor([0.13, 0.0, 1.65, 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
    state_sized = state_template.unsqueeze(0).repeat(observations.shape[0], 1) #[batch, state]
    rot_differences = quat_diff_rad(observations[:, 3:7], state_sized[:, 3:7]).unsqueeze(-1) #[batch]
    print(rot_differences)
    mask = torch.all(rot_differences <= 0.1, dim=1) #0.01 is too little
    indices = torch.nonzero(mask).squeeze()
    observations = observations[indices, :]
    actions = actions[indices, :]
    
    max_values = fix_peg_pos_state(torch.tensor([[0.145, 0.001, 1.070, 0.0, 0.0, 0.0, 1.0]]))[0]
    min_values = fix_peg_pos_state(torch.tensor([[0.115, -0.001, 1.067, 0.0, 0.0, 0.0, 1.0]]))[0]
    mask = torch.all((observations[:, 0:3] >= min_values) & (observations[:, 0:3] <= max_values), dim=1)
    indices = torch.nonzero(mask).squeeze()
    #print("Num samples that have this", index.numel())
    #print(indices)
    if indices.numel() > 0:
      index = indices[0]
      #print("action_pos_label", actions[indices], "location", observations[indices])#, "state", observations[index])
      #make xz states
      xz_states = torch.cat([observations[indices, 0].unsqueeze(-1), observations[indices, 2].unsqueeze(-1)], axis=1).numpy()
      vectors = torch.cat([actions[indices, 0].unsqueeze(-1), actions[indices, 2].unsqueeze(-1)], axis=1).numpy()
      print(xz_states)
      print(vectors)
      plt.rcParams['figure.dpi'] = 150
      plt.quiver([point[0] for point in xz_states], 
           [point[1] for point in xz_states], 
           [vector[0] / max(np.abs(vector[0]), np.abs(vector[1])) if max(np.abs(vector[0]), np.abs(vector[1])) != 0 else 0 for vector in vectors], 
           [vector[1] / max(np.abs(vector[0]), np.abs(vector[1])) if max(np.abs(vector[0]), np.abs(vector[1])) != 0 else 0 for vector in vectors], scale=120.0)
      #plt.quiver(x, y, u, v, scale=40.0)
      plt.gca().set_aspect('equal', adjustable='box')
      plt.savefig("data_vector_field_x")
      return
      actions = actions[index].unsqueeze(0)
      policy_out = actor(observations[index].unsqueeze(0))
      action_quat, action_pos = axis_angle_to_quaternion(actions)
      pi_pos=policy_out
      print("outputted actions", pi_pos)
      actor_loss_pos = F.mse_loss(pi_pos, action_pos)
      
      bc_losses = 1e7 * actor_loss_pos# + 1e3 * actor_loss_rot
      denom=(torch.abs(action_pos)) #there is an issue when it is 0
      percent_diff=(torch.abs(pi_pos - action_pos) / torch.abs(action_pos).mean()).mean() * 100.0
      print("percent diff", percent_diff)
      print("bc_losses", bc_losses)
      
    
def plot_vf_heatmap(observations, actions, actor, actor_optimizer, t, vf_file):
    vf = ValueFunction(state_dim, hidden_dim=1024, n_hidden=5)
    #vf_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_vf_1999999.pth"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_no_noise_sparse/chp_IQL_vf_1999999.pth"
    vf_weights=torch.load(vf_file)
    vf.load_state_dict(vf_weights)
    
    with torch.no_grad():
      max_values = fix_peg_pos_state(torch.tensor([[0.15, 0.001, 1.070, 0.0, 0.0, 0.0, 1.0]]))[0]
      min_values = fix_peg_pos_state(torch.tensor([[0.11, -0.001, 1.040, 0.0, 0.0, 0.0, 1.0]]))[0]
      mask = torch.all((observations[:, 0:3] >= min_values) & (observations[:, 0:3] <= max_values), dim=1)
      indices = torch.nonzero(mask).squeeze()
      #print("Num samples that have this", index.numel())
      #print(indices)
      if indices.numel() > 0:
        index = indices[0]
        #print("action_pos_label", actions[indices], "location", observations[indices])#, "state", observations[index])
        #make xz states
        xz_states = torch.cat([observations[indices, 0].unsqueeze(-1), observations[indices, 2].unsqueeze(-1)], axis=1).numpy()
        values = vf(observations[indices, :]) #[observations matching, 1]
        values = torch.clip(values, min=-40.0, max=1000.0)
        print(xz_states)
        print(values)
        plt.rcParams['figure.dpi'] = 150
        plt.scatter([point[0] for point in xz_states], 
             [point[1] for point in xz_states],
             c=values, cmap='viridis', alpha=0.5, s=2)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("sampled_datapoints_heatmap")
        return
      
def get_reward_states(observations, actions):
    max_values = fix_peg_pos_state(torch.tensor([[0.131, 0.001, 1.045, 0.0, 0.0, 0.0, 1.0]]))[0]
    min_values = fix_peg_pos_state(torch.tensor([[0.129, -0.001, 1.040, 0.0, 0.0, 0.0, 1.0]]))[0]
    mask = torch.all((observations[:, 0:3] >= min_values) & (observations[:, 0:3] <= max_values), dim=1)
    indices = torch.nonzero(mask).squeeze()
    if indices.numel() > 2:
      index = indices[0]
      actions = actions[indices]#.unsqueeze(0)
      observations = observations[indices]
      return indices#actions, observations
    else:
      return None#, None
      
      
def test_individual_states_y(observations, actions, actor, actor_optimizer, t):
    max_values = fix_peg_pos_state(torch.tensor([[0.131, 0.015, 1.070, 0.0, 0.0, 0.0, 1.0]]))[0]
    min_values = fix_peg_pos_state(torch.tensor([[0.129, -0.015, 1.062, 0.0, 0.0, 0.0, 1.0]]))[0]
    #max_values = fix_peg_pos_state(torch.tensor([[0.131, 0.001, 1.045, 0.0, 0.0, 0.0, 1.0]]))[0]
    #min_values = fix_peg_pos_state(torch.tensor([[0.129, -0.001, 1.040, 0.0, 0.0, 0.0, 1.0]]))[0]
    
    #remove anything too far off the straight up rotation
    state_template = torch.tensor([0.13, 0.0, 1.065, 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
    state_sized = state_template.unsqueeze(0).repeat(replay_buffer.size(), 1) #[batch, state]
    rot_differences = quat_diff_rad(observations[:, 3:7], state_sized[:, 3:7]).unsqueeze(-1) #[batch]
    print(rot_differences)
    mask = torch.all(rot_differences <= 0.25, dim=1) #0.01 is too little
    indices = torch.nonzero(mask).squeeze()
    observations = observations[indices, :]
    
    mask = torch.all((observations[:, 0:3] >= min_values) & (observations[:, 0:3] <= max_values), dim=1)
    indices = torch.nonzero(mask).squeeze()
    
    #print("Num samples that have this", index.numel())
    #print(indices)
    if indices.numel() > 0:
      index = indices[0]
      #print("action_pos_label", actions[indices], "location", observations[indices])#, "state", observations[index])
      #make xz states
      xz_states = torch.cat([observations[indices, 1].unsqueeze(-1), observations[indices, 2].unsqueeze(-1)], axis=1).numpy()
      vectors = torch.cat([actions[indices, 1].unsqueeze(-1), actions[indices, 2].unsqueeze(-1)], axis=1).numpy()
      print(xz_states)
      print(vectors)
      plt.rcParams['figure.dpi'] = 250
      plt.quiver([point[0] for point in xz_states], 
           [point[1] for point in xz_states], 
           [vector[0] / max(np.abs(vector[0]), np.abs(vector[1])) if max(np.abs(vector[0]), np.abs(vector[1])) != 0 else 0 for vector in vectors], 
           [vector[1] / max(np.abs(vector[0]), np.abs(vector[1])) if max(np.abs(vector[0]), np.abs(vector[1])) != 0 else 0 for vector in vectors], scale=120.0)
      #plt.quiver(x, y, u, v, scale=40.0)
      plt.gca().set_aspect('equal', adjustable='box')
      plt.savefig("data_vector_field_y")
      return
      actions = actions[index].unsqueeze(0)
      policy_out = actor(observations[index].unsqueeze(0))
      action_quat, action_pos = axis_angle_to_quaternion(actions)
      pi_pos=policy_out
      print("outputted actions", pi_pos)
      actor_loss_pos = F.mse_loss(pi_pos, action_pos)
      
      bc_losses = 1e7 * actor_loss_pos# + 1e3 * actor_loss_rot
      denom=(torch.abs(action_pos)) #there is an issue when it is 0
      percent_diff=(torch.abs(pi_pos - action_pos) / torch.abs(action_pos).mean()).mean() * 100.0
      print("percent diff", percent_diff)
      print("bc_losses", bc_losses)
    
def train_specifically_close_states(observations, actions, actor, actor_optimizer, t):
    max_values = fix_peg_pos_state(torch.tensor([[0.135, 0.005, 1.067, 0.0, 0.0, 0.0, 1.0]]))[0]
    min_values = fix_peg_pos_state(torch.tensor([[0.125, -0.005, 1.040, 0.0, 0.0, 0.0, 1.0]]))[0]
    #max_values = torch.tensor([0.135, 0.005, 1.067])
    #min_values = torch.tensor([0.125, -0.005, 1.040])
    mask = torch.all((observations[:, 0:3] >= min_values) & (observations[:, 0:3] <= max_values), dim=1)
    indices = torch.nonzero(mask).squeeze()
    #print("Num samples that have this", indices.numel())
    #print(indices)
    if indices.numel() > 2:
      index = indices[0]
      #print("action_pos_label", actions[indices])#, "state", observations[index])
    
      actions = actions[indices]#.unsqueeze(0)
      policy_out = actor(observations[indices])
      action_quat, action_pos = axis_angle_to_quaternion(actions)
      pi_pos=policy_out
      #print("outputted actions", pi_pos)
      
      #relative direction loss, divide both by selves to just want to match the normalized values
      
      actor_loss_pos = F.mse_loss(pi_pos, action_pos)
      
      bc_losses = 1e7 * actor_loss_pos# + 1e3 * actor_loss_rot
      if t % 101 == 0:
        denom=(torch.abs(action_pos)) #there is an issue when it is 0
        percent_diff=(torch.abs(pi_pos - action_pos) / torch.abs(action_pos).mean()).mean() * 100.0
        print("percent diff", percent_diff)
        print("bc_losses", bc_losses)
        
      actor_optimizer.zero_grad()
      bc_losses.backward()
      torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
      actor_optimizer.step()
      
      return bc_losses.mean()
    else:
      return 100.0
    
def dot_product(one, two, dim):
    res = one * two
    return torch.sum(res, dim=dim)
    
def iterative_train_pos_only(observations, actions, actor, actor_optimizer, t, dot_prod=False):

    policy_out = actor(observations)
    action_quat, action_pos = axis_angle_to_quaternion(actions)
    pi_pos=policy_out
    
    if dot_prod == False:
      #loss type mse
      actor_loss_pos = F.mse_loss(pi_pos, action_pos)
      bc_losses = 1e7 * actor_loss_pos# + 1e3 * actor_loss_rot
    else:
      #loss type normalized maxing dot product
      #actor_loss_pos = dot_product(torch.nn.functional.normalize(pi_pos, dim=0), torch.nn.functional.normalize(action_pos, dim=0), dim=1))
      #The normalized values multiplying together 
      #bc_losses=-1.0 * torch.mean(actor_loss_pos)
      #This is dot product then normalized after, dot product makes sense as negative and negative make a positive so higher similarity and positive and negative make lower number so lower similarity
      actor_loss_pos = F.cosine_similarity(pi_pos, action_pos, dim=1)
      bc_losses=-1.0 * torch.mean(actor_loss_pos)
      #print(pi_pos, action_pos)
      #print(actor_loss_pos)
    
    if t % 100 == 0:
      if dot_prod == False:
        denom=(torch.abs(action_pos)) #there is an issue when it is 0
        percent_diff=(torch.abs(pi_pos - action_pos) / torch.abs(action_pos).mean()).mean() * 100.0
        print("percent diff", percent_diff)
      print("bc_losses", bc_losses)
      
    actor_optimizer.zero_grad()
    bc_losses.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
    actor_optimizer.step()
    
    return bc_losses.mean()
    
buffer_size = 10000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
rew_type = "dense"#"dense"#"sparse"
name_use="DotProduct_BC_BIGDATA_5_layer"
batch_size = 256

def train_fourier_BC():
    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer()
    print("buf size:", replay_buffer.size())

    max_action = 1.0

    name = "FourierBC_5_layer"
    checkpoints_path = os.path.join("./checkpoints/", name)
    print(f"Checkpoints path: {checkpoints_path}")
    os.makedirs(checkpoints_path, exist_ok=True)

    policy_action_dim = 6
    #hidden_dim: int = 1024, n_hidden: int = 5
    hidden_dim=1024
    n_hidden=5
    actor = DeterministicFourierPolicy(state_dim, policy_action_dim, max_action, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    torch.save(actor.B, os.path.join("./checkpoints/"+name+"/", f"chp_FBC_B.pth"))
    evaluations = []
    for t in range(int(1000000)):
      batch = replay_buffer.sample(batch_size)
      batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
      state, action = fix_batch(batch)

      state = state.to(device)
      action = action.to(device)
      log_dict = iterative_train_fourier_BC(state, action, actor, actor_optimizer) 
      #Input scaled to -1 to 1 actions and proper states. Should learn these -1 to 1 actions and that is that. In pos, axis angles
      if t % 1000 == 0: 
        print("train timestep", t, "av bc_loss", log_dict)
      if (t + 1) % 20000 == 0 or t == 0:
          print("save time")
          print("now saving here")
          torch.save(actor.state_dict(), os.path.join("./checkpoints/"+name+"/", f"chp_FBC_actor_{t}.pth"))


def train():
    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer()
    print("buf size:", replay_buffer.size())

    max_action = 1.0

    name = name_use+rew_type
    checkpoints_path = os.path.join("./checkpoints/", name)
    print(f"Checkpoints path: {checkpoints_path}")
    os.makedirs(checkpoints_path, exist_ok=True)

    policy_action_dim = 6
    #hidden_dim: int = 1024, n_hidden: int = 5
    hidden_dim=1024
    n_hidden=5
    #q_network = Qf(state_dim, policy_action_dim, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    actor = DeterministicPolicy(state_dim, 3, max_action, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)

    #policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/BC_new_pos_only_5_layerdense/chp_BC_actor_499999.pth"
    #actor_weights=torch.load(policy_file)
    #actor.load_state_dict(actor_weights)

    #q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    batch_size=10000
    evaluations = []
    for t in range(int(500000)):
      batch = replay_buffer.sample(batch_size)
      if rew_type == "dense":
        batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
      else:
        batch = [batch[0], batch[2], batch[4], batch[1], batch[4]]
      state, action = fix_batch(batch)
      batch[0] = state
      batch[1] = action
      #batch[2] += 5.0
      #remember that the data is state and next state so 28 values + 1 reward + 1 done. The action is created from these two states. 
      
      #action=batch[1]
      batch = [b.to(device) for b in batch]
      #state=state_sensitivity_fixing(state)
      state = state.to(device)
      action = action.to(device)
      train_specifically_close_states(state, action, actor, actor_optimizer, t)
      #test_individual_states(state, action, actor, actor_optimizer, t)
      #log_dict = iterative_train_pos_only(state, action, actor, actor_optimizer, t)
      #log_dict = iterative_train(batch, q_network, actor, q_optimizer, actor_optimizer) 
      #Input scaled to -1 to 1 actions and proper states. Should learn these -1 to 1 actions and that is that. In pos, axis angles
      #if t % 1000 == 0: 
        #print("train timestep", t, "av bc_loss", log_dict[0], "q_output_mean", log_dict[1])
        #print("reward mean", batch[2].mean())
      if (t + 1) % 20000 == 0 or t == 0:
          print("save time")
          print("now saving here")
          torch.save(actor.state_dict(), os.path.join("./checkpoints/"+name_use+rew_type+"/", f"chp_BC_actor_{t}.pth"),)
          
def small_train():
    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer(file_path)
    print("buf size:", replay_buffer.size())
    max_action = 1.0
    name = name_use
    checkpoints_path = os.path.join("./checkpoints/", name)
    print(f"Checkpoints path: {checkpoints_path}")
    os.makedirs(checkpoints_path, exist_ok=True)
    policy_action_dim = 3
    hidden_dim=1024
    n_hidden=5
    actor = DeterministicPolicy(state_dim, policy_action_dim, max_action, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)

    #policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/BC_oversampling_halftime_important_values_BIGDATA_5_layer/chp_BC_actor_499999.pth"
    #actor_weights=torch.load(policy_file)
    #actor.load_state_dict(actor_weights)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    
    for t in range(int(1000000)):
      if t % 2 == 0:
        batch_size=replay_buffer.size()
      else:
        batch_size=256
      batch_size=256
      batch = replay_buffer.sample(batch_size)
      batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
      state, action = fix_batch(batch)
      batch[0] = state
      batch[1] = action
      #state=state_sensitivity_fixing(state)
      state = state.to(device)
      action = action.to(device)
      #test_individual_states(state, action, actor, actor_optimizer, t)
      #test_individual_states_y(state, action, actor, actor_optimizer, t)
      #if t % 2 == 0:
      #  train_specifically_close_states(state, action, actor, actor_optimizer, t)
      #else:
      #  iterative_train_pos_only(state, action, actor, actor_optimizer, t)
      iterative_train_pos_only(state, action, actor, actor_optimizer, t, dot_prod=True)
      #test_individual_states(state, action, actor, actor_optimizer, t)
      #log_dict = iterative_train_pos_only(state, action, actor, actor_optimizer, t)
      if (t + 1) % 20000 == 0 or t == 0:
          print("save time")
          print("now saving here")
          torch.save(actor.state_dict(), os.path.join("./checkpoints/"+name_use+"/", f"chp_BC_actor_{t}.pth"),)

def plot_samples():
    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer(file_path)
    print("buf size:", replay_buffer.size())
    max_action = 1.0
    name = name_use
    checkpoints_path = os.path.join("./checkpoints/", name)
    print(f"Checkpoints path: {checkpoints_path}")
    os.makedirs(checkpoints_path, exist_ok=True)
    policy_action_dim = 3
    hidden_dim=1024
    n_hidden=5
    actor = DeterministicPolicy(state_dim, policy_action_dim, max_action, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)

    policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/BC_oversampling_halftime_important_values_BIGDATA_5_layer/chp_BC_actor_499999.pth"
    actor_weights=torch.load(policy_file)
    actor.load_state_dict(actor_weights)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    
    for t in range(int(1000000)):
      batch_size=replay_buffer.size()
      batch = replay_buffer.sample(batch_size)
      batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
      state, action = fix_batch(batch)
      batch[0] = state
      batch[1] = action
      action = action.to(device)
      vf_file="/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_vf_3999999.pth"
      plot_vf_heatmap(state, action, actor, actor_optimizer, t, vf_file)
      return
      
def plot_data():
    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer(file_path)
    print("buf size:", replay_buffer.size())
    max_action = 1.0
    name = name_use
    checkpoints_path = os.path.join("./checkpoints/", name)
    print(f"Checkpoints path: {checkpoints_path}")
    os.makedirs(checkpoints_path, exist_ok=True)
    policy_action_dim = 3
    hidden_dim=1024
    n_hidden=5
    actor = DeterministicPolicy(state_dim, policy_action_dim, max_action, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    batch_size=replay_buffer.size()
    batch = replay_buffer.sample(batch_size)
    batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
    state, action = fix_batch(batch)
    batch[0] = state
    batch[1] = action
    state = state.to(device)
    action = action.to(device)
    print(batch[4].mean())
    test_individual_states(state, action, actor, actor_optimizer, 1)
    #test_individual_states_y(state, action, actor, actor_optimizer, 1)

              
if __name__ == "__main__":
  #train()
  #train_fourier_BC()
  #max_values = fix_peg_pos_state(torch.tensor([[0.135, 0.005, 1.067, 0.0, 0.0, 0.0, 1.0]]))[0]
  #min_values = fix_peg_pos_state(torch.tensor([[0.125, -0.005, 1.040, 0.0, 0.0, 0.0, 1.0]]))[0]
  #print(max_values)
  #print(fix_peg_pos_state(torch.tensor([[0.13, 0.00, 1.065, 0.0, 0.0, 0.0, 1.0]]))[0])
  #print(min_values)
  """
  tensor([[0.5050, 0.0050, 0.0270]])
  tensor([[ 5.0000e-01, -8.0718e-11,  2.5000e-02]])
  tensor([[ 0.4950, -0.0050,  0.0000]])
  """
  #small_train()
  plot_samples()
  #plot_data()            