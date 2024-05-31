from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np
import wandb
#from torch.quaternion import as_quat_array, as_float_array
#import warp as wp
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/')
sys.path.append('../../isaacgym/python/isaacgym')#"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
sys.path.append('../../../isaacgym/python/isaacgym')
#for path in sys.path:
#    print(path)
#print(sys.path)

#from IL_RRT_Star import *#ReplayBuffer, fill_buffer #export LD_LIBRARY_PATH=/common/home/jhd79/.conda/envs/RL/lib:$LD_LIBRARY_PATH
#import torch_utils
#from isaacgym import gymapi, gymtorch, torch_utils
#from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
#from isaacgymenvs.tasks.factory.factory_schema_config_task import (FactorySchemaConfigTask,)
#import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
#from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
#from isaacgymenvs.utils.torch_jit_utils import *

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

#import matplotlib.pyplot as plt
#from IPython.display import clear_output

from CORL.algorithms.offline.td3_bc import Critic, TD3_BC #, Actor

buffer_size = 10000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self._initialize_weights

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use 'relu' for nonlinearity as it's a common choice for ELU as well
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.max_action * self.net(state)
        #action = self.net(state)
        #action[0:3] *= self.max_action
        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
        
class BC_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, internal_layer: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, internal_layer),
            nn.ReLU(),
            nn.Linear(internal_layer, int(internal_layer/2)),
            nn.ReLU(),
            nn.Linear(int(internal_layer/2), int(internal_layer/4)),
            nn.ReLU(),
            nn.Linear(int(internal_layer/4), action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self._initialize_weights

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use 'relu' for nonlinearity as it's a common choice for ELU as well
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.max_action * self.net(state)
        #action = self.net(state)
        #action[0:3] *= self.max_action
        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

@dataclass
class TrainConfig:
    """ Training config for Machine Learning """
    alpha: float = 0.000001
    batch_size: int = 256
    buffer_size: int = 10000000
    checkpoints_path: str = "./checkpoints/"
    device: str = "cuda"
    discount: float = 0.99
    env: str = "peg_insertion"
    eval_freq: int = 5000
    expl_noise: float = 0.1
    group: str = "td3-bc-peg-insertion"
    load_model: str = ''
    max_timesteps: int = 1000000
    n_episodes: int = 10
    name: str = "TD3-BC"
    noise_clip: float = 0.5
    normalize: bool = True
    normalize_reward: bool = False
    policy_freq: int = 1
    policy_noise: float = 0.2
    project: str = "CORL"
    seed: int = 0
    tau: float = 0.005

"""    
@dataclass
class BCConfig:
    # Training config for Machine Learning
    lr: float = 0.0003
    batch_size: int = 256
    buffer_size: int = 10000000
    checkpoints_path: str = "./checkpoints/"
    max_timesteps: int = 1000000
    internal_layer: int = 512
""" 

def fill_buffer():
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    with open('./using_planner/interpolated_RRT_Star_4.txt', 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
            # Splitting the data into respective parts
            peg_state = torch.tensor(data[0:7]).to(device)
            hole_state = torch.tensor(data[7:14]).to(device)
            next_peg_state = torch.tensor(data[14:21]).to(device)
            action = torch.tensor(data[21:28]).to(device)
            reward = torch.tensor([data[28]]).to(device)
            done = torch.tensor([data[29]]).to(device)

            current_state = torch.cat((peg_state, hole_state))
            next_state = torch.cat((next_peg_state, hole_state))

            # Adding to the replay buffer
            replay_buffer.add(current_state, next_state, action, reward, done)
    
    return replay_buffer
    
class ReplayBuffer:

    def __init__(self, state_dim: int, action_dim: int, device: str):

        self._pointer = 0
        self._size = 0

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device

    def add(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):

        index = self._pointer
        self._states[index] = state.clone().cpu()
        self._next_states[index] = next_state.clone().cpu()
        self._actions[index] = action.clone().cpu()
        self._rewards[index] = reward.clone().cpu()
        self._dones[index] = done.clone().cpu()

        self._pointer += 1
        self._size = self._size + 1

    def sample(self, batch_size: int) -> List[torch.Tensor]:
    
        if batch_size > self._size:
            batch_size = self._size  # Set batch_size to the size of the buffer if it's greater
        indices = np.random.randint(0, self._size, size = batch_size)
        return [self._states[indices], self._next_states[indices], self._actions[indices], self._rewards[indices], self._dones[indices]]

    def split_buffer(self, train_frac=0.8):

        split_index = int(self._size * train_frac)
        train_indices = np.random.choice(self._size, size=split_index, replace=False)
        val_indices = np.setdiff1d(np.arange(self._size), train_indices)

        train_data = [self._states[train_indices], self._next_states[train_indices], 
                      self._actions[train_indices], self._rewards[train_indices], 
                      self._dones[train_indices]]
        
        val_data = [self._states[val_indices], self._next_states[val_indices], 
                    self._actions[val_indices], self._rewards[val_indices], 
                    self._dones[val_indices]]

        return train_data, val_data

    def size(self):
        return self._size


@pyrallis.wrap()
def train_from_repo(config: TrainConfig):
    
    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("buf size:", replay_buffer.size())

    #positions, actions = extract_actions_and_positions(replay_buffer)
    #plot_3d_vector_field(positions, actions, 'replay_buffer_vector_field_4.png', sampling_rate=0.005, vector_length=0.005)

    # Set seeds
    seed = config.seed
    #set_seed(seed)

    max_action = 0.01

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-3)
    critic_2 = Critic(state_dim, action_dim).to(device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-3)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    #print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    #wandb_init(asdict(config))

    for t in range(int(config.max_timesteps)):
        if t % 1000 == 0: print("train timestep", t)
        batch = replay_buffer.sample(config.batch_size)
        batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
        #Convert to axis angles:
        angle, axis = quat_to_angle_axis(batch[1]) #angle, axis output
        batch[1] = angle * axis
        
        #batch[1][:, 0:3] *= 10000.0
        #batch[1][:, 3:6] *= 100.0
        #batch = [batch[0], batch[2], batch[3], batch[1], batch[4]]
        batch = [b.to(device) for b in batch]
        #It seems to be the case here that the state input is 256x1 instead of 256(which is the batch size) x 14(which is the state size)
        state, action, reward, next_state, done = batch
        #print(state.shape, next_state.shape) #need to change this in the CORL to be how our replay buffer is which is [state, next_state, action, reward, done]
        log_dict = trainer.train(batch) #TRAINING HERE
        #wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode, Cannot actually evaluate mid-step here because we don't have access to the actual environment to test the policy on?
        if (t + 1) % config.eval_freq == 0:
            print("save time")
            if config.checkpoints_path is not None:
                print("now saving here")
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

max_action=0.01
class Loaded_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Loaded_Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        print("state:", state)
        out = self.net(state) #self.max_action * 
        print("out")
        out = out.to(device)
        return out

def test_accuracy():
  trainer = Loaded_Actor(state_dim, action_dim, max_action)
  trainer = trainer.to(device)
  policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_234999.pt"
  print(f"Loading model from {policy_file}")
  actor_weights=torch.load(policy_file)['actor']
  trainer.load_state_dict(actor_weights)
  file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
  replay_buffer = fill_buffer()
  for t in range(10):
    batch = replay_buffer.sample(1)
    state, next_state, action, reward, done = batch
    pi = trainer(state)
    pos_actions = pi[:, :3]
    rot_actions = pi[:, 3:]
    
    pos_action_should = action[:,:3]
    quat_action_should = action[:,3:7]
    print("pos_actions", pos_actions)
    print("rot_actions", rot_actions)
    print("pos_action_should", pos_action_should)
    print("quat_action_should", quat_action_should)

""" 
# BC --------------------------------------------------------------------------------------------------------------------------#

@pyrallis.wrap()
def train_BC(config: BCConfig):
    print(config.max_timesteps)
    return
    #Here we run iterations number of loops on Actor defined here
    max_action = 1.0
    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("buf size:", replay_buffer.size())
    actor = BC_Actor(state_dim, action_dim, max_action, config.internal_layer).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
      
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
           
    iterations=config.max_timesteps
    for i in range(iterations):
      batch = replay_buffer.sample(config.batch_size)
      _, loss=train_iterative_BC(actor, actor_optimizer, batch)
      if i % 100000 == 0:
        print(i, loss)
    if config.checkpoints_path is not None:
      print("now saving here")
      torch.save(
          actor.state_dict(),
          os.path.join(config.checkpoints_path, f"checkpoint_BC.pt"),
      )
  
#To iteratively train a behavior cloning
def train_iterative_BC(actor, actor_optimizer, batch):

        log_dict = {}

        state, next_state, action, reward, done = batch
        #next_state = normalize_states(next_state, self.actor.state_mean, self.actor.state_std)

        # Compute actor loss 
        pi = actor(state)
        #The actions within IsaacGym is usually -1 to 1 then scaled by 0.01 right before usage. In the data they are already scaled down except for the last value.
        #Scale them back to -1 to 1 here for training.
        action[:, 0:3] *= 100.0
        action[:, 3:6] *= 100.0
        actor_loss = F.mse_loss(pi, action).mean()
        actor_loss = actor_loss

        log_dict["train_loss"] = actor_loss.item()
        #plot_data['train_loss'].append(actor_loss.item())

        # Optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Clip gradients: gradients are modified in-place
        clip_grad_norm_(actor.net.parameters(), max_norm=1.0)

        #for name, param in actor.named_parameters():
        #    if param.grad is not None:
        #        check_nan(param.grad, f"Gradient of {name}")

        actor_optimizer.step()

        return log_dict, actor_loss.mean()

# BC --------------------------------------------------------------------------------------------------------------------------#
""" 
def coordinator(inp : str):
    if inp == "BC":
      cfg = pyrallis.parse(config_class=BCConfig)
      train_BC()
    elif inp == "TD3-BC":
      cfg = pyrallis.parse(config_class=TrainConfig)
      train_from_repo()
    elif inp == "test":
      test_accuracy()
    return

if __name__ == "__main__":
      #python train_model.py --config_path=some_config.yaml
      #test_accuracy()
      train_from_repo()
      #train_BC()