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

from utilities import fill_buffer, fix_batch, get_scaled_quaternion, quat_diff_rad

buffer_size = 10000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
 
class BC_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, internal_layer: int):
        super(BC_Actor, self).__init__()

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
        action = self.net(state)
        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

    
@dataclass
class BCConfig:
    """ Training config for Machine Learning """
    lr: float = 0.0003
    batch_size: int = 256
    buffer_size: int = 10000000
    checkpoints_path: str = "./checkpoints/"
    max_timesteps: int = 1000000
    internal_layer: int = 512

# BC --------------------------------------------------------------------------------------------------------------------------#

@pyrallis.wrap()
def train_BC(config: BCConfig):
    #Here we run iterations number of loops on Actor defined here
    max_action = 1.0
    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("buf size:", replay_buffer.size())
    policy_action_dim=6 #3 pos + 3 axis angles
    actor = BC_Actor(state_dim, policy_action_dim, max_action, config.internal_layer).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), 3e-4) #lr=config.lr
      
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
           
    iterations=config.max_timesteps
    for i in range(iterations):
      batch = replay_buffer.sample(config.batch_size)
      _, loss=train_iterative_BC(actor, actor_optimizer, batch)
      
      #print(i)
      if i % 100 == 0:
        print(i, loss)
      if i % 100000 == 0 and i != 0:
        print(i, loss)
        if config.checkpoints_path is not None:
          print("now saving here")
          hyp=str(i)+"lr"+str(config.lr)+"_size"+str(config.internal_layer)+"_"
          torch.save(
              actor.state_dict(),
              os.path.join(config.checkpoints_path, hyp+"checkpoint_BC.pt"),
          )
          print(os.path.join(config.checkpoints_path, hyp+"checkpoint_BC.pt"))
  
#To iteratively train a behavior cloning
def train_iterative_BC(actor, actor_optimizer, batch):

    log_dict = {}
    
    #Just getting robot frame state and axis angles action
    #state, next_state, action, reward, done = batch
    #batch = replay_buffer.sample(config.batch_size)
    batch = [batch[0], batch[2], batch[3], batch[1], batch[4]]
    action_vals=batch[1].clone()
    state, action = fix_batch(batch) #brings to -1 to 1 scale and to axis angles from quaternion
    
    # Compute actor loss
    pi = actor(state)
    #-----------BC term, need to turn axis angles (which we turned to axis angles to scale so network can learn -1 to 1) back to quaternion to make a loss-------------------#
    pi_quat, pi_pos = get_scaled_quaternion(pi)
    #action_quat, action_pos = get_scaled_quaternion(action) #Gets the quaternion from the axis angles which are scaled by the low max action down
    #print(action_quat[0], action_vals[0])
    action_quat=action_vals[:, 3:]
    action_pos=action_vals[:, :3]
    #print(action_quat[0], action_pos[0], action_quat1[0], action_pos1[0]) #works!
    actor_loss_pos = F.mse_loss(pi_pos, action_pos)
    print(pi_pos)
    print(actor_loss_pos)
    actor_loss_rot = quat_diff_rad(pi_quat, action_quat).mean()
    actor_loss = 1e8 * actor_loss_pos + 1e3 * actor_loss_rot #1e8, 1e3
    #-----------BC term, need to turn axis angles (which we turned to axis angles to scale so network can learn -1 to 1) back to quaternion to make a loss-------------------#
    
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

#python using_planner/BC.py --config_path=./using_planner/cfg/BC_config.yaml --max_timesteps=1.25

if __name__ == "__main__":
      cfg = pyrallis.parse(config_class=BCConfig)
      train_BC()
