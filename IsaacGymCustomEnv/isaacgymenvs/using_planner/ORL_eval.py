from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np
#import pyrallis
import wandb
#from torch.quaternion import as_quat_array, as_float_array
#import warp as wp

import sys
sys.path.append('..')
sys.path.append('../..')
#sys.path.append('../../isaacgym/python/isaacgym')#"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
#sys.path.append('../../../isaacgym/python/isaacgym')
#sys.path.append('../../isaacgym/python')#"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
#sys.path.append('../../../isaacgym/python')
print("1")

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.utils.torch_jit_utils import *
#import torch_utils
print("2")
#from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
#from isaacgymenvs.tasks.factory.factory_schema_config_task import (FactorySchemaConfigTask,)
#import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
print("3")
#from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils.torch_jit_utils import *

print(sys.path)

import torch
import torch.nn as nn
import time
#import torch.nn.functional as F
#import torch.nn.init as init
#from torch.nn.utils import clip_grad_norm_
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
max_action=0.01
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

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
    #Member of nn.Module, use .load_state_dict(torch.load(policy_file))

#@pyrallis.wrap()
def eval():
    
    # Set seeds
    #seed = config.seed
    #set_seed(seed)

    trainer = Loaded_Actor(state_dim, action_dim, max_action)
    trainer = trainer.to(device)
    policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_29999.pt"
    print(f"Loading model from {policy_file}")
    actor_weights=torch.load(policy_file)['actor']
    print(actor_weights)
    trainer.load_state_dict(actor_weights)

    for param in trainer.parameters():
        param.requires_grad = False 
    #trainer.actor.eval()

    init_pos = torch.tensor([.127, -0.002, 1.065], dtype=torch.float32, device=device).unsqueeze(0)
    init_quat = torch.tensor([-0.001, 0.005, -0.053, 0.999], dtype=torch.float32, device=device).unsqueeze(0)
    
    hole_pos = torch.tensor([1.3000e-01, 4.3656e-11, 1.0400e+00], dtype=torch.float32, device=device).unsqueeze(0)
    hole_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0)
    
    init_state = torch.cat((init_pos, init_quat), dim=1)
    hole_state = torch.cat((hole_pos, hole_quat), dim=1)
    peg_state = init_state.clone()
    
    d_pos = torch.abs(hole_state[:,3] - peg_state[:,3])
    d_rot = quat_diff_rad(hole_state[:,3:], peg_state[:,3:])
    
    state = torch.cat((peg_state, hole_state), dim=1)
    time.sleep(.1)

    step = 0

    if os.path.exists("learned_RRTStar_5.txt"):
        os.remove("learned_RRTStar_5.txt")
    
    with open('learned_RRTStar_5.txt', 'w') as file:    
        #while d_pos > 0.: # and d_rot >= 0.:
        for i in range(200):
            #print("step: ", step)
            #print("peg_state: ", peg_state)
            #print("hole_state: ", hole_state)

            step += 1

            peg_state = state[:,0:7]
            hole_state = state[:,7:14]
            state=state.to(device)

            # Compute actor loss 
            pi = trainer(state)
            #print(pi)
            
            pos_actions = pi[:, :3]
            rot_actions = pi[:, 3:]
    
            scale_factor_pos = max_action * torch.ones(pos_actions.size(1), device='cuda:0')
            scale_factor_rot = max_action * torch.ones(rot_actions.size(1), device='cuda:0')
            diag_w_pos = torch.diag(scale_factor_pos)
            diag_w_rot = torch.diag(scale_factor_rot)
            
            # Scale the rotational actions
            pos_actions = pos_actions @ diag_w_pos
            rot_actions = rot_actions @ diag_w_rot
            print(pos_actions)
    
            #next_state_pred_pos = state[:,:3] + pos_actions 
    
            # Context actions
            #pi_angle = torch.norm(rot_actions, p=2, dim=-1)
            #pi_axis = rot_actions / (pi_angle.unsqueeze(-1) + 1e-8)  # Avoid divide by zero
            #pi_quat = quat_from_angle_axis(pi_angle, pi_axis)
            pi_quat=rot_actions
            #pi_quat = torch.where(pi_angle.unsqueeze(-1).repeat(1, 4) > 1e-08,
            #                                pi_quat,
            #                                torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(rot_actions.size()[0], 1),)

            #The pi_quat and state[:,3:7] need to be the same size
            #print(pi_quat)
            #print(state[:,3:7])
            peg_state = torch.cat((state[:,0:3] + pos_actions, quat_mul(pi_quat, state[:,3:7])), dim=1)
            peg_state_print = peg_state.clone()
            peg_state_print = peg_state_print[:,[0,1,2,6,3,4,5]]

            # Format peg_state_print as a string
            peg_state_str = ' '.join(f'{x:.5f}' for x in peg_state_print.squeeze(0).tolist())
            # Write the formatted string to the file
            file.write(peg_state_str + '\n')

            state = torch.cat((peg_state, hole_state), dim=1)
            
            #print("peg_state: ", state[:,:7])
            #print("hole_state: ", hole_state)
            #print("step: ", step)
            #time.sleep(0.1)

            #print(hole_state[:,:])
            #print(hole_state[:,3])
            d_pos = torch.sum(torch.abs(hole_state[:,:3] - peg_state[:,:3]))
            #print(hole_state[:,3])
            #print(peg_state[:,3])
            d_rot = quat_diff_rad(hole_state[:,3:], peg_state[:,3:])
        
            print("d_pos: ", d_pos)
            print("d_rot: ", d_rot)


    print("Final Position: ", state[:,:3])
    print("Final Rotation: ", state[:,3:7])

    print("Goal Position: ", hole_state[:,:3])
    print("Goal Rotation: ", hole_state[:,3:7])
    
if __name__ == "__main__":
    eval()