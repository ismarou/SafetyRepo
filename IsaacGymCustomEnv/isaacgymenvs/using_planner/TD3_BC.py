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

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

TensorBatch = List[torch.Tensor]

import gym
import wandb
import pyrallis
from utilities import fill_buffer, quat_to_angle_axis, pose_world_to_robot_base, quat_mul, quat_from_angle_axis, quat_diff_rad

buffer_size = 10000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, internal_layer: int = 512):
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
        action = self.net(state)
        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, internal_layer: int = 512):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, internal_layer),
            nn.ReLU(),
            nn.Linear(internal_layer, int(internal_layer/2)),
            nn.ReLU(),
            nn.Linear(int(internal_layer/2), int(internal_layer/4)),
            nn.ReLU(),
            nn.Linear(int(internal_layer/4), 1),
            nn.Tanh(),
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use 'relu' for nonlinearity as it's a common choice for ELU as well
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], 1)
        #print(self.net)
        #print(state_action.shape)
        return self.net(state_action)

@dataclass
class TrainConfig:
    """ Training config for Machine Learning """
    alpha: float = 2.5
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
    policy_freq: int = 2
    policy_noise: float = 0.2
    project: str = "CORL"
    seed: int = 0
    tau: float = 0.004

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def fix_batch(batch):
    state, action, next_state, reward, done = batch
    #--------------------------------------Turn the state into the robot frame
    peg_pos = state[:,0:3]
    peg_quat = state[:,3:7]
    hole_pos = state[:,7:10]
    hole_quat = state[:,10:14]

    peg_pos_ = pose_world_to_robot_base(peg_pos, peg_quat)[0],  # 3
    peg_quat_ = pose_world_to_robot_base(peg_pos, peg_quat)[1],  # 4
    hole_pos_ = pose_world_to_robot_base(hole_pos, hole_quat)[0],  # 3
    hole_quat_ = pose_world_to_robot_base(hole_pos, hole_quat)[1],  # 4

    peg_pos = peg_pos_[0].clone()
    peg_quat = peg_quat_[0].clone()
    hole_pos = hole_pos_[0].clone()
    hole_quat = hole_quat_[0].clone()

    state = torch.cat((peg_pos, peg_quat, hole_pos, hole_quat), dim=1)
    
    #--------------------------------------Create the actions (either by going to next state and recovering action or converting quaternion from the action into axis angles)
    pos_actions = action[:, :3]
    rot_actions = action[:, 3:] #quaternion from the dataset, this is dim of 7
    angle, axis = quat_to_angle_axis(rot_actions) #angle, axis output
    #print(angle.unsqueeze(1).shape)
    #print(axis.shape)
    #Ensure that this is doing the correct conversion
    rot_actions = angle.unsqueeze(1) * axis
    
    #now in axis angles, can all be scaled by 100 to counteract the already scaled down by 0.01
    scale_factor_pos = 100.0 * torch.ones(pos_actions.size(1), device=device)
    scale_factor_rot = 100.0 * torch.ones(rot_actions.size(1), device=device)
    diag_w_pos = torch.diag(scale_factor_pos)
    diag_w_rot = torch.diag(scale_factor_rot)
    # Scale the rotational actions
    pos_actions = pos_actions @ diag_w_pos
    rot_actions = rot_actions @ diag_w_rot
    
    action_out = torch.cat((pos_actions, rot_actions), dim=1)
    
    return state.clone(), action_out.clone()

class TD3_BC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        #print("state", state[0])
        #print("action", action[0])

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            #print("action for targetQ", next_action[0])
            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = 1.5 + (reward) + not_done * self.discount * target_q
            
        # Get current Q estimates
        #print("ground truth action", action[0])
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            q_test = self.critic_1(state, action)
            lmbda = self.alpha / q.abs().mean().detach()
            
            #-----------BC term, need to turn axis angles (which we turned to axis angles to scale so network can learn -1 to 1) back to quaternion to make a loss-------------------#
            pi_quat, pi_pos = self.get_scaled_quaternion(pi)
            action_quat, action_pos = self.get_scaled_quaternion(action)
            actor_loss_pos = F.mse_loss(pi_pos, action_pos)
            actor_loss_rot = quat_diff_rad(pi_quat, action_quat).mean()
            actor_loss_BC = 1e5 * actor_loss_pos + 1e3 * actor_loss_rot #1e8, 1e3
            #-----------BC term, need to turn axis angles (which we turned to axis angles to scale so network can learn -1 to 1) back to quaternion to make a loss-------------------#

            actor_loss = -lmbda * q.mean() + actor_loss_BC
  
            #--------My print--------#
            if self.total_it % (self.policy_freq*1000) == 0:
              #print("output action", pi[0], "\n target q reward output", target_q[0])
              print("What action we outputted", pi[0], q.mean(), "what the action given rewards are", q_test.mean(), action[0])
            if self.total_it % (self.policy_freq*100) == 0:
              print("Difference from BC:", actor_loss_BC.mean())
            if self.total_it % (self.policy_freq*500) == 0:
              print("Average reward from target q:", q.mean(), "current target q with addition of immediate reward", target_q[0], "critic loss", critic_loss.mean())
            #--------My print--------#

            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            p2i = self.actor(state)
            
            #--------My print--------#
            #if self.total_it % (self.policy_freq*600) == 0:
            #  print("updated action", p2i[0])
            #--------My print--------#

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    #Edited in, this is for turning axis angle to quaternion to calculate loss properly
    def get_scaled_quaternion(self, pi):
        #----------------------------------------------------------------------------For scaling actions and turning axis angle to quat
        pos_actions = pi[:, :3]
        rot_actions = pi[:, 3:]
    
        scale_factor_pos = 0.01 * torch.ones(pos_actions.size(1), device=device)
        scale_factor_rot = 0.01 * torch.ones(rot_actions.size(1), device=device)
        diag_w_pos = torch.diag(scale_factor_pos)
        diag_w_rot = torch.diag(scale_factor_rot)
        
        # Scale the rotational actions
        pos_actions = pos_actions @ diag_w_pos
        rot_actions = rot_actions @ diag_w_rot
    
        # Context actions
        pi_angle = torch.norm(rot_actions, p=2, dim=-1)
        pi_axis = rot_actions / (pi_angle.unsqueeze(-1) + 1e-8)  # Avoid divide by zero
        pi_quat = quat_from_angle_axis(pi_angle, pi_axis)
        pi_quat = torch.where(pi_angle.unsqueeze(-1).repeat(1, 4) > 1e-08,
                                        pi_quat,
                                        torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(rot_actions.size()[0], 1),)
        #next_state_pred_quat = quat_mul(pi_quat, state[:,3:7]) #pi_quat is just the quaternion here and we don't have to apply to the state
        return pi_quat, pos_actions
        #----------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]

@pyrallis.wrap()
def train_from_repo(config: TrainConfig):
    
    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()

    max_action = 1.0 #It is important that TD3-BC has max action of 1.0 so it clamps around there because we scale it down only in the IsaacGym
    policy_action_dim = 6 #3 pos, 3 axis angles (not quaternion)
    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    actor = Actor(state_dim, policy_action_dim, max_action).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, policy_action_dim).to(device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, policy_action_dim).to(device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

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
    print("TRAINING")
    #The system's relation to the state is through data which I fix with my function
    #The actions are considered as axis angles so 6 dim values
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
        batch = [batch[0], batch[2], batch[4], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
        state, action = fix_batch(batch)
        batch[0] = state
        batch[1] = action
        #remember that the data is state and next state so 28 values + 1 reward + 1 done. The action is created from these two states. 

        batch = [b.to(device) for b in batch]
        #state, action, reward, next_state, done = batch
        log_dict = trainer.train(batch) #TRAINING HERE

        if (t + 1) % config.eval_freq == 0:
            print("save time")
            if config.checkpoints_path is not None:
                print("now saving here")
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_TD3_Sparse_{t}.pt"),
                )

from CORL.algorithms.offline.iql import DeterministicPolicy

@pyrallis.wrap()
def eval(config: TrainConfig):
    iql = True
    if iql == True:
      print("USING IQL", iql)
      policy = (DeterministicPolicy(14, 6, 1.0, dropout=None))
      policy = policy.to(device)
      policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/IQL_5_layer_vfqfpolicy_laynorm_dense/chp_IQL_actor_399999.pth"
      print(f"Loading model from {policy_file}")
      actor_weights=torch.load(policy_file)#['actor']
      policy.load_state_dict(actor_weights)
      policy.eval()
    else:
      #Added---------------------------------------------------------------------------------------------------------------------------#
      policy = Actor(14, 6, 1.0)
      policy = policy.to(device)
      policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/properly_trained_results/BC_300000lr0.0003_size512_checkpoint.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_999999.pt" #"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/100000lr0.0003_size512_checkpoint_BC.pt""/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/pretty_much_working_BC100000lr0.0003_size512_checkpoint_BC.pt"#
      print(f"Loading model from {policy_file}")
      actor_weights=torch.load(policy_file)#['actor']
      policy.load_state_dict(actor_weights)
      #Added---------------------------------------------------------------------------------------------------------------------------#
      #The eval() function doesn't acutally have the conversions that are correct. The eval() function works with BC because BC just goes down.
    
    init_pos = torch.tensor([.127, -0.002, 1.065 + 0.01], dtype=torch.float32, device=device).unsqueeze(0)
    init_quat = torch.tensor([-0.001, 0.005, -0.053, 0.999], dtype=torch.float32, device=device).unsqueeze(0)
    
    hole_pos = torch.tensor([1.3000e-01, 4.3656e-11, 1.0400e+00], dtype=torch.float32, device=device).unsqueeze(0)
    hole_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0)
    
    init_state = torch.cat((init_pos, init_quat), dim=1)
    hole_state = torch.cat((hole_pos, hole_quat), dim=1)
    peg_state = init_state.clone()
    
    d_pos = torch.abs(hole_state[:,3] - peg_state[:,3])
    d_rot = quat_diff_rad(hole_state[:,3:], peg_state[:,3:])
    
    state = torch.cat((peg_state, hole_state), dim=1)
    
    #------Test
    #pi=policy(state)
    #print(pi)
    #------Test
    time.sleep(.1)

    step = 0

    if os.path.exists("learned_RRTStar_5.txt"):
        os.remove("learned_RRTStar_5.txt")
    
    with open('learned_RRTStar_5.txt', 'w') as file:    
        #while d_pos >= 0.: # and d_rot >= 0.:
        while peg_state[0,2]>=1.043:
            print("step: ", step)
            print("peg_state: ", peg_state)
            print("hole_state: ", hole_state)

            step += 1

            peg_state = state[:,0:7]
            hole_state = state[:,7:14] 

            # Compute actor loss 
            print(state.shape)
            pi = policy(fix_state(state))
            
            pos_actions = pi[:, :3]
            rot_actions = pi[:, 3:]
            max_action=0.01
            scale_factor_pos = max_action * torch.ones(pos_actions.size(1), device=device)
            scale_factor_rot = max_action * torch.ones(rot_actions.size(1), device=device)
            diag_w_pos = torch.diag(scale_factor_pos)
            diag_w_rot = torch.diag(scale_factor_rot)
            
            # Scale the rotational actions
            pos_actions = pos_actions @ diag_w_pos
            rot_actions = rot_actions @ diag_w_rot

            # Context actions
            pi_angle = torch.norm(rot_actions, p=2, dim=-1)
            pi_axis = rot_actions / pi_angle.unsqueeze(-1)
            pi_quat = quat_from_angle_axis(pi_angle, pi_axis)
            
            print("pos actions", pos_actions)

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
            time.sleep(0.1)

            d_pos = torch.abs(hole_state[:,3] - peg_state[:,3])
            d_rot = quat_diff_rad(hole_state[:,3:], peg_state[:,3:])
        
            print("d_pos: ", d_pos)
            print("d_rot: ", d_rot)


    print("Final Position: ", state[:,:3])
    print("Final Rotation: ", state[:,3:7])

    print("Goal Position: ", hole_state[:,:3])
    print("Goal Rotation: ", hole_state[:,3:7])

def fix_state(state):
    #--------------------------------------Turn the state into the robot frame
    peg_pos = state[:,0:3]
    peg_quat = state[:,3:7]
    hole_pos = state[:,7:10]
    hole_quat = state[:,10:14]

    peg_pos_ = pose_world_to_robot_base(peg_pos, peg_quat)[0],  # 3
    peg_quat_ = pose_world_to_robot_base(peg_pos, peg_quat)[1],  # 4
    hole_pos_ = pose_world_to_robot_base(hole_pos, hole_quat)[0],  # 3
    hole_quat_ = pose_world_to_robot_base(hole_pos, hole_quat)[1],  # 4

    peg_pos = peg_pos_[0].clone()
    peg_quat = peg_quat_[0].clone()
    hole_pos = hole_pos_[0].clone()
    hole_quat = hole_quat_[0].clone()

    state = torch.cat((peg_pos, peg_quat, hole_pos, hole_quat), dim=1)
    return state

import matplotlib.pyplot as plt
def plot_policy_function():
    policy = Actor(14, 6, 1.0)
    policy = policy.to(device)
    policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/200000lr0.0003_size512_checkpoint_BC.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/pretty_much_working_BC100000lr0.0003_size512_checkpoint_BC.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/200000lr0.0003_size512_checkpoint_BC.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/pretty_much_working_BC100000lr0.0003_size512_checkpoint_BC.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/max_trained_TD3_BC_Working_999999" #"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/max_trained_TD3_BC_Working_999999"
    print(f"Loading model from {policy_file}")
    actor_weights=torch.load(policy_file)#['actor']
    policy.load_state_dict(actor_weights)
    
    #State is plug_pos, plug_quat, socket_pos, socket_quat, fix everything except the plug_pos's y and z
    y=10.0
    z=10.0
    state_template = torch.tensor([1.3000e-01, y, z, -0.022, -0.023, 0.005, 0.999, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
    
    #So fix the x
    #Then across horizontally have y shift
    #Then on a new list shift z
    #At each value, a value for x, y, u, v
    num_spec=10
    x_starting_point=-0.013 #really is y starting point for the robot
    x_ending_point=0.013
    x_points=np.linspace(x_starting_point, x_ending_point, num_spec)
    y_starting_point=1.075 #really is y starting point for the robot
    y_ending_point=1.045
    y_points=np.linspace(y_starting_point, y_ending_point, num_spec)
    x = [[0 for j in range(num_spec)] for i in range(num_spec)] #y
    y = [[0 for j in range(num_spec)] for i in range(num_spec)] #z
    u = [[0 for j in range(num_spec)] for i in range(num_spec)]
    v = [[0 for j in range(num_spec)] for i in range(num_spec)]
    for xx in range(num_spec):
      for yy in range(num_spec):
        x[xx][yy] = x_points[xx]
        y[xx][yy] = y_points[yy]
        state_inp = state_template.clone()
        state_inp[1] = x_points[xx]
        state_inp[2] = y_points[yy]
        state_inp=fix_state(state_inp.unsqueeze(0))
        print(state_inp)
        action=policy(state_inp) #[x,y,z,rotx,roty,rotz]
        action=action.detach()
        print(action)
        u[xx][yy]=action[:,1].numpy()[0]
        v[xx][yy]=action[:,2].numpy()[0]
        
    print(u)
    print(v)
    #print(x)
    plt.quiver(x, y, u, v, scale=1.0)
    plt.savefig("plot.png")

if __name__ == "__main__":
    #rot_actions = torch.tensor([0.0005, 0.00097, -0.00527, 0.9999997])
    #angle, axis = quat_to_angle_axis(rot_actions) #angle, axis output
    #rot_actions = angle * axis
    #print(rot_actions)
    
    #cfg = pyrallis.parse(config_class=TrainConfig)
    eval()
    #train_from_repo()
    #plot_policy_function()
