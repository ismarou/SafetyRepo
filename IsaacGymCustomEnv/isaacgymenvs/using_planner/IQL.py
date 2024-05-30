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
from utilities import fill_buffer, fix_batch, get_scaled_quaternion, quat_diff_rad, pose_world_to_robot_base

#import matplotlib.pyplot as plt
#from IPython.display import clear_output

from CORL.algorithms.offline.iql import TwinQ, ValueFunction, DeterministicPolicy, GaussianPolicy, ImplicitQLearning, compute_mean_std, normalize_states
from utilities import get_scaled_quaternion, quat_diff_rad, get_euler_xyz, quat_from_euler_xyz, quat_mul
from QBC import get_reward_states

buffer_size = 10000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
rew_type = "sparse"#"dense"#"sparse"
name_use="LowerActionQFixed_1000times_IQL_5_layer_laynorm_larger_data_"

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(2.5e5)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(2e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = "./checkpoints/"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.999  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss, as it approaches 0.5 it approaches being the mean
    iql_deterministic: bool = True  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL"

    def __post_init__(self):
        #self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        self.name = name_use+rew_type
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


    
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
def train(config: TrainConfig):
    #env = gym.make(config.env)

    #state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.shape[0]

    #dataset = d4rl.qlearning_dataset(env)

    """
    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    """
    #Try normalization, higher gamma, deterministic policy

    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer(file_path)
    print("buf size:", replay_buffer.size())

    max_action = 1.0

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    policy_action_dim = 6
    action_dim = 7
    #hidden_dim: int = 1024, n_hidden: int = 5
    hidden_dim=1024
    n_hidden=5
    q_network = TwinQ(state_dim, action_dim, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    
    v_network = ValueFunction(state_dim, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    
    actor = (
        DeterministicPolicy(
            state_dim, policy_action_dim, max_action, dropout=config.actor_dropout, hidden_dim=hidden_dim, n_hidden=n_hidden
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, policy_action_dim, max_action, dropout=config.actor_dropout, hidden_dim=hidden_dim, n_hidden=n_hidden
        )
    ).to(device)
    
    #weights=torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_qf_1999999.pth")
    #q_network.load_state_dict(weights)
    
    #weights=torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_vf_1999999.pth")
    #v_network.load_state_dict(weights)
    
    #weights=torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_actor_1999999.pth")
    #actor.load_state_dict(weights)
    
    print("Is deterministic?: ", config.iql_deterministic)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
        "noise": 0.0,
        "plug_rot_noise": 0.0,
    }

    #print("---------------------------------------")
    #print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    #print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    #wandb_init(asdict(config))

    frequency_sample_reward=100.0
    evaluations = []
    for t in range(int(config.max_timesteps)):
      if t % 1000 == 0: print("train timestep", t)
      
      if frequency_sample_reward != -1 and t % frequency_sample_reward == 0:
        batch = replay_buffer.sample(replay_buffer.size())
      else:
        batch = replay_buffer.sample(config.batch_size)
      
      if rew_type == "dense":
        batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
      else:
        batch = [batch[0], batch[2], batch[4], batch[1], batch[4]]
        batch[2] *= 100.0

      state, action = fix_batch(batch)
      batch[0] = state
      #batch[1] = action #This action is between -0.01 and 0.01 so that it matches scale with state
      batch[3] = fix_state(batch[3]) #these good
      #remember that the data is state and next state so 28 values + 1 reward + 1 done. The action is created from these two states.
      
      if frequency_sample_reward != -1 and t % frequency_sample_reward == 0:
        indices = get_reward_states(batch[0], batch[1])
        if indices != None:
          batch[0] = batch[0][indices]
          batch[1] = batch[1][indices]
          batch[2] = batch[2][indices]
          batch[3] = batch[3][indices]
          batch[4] = batch[4][indices]
          print("rewards in the reward batch", batch[2].mean(), q_network.both(batch[0], batch[1]))
        else:
          continue
        
      #if (t+1) % 10000 == 0:
      #  torch.save(
      #    trainer.state_dict()['actor'],
      #    os.path.join("./checkpoints/"+name_use+rew_type+"/", f"chp_IQL_actor_trained_after_{t}_.pth"),
      #    )

      batch = [b.to(device) for b in batch]
      #print(batch[0].shape)
      log_dict = trainer.train(batch) #Input scaled to -1 to 1 actions and proper states. Should learn these -1 to 1 actions and that is that. In pos, axis angles
      if (t + 1) % config.eval_freq == 0 or t == 0:
          print("save time")
          if config.checkpoints_path is not None:
              print("now saving here")
              torch.save(
                  trainer.state_dict()['qf'],
                  os.path.join("./checkpoints/"+name_use+rew_type+"/", f"chp_IQL_qf_{t}.pth"),
              )
              torch.save(
                  trainer.state_dict()['vf'],
                  os.path.join("./checkpoints/"+name_use+rew_type+"/", f"chp_IQL_vf_{t}.pth"),
              )
              torch.save(
                  trainer.state_dict()['actor'],
                  os.path.join("./checkpoints/"+name_use+rew_type+"/", f"chp_IQL_actor_{t}.pth"),
              )
              
@pyrallis.wrap()
def train_just_policy(config: TrainConfig):

    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer(file_path)
    print("buf size:", replay_buffer.size())

    max_action = 1.0
    
    checkpoint_path="actor_only_"+name_use+rew_type
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    policy_action_dim = 6
    action_dim = 7
    #hidden_dim: int = 1024, n_hidden: int = 5
    hidden_dim=1024
    n_hidden=5
    q_network = TwinQ(state_dim, action_dim, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    
    v_network = ValueFunction(state_dim, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    
    actor = (
        DeterministicPolicy(
            state_dim, policy_action_dim, max_action, dropout=config.actor_dropout, hidden_dim=hidden_dim, n_hidden=n_hidden
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, policy_action_dim, max_action, dropout=config.actor_dropout, hidden_dim=hidden_dim, n_hidden=n_hidden
        )
    ).to(device)
    
    weights=torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_1000times_IQL_5_layer_laynorm_larger_data_sparse/chp_IQL_qf_1749999.pth")
    q_network.load_state_dict(weights)
    
    weights=torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_1000times_IQL_5_layer_laynorm_larger_data_sparse/chp_IQL_vf_1749999.pth")
    v_network.load_state_dict(weights)
    
    #weights=torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_actor_1999999.pth")
    #actor.load_state_dict(weights)
    
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    evaluations = []
    for t in range(int(config.max_timesteps)):
      if t % 1000 == 0: print("train timestep", t)
      batch = replay_buffer.sample(config.batch_size)
      if rew_type == "dense":
        batch = [batch[0], batch[2], batch[3], batch[1], batch[4]] #need to be sadsd, was originally [state, next_state, action, reward, done], want to give sparse rewards so only reward of done
      else:
        batch = [batch[0], batch[2], batch[4], batch[1], batch[4]]
        batch[2] *= 1000.0
      state, action = fix_batch(batch)
      batch[0] = state
      #batch[1] = action #This action is between -0.01 and 0.01 so that it matches scale with state
      batch[3] = fix_state(batch[3]) #these good

      batch = [b.to(device) for b in batch]
      #print(batch[0].shape)
      #log_dict = trainer.train(batch) #Input scaled to -1 to 1 actions and proper states. Should learn these -1 to 1 actions and that is that. In pos, axis angles
      bc_loss, actor_loss_rot, actor_loss_pos = _update_policy(batch[0], batch[1], batch[3], actor, actor_optimizer, q_network, v_network)
      if t % 1000 == 0: print("bc_loss", bc_loss, "rot loss", actor_loss_rot, "pos loss", actor_loss_pos)
      if (t + 1) % config.eval_freq == 0 or t == 0:
          print("save time")
          if config.checkpoints_path is not None:
              print("now saving here")
              torch.save(
                  actor.state_dict(),
                  os.path.join("./checkpoints/"+name_use+rew_type+"/", f"chp_IQL_actor_action_trained_only_{t}.pth"),
              )
              
def _update_policy(observations, actions, next_state, actor, actor_optimizer, q_target, vf):
    with torch.no_grad():
      target_q = q_target(observations, actions).unsqueeze(-1) #shape is [batch, 1]
      v = vf(observations) #shape is [batch, 1]          
      adv = target_q - v
    
    policy_out=actor(observations)
    exp_adv = torch.exp(3.0 * adv).clamp(max=100.0) #e to the power of the scaled advantage of the action value versus the value function value
    pi_quat, pi_pos = get_scaled_quaternion(policy_out, 0.01)
    action_pos=actions[:,0:3]
    action_quat=actions[:,3:7]
    next_state_pred_pos = observations[:,0:3] + pi_pos
    next_state_pred_quat = quat_mul(observations[:,3:7], pi_quat)
    
    #actor_loss_pos = F.mse_loss(next_state_pred_pos, next_state[:,0:3])
    actor_loss_pos = F.cosine_similarity(pi_pos, action_pos, dim=1)
    actor_loss_pos = -1.0 * torch.mean(actor_loss_pos)
    
    actor_loss_rot = quat_diff_rad(next_state_pred_quat, next_state[:,3:7]).mean()
    
    bc_losses = 1 * actor_loss_pos + 1e1 * actor_loss_rot
    policy_loss = torch.mean(exp_adv * bc_losses) #so its a Q/V advantage term where it pretty much learns directly from the Q and behavior cloning term
    actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100.0)
    actor_optimizer.step()
    #self.actor_lr_schedule.step()
    return bc_losses.mean(), actor_loss_rot, actor_loss_pos
              
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
def plot_value_function():
    vf = ValueFunction(state_dim, hidden_dim=1024, n_hidden=5)
    vf_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_vf_3999999.pth"
    print(f"Loading model from {vf_file}")
    print(torch.load(vf_file).keys())
    print(torch.load(vf_file))
    #print(torch.load(vf_file)['actor'])
    vf_weights=torch.load(vf_file)#['vf']
    vf.load_state_dict(vf_weights)
    
    #So fix the x
    #Then across horizontally have y shift
    #Then on a new list shift z
    #I want to print the direction of the movement from the policy at each point. 
    #state_template = torch.tensor([1.3100e-01, y, z, -0.022, -0.023, 0.005, 0.999, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
    
    #So fix the x
    #Then across horizontally have y shift
    #Then on a new list shift z
    #At each value, a value for x, y, u, v
    num_spec=100
    x_starting_point=-0.01 #really is y starting point for the robot
    x_ending_point=0.01
    x_points=np.linspace(x_starting_point, x_ending_point, num_spec)
    #print("xpoinst", x_points)
    y_starting_point=1.075 #really is y starting point for the robot
    y_ending_point=1.040
    y_points=np.linspace(y_starting_point, y_ending_point, num_spec)
    #print("y points", y_points)
    values = [[0 for j in range(num_spec)] for i in range(num_spec)]
    #print("values size", np.asarray(values).shape)
    for xx in range(num_spec): #y
      for yy in range(num_spec): #z
        state_inp = torch.tensor([1.3000e-01, float(x_points[xx]), float(y_points[yy]), 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
        print("state inp original", state_inp)
        state_inp=fix_state(state_inp.unsqueeze(0))
        print("state inp fixed", state_inp)
        v=vf(state_inp)
        v=v.detach()
        print("value", v, "y value", x_points[xx], "z value", y_points[yy])
        values[yy][xx]=v.numpy()[0]
        #if x_points[xx] > -0.005 and x_points[xx] < 0.005:
        #  values[yy][xx] = v.numpy()[0] * 0.0
        print(values[yy][xx])
    
    values=np.asarray(values)
    #values=(values-np.min(values)) / (np.max(values)-np.min(values))
    #for xx in range(num_spec): #y
    #  for yy in range(num_spec): #z
    #    print(values[xx][yy])
    X, Y = np.meshgrid(x_points, y_points)
    plt.scatter(X, Y, c=values, cmap='viridis', s=2)
    #plt.imshow(values, cmap='viridis', interpolation='nearest')#, extent=[x_starting_point, x_ending_point, y_starting_point, y_ending_point], vmin=x_starting_point, vmax=x_ending_point)
    plt.colorbar()
    plt.savefig("plot_heatmap.png")

#The idea is that we only look at very straight up poses then for nearest neighbor we plot their actual location on the x/y z plot
def plot_nearest_neighbor_data_policy(use_x=True, in_original_spot=False, se3_dist=False):
    file_path = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_rectangular_16mm_0001_Brief_Really_Biased_Edgar_Test.txt"
    replay_buffer = fill_buffer()
    states = replay_buffer._states[0:replay_buffer.size(), :]
    actions = replay_buffer._actions[0:replay_buffer.size(), :]
    y=0.0
    z=1.065
    state_template = torch.tensor([0.13, y, z, 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
    #remove anything too far off the straight up rotation
    state_sized = state_template.unsqueeze(0).repeat(replay_buffer.size(), 1) #[batch, state]
    rot_differences = quat_diff_rad(states[:, 3:7], state_sized[:, 3:7]).unsqueeze(-1) #[batch]
    print(rot_differences)
    mask = torch.all(rot_differences <= 0.25, dim=1) #0.01 is too little
    indices = torch.nonzero(mask).squeeze()
    states = states[indices, :]
    actions = actions[indices, :]
    print(states.shape[0])
    
    num_spec=30
    if use_x == False:
      x_starting_point=-0.005#0.125#-0.01 #really is y starting point for the robot
      x_ending_point=0.005#0.135#0.01
    else:
      x_starting_point=0.125#-0.01 #really is y starting point for the robot
      x_ending_point=0.135#0.01
    x_points=np.linspace(x_starting_point, x_ending_point, num_spec)
    y_starting_point=1.07 #really is y starting point for the robot
    y_ending_point=1.040
    y_points=np.linspace(y_starting_point, y_ending_point, num_spec)
    x = [[0 for j in range(num_spec)] for i in range(num_spec)] #y
    y = [[0 for j in range(num_spec)] for i in range(num_spec)] #z
    u = [[0 for j in range(num_spec)] for i in range(num_spec)]
    v = [[0 for j in range(num_spec)] for i in range(num_spec)]
    alpha = [[0 for j in range(num_spec)] for i in range(num_spec)]
    for xx in range(num_spec):
      for yy in range(num_spec):
        x[xx][yy] = x_points[xx]
        y[xx][yy] = y_points[yy]
        state_inp = state_template.clone()
        if use_x == False:
          state_inp[1] = x_points[xx]
        else:
          state_inp[0] = x_points[xx]
        state_inp[2] = y_points[yy]
        #state_inp=fix_state(state_inp.unsqueeze(0))
        #action=policy(state_inp) #[x,y,z,rotx,roty,rotz]
        #action=action.detach()
        
        #states are [batch, 14], in the original 0.13 frame
        #dist across dim=0
        state_inp = state_inp.unsqueeze(0).repeat(states.shape[0], 1)
        #diff_pos = torch.sum(torch.pow(states[:, 0:3] - state_inp[:, 0:3], 2), dim=1)
        diff_pos = torch.norm(states[:, 0:3] - state_inp[:, 0:3], p=2, dim=1)
        diff_rot = 1e-5 * quat_diff_rad(states[:, 3:7], state_inp[:, 3:7]) #1-e5 works well
        d = diff_pos+diff_rot
        ind = torch.min(diff_pos+diff_rot, dim=0).indices #dim to reduce, returns indices
        print(ind)
        #print(diff_pos, diff_rot, diff_pos + diff_rot)
        #print(torch.min(diff_rot, dim=0))
        action=actions[ind].unsqueeze(0)
        #print(states[ind, 3:7].shape)
        #print(state_inp[0, 3:7].shape)
        #print(torch.norm(states[ind, 0:3].unsqueeze(0) - state_inp[ind, 0:3].unsqueeze(0), p=2, dim=1).shape)
        if se3_dist == True:
          alpha[xx][yy] = 1e-2 * quat_diff_rad(states[ind, 3:7].unsqueeze(0), state_inp[0, 3:7].unsqueeze(0)).numpy() + torch.norm(states[ind, 0:3].unsqueeze(0) - state_inp[ind, 0:3].unsqueeze(0), p=2, dim=1).numpy()
        else:
          alpha[xx][yy] = quat_diff_rad(states[ind, 3:7].unsqueeze(0), state_inp[0, 3:7].unsqueeze(0)).numpy()
        print(alpha[xx][yy])
        
        #Here set them actually to the location found in the data
        if in_original_spot == True:
          tar_dim=0
          if use_x == False:
            tar_dim=1
          x[xx][yy] = states[ind, tar_dim]
          y[xx][yy] = states[ind, 2]
        
        if use_x == False:
          if (x_points[xx] < 0.001 and x_points[xx] > -0.001) or y_points[yy] > 1.06:
            u[xx][yy]=action[:,1].numpy()[0]
            v[xx][yy]=action[:,2].numpy()[0]
            maximum=np.abs(u[xx][yy])
            if np.abs(v[xx][yy]) > maximum:
              maximum=np.abs(v[xx][yy])
            u[xx][yy] /= maximum
            v[xx][yy] /= maximum
          else:
            u[xx][yy]=0.0
            v[xx][yy]=0.0
        else:
          if (x_points[xx] < 0.131 and x_points[xx] > 0.129) or y_points[yy] > 1.06:
            u[xx][yy]=action[:,0].numpy()[0]
            v[xx][yy]=action[:,2].numpy()[0]
            maximum=np.abs(u[xx][yy])
            if np.abs(v[xx][yy]) > maximum:
              maximum=np.abs(v[xx][yy])
            u[xx][yy] /= maximum
            v[xx][yy] /= maximum
          else:
            u[xx][yy]=0.0
            v[xx][yy]=0.0
        
     
    maximum=0.0
    for xx in range(num_spec):
      for yy in range(num_spec):
        if alpha[xx][yy] > maximum:
          maximum=alpha[xx][yy]
    print("maximum", maximum)
    for xx in range(num_spec):
      for yy in range(num_spec):
        alpha[xx][yy] /= maximum
    print(alpha)
    #print(u)
    #print(v)
    plt.quiver(x, y, u, v, alpha=alpha, scale=40.0)#0.0025)
    #plt.gca().set_aspect('equal', adjustable='box')
    if use_x == False:
      plt.savefig("dataNN_vector_field_plot_y.png")
    else:
      plt.savefig("dataNN_vector_field_plot_x.png")
    plt.clf()

from QBC import state_sensitivity_fixing
   
def plot_policy_function(policy_file,use_x=False,use_Fourier=False):
    if use_Fourier == False:
      policy = DeterministicPolicy(14, 6, 1.0, hidden_dim=1024, n_hidden=5)
      #policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_no_noise_sparse/chp_IQL_actor_1999999.pth"
    else:
      policy = DeterministicFourierPolicy(14, 6, 1.0, hidden_dim=1024, n_hidden=5)
      #policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/FourierBC_5_layer/chp_FBC_actor_79999.pth"
    print(f"Loading model from {policy_file}")
    actor_weights=torch.load(policy_file)
    policy.load_state_dict(actor_weights)
    if use_Fourier == True:
      B_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/FourierBC_5_layer/chp_FBC_B.pth"
      B_matrix=torch.load(B_file)
      policy.B = B_matrix
    
    #State is plug_pos, plug_quat, socket_pos, socket_quat, fix everything except the plug_pos's y and z
    y=0.0
    z=1.065
    state_template = torch.tensor([0.13, y, z, 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])
    
    #use_x = True
    
    #So fix the x
    #Then across horizontally have y shift
    #Then on a new list shift z
    #At each value, a value for x, y, u, v
    num_spec=30
    if use_x == False:
      x_starting_point=-0.005#0.125#-0.01 #really is y starting point for the robot
      x_ending_point=0.005#0.135#0.01
    else:
      x_starting_point=0.125#-0.01 #really is y starting point for the robot
      x_ending_point=0.135#0.01
    x_points=np.linspace(x_starting_point, x_ending_point, num_spec)
    y_starting_point=1.07 #really is y starting point for the robot
    y_ending_point=1.040
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
        if use_x == False:
          state_inp[1] = x_points[xx]
        else:
          state_inp[0] = x_points[xx]
        state_inp[2] = y_points[yy]
        state_inp=fix_state(state_inp.unsqueeze(0))
        #print(state_inp)
        #state_inp=state_sensitivity_fixing(state_inp)
        action=policy(state_inp) #[x,y,z,rotx,roty,rotz]
        action=action.detach()
        #print("state", state_inp, "action", action)
        #print(action)
        if use_x == False:
          if (x_points[xx] < 0.001 and x_points[xx] > -0.001) or y_points[yy] > 1.06:
            u[xx][yy]=action[:,1].numpy()[0]
            v[xx][yy]=action[:,2].numpy()[0]
            maximum=np.abs(u[xx][yy])
            if np.abs(v[xx][yy]) > maximum:
              maximum=np.abs(v[xx][yy])
            #magnitude=np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2)
            #print(u[xx][yy], "divided by", np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2), u[xx][yy] / np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2))
            u[xx][yy] /= maximum# magnitude#divide by the average of the total vector's magnitude, divided by the same value the direction is still saved, the loss would be divided self for the action
            v[xx][yy] /= maximum# magnitude
          else:
            u[xx][yy]=0.0
            v[xx][yy]=0.0
        else:
          if (x_points[xx] < 0.131 and x_points[xx] > 0.129) or y_points[yy] > 1.06:
            u[xx][yy]=action[:,0].numpy()[0]
            v[xx][yy]=action[:,2].numpy()[0]
            maximum=np.abs(u[xx][yy])
            if np.abs(v[xx][yy]) > maximum:
              maximum=np.abs(v[xx][yy])
            #print(u[xx][yy], "divided by", np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2), u[xx][yy] / np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2))
            u[xx][yy] /= maximum #magnitude #divide by the average of the total vector's magnitude, divided by the same value the direction is still saved, the loss would be divided self for the action
            v[xx][yy] /= maximum
          else:
            u[xx][yy]=0.0
            v[xx][yy]=0.0
        
    #What this should be is all arrows the same length. So 
        
    #print(u)
    #print(v)
    #av1 = (torch.abs(torch.tensor(u)).mean(dim=1)).unsqueeze(-1)
    #av2 = (torch.abs(torch.tensor(v)).mean(dim=1)).unsqueeze(-1)
    #u = (torch.tensor(u) / av1).numpy()
    #v = (torch.tensor(v) / av2).numpy()
    #For each batch value, divide it by the average value between all this to maintain the relative scale but bring all to the same scale
    #print(x)
    plt.quiver(x, y, u, v, scale=40.0)#0.0025)
    #plt.gca().set_aspect('equal', adjustable='box')
    if use_x == False:
      plt.savefig("vector_field_plot_y.png")
    else:
      plt.savefig("vector_field_plot_x.png")
    plt.clf()
        
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
        

if __name__ == "__main__":
    #train()
    #train_just_policy()
    plot_value_function()
    plot_nearest_neighbor_data_policy(use_x=True, se3_dist=True)
    plot_nearest_neighbor_data_policy(use_x=False, se3_dist=True)
    #policy_file="/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_IQL_5_layer_laynorm_larger_data_dense/chp_IQL_actor_action_trained_only_499999.pth"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/DotProduct_BC_BIGDATA_5_layer/chp_BC_actor_999999.pth"
    #plot_policy_function(policy_file, False, False)
    #plot_policy_function(policy_file, True, False)
