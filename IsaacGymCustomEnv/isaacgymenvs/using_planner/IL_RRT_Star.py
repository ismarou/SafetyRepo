from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np
import pyrallis
import wandb
#from torch.quaternion import as_quat_array, as_float_array
#import warp as wp

import sys
sys.path.append('..')
sys.path.append('../../isaacgym/python/isaacgym')#"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
sys.path.append('../../../isaacgym/python/isaacgym')
sys.path.append('../../isaacgym/python')#"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
sys.path.append('../../../isaacgym/python')

#from isaacgym import torch_utils#gymapi, gymtorch, torch_utils
import torch_utils
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

import matplotlib.pyplot as plt
from IPython.display import clear_output

l_w_tr = 1
l_w_rot = 1 

# Compute Loss
l_w_pos_0 = 1e8
l_w_rot_0 = 1e3

# Create the replay buffer
state_dim = 7 + 7  # 3D position + 4D quaternion for peg and hole
action_dim = 7  # Assuming 7D action space (3 for position and 4 for quaternion orientation)


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)

def extract_actions_and_positions(replay_buffer):
    actions = replay_buffer._actions[:, :3].cpu().numpy()  # Extract the first 3 components (3D action components)
    positions = replay_buffer._states[:, :3].cpu().numpy()  # Extract the first 3 components (3D position components)
    return positions, actions

def plot_3d_vector_field(positions, actions, save_path, sampling_rate=0.00000001, vector_length=0.01):
    # Sparsely sample the data
    num_samples = int(len(actions) * sampling_rate)
    indices = np.random.choice(len(actions), num_samples, replace=False)
    sampled_positions = positions[indices]
    sampled_actions = actions[indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extracting components for clarity
    x, y, z = sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2]
    u, v, w = sampled_actions[:, 0], sampled_actions[:, 1], sampled_actions[:, 2]

    # Plotting the vector field
    ax.quiver(x, y, z, u, v, w, length=vector_length, normalize=True)

    # Setting the labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Vector Field of Actions in Replay Buffer')

    # Scaling the axes
    ax.auto_scale_xyz([np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)])

    plt.savefig(save_path)

# Add sample_batch function to sample a batch of data
def sample_batch(data, batch_size):
    indices = np.random.randint(0, len(data[0]), size=batch_size)
    return [d[indices] for d in data]
    

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")


@torch.jit.script
def quat_diff_rad_Custom(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    return 2 * torch.acos(torch.abs(mul[:, -1]))
    # return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 0:3], p=2, dim=-1), max=1.0))

# Set max_action to 0.001
max_action = 0.01
max_action_transl = max_action
max_action_rot = max_action


'''
def live_plot(data_dict, figsize=(10,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('BC_Custom_RRT_Star_4.png')
    plt.show()
'''

def live_plot(data_dict, figsize=(10,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)

    # Plot training loss
    plt.plot(data_dict['train_loss'], label='Train Loss')

    # Plot validation loss
    plt.plot(data_dict['val_loss'], label='Validation Loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('BC_Custom_RRT_Star_Train_Val_Billion_4.png')
    plt.show()


TensorBatch = List[torch.Tensor]

plot_data = {
                'train_loss': [],
                'val_loss': [],  
                'sigma1': [],
                'sigma2': [],
                'actor_loss_pos': [],
                'actor_loss_rot': []
                                                }



def convert_to_same_base(value1, value2):
    """
    Convert two float numbers to the same base exponent format.
    
    Args:
    value1 (float): The first number.
    value2 (float): The second number.

    Returns:
    tuple: A tuple containing the two numbers in the same base exponent format.
    """
    import math

    # Extracting the exponents of the two numbers
    exponent1 = math.floor(math.log10(abs(value1)))
    exponent2 = math.floor(math.log10(abs(value2)))

    # Finding the difference in exponents
    exponent_diff = exponent1 - exponent2

    # Adjusting the numbers to the same base
    converted_value1 = value1 * (10 ** -exponent_diff)
    converted_value2 = value2

    return converted_value1, converted_value2

@dataclass
class TrainConfig:
    # Experiment settings
    device: str = "cuda:0"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    max_timesteps: int = int(1e9)  # Max time steps to run environment
    checkpoints_path: str = "./models/bc"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    gamma: float = 0.99  # Discount factor
    normalize: bool = True  # Normalize states
    
    # Wandb logging
    project: str = "CORL"
    group: str = "BC-CUSTOM"
    name: str = "Custom_BC"
    lambda_weight: float = 0.5  # Weighting parameter for the loss function


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
                config=config,
                project=config["project"],
                group=config["group"],
                name=config["name"],
                id=str(uuid.uuid4()),
                                            )
    wandb.run.save()




def keep_best_trajectories(
                            dataset: Dict[str, np.ndarray],
                            frac: float,
                            gamma: float,
                            max_episode_steps: int = 1000,
                                                                            ):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= gamma
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    theta = torch.acos(cos)
    #theta = torch.min(theta, 2*np.pi - theta)    
    return theta



# Initialize the replay buffer
buffer_size = 10000000  # Set your desired buffer size
state_dim = 7 + 7  # Fixed state dimension
state_dim_net = 14  # Fixed state dimension for NN
action_dim = 7  # Fixed action dimension
# for 6D Continuous Rotation Action Space:
action_dim_net = 6  # Fixed action dimension

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Set max_action to 0.001
max_action = 0.01
max_action_transl = max_action
max_action_rot = max_action


def pose_world_to_robot_base(pos, quat):
    
    """Convert pose from world frame to robot base frame."""

    robot_base_pos = torch.tensor([-3.7000e-01,  4.3656e-11,  1.0400e+00], dtype = torch.float32, device = device ).unsqueeze(0)
    robot_base_quat = torch.tensor([-2.2582e-11,  6.2217e-10,  3.5933e-11,  1.0000e+00], dtype=torch.float32, device = device).unsqueeze(0)

    robot_base_transform_inv = torch_utils.tf_inverse(robot_base_quat, robot_base_pos)
    robot_base_transform_inv_quat = robot_base_transform_inv[0].repeat(pos.shape[0], 1)
    robot_base_transform_inv_pos = robot_base_transform_inv[1].repeat(pos.shape[0], 1)

    quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(robot_base_transform_inv_quat, robot_base_transform_inv_pos, quat, pos)
    
    return pos_in_robot_base, quat_in_robot_base

def format_and_save_trajectory(state, next_state, action, reward, done, file_path):
    
    """
    Formats and saves the trajectory data to a file.

    Args:
    next_nodes (torch.Tensor): The trajectory data.
    file_path (str): The path of the file to save the data.
    """

    with open(file_path, 'a') as file:
        peg_state = state[:,0:7] 
        hole_state = state[:,7:14]
        next_peg_state = next_state[:,0:7]
        done = done.type(torch.int32)
        reward = reward.unsqueeze(0)
        done = done.unsqueeze(0)
        full_state = torch.cat((peg_state, hole_state, next_peg_state, action, reward, done), dim=1)

        # Format each element of peg_state to 5 decimal places and write to file
        for state in full_state:
            formatted_state = ' '.join(f'{x:.5f}' for x in state)  # Format each element individually
            file.write(formatted_state + '\n')


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def normalize_states(states: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (states - mean) / (std + 1e-8)


def wrap_env(
                env: gym.Env,
                state_mean: Union[np.ndarray, float] = 0.0,
                state_std: Union[np.ndarray, float] = 1.0,
                reward_scale: float = 1.0,
                                                                    ) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):

        return (state - state_mean) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class Actor(nn.Module):
    def __init__(self, state_dim_net: int, action_dim_net: int, max_action: float):

        super(Actor, self).__init__()
        self.net = nn.Sequential(
                                    nn.Linear(state_dim_net, 512),
                                    #nn.BatchNorm1d(256),  # Batch normalization
                                    nn.ELU(),
                                    nn.Linear(512, 256),
                                    #nn.BatchNorm1d(256),  # Batch normalization
                                    nn.ELU(),
                                    nn.Linear(256, 128),
                                    #nn.BatchNorm1d(256),  # Batch normalization
                                    nn.ELU(),
                                    nn.Linear(128, 6),
                                    nn.Tanh()
                                                                                    )


        self.sigma1 = nn.Parameter(torch.zeros(1))
        self.sigma2 = nn.Parameter(torch.zeros(1))

        self._initialize_weights()
        self.state_mean = 0.0
        self.state_std = 1.0

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use 'relu' for nonlinearity as it's a common choice for ELU as well
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

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
        
        # Check outputs of each layer
        x = state.clone()

        for i, layer in enumerate(self.net):
            x = layer(x)
            #print(f"Output after layer {i} ({layer.__class__.__name__}):", x)
            if torch.isnan(x).any():
                print(f"NaN detected after layer {i} ({layer.__class__.__name__})")
                break  # Optional: break the loop if NaN is detected

        return x

class BC:
    def __init__(
                    self,
                    max_action: np.ndarray,
                    actor: nn.Module,
                    actor_optimizer: torch.optim.Optimizer,
                    gamma: float = 0.99,
                    device: str = "cuda:0",
                                                                            ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.gamma = gamma

        self.total_it = 0
        self.device = device

    def validate(self, batch: TensorBatch) -> Dict[str, float]:

        """
        Validate the model on a batch of data.

        Args:
            batch (TensorBatch): A batch of data for validation.
        
        Returns:
            Dict[str, float]: A dictionary containing validation metrics.
        """
        state, next_state, action, reward, done = batch

        with torch.no_grad():
            # Forward pass through the actor network
            pi, _, _ = self.actor(state)

            # Split predicted actions
            pos_actions = pi[:, :3]
            rot_actions = pi[:, 3:]

            # Scale the actions
            scale_factor_pos = self.max_action * torch.ones(pos_actions.size(1), device=self.device)
            scale_factor_rot = self.max_action * torch.ones(rot_actions.size(1), device=self.device)
            diag_w_pos = torch.diag(scale_factor_pos)
            diag_w_rot = torch.diag(scale_factor_rot)
            
            pos_actions = pos_actions @ diag_w_pos
            rot_actions = rot_actions @ diag_w_rot

            # Compute predicted next state
            next_state_pred_pos = state[:, :3] + pos_actions
            pi_angle = torch.norm(rot_actions, p=2, dim=-1)
            pi_axis = rot_actions / (pi_angle.unsqueeze(-1) + 1e-8)
            pi_quat = quat_from_angle_axis(pi_angle, pi_axis)
            next_state_pred_quat = quat_mul(pi_quat, state[:, 3:7])

            # Compute loss
            actor_loss_pos = F.mse_loss(next_state[:, :3], next_state_pred_pos)
            actor_loss_rot = quat_diff_rad(next_state_pred_quat, next_state[:, 3:7]).mean()
            val_loss = l_w_pos_0 * actor_loss_pos + l_w_rot_0 * actor_loss_rot

        return {"val_loss": val_loss.item()}


    def train(self, batch: TensorBatch) -> Dict[str, float]:

        log_dict = {}
        self.total_it += 1

        state, next_state, action, reward, done = batch
        #next_state = normalize_states(next_state, self.actor.state_mean, self.actor.state_std)

        # Compute actor loss 
        pi, s1, s2 = self.actor(state)

        pos_actions = pi[:, :3]
        rot_actions = pi[:, 3:]

        scale_factor_pos = max_action * torch.ones(pos_actions.size(1), device='cuda:0')
        scale_factor_rot = max_action * torch.ones(rot_actions.size(1), device='cuda:0')
        diag_w_pos = torch.diag(scale_factor_pos)
        diag_w_rot = torch.diag(scale_factor_rot)
        
        # Scale the rotational actions
        pos_actions = pos_actions @ diag_w_pos
        rot_actions = rot_actions @ diag_w_rot

        next_state_pred_pos = state[:,:3] + pos_actions 

        # Context actions
        pi_angle = torch.norm(rot_actions, p=2, dim=-1)
        pi_axis = rot_actions / (pi_angle.unsqueeze(-1) + 1e-8)  # Avoid divide by zero
        pi_quat = quat_from_angle_axis(pi_angle, pi_axis)
        pi_quat = torch.where(pi_angle.unsqueeze(-1).repeat(1, 4) > 1e-08,
                                        pi_quat,
                                        torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(rot_actions.size()[0], 1),)



        next_state_pred_quat = quat_mul(pi_quat, state[:,3:7])

        next_state_pos = next_state[:,:3]
        next_state_quat = next_state[:,3:7]


        print("*-------------------------------------------------------------------------------------------*")
        print("NN Output: ")
        print(" ")
        print(" ")
        print(" ")
        print("state pos: ", state[:5,:3])
        print("hole pos: ", state[:5,7:10])
        print("pos_actions: ", pos_actions[:5]) 
        print("next_state_pred_pos: ", next_state_pred_pos[:5])
        print("next_state pos GT: ", next_state[:5,:3])
        print(" ")
        print(" ")
        print(" ")
        print("state quat: ", state[:5,3:7])
        print("hole quat: ", state[:5,10:14])
        print("pi_quat: ", pi_quat[:5])
        print("next_state_pred_quat: ", next_state_pred_quat[:5])
        print("next_state quat GT: ", next_state[:5,3:7])
        print(" ")
        print("*-------------------------------------------------------------------------------------------*")

        actor_loss_pos = F.mse_loss(next_state_pred_pos, next_state_pos)
        actor_loss_rot = quat_diff_rad(next_state_pred_quat, next_state_quat).mean()
        actor_loss = l_w_pos_0 * actor_loss_pos + l_w_rot_0 * actor_loss_rot

        # Update log_dict with the correct key
        log_dict["train_loss"] = actor_loss.item()

        print("*-------------------------------------------------------------------------------------------*")
        print("weighted actor_loss_pos: ", actor_loss_pos.item())
        print("weighted actor_loss_rot: ", actor_loss_rot.item())
        print(" ")
        print("weighted actor_loss_pos: ", l_w_pos_0 * actor_loss_pos.item())
        print("weighted actor_loss_rot: ", l_w_rot_0 * actor_loss_rot.item())
        print("actor_loss: ", actor_loss.item())
        print("*-------------------------------------------------------------------------------------------*")

        plot_data['train_loss'].append(actor_loss.item())

        if self.total_it % 1e02 == 0:# and self.total_it>500:  # Update the plot every 100 iterations
            live_plot(plot_data, title='Training Progress')

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Clip gradients: gradients are modified in-place
        clip_grad_norm_(self.actor.net.parameters(), max_norm=1.0)

        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                check_nan(param.grad, f"Gradient of {name}")

        self.actor_optimizer.step()

        return log_dict


    def state_dict(self) -> Dict[str, Any]:
        return {
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "total_it": self.total_it,
                                                                                    }


    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]



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


def compute_mean_std(replay_buffer: ReplayBuffer, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    buffer_size = min(10000, replay_buffer._size)  # Use up to 10,000 states
    random_indices = np.random.choice(replay_buffer._size, size=buffer_size, replace=False)
    states = replay_buffer._states[random_indices].cpu().numpy()  # Extract random states
    print("states in compute_mean_std: ", states)
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0) + eps
    return mean, std



def torch_quaternion_slerp(q1, q2, max_steps):
    """Spherical linear interpolation between two quaternions"""
    
    # Calculate interpolation fractions
    fractions = np.linspace(0, 1, max_steps)

    # Initialize interpolated quaternion array
    interp_quats = []
    # Interpolate along spherical path    
    for f in fractions:
        f = torch.tensor(f, dtype=torch.float32, device=device)
        interp_quats.append(slerp(q1, q2, f))
        
    return interp_quats


def quaternion_slerp(q1, q2, max_steps):
    """Spherical linear interpolation between two quaternions"""
    
    # Calculate interpolation fractions
    fractions = np.linspace(0, 1, max_steps)
    
    # Initialize interpolated quaternion array
    interp_quats = []
    
    # Interpolate along spherical path    
    for f in fractions:
        interp_quats.append(quaternion_interpolate(q1, q2, f))
        
    return interp_quats


def quaternion_interpolate(q1, q2, f):
    """Interpolate between two quaternions"""
    
    # Calculate angle between quaternions    
    dot = np.dot(q1, q2)
    
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Interpolate        
    theta = np.arccos(dot) * f
    return (q2*np.sin(theta) + q1*np.cos(theta)) / np.sin(theta) + np.finfo(float).eps


def read_and_process_data_RRT_Star(max_action: float, device: str) -> ReplayBuffer:
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    
    base_path = "/common/home/im316/RL4Insertion_code/peg_in_hole2/peg_in_hole/Train"
    traj_file = "pih_0001_tree.txt"
    trajectories_file_path = "interpolated_RRT_Star_4.txt"
    
    if os.path.exists(trajectories_file_path):
        os.remove(trajectories_file_path)

    traj_file_path = os.path.join(base_path, traj_file)
    print("Processing file: ", traj_file_path)

    next_node = None    
    lines_dict = {}

    with open(traj_file_path, 'r') as file:
        lines = file.readlines()

        hole_line = lines[1].strip()  # Remove leading/trailing whitespace
        hole_values = torch.tensor([float(x) for x in hole_line.split()], dtype=torch.float32)

        hole_parent_idx = int(hole_values[0])
        hole_edge_idx = int(hole_values[1])
        hole_node_idx = int(hole_values[2])
        
        hole_pose = hole_values[3:10]  # Extract 3D position and 4D quaternion (wxyz)
        hole_position = hole_pose[:3].unsqueeze(0)  # Extract 3D position
        hole_position = hole_position / 1000.0  # Convert from mm to m

        print("hole_position I read:", hole_position)
        #hole_position += torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)
        hole_position = torch.tensor([1.3000e-01, 4.3656e-11, 1.0400e+00], dtype=torch.float32).unsqueeze(0) 
        hole_rotation = hole_pose[3:7].unsqueeze(0)  # Extract 4D quaternion (wxyz)
        hole_rotation = hole_rotation[:,[1,2,3,0]]  # Convert quaternion from wxyz to xyzw
        hole_rotation = torch.tensor([0., 0., 0., 1.], dtype=torch.float32).unsqueeze(0) 
        hole_rotation_angle, hole_rotation_axis = quat_to_angle_axis(hole_rotation)
        hole_rotation_angle_axis = hole_rotation_angle * hole_rotation_axis
        hole_state = torch.cat((hole_position, hole_rotation), dim=1)  # Combine position and rotation of hole
        lines_dict[hole_node_idx] = hole_state.numpy()

        for k in range(2,len(lines)):
            line = lines[k]
            # Skip lines that are empty or start with '#'
            if not line or line.startswith('#'):
                print("line skipped:", line)
                continue

            # Split line into components and extract node_idx
            line_components = line.split()

            values = [float(x) for x in line_components]
            parent_idx = int(values[0])
            edge_idx = int(values[1])
            node_idx = int(values[2])
            
            values = torch.tensor(values, dtype=torch.float32)
            node_pose = values[3:10]  # Extract 3D position and 4D quaternion (wxyz)
            position = node_pose[:3].unsqueeze(0)  # Extract 3D position
            position = position / 1000.0  # Convert from mm to m
            position -= torch.tensor([0.0, 0.0, 0.0260], dtype=torch.float32).unsqueeze(0)
            position += hole_position
            rotation = node_pose[3:7].unsqueeze(0)  # Extract 4D quaternion (wxyz)
            rotation = rotation[:,[1,2,3,0]]  # Convert quaternion from wxyz to xyzw
            rotation_angle, rotation_axis = quat_to_angle_axis(rotation)
            rotation_angle_axis = rotation_angle * rotation_axis

            peg_state = torch.cat((position, rotation), dim=1)  # Combine position and rotation of peg
            lines_dict[node_idx] = peg_state.numpy()


        print("First pass finished")
        #time.sleep(10)
        for k in range(2,len(lines)):
            line = lines[k]
            # Skip lines that are empty or start with '#'
            if not line or line.startswith('#'):
                print("line skipped:", line)
                continue

            # Split line into components and extract node_idx
            line_components = line.split()

            values = [float(x) for x in line_components]
            parent_idx = int(values[0])
            edge_idx = int(values[1])
            node_idx = int(values[2])
            
            values = torch.tensor(values, dtype=torch.float32)
            node_pose = values[3:10]  # Extract 3D position and 4D quaternion (wxyz)
            position = node_pose[:3].unsqueeze(0)  # Extract 3D position
            position = position / 1000.0  # Convert from mm to m
            position -= torch.tensor([0., 0., 0.0260], dtype=torch.float32).unsqueeze(0)
            position += hole_position
            rotation = node_pose[3:7].unsqueeze(0)  # Extract 4D quaternion (wxyz)
            rotation = rotation[:,[1,2,3,0]]  # Convert quaternion from wxyz to xyzw
            rotation_angle, rotation_axis = quat_to_angle_axis(rotation)
            rotation_angle_axis = rotation_angle * rotation_axis

            peg_state = torch.cat((position, rotation), dim=1)  # Combine position and rotation of peg
            lines_dict[node_idx] = peg_state.numpy()

            current_node = torch.cat((peg_state, hole_state), dim=1)  # Combine position and rotation of peg with hole_pose

            next_values = torch.tensor([x for x in lines_dict[parent_idx]], dtype=torch.float32)
            next_values = next_values.squeeze(0)
            next_parent_idx = int(next_values[0])
            next_edge_idx = int(next_values[1])
            next_node_idx = int(next_values[2])
            next_node_pose = next_values[3:10]  # Extract 3D position and 4D quaternion (xyzw)
            next_position = torch.tensor(next_values[:3]).unsqueeze(0)
            next_rotation = torch.tensor(next_values[3:7]).unsqueeze(0)
            next_rotation_angle, next_rotation_axis = quat_to_angle_axis(next_rotation)
            next_rotation_angle_axis = next_rotation_angle * next_rotation_axis
            
            next_peg_state = torch.cat((next_position, next_rotation), dim=1)  # Combine position and rotation of peg
            next_node = torch.cat((next_peg_state, hole_state), dim=1)  # Combine position and rotation of peg with hole_pose

            dt_x = abs(next_position[:,0].cpu().numpy() - position[:,0].cpu().numpy())
            dt_y = abs(next_position[:,1].cpu().numpy() - position[:,1].cpu().numpy())
            dt_z = abs(next_position[:,2].cpu().numpy() - position[:,2].cpu().numpy())

            dr_x = abs(next_rotation_angle_axis[:,0].cpu().numpy() - rotation_angle_axis[:,0].cpu().numpy())
            dr_y = abs(next_rotation_angle_axis[:,1].cpu().numpy() - rotation_angle_axis[:,1].cpu().numpy())
            dr_z = abs(next_rotation_angle_axis[:,2].cpu().numpy() - rotation_angle_axis[:,2].cpu().numpy())

            steps_x = int(dt_x / max_action_transl)
            steps_y = int(dt_y / max_action_transl)
            steps_z = int(dt_z / max_action_transl)

            steps_rx = int(dr_x / max_action_rot)
            steps_ry = int(dr_y / max_action_rot)
            steps_rz = int(dr_z / max_action_rot)

            max_steps = max(steps_x, steps_y, steps_z, steps_rx, steps_ry, steps_rz) + 1

            next_node_xs = np.linspace(position[:,0].squeeze(0).cpu().numpy(), next_position[:,0].squeeze(0).cpu().numpy(), max_steps, endpoint=True)
            next_node_ys = np.linspace(position[:,1].squeeze(0).cpu().numpy(), next_position[:,1].squeeze(0).cpu().numpy(), max_steps, endpoint=True)
            next_node_zs = np.linspace(position[:,2].squeeze(0).cpu().numpy(), next_position[:,2].squeeze(0).cpu().numpy(), max_steps, endpoint=True)

            print("next_node_zs:", next_node_zs)

            diff_xs = np.diff(next_node_xs)
            diff_ys = np.diff(next_node_ys)
            diff_zs = np.diff(next_node_zs)

            next_node_xs = torch.tensor(next_node_xs, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            next_node_ys = torch.tensor(next_node_ys, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            next_node_zs = torch.tensor(next_node_zs, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            node_q = rotation.squeeze(0).cpu().numpy()
            next_node_q = next_rotation.squeeze(0).cpu().numpy()        

            node_q = torch.tensor(node_q, dtype=torch.float32, device=device ).unsqueeze(0)
            next_node_q = torch.tensor(next_node_q, dtype=torch.float32, device=device).unsqueeze(0)

            next_node_qs = torch_quaternion_slerp(node_q, next_node_q, max_steps)

            for i, tensor in enumerate(next_node_qs):
                next_node_qs[i] = tensor.squeeze(0) 

            next_node_qs = torch.stack(next_node_qs).unsqueeze(0)
            next_node_qs = next_node_qs.cpu()
            next_nodes = torch.cat((next_node_xs, next_node_ys, next_node_zs, next_node_qs), dim=2)

            for st in range(max_steps-1):

                state = next_nodes[:,st,:].squeeze(1)
                next_state = next_nodes[:,st+1,:].squeeze(1)

                print("state_z:", state[:,2])
                print("next_state_z:", next_state[:,2])

                state = torch.cat((state, hole_state), dim=1)
                next_state = torch.cat((next_state, hole_state), dim=1)

                state_pos = state[:,0:3]
                state_quat = state[:,3:7]
                next_state_pos = next_state[:,0:3]
                next_state_quat = next_state[:,3:7]

                action_pos = next_state_pos - state_pos
                action_quat = quat_mul(next_state_quat, quat_conjugate(state_quat))
                action = torch.cat((action_pos, action_quat), dim=1)

                reward_pos = torch.norm(next_state_pos - hole_position, p=2)
                reward_rot = quat_diff_rad(next_state_quat, hole_rotation)
                reward = - ( l_w_tr*reward_pos + l_w_rot*reward_rot )

                if parent_idx == hole_node_idx:
                    if st == max_steps-2:
                        done = torch.tensor([1.0], dtype=torch.float32)
                    else:
                        done = torch.tensor([0.0], dtype=torch.float32)
                else:
                    done = torch.tensor([0.0], dtype=torch.float32)

                
                print("******************************************")
                print("state: ", state)
                print("next_state: ", next_state)
                print("action: ", action)
                print("reward: ", reward)
                print("done: ", done)
                print("******************************************")
                
                format_and_save_trajectory(state, next_state, action, reward, done, trajectories_file_path)


@pyrallis.wrap()
def eval(config: TrainConfig):
    
    # Open the file for reading
    with open("interpolated_RRT_Star_4.txt", "r") as file:
        # Read the lines
        lines = file.readlines()
        # Initialize the states list
        states_list = []
        # Loop through each linec
        for line in lines:
            # Convert the line to a list of floats
            state_l = [float(x) for x in line.strip().split()]

            # Convert the list to a PyTorch tensor on the specified device
            state_l = torch.tensor(state_l, dtype=torch.float, device='cuda:0')
            state_l = state_l[[0,1,2,4,5,6,3]]
            
            # Append the state to the states list
            states_list.append(state_l)


    # Set seeds
    seed = config.seed
    set_seed(seed)
    
    actor = Actor(state_dim_net, action_dim_net, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(),lr=1e-03, weight_decay=1e-5) # Including L2 regularization (weight decay)


    state_mean = np.loadtxt('state_mean_hole_4.txt')
    state_std = np.loadtxt('state_std_hole_4.txt')
    state_mean = torch.tensor(state_mean, device=config.device, dtype=torch.float32)
    state_std = torch.tensor(state_std, device=config.device, dtype=torch.float32)
    actor.state_mean = state_mean
    actor.state_std = state_std

    kwargs = {
                "max_action": max_action,
                "actor": actor,
                "actor_optimizer": actor_optimizer,
                "gamma": config.gamma,
                "device": config.device,
                                                                }


    trainer = BC(**kwargs)
    policy_file = "/common/home/im316/RL4Insertion_code/CORL/algorithms/checkpoint_BC_RRT_Star_Train5_120000.pth"
    print(policy_file)
    print(f"Loading model from {policy_file}")
    trainer.load_state_dict(torch.load(policy_file))

    for param in trainer.actor.parameters():
        param.requires_grad = False 
    trainer.actor.eval()

    #init_pos = torch.tensor([0.4999, -0.0006,  0.0576], dtype=torch.float32, device=device).unsqueeze(0)
    
    #init_pos = torch.tensor([1.3000e-01 - 0.01, 4.3656e-11 + 0.003, 1.0400e+00 + 0.062], dtype=torch.float32, device=device).unsqueeze(0)
    #init_quat = torch.tensor([0.707, 0.000, -0.707, 0.], dtype=torch.float32, device=device).unsqueeze(0)
    
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
        while d_pos >= 0.: # and d_rot >= 0.:

            print("step: ", step)
            print("peg_state: ", peg_state)
            print("hole_state: ", hole_state)

            step += 1

            peg_state = state[:,0:7]
            hole_state = state[:,7:14] 

            # Compute actor loss 
            pi, _, _ = trainer.actor(state)
            
            pos_actions = pi[:, :3]
            rot_actions = pi[:, 3:]

            scale_factor_pos = max_action * torch.ones(pos_actions.size(1), device='cuda:0')
            scale_factor_rot = max_action * torch.ones(rot_actions.size(1), device='cuda:0')
            diag_w_pos = torch.diag(scale_factor_pos)
            diag_w_rot = torch.diag(scale_factor_rot)
            
            # Scale the rotational actions
            pos_actions = pos_actions @ diag_w_pos
            rot_actions = rot_actions @ diag_w_rot

            # Context actions
            pi_angle = torch.norm(rot_actions, p=2, dim=-1)
            pi_axis = rot_actions / pi_angle.unsqueeze(-1)
            pi_quat = quat_from_angle_axis(pi_angle, pi_axis)

            peg_state = torch.cat((state[:,0:3] + pos_actions, quat_mul(pi_quat, state[:,3:7])), dim=1)
            peg_state_print = peg_state.clone()
            peg_state_print = peg_state_print[:,[0,1,2,6,3,4,5]]

            # Format peg_state_print as a string
            peg_state_str = ' '.join(f'{x:.5f}' for x in peg_state_print.squeeze(0).tolist())
            # Write the formatted string to the file
            file.write(peg_state_str + '\n')

            state = torch.cat((peg_state, hole_state), dim=1)
            
            print("peg_state: ", state[:,:7])
            print("hole_state: ", hole_state)
            print("step: ", step)
            time.sleep(0.1)

            d_pos = torch.abs(hole_state[:,3] - peg_state[:,3])
            d_rot = quat_diff_rad(hole_state[:,3:], peg_state[:,3:])
        
            print("d_pos: ", d_pos)
            print("d_rot: ", d_rot)


    print("Final Position: ", state[:,:3])
    print("Final Rotation: ", state[:,3:7])

    print("Goal Position: ", hole_state[:,:3])
    print("Goal Rotation: ", hole_state[:,3:7])


def fill_buffer():
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    with open('interpolated_RRT_Star_4.txt', 'r') as file:
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

@pyrallis.wrap()
def train(config: TrainConfig):

    #read_and_process_data_RRT_Star(max_action, device)

    file_path = '/common/home/im316/RL4Insertion_code/CORL/algorithms/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("RB size:", replay_buffer.size())

    positions, actions = extract_actions_and_positions(replay_buffer)
    plot_3d_vector_field(positions, actions, 'replay_buffer_vector_field_4.png', sampling_rate=0.005, vector_length=0.005)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    actor = Actor(state_dim_net, action_dim_net, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-04, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='min', factor=0.1, patience=10, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    kwargs = {
                "max_action": max_action,
                "actor": actor,
                "actor_optimizer": actor_optimizer,
                "gamma": config.gamma,
                "device": config.device,
                                                                }

    
    if config.normalize:
        state_mean, state_std = compute_mean_std(replay_buffer, eps=1e-3)
        state_mean = torch.tensor(state_mean, device=device, dtype=torch.float32)
        state_std = torch.tensor(state_std, device=device, dtype=torch.float32)
    else:
        # The size here should match the feature size of your states
        num_features = 14  # replace with your actual number of features in your data samples
        state_mean = torch.zeros(num_features, device=device, dtype=torch.float32)
        state_std = torch.ones(num_features, device=device, dtype=torch.float32)

    np.savetxt('state_mean_hole_4.txt', state_mean.cpu().numpy())
    np.savetxt('state_std_hole_4.txt', state_std.cpu().numpy())

    state_mean = np.loadtxt('state_mean_hole_4.txt')
    state_std = np.loadtxt('state_std_hole_4.txt')
    state_mean = torch.tensor(state_mean, device=device, dtype=torch.float32)
    state_std = torch.tensor(state_std, device=device, dtype=torch.float32)

    # Remaining code for checkpoints
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    print("---------------------------------------")
    print(f"Training BC ,Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)
    trainer.scheduler = scheduler

    #if config.load_model != "":
    #    policy_file = Path(config.load_model)
    #    trainer.load_state_dict(torch.load(policy_file))
    
    trainer.actor.state_mean = state_mean
    trainer.actor.state_std = state_std
    trainer.actor.train()
    for param in trainer.actor.parameters():
        param.requires_grad = True


    # Early stopping parameters
    early_stopping_patience = config.max_timesteps
    no_improvement_epochs = 0
    best_val_loss = float('inf')

    # Split the data
    train_data, val_data = replay_buffer.split_buffer()

    #train_batch = sample_batch(train_data, config.batch_size)
    #train_batch = sample_batch(train_data, 1)
    for epoch in range(config.max_timesteps):
        # Training step
        train_batch = sample_batch(train_data, config.batch_size)
        train_log_dict = trainer.train(train_batch)

        # Validation step
        val_batch = sample_batch(val_data, config.batch_size)
        val_log_dict = trainer.validate(val_batch)

        # Update plot data
         # Append the correct loss value to the plot_data dictionary
        plot_data['train_loss'].append(train_log_dict['train_loss'])
        plot_data['val_loss'].append(val_log_dict['val_loss'])

        # Early stopping check
        if val_log_dict['val_loss'] < best_val_loss:
            best_val_loss = val_log_dict['val_loss']
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if epoch % 1e04 == 0:
            torch.save(trainer.state_dict(), os.path.join("/common/home/im316/RL4Insertion_code/CORL/algorithms/", f"checkpoint_BC_RRT_Star_Train5_{epoch}.pth"),)

        if no_improvement_epochs >= early_stopping_patience:
            torch.save(trainer.state_dict(), os.path.join("/common/home/im316/RL4Insertion_code/CORL/algorithms/", "checkpoint_BC_RRT_Star_Early_5.pth"),)
            print("Early stopping triggered")
            break

        # Update live plot
        plot_update_freq = 1e02  # Update the plot every 100 iterations
        if epoch % plot_update_freq == 0:
            live_plot(plot_data, title='Training and Validation Loss')
    
    '''
    # Training Loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 10  # Early stopping criteria
    best_model = None

    #batch = replay_buffer.sample(1)
    for epoch in range(config.max_timesteps):
        batch = replay_buffer.sample(config.batch_size)
        #print("We will overfit:", batch)    
        log_dict = trainer.train(batch)
        #wandb.log(log_dict, step=trainer.total_it)        
        
        if epoch > 1e03 and epoch % 1e05 == 0:
            torch.save(trainer.state_dict(), os.path.join("/common/home/im316/RL4Insertion_code/CORL/algorithms/", f"checkpoint_BC_RRT_Star_Train{epoch}.pth"),)
        
        #if log_dict["val_loss"] < best_val_loss:
        #    best_val_loss = log_dict["val_loss"]
        #    epochs_without_improvement = 0
        #    best_model = copy.deepcopy(trainer.actor.state_dict())
    '''

if __name__ == "__main__":
    
    #train()
    eval()

