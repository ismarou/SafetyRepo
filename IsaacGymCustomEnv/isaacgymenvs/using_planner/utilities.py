from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np
import wandb

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

buffer_size = 10000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def fill_buffer(file_use="/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt"):
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    with open(file_use, 'r') as file:
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

#For converting data ---------------------------------------------
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
    
    #If we scale it up, it may devalue states and care too much about actions, especially with normalization ruining any diversity in state, especially with the Q function; always keep actions and states in same scale
    #now in axis angles, can all be scaled by 100 to counteract the already scaled down by 0.01
    #scale_factor_pos = 100.0 * torch.ones(pos_actions.size(1), device=device)
    #scale_factor_rot = 100.0 * torch.ones(rot_actions.size(1), device=device)
    #diag_w_pos = torch.diag(scale_factor_pos)
    #diag_w_rot = torch.diag(scale_factor_rot)
    # Scale the rotational actions
    #pos_actions = pos_actions @ diag_w_pos
    #rot_actions = rot_actions @ diag_w_rot
    
    action_out = torch.cat((pos_actions, rot_actions), dim=1)
    
    return state.clone(), action_out.clone()

def fix_peg_pos_quat_state(state):
    peg_pos = state[:, 0:3]
    peg_quat = state[:, 3:7]

    peg_pos_ = pose_world_to_robot_base(peg_pos, peg_quat)[0],  # 3
    peg_quat_ = pose_world_to_robot_base(peg_pos, peg_quat)[1],  # 4
    
    peg_pos = peg_pos_[0].clone()
    peg_quat = peg_quat_[0].clone()

    state = torch.cat((peg_pos, peg_quat), dim=1)
    
    return state
    
def fix_peg_pos_state(state):
    peg_pos = state[:, 0:3]
    peg_quat = state[:, 3:7]

    peg_pos_ = pose_world_to_robot_base(peg_pos, peg_quat)[0],  # 3
    
    return peg_pos_

def get_scaled_quaternion(pi, scale=0.01):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #For scaling actions and turning axis angle to quat
    pos_actions = pi[:, :3]
    rot_actions = pi[:, 3:]

    scale_factor_pos = scale * torch.ones(pos_actions.size(1), device=device)
    scale_factor_rot = scale * torch.ones(rot_actions.size(1), device=device)
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
                                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(rot_actions.size()[0], 1),)
    #next_state_pred_quat = quat_mul(pi_quat, state[:,3:7]) #pi_quat is just the quaternion here and we don't have to apply to the state
    return pi_quat, pos_actions
    
def axis_angle_to_quaternion(pi):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #For scaling actions and turning axis angle to quat
    pos_actions = pi[:, :3]
    rot_actions = pi[:, 3:]

    # Context actions
    pi_angle = torch.norm(rot_actions, p=2, dim=-1)
    pi_axis = rot_actions / (pi_angle.unsqueeze(-1) + 1e-8)  # Avoid divide by zero
    pi_quat = quat_from_angle_axis(pi_angle, pi_axis)
    pi_quat = torch.where(pi_angle.unsqueeze(-1).repeat(1, 4) > 1e-08,
                                    pi_quat,
                                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(rot_actions.size()[0], 1),)
    #next_state_pred_quat = quat_mul(pi_quat, state[:,3:7]) #pi_quat is just the quaternion here and we don't have to apply to the state
    return pi_quat, pos_actions

#For conversions --------------------------------------------------------------------------------------
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))
    
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

def pose_world_to_robot_base(pos, quat):
    
    """Convert pose from world frame to robot base frame."""

    robot_base_pos = torch.tensor([-3.7000e-01,  4.3656e-11,  1.0400e+00], dtype = torch.float32, device = device ).unsqueeze(0)
    robot_base_quat = torch.tensor([-2.2582e-11,  6.2217e-10,  3.5933e-11,  1.0000e+00], dtype=torch.float32, device = device).unsqueeze(0)

    robot_base_transform_inv = tf_inverse(robot_base_quat, robot_base_pos)
    robot_base_transform_inv_quat = robot_base_transform_inv[0].repeat(pos.shape[0], 1)
    robot_base_transform_inv_pos = robot_base_transform_inv[1].repeat(pos.shape[0], 1)

    quat_in_robot_base, pos_in_robot_base = tf_combine(robot_base_transform_inv_quat, robot_base_transform_inv_pos, quat, pos)
    
    return pos_in_robot_base, quat_in_robot_base
    
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quat_unit(a):
    return normalize(a)
    
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))
    
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)
    
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)
    
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1

def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(
        torch.clamp(
            torch.norm(
                mul[:, 0:3],
                p=2, dim=-1), max=1.0)
    )
    
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
    
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)
    
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)