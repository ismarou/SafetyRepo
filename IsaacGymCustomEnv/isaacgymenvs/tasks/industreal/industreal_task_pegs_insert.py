# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

    IndustReal: class for peg insertion task.

    Inherits IndustReal pegs environment class and Factory abstract task class (not enforced).

    Trains a peg insertion policy with Simulation-Aware Policy Update (SAPU), SDF-Based Reward, and Sampling-Based Curriculum (SBC).

    Can be executed with python train.py task=IndustRealTaskPegsInsert test=True headless=True task.env.numEnvs=256

"""


import hydra
import numpy as np
import omegaconf
import os

import torch

import warp as wp
from pysdf import SDF
import trimesh
import open3d as o3d
from urdfpy import URDF

import time


current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
import sys
sys.path.append('../../using_planner/')
repulsor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../using_planner'))
sys.path.append(repulsor_dir)
from repulsor import *
sys.path.append('..')
sys.path.append('../..')
sys.path.append('/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/')
sys.path.append('../../isaacgym/python/isaacgym')

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import ( FactorySchemaConfigTask, )
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils import torch_jit_utils
import torch.nn as nn
import random

import isaacgymenvs.tasks.factory.factory_control as fc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_clouds_parallel(plug_points, socket_points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract point data from plug_points and convert to NumPy array
    plug_points_np = plug_points.cpu().numpy()
    ax.scatter(plug_points_np[:, 0], plug_points_np[:, 1], plug_points_np[:, 2], c='blue', marker='o', label='Plug')

    # Extract point data from socket_points and convert to NumPy array
    socket_points_np = socket_points.cpu().numpy()
    ax.scatter(socket_points_np[:, 0], socket_points_np[:, 1], socket_points_np[:, 2], c='red', marker='^', label='Socket')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_point_clouds(plug_pc, socket_pc):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract point data from plug_pc and convert to NumPy array
    plug_points = np.array(plug_pc.vertices, dtype=np.float32)
    ax.scatter(plug_points[:, 0], plug_points[:, 1], plug_points[:, 2], c='blue', marker='o', label='Plug')

    # Extract point data from socket_pc and convert to NumPy array
    socket_points = np.array(socket_pc.vertices, dtype=np.float32)
    ax.scatter(socket_points[:, 0], socket_points[:, 1], socket_points[:, 2], c='red', marker='^', label='Socket')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_point_clouds_2(plug_pc, socket_pc):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract point data from plug_pc and convert to NumPy array
    plug_points = np.array(plug_pc, dtype=np.float32)
    ax.scatter(plug_points[:, 0], plug_points[:, 1], plug_points[:, 2], c='blue', marker='o', label='Plug')

    # Extract point data from socket_pc and convert to NumPy array
    socket_points = np.array(socket_pc, dtype=np.float32)
    ax.scatter(socket_points[:, 0], socket_points[:, 1], socket_points[:, 2], c='red', marker='^', label='Socket')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_plug_and_socket_points(plug_points, socket_points):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract point data from plug_points for the first environment
    plug_points_env0 = plug_points[0].cpu().numpy()
    ax.scatter(plug_points_env0[:, 0], plug_points_env0[:, 1], plug_points_env0[:, 2], c='blue', marker='o', label='Plug')

    # Extract point data from socket_points for the first environment
    socket_points_env0 = socket_points[0].cpu().numpy()
    ax.scatter(socket_points_env0[:, 0], socket_points_env0[:, 1], socket_points_env0[:, 2], c='red', marker='^', label='Socket')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout()
    plt.show()



"""
    ReadMe

    In the init, select the policy to load in. If you select self.potential_field=True then the potential field will be used as the policy in place of the actor loaded in.
    In the init if potential_field==True then create pegs and sockets [0,0,0,0,0,0,1] to be used as template for parallel changes later

    Within prephysics step actions are chosen:
    Actions are passed into function of RL games network. This can be used for residual RL.
    In prephysics step the peg and socket poses and quats have noise added to them. 
    Then either potential field action or policy action is used and passed to the DG controller.
    For the potential field, in parallel put the points in the proper locations. Then choose the actions with the act() function and pass to DG controller.
"""

def find_bottom_center_point(pointcloud):
    
    # Find the minimum z-coordinate
    min_z = pointcloud[:, :, 2].min(dim=1)[0]
    
    # Create a mask for points with the minimum z-coordinate
    mask = pointcloud[:, :, 2] == min_z.unsqueeze(1)
    
    # Select the points with the minimum z-coordinate
    bottom_points = pointcloud[mask].view(pointcloud.shape[0], -1, 3)
    
    # Calculate the center point of the bottom points
    bottom_center = bottom_points.mean(dim=1)
    
    return bottom_center


def transform_pointcloud(pointcloud, pos, quat):
    
    # Convert quaternion to rotation matrix
    rotation_matrix = torch_jit_utils.quaternion_to_matrix(quat)
    
    # Transform the pointcloud
    transformed_pointcloud = torch.matmul(pointcloud, rotation_matrix.transpose(1, 2)) + pos.unsqueeze(1)
    
    return transformed_pointcloud


class IndustRealTaskPegsInsert(IndustRealEnvPegs, FactoryABCTask):
    def __init__(
                        self,
                        cfg,
                        rl_device,
                        sim_device,
                        graphics_device_id,
                        headless,
                        virtual_screen_capture,
                        force_render,
                    
                                                                                    ):
        
        """
            Initialize instance variables. 
            Initialize task superclass.
        """

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__( cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, )


        self.identity_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.identity_quat = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
        self.identity_pose = torch.cat([self.identity_pos, self.identity_quat], dim=1)
        self.single_identity_pose = torch.cat([self.identity_pos[0, :].unsqueeze(0), self.identity_quat[0, :].unsqueeze(0)], dim=1)

        self._acquire_task_tensors()
        self.parse_controller_spec()


        # Get Warp mesh objects for SAPU and SDF-based reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        ( self.wp_plug_meshes, self.wp_plug_meshes_sampled_points, self.wp_socket_meshes, self.wp_socket_meshes_sampled_points) = algo_utils.load_asset_meshes_in_warp_Ismarou(
                                                                                                                                                                                    plug_files = self.plug_files,
                                                                                                                                                                                    socket_files = self.socket_files,
                                                                                                                                                                                    num_samples = self.cfg_task.rl.sdf_reward_num_samples,
                                                                                                                                                                                    device = self.wp_device,
                                                                                                                                                                                                                                                    )

        self.plug_scale = self.cfg_task.env.plug_scale
        self.socket_scale = self.cfg_task.env.socket_scale

        self.plug_pc_template_list = []
        self.socket_pc_template_list = []

        # Convert the lists of point clouds to tensors
        self.plug_points_template_tensor = torch.zeros((self.num_envs, self.cfg_task.rl.sdf_reward_num_samples, 3), device=self.device)
        self.socket_points_template_tensor = torch.zeros((self.num_envs, self.cfg_task.rl.sdf_reward_num_samples, 3), device=self.device)

        for env_idx in range(len(self.plug_files)):

            self.plug_trimesh_urdf = URDF.load(self.plug_files[env_idx])
            self.plug_trimesh_template = self.plug_trimesh_urdf.links[0].collision_mesh
            self.plug_trimesh_template = self.plug_trimesh_template.apply_scale(self.plug_scale)

            '''
            # Assuming you want to scale along the Y-axis (index 1)
            
            scale_factor = self.socket_scale
            scale_axis = 1

            # Get the vertices of the trimesh
            vertices = self.socket_trimesh_template.vertices

            # Scale the vertices along the specified axis
            vertices[:, scale_axis] *= scale_factor

            # Update the vertices of the trimesh
            self.socket_trimesh_template.vertices = vertices
            
            '''

            self.pointcloud_plug_template = trimesh.sample.sample_surface_even(self.plug_trimesh_template, self.cfg_task.rl.sdf_reward_num_samples, seed = 0)[0]
            self.plug_pc_template = trimesh.points.PointCloud(self.pointcloud_plug_template.copy())
            self.plug_pc_template_list.append(self.plug_pc_template)
            self.plug_points_template_tensor[env_idx, :, :] = torch.tensor(self.pointcloud_plug_template, device=self.device)

            self.socket_trimesh_urdf = URDF.load(self.socket_files[env_idx])
            self.socket_trimesh_template = self.socket_trimesh_urdf.links[0].collision_mesh
            self.socket_trimesh_template = self.socket_trimesh_template.apply_scale(self.socket_scale)
            self.pointcloud_plug_template = trimesh.sample.sample_surface_even(self.socket_trimesh_template, self.cfg_task.rl.sdf_reward_num_samples, seed = 0)[0]
            self.socket_pc_template = trimesh.points.PointCloud(self.pointcloud_plug_template.copy())
            self.socket_pc_template_list.append(self.socket_pc_template)
            self.socket_points_template_tensor[env_idx, :, :] = torch.tensor(self.pointcloud_plug_template, device=self.device)

        #visualize_point_clouds(self.plug_pc_template, self.socket_pc_template)


        if self.viewer != None:
            self._set_viewer_params()
        

        # -------------------------------------------- POTENTIAL FIELD -----------------------------------------------------------------------#
        
        # A. Create Socket Point Cloud

        # i) Joe's version: Create it at every iteration

        '''
        resolution = self.cfg_task.env.resolution
        
        self.socket_points_template_Joe = create_socket(
                                                        plug_type = self.plug_types[0], 
                                                        plug_width = self.plug_widths[0],
                                                        plug_depth = self.plug_depths[0],
                                                        plug_length = self.plug_heights[0],
                                                        socket_width = self.socket_widths[0], 
                                                        socket_depth = self.socket_depths[0], 
                                                        socket_length = self.socket_heights[0], 
                                                        device = self.device,
                                                        pose = self.single_identity_pose, 
                                                        resolution=resolution
                                                                                                                          ).unsqueeze(0)
                                                                                                            
        for env_idx in range(1, self.num_envs):
            self.socket_points_template_Joe = torch.cat([self.socket_points_template_Joe, create_socket(                                                
                                                                                                    plug_type = self.plug_types[env_idx], 
                                                                                                    plug_width = self.plug_widths[env_idx],
                                                                                                    plug_depth = self.plug_depths[env_idx],
                                                                                                    plug_length = self.plug_heights[env_idx],
                                                                                                    socket_width = self.socket_widths[env_idx], 
                                                                                                    socket_depth = self.socket_depths[env_idx], 
                                                                                                    socket_length = self.socket_heights[env_idx], 
                                                                                                    device = self.device,
                                                                                                    pose = self.single_identity_pose, 
                                                                                                    resolution=resolution

                                                                                                                                                                    ).unsqueeze(0)], dim=0)
        self.socket_points_template_Joe.to(self.device)
        print("socket points shape", self.socket_points_template_Joe.shape)
        '''
        

        # ii) Sampling based version: using Nvidia's algo_utils


        # B. Create Plug Point Cloud

        # i) Joe's version: Create it at every iteration

        
        '''
        #Always create new plugs

        self.plug_points_template_Joe = create_peg_points(
                                                                plug_type = self.plug_types[0], 
                                                                plug_width = self.plug_widths[0], 
                                                                plug_depth = self.plug_depths[0], 
                                                                plug_length = self.plug_heights[0], 
                                                                pos = self.identity_pos, 
                                                                quat = self.identity_quat, 
                                                                resolution = resolution,
                                                                remove_bottom = False,
                                                                remove_top = True,
                                                                remove_walls = False                             
                                                                                                ).unsqueeze(0)        
        for env_idx in range(1, self.num_envs):
            self.plug_points_template_Joe = torch.cat([self.plug_points_template_Joe, create_peg_points(
                                                                                                            plug_type = self.plug_types[env_idx], 
                                                                                                            plug_width = self.plug_widths[env_idx], 
                                                                                                            plug_depth = self.plug_depths[env_idx], 
                                                                                                            plug_length = self.plug_heights[env_idx], 
                                                                                                            pos = self.identity_pos, 
                                                                                                            quat = self.identity_quat, 
                                                                                                            resolution = resolution,
                                                                                                            remove_bottom = False,
                                                                                                            remove_top = True,
                                                                                                            remove_walls = False                             
                                                                                                        

                                                                                                                                    ).unsqueeze(0)], dim=0)
        self.plug_points_template_Joe.to(self.device)
        print("Plug points shape", self.plug_points_template_Joe.shape)
        '''
        
        # ii) Sampling based version: using Nvidia's algo_utils

        # Parameters for Potential Field + Residual RL options

        self.entire_action_residual = self.cfg_task.env.entire_action_residual
        self.potential_field_scale_residual = self.cfg_task.env.potential_field_scale_residual
        self.self_scaling_residual = self.cfg_task.env.self_scaling_residual
        self.simple_self_scaling_residual = self.cfg_task.env.simple_self_scaling_residual
        self.residual_scale = self.cfg_task.env.residual_scale

        self.action_residual = self.cfg_task.env.action_residual
        self.structured_curriculum_scaling_residual = self.cfg_task.env.structured_curriculum_scaling_residual
        self.DELFT_structured_curriculum_scaling_residual = self.cfg_task.env.DELFT_structured_curriculum_scaling_residual

        self.rl_weight = self.cfg_task.env.rl_weight
        
        #-------------------------------------------------POTENTIAL FIELD-----------------------------------------------------------------------#
        
        #Noise

        #self.plug_obs_pos_noise = [0., 0., 0.,] #torch.zeros([self.num_envs, 3], device=self.device)

        # Defining the (Structured) Obs Noise Curriculum

        self.num_curriculum_steps_plug_obs_pos_x = int((self.cfg_task.env.plug_pos_obs_noise[0] - self.cfg_task.env.plug_pos_obs_noise_min[0]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_plug_obs_pos_y = int((self.cfg_task.env.plug_pos_obs_noise[1] - self.cfg_task.env.plug_pos_obs_noise_min[1]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_plug_obs_pos_z = int((self.cfg_task.env.plug_pos_obs_noise[2] - self.cfg_task.env.plug_pos_obs_noise_min[2]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_plug_obs_pos = max(self.num_curriculum_steps_plug_obs_pos_x, self.num_curriculum_steps_plug_obs_pos_y, self.num_curriculum_steps_plug_obs_pos_z)

        self.num_curriculum_steps_plug_obs_rot_x = int((self.cfg_task.env.plug_rot_obs_noise[0] - self.cfg_task.env.plug_rot_obs_noise_min[0]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_plug_obs_rot_y = int((self.cfg_task.env.plug_rot_obs_noise[1] - self.cfg_task.env.plug_rot_obs_noise_min[1]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_plug_obs_rot_z = int((self.cfg_task.env.plug_rot_obs_noise[2] - self.cfg_task.env.plug_rot_obs_noise_min[2]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_plug_obs_rot = max(self.num_curriculum_steps_plug_obs_rot_x, self.num_curriculum_steps_plug_obs_rot_y, self.num_curriculum_steps_plug_obs_rot_z)

        #self.num_curriculum_steps_plug_obs = max(self.num_curriculum_steps_plug_obs_pos, self.num_curriculum_steps_plug_obs_rot)
        self.num_curriculum_steps_plug_obs = self.num_curriculum_steps_plug_obs_pos

        self.num_curriculum_steps_socket_obs_pos_x = int((self.cfg_task.env.socket_pos_obs_noise[0] - self.cfg_task.env.socket_pos_obs_noise_min[0]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_socket_obs_pos_y = int((self.cfg_task.env.socket_pos_obs_noise[1] - self.cfg_task.env.socket_pos_obs_noise_min[1]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_socket_obs_pos_z = int((self.cfg_task.env.socket_pos_obs_noise[2] - self.cfg_task.env.socket_pos_obs_noise_min[2]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_socket_obs_pos = max(self.num_curriculum_steps_socket_obs_pos_x, self.num_curriculum_steps_socket_obs_pos_y, self.num_curriculum_steps_socket_obs_pos_z)

        self.num_curriculum_steps_socket_obs_rot_x = int((self.cfg_task.env.socket_rot_obs_noise[0] - self.cfg_task.env.socket_rot_obs_noise_min[0]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_socket_obs_rot_y = int((self.cfg_task.env.socket_rot_obs_noise[1] - self.cfg_task.env.socket_rot_obs_noise_min[1]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_socket_obs_rot_z = int((self.cfg_task.env.socket_rot_obs_noise[2] - self.cfg_task.env.socket_rot_obs_noise_min[2]) / self.cfg_task.env.obs_noise_curr_step_size) + 1
        self.num_curriculum_steps_socket_obs_rot = max(self.num_curriculum_steps_socket_obs_rot_x, self.num_curriculum_steps_socket_obs_rot_y, self.num_curriculum_steps_socket_obs_rot_z)
        
        #self.num_curriculum_steps_socket_obs = max(self.num_curriculum_steps_socket_obs_pos, self.num_curriculum_steps_socket_obs_rot)
        self.num_curriculum_steps_socket_obs = self.num_curriculum_steps_socket_obs_pos

        self.num_curriculum_steps_obs = max(self.num_curriculum_steps_plug_obs, self.num_curriculum_steps_socket_obs)

        print("num_curriculum_steps_obs", self.num_curriculum_steps_obs)
        time.sleep(10)

        self.socket_obs_pos_curriculum_steps = [torch.linspace(min_val, max_val, steps=self.num_curriculum_steps_obs, device = self.device) for min_val, max_val in zip(self.cfg_task.env.socket_pos_obs_noise_min, self.cfg_task.env.socket_pos_obs_noise)]
        self.socket_obs_rot_curriculum_steps = [torch.linspace(min_val, max_val, steps=self.num_curriculum_steps_obs, device = self.device) for min_val, max_val in zip(self.cfg_task.env.socket_rot_obs_noise_min, self.cfg_task.env.socket_rot_obs_noise)]
        self.plug_obs_pos_curriculum_steps = [torch.linspace(min_val, max_val, steps=self.num_curriculum_steps_obs, device = self.device) for min_val, max_val in zip(self.cfg_task.env.plug_pos_obs_noise_min, self.cfg_task.env.plug_pos_obs_noise)]
        self.plug_obs_rot_curriculum_steps = [torch.linspace(min_val, max_val, steps=self.num_curriculum_steps_obs, device = self.device) for min_val, max_val in zip(self.cfg_task.env.plug_rot_obs_noise_min, self.cfg_task.env.plug_rot_obs_noise)]

        self.socket_obs_pos_curriculum_steps = torch.stack(self.socket_obs_pos_curriculum_steps, dim=1)
        self.socket_obs_rot_curriculum_steps = torch.stack(self.socket_obs_rot_curriculum_steps, dim=1)
        self.plug_obs_pos_curriculum_steps = torch.stack(self.plug_obs_pos_curriculum_steps, dim=1)
        self.plug_obs_rot_curriculum_steps = torch.stack(self.plug_obs_rot_curriculum_steps, dim=1)

        # Defining the (Structured) Action Curriculum based on Obs Noise
        self.beta_interpol = torch.linspace(0.0, 1.0, self.num_curriculum_steps_obs, device=self.device)

        self.obs_noise_curricum_stage = 0


        if self.cfg_task.env.correlated_noise == True:
          self.correlated_peg_obs_pos_noise = torch.zeros([self.num_envs, 3], device=self.device)
          self.correlated_peg_obs_rot_noise = torch.zeros([self.num_envs, 3], device=self.device)          
       
        self.pose_tensor = torch.zeros((1, 7), dtype=torch.float32).to(self.device)


    def _get_task_yaml_params(self):
        
        """
            Initialize instance variables from YAML files.
        """

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (self.cfg_task.rl.max_episode_length)  # required instance var for VecTask

        ppo_path = os.path.join("train/IndustRealTaskPegsInsertPPO.yaml")  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting


    def _acquire_task_tensors(self):

        """
            Acquire tensors.
        """

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_pos_local = torch.tensor( [ [ 0.0, 0.0, (self.cfg_task.env.socket_base_height + self.plug_grasp_offsets[i]), ] for i in range(self.num_envs)], device=self.device,)
        self.gripper_goal_quat_local = self.identity_quat.clone()

        self.socket_top_pos_local = torch.tensor([[0.0, 0.0, self.socket_heights[i]] for i in range(self.num_envs)],device=self.device,)
        self.socket_quat_local = self.identity_quat.clone()

        # Define keypoint tensors
        self.keypoint_offsets = (algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device) * self.cfg_task.rl.keypoint_scale)
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32, device=self.device,)
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions_RL), device=self.device)

        self.curr_max_disp = self.cfg_task.rl.initial_max_disp


    def _refresh_task_tensors(self):
        
        """
            Refresh tensors.
        """

        # Compute pose of gripper goal and top of socket in global frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_jit_utils.tf_combine(
                                                                                        self.socket_quat,
                                                                                        self.socket_pos,
                                                                                        self.gripper_goal_quat_local,
                                                                                        self.gripper_goal_pos_local,
                                                                                                                            )
                                                                                
        self.socket_top_quat, self.socket_top_pos = torch_jit_utils.tf_combine(
                                                                                    self.socket_quat,
                                                                                    self.socket_pos,
                                                                                    self.socket_quat_local,
                                                                                    self.socket_top_pos_local,
                                                                                                                            )
        if not self.cfg_task.env.obs_noise_curriculum:                                            
            
            # Add observation noise to socket pos
            self.noisy_socket_pos = torch.zeros_like( self.socket_pos, dtype=torch.float32, device=self.device )
            socket_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(torch.tensor(self.socket_pos_obs_noise, dtype=torch.float32, device=self.device,))

            self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
            self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
            self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

            # Add observation noise to socket rot
            socket_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            socket_obs_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)- 0.5)
            socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(torch.tensor( self.cfg_task.env.socket_rot_obs_noise, dtype=torch.float32, device=self.device,))

            socket_obs_rot_noise_quat = torch_utils.quat_from_euler_xyz( socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2], )
            self.noisy_socket_quat = torch_utils.quat_mul( socket_obs_rot_noise_quat, self.socket_quat)
            
            # Add observation noise to plug pos
            self.noisy_plug_pos = torch.zeros_like( self.plug_pos, dtype=torch.float32, device=self.device )
            plug_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            plug_obs_pos_noise = plug_obs_pos_noise @ torch.diag(torch.tensor(self.plug_pos_obs_noise, dtype=torch.float32, device=self.device,))

            self.noisy_plug_pos[:, 0] = self.plug_pos[:, 0] + plug_obs_pos_noise[:, 0]
            self.noisy_plug_pos[:, 1] = self.plug_pos[:, 1] + plug_obs_pos_noise[:, 1]
            self.noisy_plug_pos[:, 2] = self.plug_pos[:, 2] + plug_obs_pos_noise[:, 2]
            
        
            # Add observation noise to plug rot

            plug_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            plug_obs_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            plug_obs_rot_noise = plug_obs_rot_noise @ torch.diag(torch.tensor(self.plug_rot_obs_noise, dtype=torch.float32, device=self.device,))

            plug_obs_rot_noise_quat = torch_utils.quat_from_euler_xyz(plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2])
            self.noisy_plug_quat = torch_utils.quat_mul(plug_obs_rot_noise_quat, self.plug_quat)

        else:

            # Add observation noise to socket pos
            self.noisy_socket_pos = torch.zeros_like( self.socket_pos, dtype=torch.float32, device=self.device )
            socket_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)

            socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(torch.tensor(self.socket_obs_pos_curriculum_steps[self.obs_noise_curricum_stage, :].tolist() , dtype=torch.float32, device=self.device,))

            self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
            self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
            self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

            # Add observation noise to socket rot
            socket_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            socket_obs_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)- 0.5)
            socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(torch.tensor( self.socket_obs_rot_curriculum_steps[self.obs_noise_curricum_stage,:].tolist(), dtype=torch.float32, device=self.device,))

            socket_obs_rot_noise_quat = torch_utils.quat_from_euler_xyz( socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2], )
            self.noisy_socket_quat = torch_utils.quat_mul( socket_obs_rot_noise_quat, self.socket_quat)
            
            # Add observation noise to plug pos
            self.noisy_plug_pos = torch.zeros_like( self.plug_pos, dtype=torch.float32, device=self.device )
            plug_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            self.plug_obs_pos_noise = plug_obs_pos_noise @ torch.diag(torch.tensor(self.plug_obs_pos_curriculum_steps[self.obs_noise_curricum_stage,:].tolist(), dtype=torch.float32, device=self.device,))

            self.noisy_plug_pos[:, 0] = self.plug_pos[:, 0] + self.plug_obs_pos_noise[:, 0]
            self.noisy_plug_pos[:, 1] = self.plug_pos[:, 1] + self.plug_obs_pos_noise[:, 1]
            self.noisy_plug_pos[:, 2] = self.plug_pos[:, 2] + self.plug_obs_pos_noise[:, 2]
            
        
            # Add observation noise to plug rot

            plug_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            plug_obs_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            plug_obs_rot_noise = plug_obs_rot_noise @ torch.diag(torch.tensor(self.plug_obs_rot_curriculum_steps[self.obs_noise_curricum_stage, :].tolist(), dtype=torch.float32, device=self.device,))

            plug_obs_rot_noise_quat = torch_utils.quat_from_euler_xyz(plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2])
            self.noisy_plug_quat = torch_utils.quat_mul(plug_obs_rot_noise_quat, self.plug_quat)


        if self.cfg_task.env.correlated_noise == True:
          self.noisy_plug_pos[:, :] += self.correlated_peg_obs_pos_noise[:, :]


        # Compute observation noise on socket
        ( self.noisy_gripper_goal_quat,  self.noisy_gripper_goal_pos,  ) = torch_jit_utils.tf_combine(
                                                                                                                self.noisy_socket_quat,
                                                                                                                self.noisy_socket_pos,
                                                                                                                self.gripper_goal_quat_local,
                                                                                                                self.gripper_goal_pos_local,
                                                                                                                                                        )

        # Compute pos of keypoints on plug and socket in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):

            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(
                                                                            self.plug_quat,
                                                                            self.plug_pos,
                                                                            self.identity_quat,
                                                                            keypoint_offset.repeat(self.num_envs, 1),
                                                                                                                                )[1]

            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(
                                                                                self.socket_quat,
                                                                                self.socket_pos,
                                                                                self.identity_quat,
                                                                                keypoint_offset.repeat(self.num_envs, 1),
                                                                                                                                )[1]

    
        self.noisy_plug_pos_R = self.pose_world_to_robot_base(self.noisy_plug_pos, self.noisy_plug_quat)[0]
        self.noisy_plug_quat_R = self.pose_world_to_robot_base(self.noisy_plug_pos, self.noisy_plug_quat)[1]

        self.noisy_socket_pos_R = self.pose_world_to_robot_base(self.noisy_socket_pos, self.noisy_socket_quat)[0]
        self.noisy_socket_quat_R = self.pose_world_to_robot_base(self.noisy_socket_pos, self.noisy_socket_quat)[1]

        self.noisy_gripper_goal_pos_R = self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0]
        self.noisy_gripper_goal_quat_R = self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1]




    def prepare_state(self, state):

        state = torch.tensor(state, dtype=torch.float32)
        
        peg_pos = state[:,0:3]
        peg_quat = state[:,3:7]
        hole_pos = state[:,7:10]
        hole_quat = state[:,10:14]

        peg_pos_ = self.pose_world_to_robot_base(peg_pos, peg_quat)[0],  # 3
        peg_quat_ = self.pose_world_to_robot_base(peg_pos, peg_quat)[1],  # 4
        hole_pos_ = self.pose_world_to_robot_base(hole_pos, hole_quat)[0],  # 3
        hole_quat_ = self.pose_world_to_robot_base(hole_pos, hole_quat)[1],  # 4

        peg_pos = peg_pos_[0].clone()
        hole_pos = hole_pos_[0].clone()
        peg_quat = peg_quat_[0].clone()
        hole_quat = hole_quat_[0].clone()

        state = torch.cat((peg_pos, peg_quat, hole_pos, hole_quat), dim=1)
        
        # Normalize States
        #state = normalize_states(state, self.state_mean, self.state_std)

        state_input = state
        
        return torch.tensor(state_input, dtype=torch.float32)


    def create_rotational_randomization_axis_angle(self, plug_rot_quat=None, target_position=None):
        
        if plug_rot_quat == None:
          plug_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5 )
          plug_rot_noise = plug_rot_noise @ torch.diag( torch.tensor( self.cfg_task.randomize.plug_rot_noise, dtype=torch.float32, device=self.device, ))
          plug_rot_euler = (torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) + plug_rot_noise )
          plug_rot_quat = torch_utils.quat_from_euler_xyz(plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2] )
          
        quat_total_rotation=torch_utils.quat_mul(plug_rot_quat, self.identity_quat.clone())
        action_towards = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        
        if target_position != None:
          action_towards[:, 0:3] = target_position - self.plug_pos
        action_towards[:, 3:6] = self.quaternion_towards(quat_total_rotation, self.plug_quat)
        
        return action_towards, plug_rot_quat


    def quaternion_towards(self, a: torch.Tensor, b: torch.Tensor): 
        
        #b to a, quaternion relative rotation to move
        
        #b is q1 and a is q2. This is b to a rotation
        b_conj = torch_jit_utils.quat_conjugate(b)
        rotation_towards = torch_jit_utils.quat_mul(b_conj, a)
        
        rotation_towards = rotation_towards / torch.norm(rotation_towards, dim=-1, keepdim=True)
        angle, axis = torch_jit_utils.quat_to_angle_axis(rotation_towards)
        rot_actions = angle.unsqueeze(1) * axis
        
        return rot_actions


    def residual_servoing_attractor(self, plug_points, socket_points, trajectory, actions_residual, rl_weight):
    
        #Input is peg: [envs, points, 3], socket: [envs, points, 3], peg_pose: [envs, 7], trajectory: [points, 3]

        # Actions of Potential Field

        fixed_w_pos = torch.tensor([self.cfg_task.Potential_Field.repulsive_weight_pos], device=self.device)#.repeat(self.num_envs, 1)
        fixed_w_rot = torch.tensor([self.cfg_task.Potential_Field.repulsive_weight_rot], device=self.device)#.repeat(self.num_envs, 1)
        
        new_actions = Potential_Field_actions_Conditional(
                                                                self.plug_points, 
                                                                self.socket_points, 
                                                                self.noisy_plug_pos, 
                                                                self.noisy_plug_quat,
                                                                trajectory, 
                                                                self.cfg_task.Potential_Field.Carrot_threshold, 
                                                                self.cfg_task.Potential_Field.repulse_dist_th, 
                                                                device = self.device,
                                                                w = self.cfg_task.Potential_Field.learned_weight_Potential_Field, 
                                                                residual_weight = actions_residual, 
                                                                fixed_w_pos = fixed_w_pos,
                                                                fixed_w_rot = fixed_w_rot, 
                                                                t= self.progress_buf[0], 
                                                                max_episode_length=self.cfg_task.rl.max_episode_length
                                                                                                                                            )
        #print("PF actions:", new_actions[0])
        
        # Calculate Weight beta
            
        # If + Residual RL:    
        if self.action_residual == True:

            # Total actions = Potential Field actions + beta * Residual RL actions

            # Cases for beta

            # w Noise Curriculum + DELFT curriculum beta
            if self.DELFT_structured_curriculum_scaling_residual == True:
                # Structured curriculum for action scale beta
                #beta = torch.exp(self.progress_buffer[0]/self.cfg_task.rl.max_episode_length)
                beta = max((self.cfg_task.rl.max_episode_length - self.progress_buf[0])/self.cfg_task.rl.max_episode_length, 0.0)

            # w Noise Curriculum + structured curriculum beta
            if self.structured_curriculum_scaling_residual == True:
                # Structured curriculum for action scale beta
                beta = self.beta_interpol[self.obs_noise_curricum_stage]
                #print("beta", beta)

            # w Noise Curriculum + Learned beta
            if self.simple_self_scaling_residual == True:        
                # Map to [0,1]
                beta = torch.abs(actions_residual[:, 6]).unsqueeze(-1)
                beta += 1.0
                beta *= 0.5
            else:
                # w/o Noise Curriculum
                beta = torch.ones((self.num_envs, 1), device=self.device)
            
            #Apply beta-scaled residual RL actions:

            new_actions_pos = new_actions[:, 0:3]
            new_actions_rot = new_actions[:, 3:6]
            new_actions_rot_angle = torch.norm(new_actions_rot, p=2, dim=-1)
            new_actions_rot_axis = new_actions_rot / new_actions_rot_angle.unsqueeze(-1)
            new_actions_quat = torch_jit_utils.quat_from_angle_axis(new_actions_rot_angle, new_actions_rot_axis)
            
            actions_residual_pos = actions_residual[:, 0:3]
            actions_residual_rot = actions_residual[:, 3:6]
            
            beta_actions_residual_pos = beta * actions_residual_pos
            beta_actions_residual_rot = beta * actions_residual_rot
            beta_actions_residual_rot_angle = torch.norm(beta_actions_residual_rot, p=2, dim=-1)
            beta_actions_residual_rot_axis = beta_actions_residual_rot / beta_actions_residual_rot_angle.unsqueeze(-1)
            beta_actions_residual_quat = torch_jit_utils.quat_from_angle_axis(beta_actions_residual_rot_angle, beta_actions_residual_rot_axis)

            new_actions_pos = new_actions_pos + beta_actions_residual_pos
            new_actions_quat = torch_utils.quat_mul(new_actions_quat, beta_actions_residual_quat)
            new_actions_rot_angle, new_actions_rot_axis = torch_jit_utils.quat_to_angle_axis(new_actions_quat)
            new_actions_rot = new_actions_rot_angle.unsqueeze(-1) * new_actions_rot_axis
            new_actions = torch.cat([new_actions_pos, new_actions_rot], dim=-1)

            # Map to [-1,1] again
            max_val, _ = torch.max(torch.abs(new_actions), dim=1, keepdims=True)
            max_val[max_val==0.0] = 1.0
            new_actions /= max_val
    
            #print("Total actions:", new_actions[0])

        # If you did not satisfy the condition, new_actions = only the actions of the Potential Field, without any residual RL actions

        # Total actions
        self.actions = new_actions.clone().to(self.device)
        self._apply_actions_as_ctrl_targets_noisy(actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True)


    def pre_physics_step(self, actions_RL):

        """
            Reset environments. 
            Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains.
        """

        ############################################################################################################


        # For visualization of Pose Trajectory:

        se3_pos = torch.cat([self.plug_pos[0, :], self.plug_quat[0, :]], dim=0).to(self.device)
        self.pose_tensor = torch.cat([self.pose_tensor, se3_pos.unsqueeze(0)], dim=0).to(self.device)
    
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
          #np.savetxt('trajectory_pos_noise_{self.cfg_task.env.plug_pos_obs_noise[0]}.txt', self.pose_tensor)
          torch.save(self.pose_tensor, 'trajectory_pos_noise.pt')

        if self.cfg_task.env.printing == True:
          print("actions", self.actions)

        #############################################################################################################

        # Reset environments if needed
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        # -------------------------- POTENTIAL FIELD ------------------------------ #
              
        #Convert sockets and pegs then run residual_servoing attractor

        # C.  

        # i). Joe's version: 


        # ii). Ismarou's version:

        #print("Plug Points Template Joe", self.plug_points_template_Joe)
        #print("Socket Points Template Joe", self.socket_points_template_Joe)

        #print(" ")


        #self.plug_points = update_vertices_parallel(self.plug_points_template_tensor.clone(), self.noisy_plug_quat_R, self.noisy_plug_pos_R)
        #self.socket_points = update_vertices_parallel(self.socket_points_template_tensor.clone(), self.noisy_socket_quat_R, self.noisy_socket_pos_R)


        #self.plug_points_template_tensor_identity = self.apply_transformation(self.plug_points_template_tensor, -self.init_plug_pos, torch_jit_utils.quat_conjugate(self.init_plug_quat))
        #self.socket_points_template_tensor_identity = self.apply_transformation(self.socket_points_template_tensor, -self.init_socket_pos, torch_jit_utils.quat_conjugate(self.init_socket_quat))

        # Update plug point cloud positions
        #self.plug_points_template_tensor_identity = transform_pointcloud(self.plug_points_template_tensor, self.init_plug_pos, self.init_plug_quat)

        # Update socket point cloud positions
        #self.socket_points_template_tensor_identity = transform_pointcloud(self.socket_points_template_tensor, self.init_socket_pos, self.init_socket_quat)

        #print(" ")
        #print("Mean and Std of plug points template identity", torch.mean(self.plug_points_template_tensor_identity, dim=1), torch.std(self.plug_points_template_tensor_identity, dim=1))
        #print("Mean and Std of socket points template identity", torch.mean(self.socket_points_template_tensor_identity, dim=1), torch.std(self.socket_points_template_tensor_identity, dim=1))
        #print(" ")

        self.plug_points = self.apply_transformation(self.plug_points_template_tensor, self.noisy_plug_pos_R, self.noisy_plug_quat_R)
        self.socket_points = self.apply_transformation(self.socket_points_template_tensor, self.noisy_socket_pos_R, self.noisy_socket_quat_R)

        # Call the visualization function with self.plug_points and self.socket_points
        #visualize_plug_and_socket_points(self.plug_points, self.socket_points)


        # D. Define (Noisy) Carrot Trajectory

        # i)   
        #self.Carrot_trajectory = define_trajectory_parallel(self.noisy_socket_pos_R, self.noisy_socket_quat_R, resolution=self.cfg_task.env.Carrot_trajectory_resolution, plug_lengths = self.plug_heights)
        
        # ii): 
        
        
        self.Carrot_trajectory = define_trajectory_parallel_Ismarou(
                                                                        self.noisy_socket_pos, 
                                                                        self.noisy_socket_quat, 
                                                                        init_plug_pos = self.init_plug_pos, 
                                                                        resolution = self.cfg_task.env.Carrot_trajectory_resolution, 
                                                                                                                                                )
        

        #time.sleep(500)

        # E. Potential Field ( calculated inside function ) + Residual RL (input here)
        self.residual_servoing_attractor(self.plug_points, self.socket_points, self.Carrot_trajectory, actions_RL, self.rl_weight)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------POTENTIAL FIELD------------#
    '''
    def apply_transformation(self, pc_vertices, pos, quat):

        # pc_vertices: [envs, points, 3]
        # pos: [envs, 3]
        # quat: [envs, 4]
        
        # Broadcasting pos to [envs, points, 3]
        pos = pos.unsqueeze(1).expand(-1, pc_vertices.size(1), -1)
        
        # Broadcasting quat to [envs, points, 4]
        quat = quat.unsqueeze(1).expand(-1, pc_vertices.size(1), -1)

        # Apply transformations using tf_apply from the Isaac Gym file
        transformed_pc_vertices = torch_jit_utils.tf_apply(quat, pos, pc_vertices)

        return transformed_pc_vertices
    '''        
    
    def apply_transformation(self, pc_vertices, pos, quat):
        # pc_vertices: [envs, points, 3]
        # pos: [envs, 3]
        # quat: [envs, 4]

        # Reshape pos and quat to match the number of points in pc_vertices
        num_envs, num_points, _ = pc_vertices.shape
        pos_repeated = pos.unsqueeze(1).repeat(1, num_points, 1)
        quat_repeated = quat.unsqueeze(1).repeat(1, num_points, 1)

        # Apply transformations using tf_apply from the Isaac Gym file
        transformed_pc_vertices = torch_jit_utils.tf_apply(quat_repeated, pos_repeated, pc_vertices)

        return transformed_pc_vertices


    def apply_transformation_pc(self, pc, pos, quat):
        # Convert PointCloud object to NumPy array
        pc_vertices = np.asarray(pc.vertices)

        # Convert NumPy array to PyTorch tensor
        pc_vertices = torch.from_numpy(pc_vertices).float().to(self.device)

        # Reshape pos and quat to match the number of points in pc_vertices
        num_points = pc_vertices.shape[0]
        pos_repeated = pos.repeat(num_points, 1)
        quat_repeated = quat.repeat(num_points, 1)

        # Apply transformations using tf_apply from the Isaac Gym file
        transformed_pc_vertices = torch_jit_utils.tf_apply(quat_repeated, pos_repeated, pc_vertices)

        # Convert transformed vertices back to PointCloud object
        transformed_pc = trimesh.points.PointCloud(transformed_pc_vertices.cpu().numpy())

        return transformed_pc


    def post_physics_step(self):

        """
            Step buffers. 
            Refresh tensors. 
            Compute observations and reward.
        """

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        
        """
            Compute observations.
        """

        self.gripper_goal_pos = self.socket_pos.clone()
        self.noisy_gripper_goal_pos = self.noisy_socket_pos.clone()

        delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        delta_quat = torch_jit_utils.quat_mul(self.gripper_goal_quat, torch_jit_utils.quat_conjugate(self.fingertip_centered_quat))
        noisy_delta_quat = torch_jit_utils.quat_mul(self.noisy_gripper_goal_quat, torch_jit_utils.quat_conjugate(self.fingertip_centered_quat))

        delta_plug_pos = self.socket_pos - self.plug_pos
        noisy_delta_plug_pos = self.noisy_socket_pos - self.noisy_plug_pos

        delta_plug_quat = torch_jit_utils.quat_mul(self.socket_quat, torch_jit_utils.quat_conjugate(self.plug_quat))
        noisy_delta_plug_quat = torch_jit_utils.quat_mul(self.noisy_socket_quat, torch_jit_utils.quat_conjugate(self.noisy_plug_quat))

        # Define observations (for actor)

        obs_tensors = [
                            self.pose_world_to_robot_base_matrix(self.noisy_plug_pos, self.noisy_plug_quat)[0],  # 3
                            self.pose_world_to_robot_base_matrix(self.noisy_plug_pos, self.noisy_plug_quat)[1],  # 9
                            self.pose_world_to_robot_base_matrix(self.noisy_socket_pos, self.noisy_socket_quat)[0],  # 3
                            self.pose_world_to_robot_base_matrix(self.noisy_socket_pos, self.noisy_socket_quat)[1],  # 9
                                                                                                                                        ] # 24
        



        # Define state (for critic)
        
        state_tensors = [ 
                            #self.arm_dof_pos,  # 7
                            #self.arm_dof_vel,  # 7
                            #self.pose_world_to_robot_base_matrix(  self.fingertip_centered_pos, self.fingertip_centered_quat )[0],  # 3
                            #self.pose_world_to_robot_base_matrix(  self.fingertip_centered_pos, self.fingertip_centered_quat )[1],  # 9
                            #self.fingertip_centered_linvel,  # 3
                            #self.fingertip_centered_angvel,  # 3
                            
                            #self.pose_world_to_robot_base_matrix( self.gripper_goal_pos, self.gripper_goal_quat)[0],  # 3
                            #self.pose_world_to_robot_base_matrix( self.gripper_goal_pos, self.gripper_goal_quat)[1],  # 9
                            #self.pose_world_to_robot_base_matrix( self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0],  # 3
                            #self.pose_world_to_robot_base_matrix( self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1],  # 9
                            
                            #delta_pos,  # 3
                            #delta_quat,  # 4

                            self.pose_world_to_robot_base_matrix( self.plug_pos, self.plug_quat )[0],  # 3
                            self.pose_world_to_robot_base_matrix( self.plug_pos, self.plug_quat )[1],  # 9
                            self.pose_world_to_robot_base_matrix( self.socket_pos, self.socket_quat )[0],  # 3
                            self.pose_world_to_robot_base_matrix( self.socket_pos, self.socket_quat )[1],  # 9
                            
                            self.pose_world_to_robot_base_matrix( delta_plug_pos, delta_plug_quat )[0],  # 9
                            self.pose_world_to_robot_base_matrix( delta_plug_pos, delta_plug_quat )[1],  # 9

                            #delta_plug_pos,  # 3
                            #delta_plug_quat,  # 4

                            self.pose_world_to_robot_base_matrix( self.noisy_plug_pos, self.noisy_plug_quat )[0],  # 3
                            self.pose_world_to_robot_base_matrix( self.noisy_plug_pos, self.noisy_plug_quat )[1],  # 9
                            self.pose_world_to_robot_base_matrix( self.noisy_socket_pos, self.noisy_socket_quat )[0],  # 3
                            self.pose_world_to_robot_base_matrix( self.noisy_socket_pos, self.noisy_socket_quat )[1],  # 9
                            
                            
                            self.pose_world_to_robot_base_matrix( noisy_delta_plug_pos, noisy_delta_plug_quat )[0],  # 9
                            self.pose_world_to_robot_base_matrix( noisy_delta_plug_pos, noisy_delta_plug_quat )[1],  # 9

                            #noisy_delta_plug_pos,  # 3
                            #noisy_delta_plug_quat,  # 4

                            #noisy_delta_pos - delta_pos, # 3
                            #noisy_delta_plug_pos - delta_plug_pos, # 3
                    
                            #torch_jit_utils.quat_mul(noisy_delta_quat, torch_jit_utils.quat_conjugate(delta_quat)), # 4
                            #torch_jit_utils.quat_mul(noisy_delta_plug_quat, torch_jit_utils.quat_conjugate(delta_plug_quat)), # 4
                            
                            
                            self.pose_world_to_robot_base_matrix( noisy_delta_plug_pos - delta_plug_pos, torch_jit_utils.quat_mul(noisy_delta_plug_quat, torch_jit_utils.quat_conjugate(delta_plug_quat)) )[0],  # 9
                            self.pose_world_to_robot_base_matrix( noisy_delta_plug_pos - delta_plug_pos, torch_jit_utils.quat_mul(noisy_delta_plug_quat, torch_jit_utils.quat_conjugate(delta_plug_quat)) )[1],  # 9

                                                                                                                                             ] #84 # 139 
    


        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.states_buf = torch.cat(state_tensors, dim=-1)
        return self.obs_buf
    

    def compute_reward(self):

        """
            Detect successes and failures. 
            Update reward and reset buffers.
        """

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        
        """
        
            Compute reward at current timestep.
        
            They will apply the rewards properly later. This is just the immediate reward for the given timestep as the self.rew_buf values.
            
            Just use a sparse reward for engagement or insertion that checks each step
            
            If we high interpenerate zero the reward and if we low interpenetrate then it should stay same reward value
            
            At the end here we check for if its the final step so I make noise adjustments based on success and shift the correlated noise
        
        """

        self.prev_rew_buf = self.rew_buf.clone()

        # SAPU: Compute reward scale based on interpenetration distance
        low_interpen_envs, high_interpen_envs = [], []
        ( low_interpen_envs, high_interpen_envs, sapu_reward_scale, ) = algo_utils.get_sapu_reward_scale(
                                                                                                            asset_indices=self.asset_indices,
                                                                                                            plug_pos=self.plug_pos,
                                                                                                            plug_quat=self.plug_quat,
                                                                                                            socket_pos=self.socket_pos,
                                                                                                            socket_quat=self.socket_quat,
                                                                                                            wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
                                                                                                            wp_socket_meshes=self.wp_socket_meshes,
                                                                                                            interpen_thresh=self.cfg_task.rl.interpen_thresh,
                                                                                                            wp_device=self.wp_device,
                                                                                                            device=self.device,
                                                                                                                                                                                        )

        #After calculing interpenetration in the simualtor, check engagements:

        # Success bonus: Check which envs have plug engaged (partially inserted) or fully inserted
        is_plug_engaged_w_socket = algo_utils.check_plug_engaged_w_socket(
                                                                                plug_pos=self.plug_pos,
                                                                                socket_top_pos=self.socket_top_pos,
                                                                                keypoints_plug=self.keypoints_plug,
                                                                                keypoints_socket=self.keypoints_socket,
                                                                                cfg_task=self.cfg_task,
                                                                                progress_buf=self.progress_buf,
                                                                                                                                )
        
        is_plug_inserted_in_socket = algo_utils.check_plug_inserted_in_socket(
                                                                                    plug_pos=self.plug_pos,
                                                                                    socket_pos=self.socket_pos,
                                                                                    keypoints_plug=self.keypoints_plug,
                                                                                    keypoints_socket=self.keypoints_socket,
                                                                                    cfg_task=self.cfg_task,
                                                                                    progress_buf=self.progress_buf,
                                                                                                                                    )
        
        is_plug_halfway_inserted_in_socket = algo_utils.check_plug_halfway_inserted_in_socket(
                                                                                                    plug_pos=self.plug_pos,
                                                                                                    socket_top_pos=self.socket_top_pos,
                                                                                                    keypoints_plug=self.keypoints_plug,
                                                                                                    keypoints_socket=self.keypoints_socket,
                                                                                                    cfg_task=self.cfg_task,
                                                                                                    progress_buf=self.progress_buf,
                                                                                                                                                    )

        # Success bonus: Apply sparse rewards for engagement and insertion
        # Make sure to do an equals here first because if you grey out the other options of setting = then just adds rewards perpetually and this creates just higher rewards as you go, ruining training
        
        #self.rew_buf[:] = 1.0*is_plug_engaged_w_socket #self.cfg_task.rl.engagement_bonus 
        #self.rew_buf[:] += 100.0*is_plug_inserted_in_socket #self.cfg_task.rl.engagement_bonus

        self.rew_buf[:] = 100.0*is_plug_inserted_in_socket #self.cfg_task.rl.engagement_bonus


        if len(high_interpen_envs) > 0:
            self.rew_buf[high_interpen_envs] = 0.0 #0.1 * (self.prev_rew_buf[high_interpen_envs]) #Scale down reward of previous step I guess
        
            
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        
        if is_last_step:
          
          # Success bonus: Log success rate, ignoring environments with large interpenetration
          if len(high_interpen_envs) > 0:
              is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[low_interpen_envs]
              
              self.extras["insertion_successes"] = torch.sum(is_plug_inserted_in_socket_low_interpen.float()) / self.num_envs
              if len(low_interpen_envs) == 0:
                self.extras["insertion_successes"] = 0.0
              print("percentage_plug_inserted_in_socket with low interpenetration:", self.extras["insertion_successes"] * 100.0)
              percent_inserted_nointerpen_just_from_there=torch.mean( is_plug_inserted_in_socket_low_interpen.float() )
              print("percentage_plug_inserted_in_socket from low interpenetration that inserted:", percent_inserted_nointerpen_just_from_there * 100.0)
          else:
              self.extras["insertion_successes"] = torch.mean(is_plug_inserted_in_socket.float())
              print("percentage_plug_inserted_in_socket with low interpenetration:", self.extras["insertion_successes"] * 100.0)

          # SBC: Log current max downward displacement of plug at beginning of episode
          self.extras["curr_max_disp"] = self.curr_max_disp

          # SBC: Update curriculum difficulty based on success rate
          self.curr_max_disp = algo_utils.get_new_max_disp( curr_success=self.extras["insertion_successes"], cfg_task=self.cfg_task, curr_max_disp=self.curr_max_disp,)

          '''
          Calculate and Print the percentage of pegs that were engaged with the socke
          '''

          is_plug_engaged_w_socket_episode = torch.sum(is_plug_engaged_w_socket).item()
          print("is_plug_engaged_w_socket_episode:", is_plug_engaged_w_socket_episode, "out of", self.num_envs)
          
          # Calculate the percentage of pegs that were engaged with the socket
          percentage_plug_engaged_w_socket = (is_plug_engaged_w_socket_episode/self.num_envs)*100
          print("percentage_plug_engaged_w_socket:", percentage_plug_engaged_w_socket)

          is_plug_halfway_inserted_in_socket_episode = torch.sum(is_plug_halfway_inserted_in_socket).item()
          print("is_plug_halfway_inserted_in_socket_episode:", is_plug_halfway_inserted_in_socket_episode, "out of", self.num_envs)
          
          # Calculate the percentage of pegs that were engaged with the socket
          percentage_is_plug_halfway_inserted_in_socket = (is_plug_halfway_inserted_in_socket_episode/self.num_envs)*100
          print("percentage_is_plug_halfway_inserted_in_socket:", percentage_is_plug_halfway_inserted_in_socket)

          '''
            Calculate and Print the percentage of pegs that were inserted in the socket
          '''
          
          is_plug_inserted_in_socket_episode = torch.sum(is_plug_inserted_in_socket).item()
          print("is_plug_inserted_in_socket_episode:", is_plug_inserted_in_socket_episode, "out of", self.num_envs)
          
          # Calculate the percentage of pegs that were inserted in the socket
          percentage_plug_inserted_in_socket = (is_plug_inserted_in_socket_episode/self.num_envs)*100
          self.extras["sapu_adjusted_reward"] = percentage_plug_inserted_in_socket #Just use this so we can see the value on wandb
          print("percentage_plug_inserted_in_socket:", percentage_plug_inserted_in_socket)
          
          #Updates that it is the last step
          #Based on the final success rate adjust the noise parameters
          if self.cfg_task.env.noise_shift_with_success_rate == True:
          
            print("Current Obs Noise Curriculum Stage:", self.obs_noise_curricum_stage, self.socket_obs_pos_curriculum_steps[self.obs_noise_curricum_stage, 0])
            self.extras["sdf_reward"] = self.plug_obs_pos_curriculum_steps[self.obs_noise_curricum_stage, 0] #Use this as a proxy for current peg pos noise
          
            #Scale then print the noise values. So we have it as a scale of percentages. So we choose a max noise value and the percent success times that is the noise we choose
            scaling_percent_inserted_nointerpen = torch.sum(is_plug_inserted_in_socket.float()).item() / self.num_envs
            scaling_percent_inserted_not_considering_interpen = scaling_percent_inserted_nointerpen
            if len(high_interpen_envs) > 0:
              if len(low_interpen_envs) > 0: #If we still have some low_interpen_envs
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[low_interpen_envs]
              else: #All high interpen so just zeros
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket * 0.0
              scaling_percent_inserted_nointerpen = torch.sum(is_plug_inserted_in_socket_low_interpen.float()).item() / self.num_envs
            

            # Noise Curriculum: Update Obs Noise curriculum difficulty based on success rate
            self.obs_noise_curricum_stage = algo_utils.get_new_obs_noise_curr_step(
                                                                                        curr_success = is_plug_inserted_in_socket_episode/self.num_envs, 
                                                                                        cfg_task=self.cfg_task, 
                                                                                        current_obs_noise_curricum_stage = self.obs_noise_curricum_stage, 
                                                                                        max_curriculum_steps = self.num_curriculum_steps_obs
                                                                                                                                                            
                                                                                                                                                                      )



    def _update_reset_buf(self):
        
        """
            Assign environments for reset if maximum episode length has been reached.
        """

        self.reset_buf[:] = torch.where(
                                            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                            torch.ones_like(self.reset_buf),
                                            self.reset_buf,
                                                                                                                                )

    def reset_idx(self, env_ids):
        
        """
            Reset specified environments.
        """

        self._reset_franka()

        # Close gripper onto plug
        self.disable_gravity()  # to prevent plug from falling
        self._reset_object()
        self._move_gripper_to_grasp_pose(sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
        self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        
        
        steps_total = self.cfg_task.env.init_rot_random_steps
        plug_rot_quat = None

        target_position = self.plug_pos.clone()

        for i in range(steps_total):
        
          fix_actions, plug_rot_quat = self.create_rotational_randomization_axis_angle(plug_rot_quat = plug_rot_quat, target_position = target_position)
        
          fix_actions[:, 0:3] *= 10.0
          fix_actions[:, 3:6] *= 10.0
        
          if i > int(steps_total/2):
            fix_actions[:, 0:3] *= 20.0
            fix_actions[:, 3:6] *= 0.0

          self._apply_actions_as_ctrl_targets_noisy(actions = fix_actions, ctrl_target_gripper_dof_pos = 0.0, do_scale=True)
          self.simulate_and_refresh()

        print("######################FINISHED SETTING##########################################\n")


        self.init_plug_pos = self.plug_pos.clone()        
        self.init_plug_quat = self.plug_quat.clone()
        self.init_socket_pos = self.socket_pos.clone()
        self.init_socket_quat = self.socket_quat.clone()
        
        self.init_plug_pos_R = self.pose_world_to_robot_base(self.init_plug_pos, self.init_plug_quat)[0]
        self.init_plug_quat_R = self.pose_world_to_robot_base(self.init_plug_pos, self.init_plug_quat)[1]
        self.init_socket_pos_R = self.pose_world_to_robot_base(self.init_socket_pos, self.init_socket_quat)[0]
        self.init_socket_quat_R = self.pose_world_to_robot_base(self.init_socket_pos, self.init_socket_quat)[1]

        self.enable_gravity()

        # Get plug SDF in goal pose for SDF-based reward
        self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
                                                                wp_plug_meshes = self.wp_plug_meshes,
                                                                asset_indices = self.asset_indices,
                                                                socket_pos = self.socket_pos,
                                                                socket_quat = self.socket_quat,
                                                                wp_device = self.wp_device,
                                                                                                            )

        self._reset_buffers()


    def _reset_franka(self):

        """
            Reset DOF states, DOF torques, and DOF targets of Franka.
        """

        # Randomize DOF pos
        self.dof_pos[:] = torch.cat(
                                        (
                                            torch.tensor( self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device, ),
                                            torch.tensor( [self.asset_info_franka_table.franka_gripper_width_max], device=self.device,),
                                            torch.tensor( [self.asset_info_franka_table.franka_gripper_width_max], device=self.device,),
                                                                                                                                            ), dim=-1,).unsqueeze(0)  # shape = (num_envs, num_dofs)

        # Stabilize Franka
        self.dof_vel[:, :] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[:, :] = 0.0
        self.ctrl_target_dof_pos = self.dof_pos.clone()
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

        # Set DOF state
        franka_actor_ids_sim = self.franka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
                                                    self.sim,
                                                    gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(franka_actor_ids_sim),
                                                    len(franka_actor_ids_sim),
                                                                                                                                )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
                                                            self.sim,
                                                            gymtorch.unwrap_tensor(self.dof_torque),
                                                            gymtorch.unwrap_tensor(franka_actor_ids_sim),
                                                            len(franka_actor_ids_sim),
                                                                                                                            )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_object(self):

        """
            Reset root state of plug and socket.
        """

        self._reset_socket()
        self._reset_plug(before_move_to_grasp=True)


    def _reset_socket(self):
        
        """
            Reset root state of socket.
        """

        # Randomize socket pos
        socket_noise_xy = 2 * ( torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5 )
        socket_noise_xy = socket_noise_xy @ torch.diag( torch.tensor( self.cfg_task.randomize.socket_pos_xy_noise, dtype=torch.float32, device=self.device,))
        socket_noise_z = torch.zeros( (self.num_envs), dtype=torch.float32, device=self.device )
        socket_noise_z_mag = (self.cfg_task.randomize.socket_pos_z_noise_bounds[1] - self.cfg_task.randomize.socket_pos_z_noise_bounds[0] )
        socket_noise_z = ( socket_noise_z_mag * torch.rand((self.num_envs), dtype=torch.float32, device=self.device) + self.cfg_task.randomize.socket_pos_z_noise_bounds[0] )

        self.socket_pos[:, 0] = ( self.robot_base_pos[:, 0] + self.cfg_task.randomize.socket_pos_xy_initial[0] + socket_noise_xy[:, 0])
        self.socket_pos[:, 1] = ( self.robot_base_pos[:, 1] + self.cfg_task.randomize.socket_pos_xy_initial[1] + socket_noise_xy[:, 1] )
        self.socket_pos[:, 2] = self.cfg_base.env.table_height + socket_noise_z

        # Randomize socket rot
        socket_rot_noise = 2 * ( torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5 )
        socket_rot_noise = socket_rot_noise @ torch.diag(torch.tensor( self.cfg_task.randomize.socket_rot_noise, dtype=torch.float32, device=self.device,))

        socket_rot_euler = ( torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) + socket_rot_noise )
        
        socket_rot_quat = torch_utils.quat_from_euler_xyz(socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2])
        self.socket_quat[:, :] = socket_rot_quat.clone()

        # Stabilize socket
        self.socket_linvel[:, :] = 0.0
        self.socket_angvel[:, :] = 0.0

        # Set socket root state
        socket_actor_ids_sim = self.socket_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
                                                        self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state),
                                                        gymtorch.unwrap_tensor(socket_actor_ids_sim),
                                                        len(socket_actor_ids_sim),
                                                                                                                        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_plug(self, before_move_to_grasp):
        
        """
            Reset root state of plug.
        """

        if before_move_to_grasp:
        
            self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[0] #ADDED----------------------------------------------------------------------
        
            # Generate plug pos noise
            self.plug_pos_xy_noise = 2 * ( torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5 )
            self.plug_pos_xy_noise = self.plug_pos_xy_noise @ torch.diag( torch.tensor( self.cfg_task.randomize.plug_pos_xy_noise, dtype=torch.float32, device=self.device,))

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement

        self.plug_pos[:, :] = self.socket_pos.clone()
        self.plug_pos[:, 2] += self.socket_heights
        self.plug_pos[:, 2] -= self.curriculum_disp
        
        #------------------------------------------------------------------Added

        # Apply XY noise to plugs not partially inserted into sockets
        socket_top_height = self.socket_pos[:, 2] + self.socket_heights
        plug_partial_insert_idx = np.argwhere(self.plug_pos[:, 2].cpu().numpy() > socket_top_height.cpu().numpy() ).squeeze()
        self.plug_pos[plug_partial_insert_idx, :2] += self.plug_pos_xy_noise[plug_partial_insert_idx]

        self.plug_quat[:, :] = self.identity_quat.clone()
        
        # Randomize plug rot -----------------------------------------------------------------------------------------------------
        plug_rot_noise = 2 * ( torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5 )
        plug_rot_noise = plug_rot_noise @ torch.diag( torch.tensor( self.cfg_task.randomize.plug_rot_noise, dtype=torch.float32, device=self.device,))
        plug_rot_euler = (torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) + plug_rot_noise)
        plug_rot_quat = torch_utils.quat_from_euler_xyz(plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2] )
        
        #--------------------------------------------------------------------------------------------------------------------------

        # Stabilize plug
        self.plug_linvel[:, :] = 0.0
        self.plug_angvel[:, :] = 0.0

        # Set plug root state
        plug_actor_ids_sim = self.plug_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
                                                        self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state),
                                                        gymtorch.unwrap_tensor(plug_actor_ids_sim),
                                                        len(plug_actor_ids_sim),
                                                                                                                    )

        # Simulate one step to apply changes
        self.simulate_and_refresh()


    def _reset_buffers(self):
        
        """
            Reset buffers.
        """

        self.reset_buf[:] = 0
        self.progress_buf[:] = 0

    def _set_viewer_params(self):
        
        """
            Set viewer parameters.
        """

        cam_pos = gymapi.Vec3(-1.0, -1.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    
    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        
        # Apply actions from policy as position/rotation targets.

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_centered_pos = (self.fingertip_centered_pos + pos_actions)
        
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                                                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                                rot_actions_quat,
                                                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
                                                                                                                                            )
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()
    

    def DG_IndustReal(self):

        inv_Pose_Gripper_quat, inv_Pose_Gripper_t = torch_jit_utils.tf_inverse(self.fingertip_centered_quat, self.fingertip_centered_pos)

        curr_plug_pos = self.noisy_plug_pos
        curr_plug_quat = self.noisy_plug_quat

        DeltaG_quat, DeltaG_t = torch_jit_utils.tf_combine(inv_Pose_Gripper_quat, inv_Pose_Gripper_t, curr_plug_quat, curr_plug_pos)
        inv_DeltaG_quat, inv_DeltaG_t = torch_jit_utils.tf_inverse(DeltaG_quat, DeltaG_t)

        return inv_DeltaG_t, inv_DeltaG_quat
    
    def apply_invDG_IndustReal_insert(self, target_pos, target_quat, inv_DeltaG_t, inv_DeltaG_quat, device):

        target_Pose_Gripper_quat, target_Pose_Gripper_t = torch_jit_utils.tf_combine(target_quat, target_pos, inv_DeltaG_quat, inv_DeltaG_t)
        return target_Pose_Gripper_t, target_Pose_Gripper_quat

    def refresh_and_acquire_tensors(self):
        
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self._acquire_env_tensors()
    
    def _apply_actions_as_ctrl_targets_noisy(self, actions, ctrl_target_gripper_dof_pos, do_scale):
            
            """
                Apply actions from policy as position/rotation targets.
            """

            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
            

            ############ Control Object Pose ############
            target_pos = self.noisy_plug_pos + pos_actions
            ############################################
            
            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6]
            if do_scale:
                rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

            # Convert to quat and set rot target
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(
                                                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                                rot_actions_quat,
                                                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
                                                                                                                                        )
            
            

            ####################### Control Object Pose ################################
            target_quat = torch_utils.quat_mul(rot_actions_quat, self.noisy_plug_quat)
            ############################################################################
            
            target_pos = target_pos.to(self.device)
            target_quat = target_quat.to(self.device)
            target_pos = target_pos.unsqueeze(0) if target_pos.dim() == 1 else target_pos
            target_quat = target_quat.unsqueeze(0) if target_quat.dim() == 1 else target_quat
                
            ############################## Control Object Pose (DG Transformation ###############################################
            inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal()
    
            self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
            #####################################################################################################################
    
            s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                                                                            fingertip_midpoint_pos = self.fingertip_centered_pos,
                                                                            fingertip_midpoint_quat = self.fingertip_centered_quat,
                                                                            ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_centered_pos,
                                                                            ctrl_target_fingertip_midpoint_quat = self.ctrl_target_fingertip_centered_quat,
                                                                            jacobian_type=self.cfg_ctrl['jacobian_type'],
                                                                            rot_error_type='axis_angle'
                                                                                                                                                                    )

            s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
            s_t_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions_control), device = self.device)
            s_t_actions[:, :6] = s_t_delta_hand_pose
        
            s_t_actions_pos = s_t_actions[:, :3]
            self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + s_t_actions_pos

            s_t_actions_rot = s_t_actions[:, 3:]
            s_t_angle = torch.norm(s_t_actions_rot, p=2, dim=-1)
            s_t_axis = s_t_actions_rot / s_t_angle.unsqueeze(-1)
            s_t_rot_actions_quat = torch_utils.quat_from_angle_axis(s_t_angle, s_t_axis)
            self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(s_t_rot_actions_quat, self.fingertip_centered_quat)

            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

            self.generate_ctrl_signals()
            self.refresh_and_acquire_tensors()

                
    def _apply_actions_as_ctrl_targets_randomize_rotations(self, actions, ctrl_target_gripper_dof_pos, do_scale):
            
            """
                Apply actions from policy as position/rotation targets.
            """

            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))


            target_pos = self.plug_pos + pos_actions
            
            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6]
            if do_scale:
                rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

            # Convert to quat and set rot target
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(
                                                    angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                                    rot_actions_quat,
                                                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
                                                                                                                                                        )

            target_quat = torch_utils.quat_mul(rot_actions_quat, self.noisy_plug_quat)

                                                        
            target_pos = target_pos.to(self.device)
            target_quat = target_quat.to(self.device)
            target_pos = target_pos.unsqueeze(0) if target_pos.dim() == 1 else target_pos
            target_quat = target_quat.unsqueeze(0) if target_quat.dim() == 1 else target_quat

            
            inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal()
            self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)

            s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                                                                        fingertip_midpoint_pos = self.fingertip_centered_pos,
                                                                        fingertip_midpoint_quat = self.fingertip_centered_quat,
                                                                        ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_centered_pos,
                                                                        ctrl_target_fingertip_midpoint_quat = self.ctrl_target_fingertip_centered_quat,
                                                                        jacobian_type=self.cfg_ctrl['jacobian_type'],
                                                                        rot_error_type='axis_angle'
                                                                                                                                                                )

            s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
            s_t_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions_control), device = self.device)
            s_t_actions[:, :6] = s_t_delta_hand_pose
        
            s_t_actions_pos = s_t_actions[:, :3]
            self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + s_t_actions_pos

            s_t_actions_rot = s_t_actions[:, 3:]
            s_t_angle = torch.norm(s_t_actions_rot, p=2, dim=-1)
            s_t_axis = s_t_actions_rot / s_t_angle.unsqueeze(-1)
            s_t_rot_actions_quat = torch_utils.quat_from_angle_axis(s_t_angle, s_t_axis)
            self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(s_t_rot_actions_quat, self.fingertip_centered_quat)

            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

            self.generate_ctrl_signals()
            self.refresh_and_acquire_tensors()

            self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
        
            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
            self.generate_ctrl_signals()


    def _move_gripper_to_grasp_pose(self, sim_steps):
        
        """
            Define grasp pose for plug and move gripper to pose.
        """

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = self.plug_pos.clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += self.plug_grasp_offsets

        # Set target rot
        ctrl_target_fingertip_centered_euler = (torch.tensor(self.cfg_task.randomize.fingertip_centered_rot_initial, device=self.device,).unsqueeze(0).repeat(self.num_envs, 1))

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
                                                                                        ctrl_target_fingertip_centered_euler[:, 0],
                                                                                        ctrl_target_fingertip_centered_euler[:, 1],
                                                                                        ctrl_target_fingertip_centered_euler[:, 2],
                                                                                                                                                )

        self.move_gripper_to_target_pose( gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max, sim_steps=sim_steps, )

        # Reset plug in case it is knocked away by gripper movement
        self._reset_plug(before_move_to_grasp=False)


    def pose_world_to_robot_base_matrix(self, pos, quat):
        
        """
            Convert pose from world frame to robot base frame.
        """

        # Convert
        robot_base_transform_inv = torch_utils.tf_inverse(self.robot_base_quat, self.robot_base_pos)
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos)
        return pos_in_robot_base, torch_jit_utils.quaternion_to_matrix(quat_in_robot_base).reshape(self.num_envs, -1)
        

    def pose_world_to_hand_base_matrix(self, pos, quat):
        
        """
            Convert pose from world frame to robot base frame.
        """

        # Convert TODO should we clone here?
        robot_base_transform_inv = torch_utils.tf_inverse(self.fingertip_centered_quat.clone(), self.fingertip_centered_pos.clone())
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos)
        return pos_in_robot_base, torch_jit_utils.quaternion_to_matrix(quat_in_robot_base).reshape(self.num_envs, -1)
