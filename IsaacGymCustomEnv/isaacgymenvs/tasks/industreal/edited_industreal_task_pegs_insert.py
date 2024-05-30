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

"""IndustReal: class for peg insertion task.

    Inherits IndustReal pegs environment class and Factory abstract task class (not enforced).

    Trains a peg insertion policy with Simulation-Aware Policy Update (SAPU), SDF-Based Reward, and Sampling-Based Curriculum (SBC).

    Can be executed with python train.py task=IndustRealTaskPegsInsert.
"""

import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils import torch_jit_utils
"""
import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp

from scipy.spatial import KDTree

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np
#import pyrallis
import wandb
#from torch.quaternion import as_quat_array, as_float_array
import warp as wp

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (FactorySchemaConfigTask,)
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils.torch_jit_utils import *
import isaacgymenvs.tasks.factory.factory_control as fc

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

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (FactorySchemaConfigTask,)
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils import torch_jit_utils


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm_


import time
import re
import os

import cv2

import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import omegaconf
import matplotlib.pyplot as plt

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

torch.set_printoptions(sci_mode=False)
"""


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
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
        #action = self.max_action * self.net(state)
        action = self.net(state)
        #action[0:3] *= self.max_action
        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


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
            """Initialize instance variables. Initialize task superclass."""

            self.cfg = cfg
            self._get_task_yaml_params()

            super().__init__(
                                cfg,
                                rl_device,
                                sim_device,
                                graphics_device_id,
                                headless,
                                virtual_screen_capture,
                                force_render,
                                                                )

            self._acquire_task_tensors()
            self.parse_controller_spec()

            # Get Warp mesh objects for SAPU and SDF-based reward
            wp.init()
            self.wp_device = wp.get_preferred_device()
            ( self.wp_plug_meshes, self.wp_plug_meshes_sampled_points, self.wp_socket_meshes, ) = algo_utils.load_asset_meshes_in_warp(
                                                                                                                                            plug_files=self.plug_files,
                                                                                                                                            socket_files=self.socket_files,
                                                                                                                                            num_samples=self.cfg_task.rl.sdf_reward_num_samples,
                                                                                                                                            device=self.wp_device,
                                                                                                                                                                                                    )

            if self.viewer != None:
                self._set_viewer_params()

            self.reset_threshold = 200  # Define the threshold for reset
            self.iterations_since_last_reset = 0  # Track the iterations


            self.run_actions = False

            self.residuals = True
            self.sparse_rewards = True
            self.RRT_Star_Reset = True

            self.replan_states = []
            self.reset_replan_episodes = 50
            self.replan_episodes = 0


            # In your IndustRealTaskPegsInsert class initializer
            actor =  Actor(state_dim_net, action_dim_net, max_action).to("cuda:0")
            trainer = Loaded_Actor(state_dim, action_dim, max_action)
            trainer = trainer.to(device)
            policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_29999.pt"
            print(f"Loading model from {policy_file}")
            actor_weights=torch.load(policy_file)['actor']

            self.actor_model = .load_state_dict(actor_weights)"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_49999.pt")
            

            

            '''

            self.num_cameras = 2  # Number of cameras
            self.camera_handles = []
            self.rgb_tensors = []

            # Camera properties
            self.camera_eye_list = [[0.5, 0.0, 0.3], [-0.5, 0.0, 0.3]]
            self.camera_lookat_list = [[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
            self.camera_props = gymapi.CameraProperties()

            for i_env in range(self.num_envs):
                env_ptr = self.gym.create_env(self.gym, gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1), i_env)
                for i_cam in range(self.num_cameras):
                    camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                    self.camera_handles.append(camera_handle)

                    camera_eye = self.camera_eye_list[i_cam]
                    camera_lookat = self.camera_lookat_list[i_cam]
                    self.gym.set_camera_location(camera_handle, env_ptr, camera_eye, camera_lookat)

                    camera_tensor_rgb = self.gym.get_camera_image_gpu_tensor(self.gym, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                    torch_cam_rgb_tensor = gymtorch.wrap_tensor(camera_tensor_rgb)
                    self.rgb_tensors.append(torch_cam_rgb_tensor)

            # Video writing setup
            self.width, self.height = 640, 480  # Set the width and height according to your camera settings
            self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.video_writer = cv2.VideoWriter('simulation_video_BC.mp4', self.fourcc, 30.0, (self.width, self.height))


            # From DexterousManipulation github repo:

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 768
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            '''

            self.file_path = '/home/ismarou/Downloads/IsaacGymEnvs_Test/IsaacGymEnvs_Test/IsaacGymEnvs/isaacgymenvs/interpolated_RRT_Star_4.txt'
            self.file_path_RRT = '/home/ismarou/Downloads/IsaacGymEnvs_Test/IsaacGymEnvs_Test/IsaacGymEnvs/isaacgymenvs/pih_0001_tree.txt'
            
            self.replay_buffer = fill_buffer(self.file_path, state_dim, action_dim, buffer_size)
            self.replay_buffer_RRT = fill_buffer_RRT(self.file_path_RRT, self.socket_pos, self.socket_quat, state_dim, buffer_size)

            # After populating the buffer, build a KD Tree
            self.replay_buffer.build_kd_tree()
            self.replay_buffer_RRT.build_kd_tree()


        def _get_task_yaml_params(self):
            """Initialize instance variables from YAML files."""

            cs = hydra.core.config_store.ConfigStore.instance()
            cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

            self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
            self.max_episode_length = (self.cfg_task.rl.max_episode_length)  # required instance var for VecTask

            ppo_path = os.path.join("train/IndustRealTaskPegsInsertPPO.yaml")  # relative to Gym's Hydra search path (cfg dir)
            self.cfg_ppo = hydra.compose(config_name=ppo_path)
            self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

        def _acquire_task_tensors(self):
            """Acquire tensors."""

            self.identity_quat = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))

            # Compute pose of gripper goal and top of socket in socket frame
            self.gripper_goal_pos_local = torch.tensor([[0.0,0.0,(self.cfg_task.env.socket_base_height + self.plug_grasp_offsets[i]),] for i in range(self.num_envs)], device=self.device,)
            self.gripper_goal_quat_local = self.identity_quat.clone()

            self.socket_top_pos_local = torch.tensor([[0.0, 0.0, self.socket_heights[i]] for i in range(self.num_envs)], device=self.device,)
            self.socket_quat_local = self.identity_quat.clone()

            # Define keypoint tensors
            self.keypoint_offsets = (algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device) * self.cfg_task.rl.keypoint_scale)
            self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32, device=self.device,)
            self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

            self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

            self.curr_max_disp = self.cfg_task.rl.initial_max_disp

        def _refresh_task_tensors(self):
            """Refresh tensors."""

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

            # Add observation noise to socket pos
            self.noisy_socket_pos = torch.zeros_like( self.socket_pos, dtype=torch.float32, device=self.device )
            socket_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(torch.tensor(self.cfg_task.env.socket_pos_obs_noise, dtype=torch.float32, device=self.device,))

            self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
            self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
            self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

            # Add observation noise to socket rot
            socket_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            socket_obs_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(torch.tensor(self.cfg_task.env.socket_rot_obs_noise, dtype=torch.float32, device=self.device,))

            socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
            self.noisy_socket_quat = torch_utils.quat_from_euler_xyz(socket_obs_rot_euler[:, 0],socket_obs_rot_euler[:, 1],socket_obs_rot_euler[:, 2],)

            # Compute observation noise on socket
            (self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos,) = torch_jit_utils.tf_combine(
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

        def reset_to_state(self, new_state):
            # Function to reset the environment to a new state
            # This will depend on how your environment's reset mechanics work
            # For example, setting new positions and orientations of objects
            old_plug_pos = self.plug_pos
            old_plug_quat = self.plug_quat
            target_pos, target_quat = new_state[:3], new_state[3:7]
            print("I was stuck in: ", old_plug_pos, old_plug_quat)
            time.sleep(1)
            print("Now I go to my neighbour: ", target_pos, target_quat)
            time.sleep(1)
            # You might need to call any additional environment reset functions here
            #return [self.plug_pos, self.plug_quat, self.socket_pos, self.socket_quat]
            return target_pos, target_quat

        def pre_physics_step(self, actions_expl):
            
            self.BC = True

            """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.run_actions = True
            
            print("socket pose before tf:", self.socket_pos, self.socket_quat)
            #print("socket pose after tf:", self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[0], self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[1])
            print("gripper goal pose before tf:", self.gripper_goal_pos, self.gripper_goal_quat)
            #print("gripper goal pose after tf:", self.pose_world_to_robot_base(self.gripper_goal_pos, self.gripper_goal_quat)[0], self.pose_world_to_robot_base(self.gripper_goal_pos, self.gripper_goal_quat)[1])

            current_state = [
                                self.plug_pos,
                                self.plug_quat,
                                self.socket_pos,
                                self.socket_quat,
                                                    ]
                                                
            current_state = torch.cat(current_state, dim=-1).to(self.device) 
            
            print("************************************************************")
            print(" ")
            print("current_state:", current_state)
            print(" ")
            print("************************************************************")

            prepared_state = self.prepare_state(current_state)
            print("prepared_state:", prepared_state.size())
            # Predict actions using the loaded model
            self.actor_model.eval()
            actions_beh, _, _ = self.actor_model(prepared_state)
            print("actions:", actions_beh.size())   

            self.actions_beh = actions_beh.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

            print("Socket Pos: ", self.socket_pos)
            print("Socket Quat: ", self.socket_quat)
            #time.sleep(10)


            if self.residuals:
                self.actions_pos = self.actions_beh[:,:3] + actions_expl[:,:3]
                
                # Convert to quat and set rot target
                rot_actions_expl = actions_expl[:,3:]
                angle_expl = torch.norm(rot_actions_expl, p=2, dim=-1)
                axis_expl = rot_actions_expl / angle_expl.unsqueeze(-1)
                rot_actions_expl_quat = torch_utils.quat_from_angle_axis(angle_expl, axis_expl)
                self.actions_quat = torch_utils.quat_mul(rot_actions_expl_quat, self.actions_beh[:,:4])
                
                self.actions = torch.cat((self.actions_pos, self.actions_quat), dim=-1)
            else:
                self.actions = self.actions_beh.clone()

            self._apply_actions_as_ctrl_targets(actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True)

        def post_physics_step(self):
            """Step buffers. Refresh tensors. Compute observations and reward."""

            
            self.progress_buf[:] += 1

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            print("self.progress_buf size:", self.progress_buf.size())
            print("self.progress_buf:", self.progress_buf)

            print("Size of plug_pos, plug_quat:", self.plug_pos.size(), self.plug_quat.size())
            print("plug_pos at time t:", self.plug_pos)
            print("plug_quat at time t:",self.plug_quat)

            self.compute_observations()
            self.compute_reward()

            self._render_headless()


        def compute_observations(self):
            """Compute observations."""

            delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
            noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

            '''
            # Define observations (for actor)
            obs_tensors = [
                              
                                self.arm_dof_pos,  # 7
                                self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0],  # 3
                                self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1],  # 4
                                self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0],  # 3
                                self.pose_world_to_robot_base(self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1],  # 4
                                noisy_delta_pos, # 3
                                                                                                                                            ]

            # Define state (for critic)
            state_tensors = [
                                self.arm_dof_pos,  # 7
                                self.arm_dof_vel,  # 7
                                self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[0],  # 3
                                self.pose_world_to_robot_base(self.fingertip_centered_pos, self.fingertip_centered_quat)[1],  # 4
                                self.fingertip_centered_linvel,  # 3
                                self.fingertip_centered_angvel,  # 3
                                self.pose_world_to_robot_base(self.gripper_goal_pos, self.gripper_goal_quat)[0],  # 3
                                self.pose_world_to_robot_base(self.gripper_goal_pos, self.gripper_goal_quat)[1],  # 4
                                delta_pos,  # 3
                                self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
                                self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
                                noisy_delta_pos - delta_pos, # 3
                                                                                                                                        ]

            '''

            # Define observations (for actor)
            obs_tensors = [
                            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
                            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
                            self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[0],  # 3
                            self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[1],  # 4
                                                                                                                                            ]

            # Define state (for critic)
            state_tensors = [
                            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
                            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
                            self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[0],  # 3
                            self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[1],  # 4
                                                                                                                                            ]



            self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
            self.states_buf = torch.cat(state_tensors, dim=-1)

            return self.obs_buf

        def compute_reward(self):
            """Detect successes and failures. Update reward and reset buffers."""

            self._update_rew_buf()
            self._update_reset_buf()

        def _update_rew_buf(self):
            
            """Compute reward at current timestep."""

            self.prev_rew_buf = self.rew_buf.clone()

            if not self.sparse_rewards:

                # SDF-Based Reward: Compute reward based on SDF distance
                sdf_reward = algo_utils.get_sdf_reward(
                                                    wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
                                                    asset_indices=self.asset_indices,
                                                    plug_pos=self.plug_pos,
                                                    plug_quat=self.plug_quat,
                                                    plug_goal_sdfs=self.plug_goal_sdfs,
                                                    wp_device=self.wp_device,
                                                    device=self.device,
                                                                                                                            )

            
                # SDF-Based Reward: Apply reward
                self.rew_buf[:] = self.cfg_task.rl.sdf_reward_scale * sdf_reward

                # SDF-Based Reward: Log reward
                self.extras["sdf_reward"] = torch.mean(self.rew_buf)

                # SAPU: Compute reward scale based on interpenetration distance
                low_interpen_envs, high_interpen_envs = [], []
                ( low_interpen_envs, high_interpen_envs, sapu_reward_scale,) = algo_utils.get_sapu_reward_scale(
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

                # SAPU: For envs with low interpenetration, apply reward scale ("weight" step)
                self.rew_buf[low_interpen_envs] *= sapu_reward_scale

                # SAPU: For envs with high interpenetration, do not update reward ("filter" step)
                if len(high_interpen_envs) > 0:
                    self.rew_buf[high_interpen_envs] = self.prev_rew_buf[high_interpen_envs]

                # SAPU: Log reward after scaling and adjustment from SAPU
                self.extras["sapu_adjusted_reward"] = torch.mean(self.rew_buf)


            is_last_step = self.progress_buf[0] == self.max_episode_length - 1
            if is_last_step:
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

                # Success bonus: Compute reward scale based on whether plug is engaged with socket, as well as closeness to full insertion

                if not self.sparse_rewards:
                    
                    # Success bonus: Compute reward scale based on whether plug is engaged with socket, as well as closeness to full insertion
                    engagement_reward_scale = algo_utils.get_engagement_reward_scale(
                                                                                        plug_pos=self.plug_pos,
                                                                                        socket_pos=self.socket_pos,
                                                                                        is_plug_engaged_w_socket=is_plug_engaged_w_socket,
                                                                                        success_height_thresh=self.cfg_task.rl.success_height_thresh,
                                                                                        device=self.device,
                                                                                                                                                            )

                    # Success bonus: Apply reward with reward scale
                    self.rew_buf[:] += (engagement_reward_scale * self.cfg_task.rl.engagement_bonus)

                else:

                    self.replan_episodes += 1
                    if not is_plug_inserted_in_socket:
                        self.final_state = torch.cat((self.plug_pos, self.plug_quat, self.socket_pos, self.socket_quat), dim=-1)
                        self.replan_states.append(self.final_state)

                        if self.replan_episodes == self.reset_replan_episodes:
                            
                            # if 'replan_states.txt' alread exists, delete it and create a new one
                            if os.path.exists('replan_states.txt'):
                                os.remove('replan_states.txt')
                                print("replan_states.txt already exists. Deleting it and creating a new one")
                            else:
                                # Create a new 'replan_states.txt' file
                                print("Creating a new replan_states.txt file ...")
                                # Save replan states in a .txt
                                print("Saving replan states in a .txt")
                                self.replan_states = torch.stack(self.replan_states)
                                self.replan_states = self.replan_states.cpu().numpy()
                                np.savetxt('replan_states.txt', self.replan_states, delimiter=',')

                            self.replan_episodes = 0
                            self.replan_states = []

                    is_plug_inserted_in_socket_float =  is_plug_inserted_in_socket.float()
                    is_plug_inserted_in_socket_float = torch.where(is_plug_inserted_in_socket_float > 1.0, torch.tensor(-1.0), is_plug_inserted_in_socket_float)

                    self.rew_buf[:] += is_plug_inserted_in_socket_float


                # Success bonus: Log success rate, ignoring environments with large interpenetration
                if len(high_interpen_envs) > 0:
                    is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[low_interpen_envs]
                    self.extras["insertion_successes"] = torch.mean(is_plug_inserted_in_socket_low_interpen.float())
                else:
                    self.extras["insertion_successes"] = torch.mean(is_plug_inserted_in_socket.float())

                if not self.sparse_rewards:

                    # SBC: Compute reward scale based on curriculum difficulty
                    sbc_rew_scale = algo_utils.get_curriculum_reward_scale( cfg_task = self.cfg_task, curr_max_disp=self.curr_max_disp)

                    # SBC: Apply reward scale (shrink negative rewards, grow positive rewards)
                    self.rew_buf[:] = torch.where(
                                                    self.rew_buf[:] < 0.0,
                                                    self.rew_buf[:] / sbc_rew_scale,
                                                    self.rew_buf[:] * sbc_rew_scale,
                                                                                        )

                    # SBC: Log current max downward displacement of plug at beginning of episode
                    self.extras["curr_max_disp"] = self.curr_max_disp

                    # SBC: Update curriculum difficulty based on success rate
                    self.curr_max_disp = algo_utils.get_new_max_disp(
                                                                        curr_success=self.extras["insertion_successes"],
                                                                        cfg_task=self.cfg_task,
                                                                        curr_max_disp=self.curr_max_disp,
                                                                                                                            )



        def _update_reset_buf(self):
            """Assign environments for reset if maximum episode length has been reached."""

            self.reset_buf[:] = torch.where(
                                     
                                                self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                                torch.ones_like(self.reset_buf),
                                                self.reset_buf,
                                                                                                                                )

        def reset_idx(self, env_ids):
            """Reset specified environments."""
            
            self.BC = False 
            self._reset_franka()

            # Close gripper onto plug
            self.disable_gravity()  # to prevent plug from falling
            self._reset_object()

            print("Plug Pose before grasp:")
            print("plug pos:", self.plug_pos)
            print("plug_quat:",self.plug_quat)

            self._move_gripper_to_grasp_pose(sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
            self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
            self.enable_gravity()

            # Get plug SDF in goal pose for SDF-based reward
            self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
                                                                    wp_plug_meshes=self.wp_plug_meshes,
                                                                    asset_indices=self.asset_indices,
                                                                    socket_pos=self.socket_pos,
                                                                    socket_quat=self.socket_quat,
                                                                    wp_device=self.wp_device,
                                                                                                            )


            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                # Define the filename for the video
                filename = f"simulation_video_BC.mp4"  # Replace with your preferred naming scheme
                self.save_video(self.complete_video_frames, filename=filename)
                self.complete_video_frames = []
            self.video_frames = []

            '''
            #if self.cfg_task.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                # print('Saving video')
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []
            '''
            
            self._reset_buffers()
            print("plug_pos at time 0:", self.plug_pos)
            print("plug_quat at time 0:", self.plug_quat)
            print("plug_linvel at time 0:", self.plug_linvel)
            print("plug_angvel at time 0:", self.plug_angvel)

        def _reset_franka(self):
            """Reset DOF states, DOF torques, and DOF targets of Franka."""

            # Randomize DOF pos
            self.dof_pos[:] = torch.cat(
                (
                    torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos,device=self.device,),
                    torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device,),
                    torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device,),
                                                                                                                ),dim=-1,
                                                                                                                            ).unsqueeze(0)  # shape = (num_envs, num_dofs)

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
            """Reset root state of plug and socket."""

            self._reset_socket()
            self._reset_plug(before_move_to_grasp=True)

        def _reset_socket(self):
            """Reset root state of socket."""

            # Randomize socket pos
            socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)
            socket_noise_xy = socket_noise_xy @ torch.diag(torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, dtype=torch.float32, device=self.device,))
            socket_noise_z = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
            socket_noise_z_mag = (self.cfg_task.randomize.socket_pos_z_noise_bounds[1] - self.cfg_task.randomize.socket_pos_z_noise_bounds[0])
            socket_noise_z = (socket_noise_z_mag * torch.rand((self.num_envs), dtype=torch.float32, device=self.device) + self.cfg_task.randomize.socket_pos_z_noise_bounds[0])

            self.socket_pos[:, 0] = ( self.robot_base_pos[:, 0] + self.cfg_task.randomize.socket_pos_xy_initial[0] + socket_noise_xy[:, 0])
            self.socket_pos[:, 1] = ( self.robot_base_pos[:, 1] + self.cfg_task.randomize.socket_pos_xy_initial[1] + socket_noise_xy[:, 1])
            self.socket_pos[:, 2] = self.cfg_base.env.table_height + socket_noise_z

            # Randomize socket rot
            socket_rot_noise = 2 * ( torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            socket_rot_noise = socket_rot_noise @ torch.diag(torch.tensor( self.cfg_task.randomize.socket_rot_noise, dtype=torch.float32, device=self.device,))
            socket_rot_euler = ( torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) + socket_rot_noise)
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
            """Reset root state of plug."""

            if not self.RRT_Star_Reset:

                if before_move_to_grasp:
                    # Generate randomized downward displacement based on curriculum
                    curr_curriculum_disp_range = ( self.curr_max_disp - self.cfg_task.rl.curriculum_height_bound[0] )
                    self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[0] + curr_curriculum_disp_range * (torch.rand((self.num_envs,), dtype=torch.float32, device=self.device))

                    # Generate plug pos noise
                    self.plug_pos_xy_noise = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)
                    self.plug_pos_xy_noise = self.plug_pos_xy_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.plug_pos_xy_noise, dtype=torch.float32, device=self.device,))

                # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
                # minus curriculum displacement
                self.plug_pos[:, :] = self.socket_pos.clone()
                self.plug_pos[:, 2] += self.socket_heights
                self.plug_pos[:, 2] -= self.curriculum_disp

                # Apply XY noise to plugs not partially inserted into sockets
                socket_top_height = self.socket_pos[:, 2] + self.socket_heights
                plug_partial_insert_idx = np.argwhere(self.plug_pos[:, 2].cpu().numpy() > socket_top_height.cpu().numpy()).squeeze()
                self.plug_pos[plug_partial_insert_idx, :2] += self.plug_pos_xy_noise[plug_partial_insert_idx]

                self.plug_quat[:, :] = self.identity_quat.clone()

            else:
                # Sample an initial reset state randomly from self.replay_buffer 
                sampled_reset_state = self.replay_buffer_RRT.sample()    
                # sample_reset_state = Diffuser.sample_step(self.replay_buffer_RRT) 
        
                # SOS! For multiple parallel environments, we would sample num_envs random states from replay_buffer
                self.plug_pos = sampled_reset_state[:3].repeat(self.num_envs, 1).to(self.device)
                self.plug_quat = sampled_reset_state[3:].repeat(self.num_envs, 1).to(self.device)

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
            """Reset buffers."""
            self.reset_buf[:] = 0
            self.progress_buf[:] = 0
            print("Episode starting time!:", self.progress_buf)

        def _set_viewer_params(self):
            """Set viewer parameters."""

            cam_pos = gymapi.Vec3(-1.0, -1.0, 2.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

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


        def _set_viewer_params(self):
            """Set viewer parameters."""
            #bx, by, bz = -0.0012, -0.0093, 0.4335
            bx, by, bz = self.socket_pos[0, 0].double(), self.socket_pos[0, 1].double(), self.socket_pos[0, 2].double() 
            cam_pos = gymapi.Vec3(bx - 0.2, by - 0.2, bz + 0.2)
            cam_target = gymapi.Vec3(bx, by, bz)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
            
            """
                Apply actions from policy as position/rotation targets.
            """

            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
            
            #self.ctrl_target_fingertip_centered_pos = (self.fingertip_centered_pos + pos_actions)

            ############ Control Object Pose ############
            #self.robot_plug_pos = self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0]  # 3
            target_pos = self.plug_pos + pos_actions
            #target_pos = (self.noisy_plug_pos + pos_actions)
            #target_pos = self.robot_plug_pos + pos_actions
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
            
            #self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)
            ####################### Control EEf ########################################
            #self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)
            #self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)
            ###########################################################################
            

            ####################### Control Object Pose ################################
            #self.robot_plug_quat = self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1] # 4
            
            target_quat = torch_utils.quat_mul(rot_actions_quat, self.plug_quat)
            #target_quat = torch_utils.quat_mul(rot_actions_quat, self.noisy_plug_quat)
            #target_quat = torch_utils.quat_mul(rot_actions_quat, self.robot_plug_quat)
            ############################################################################


            print("*-------------------------------------------------------------------------------------------*")
            print("NN Output: ")
            print(" ")
            print(" ")
            print(" ")
            print("Current Pos:")
            print("state pos: ", self.plug_pos)
            print("current plug pos in the Robot frame: ", self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0])
            print(" ")
            print("hole pos: ", self.socket_pos)
            print("robot_base_pos: ", self.robot_base_pos)
            print("pos_actions: ", pos_actions)
            print("next_state_pred_pos: ", target_pos)
            print("next_state pos GT: ", "I don't have one here")
            print(" ")
            print(" ")
            print(" ")
            print("Current_Quat")
            print("state quat: ", self.plug_quat)
            print("current plug quat in the Robot frame: ", self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1])
            print(" ")
            print("hole_quat ", self.socket_quat)
            print("robot_base_quat: ", self.robot_base_quat)
            print("rot_actions_quat: ", rot_actions_quat)
            print("next_state_pred_quat: ", target_quat)
            print("next_state quat GT: ", "I don't have one here")
            print(" ")
            print(" ")
            print(" ")
            print("*-------------------------------------------------------------------------------------------*")

            current_state = [
                                self.plug_pos,
                                self.plug_quat,
                                self.socket_pos,
                                self.socket_quat,
                                                    ]

            

            # Implement the kNN reset logic


            '''
            # Check if the reset condition is met
            if self.run_actions:
                if self.iterations_since_last_reset >= self.reset_threshold:
                    neighbors = find_k_nearest_neighbors(self.replay_buffer, current_state, k=10, batch_size=220000)

                    new_state = weighted_average_neighbors(neighbors)
                    new_state = new_state.to(self.device)
                    target_pos, target_quat = self.reset_to_state(new_state)
                    self.iterations_since_last_reset = 0
                else:
                    self.iterations_since_last_reset += 1
            '''

            #current_state_np = np.array(current_state)
            #current_state_tensor = torch.tensor(current_state_np, device=self.device)  # Replace with your current state
            current_state_tensor = torch.cat(current_state, dim=-1)

            if self.run_actions:
                if self.iterations_since_last_reset >= self.reset_threshold:
                    # Ensure current_state is in the correct format
                    neighbors = find_k_nearest_neighbors(self.replay_buffer, current_state_tensor, k=1000)
                    new_state = weighted_average_neighbors(neighbors)
                    new_state = new_state.to(self.device)
                    target_pos, target_quat = self.reset_to_state(new_state)
                    self.iterations_since_last_reset = 0
                else:
                    self.iterations_since_last_reset += 1
                                            
            '''
            self.plug_pos = self.plug_pos.to('cuda:0')
            self.plug_quat = self.plug_quat.to('cuda:0')
            self.socket_pos = self.socket_pos.to('cuda:0')
            self.socket_quat = self.socket_quat.to('cuda:0')

            # Ensure all tensors have the same number of dimensions
            # Assuming that pos tensors are of shape [1, 3] and quat tensors are of shape [1, 4]
            self.plug_pos = self.plug_pos.unsqueeze(0) if self.plug_pos.dim() == 1 else self.plug_pos
            self.socket_pos = self.socket_pos.unsqueeze(0) if self.socket_pos.dim() == 1 else self.socket_pos
            self.plug_quat = self.plug_quat.unsqueeze(0) if self.plug_quat.dim() == 1 else self.plug_quat
            self.socket_quat = self.socket_quat.unsqueeze(0) if self.socket_quat.dim() == 1 else self.socket_quat
            '''
            
            target_pos = target_pos.to('cuda:0')
            target_quat = target_quat.to('cuda:0')
            target_pos = target_pos.unsqueeze(0) if target_pos.dim() == 1 else target_pos
            target_quat = target_quat.unsqueeze(0) if target_quat.dim() == 1 else target_quat

            
            if self.BC:
                
                #while ( torch.norm(self.plug_pos[0, 2].double() - target_pos[0, 2].double(), p=2) > .001 ): #or torch.rad2deg(quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))) > 1: 

                for _ in range(1):
                    
                    print(" ")
                    print("In the while loop...............................................................................................................")
                    print("self.plug_pose: ", self.plug_pos, self.plug_quat)
                    print("target_pose at this step: ", target_pos, target_quat)
                    print("self.socket_pose: ", self.socket_pos, self.socket_quat)
                    print(" ")
                    
                    #self.refresh_and_acquire_tensors()

                    ############################## Control Object Pose (DG Transformation ###############################################
                    inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal(self.num_envs, self.device, jacobian_type=None)
            
                    self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)

                    #self.ctrl_target_fingertip_midpoint_pos, self.ctrl_target_fingertip_midpoint_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
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
                    s_t_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device = self.device)
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
                    #self.zero_velocities(self.num_envs)
                    self.refresh_and_acquire_tensors()

                    '''
                    # Capture and write frames from each camera
                    for i_cam in range(self.num_cameras):
                        rgb_tensor = self.rgb_tensors[i_cam]
                        frame = rgb_tensor.cpu().numpy()
                        frame = np.transpose(frame, (1, 2, 0))  # Change tensor layout if necessary
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                        self.video_writer.write(frame)
                    '''

            else:
                inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal(self.num_envs, self.device, jacobian_type=None)

                self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
            
            
                self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
                self.generate_ctrl_signals()


        
        def refresh_and_acquire_tensors(self):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self._acquire_env_tensors()

        def zero_velocities(self, env_ids):

            self.dof_vel[env_ids,:] = torch.zeros_like(self.dof_vel[env_ids])
            # Set DOF state
            multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32),
                                            len(multi_env_ids_int32))        
            self.gym.simulate(self.sim)
            self.render()



        def _move_gripper_to_grasp_pose(self, sim_steps):
            """Define grasp pose for plug and move gripper to pose."""

            # Set target_pos
            self.ctrl_target_fingertip_midpoint_pos = self.plug_pos.clone()
            self.ctrl_target_fingertip_midpoint_pos[:, 2] += self.plug_grasp_offsets

            # Set target rot
            ctrl_target_fingertip_centered_euler = (torch.tensor(self.cfg_task.randomize.fingertip_centered_rot_initial, device=self.device,).unsqueeze(0).repeat(self.num_envs, 1))

            self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(ctrl_target_fingertip_centered_euler[:, 0],ctrl_target_fingertip_centered_euler[:, 1],ctrl_target_fingertip_centered_euler[:, 2],)

            self.move_gripper_to_target_pose(gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,sim_steps=sim_steps,)

            # Reset plug in case it is knocked away by gripper movement
            self._reset_plug(before_move_to_grasp=False)

        def DG(self, num_envs, device, jacobian_type):
             
            plug_euler = quat2euler(self.plug_quat)
            fingertip_midpoint_euler = quat2euler(self.fingertip_midpoint_quat)

            Pose_Obj = torch.zeros((num_envs, 4,4), device=device)
            Pose_Gripper = torch.zeros((num_envs, 4,4), device=device)
            DeltaG = torch.zeros((num_envs, 4,4), device=device)

            M1 = quaternion_to_matrix(self.plug_quat)
            Pose_Obj[:, :3,:3] = M1
            Pose_Obj[:, 0, 3] = self.plug_pos[:, 0]
            Pose_Obj[:, 1, 3] = self.plug_pos[:, 1]
            Pose_Obj[:, 2, 3] = self.plug_pos[:, 2]
            Pose_Obj[:, 3,3] = 1
        
            M2 = quaternion_to_matrix(self.fingertip_midpoint_quat)
            Pose_Gripper[:, :3,:3] = M2
            Pose_Gripper[:, 0, 3] = self.fingertip_midpoint_pos[:, 0]
            Pose_Gripper[:, 1, 3] = self.fingertip_midpoint_pos[:, 1]
            Pose_Gripper[:, 2, 3] = self.fingertip_midpoint_pos[:, 2]
            Pose_Gripper[:, 3, 3] = 1
                                                                                                                                                           
            inv_Pose_Gripper_quat, inv_Pose_Gripper_t = tf_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
                                                                                                                                           
            DeltaG_t = DeltaG[:, :3, 3]
            DeltaG_quat = matrix_to_quaternion(DeltaG[:, :3,:3])

            curr_plug_pos = self.plug_pos
            curr_plug_quat = self.plug_quat

            #curr_plug_pos = curr_plug_pos + pos_noise
            #quat_noise = torch_utils.quat_from_euler_xyz(euler_noise[:,0], euler_noise[:,1], euler_noise[:,2])
            #curr_plug_quat = torch_utils.quat_mul(quat_noise, curr_plug_quat)
        
            DeltaG_quat, DeltaG_t = tf_combine(inv_Pose_Gripper_quat, inv_Pose_Gripper_t, curr_plug_quat, curr_plug_pos)
            inv_DeltaG_quat, inv_DeltaG_t = tf_inverse(DeltaG_quat, DeltaG_t)

            return inv_DeltaG_t, inv_DeltaG_quat

        def DG_IndustReal(self, num_envs, device, jacobian_type):

            inv_Pose_Gripper_quat, inv_Pose_Gripper_t = tf_inverse(self.fingertip_centered_quat, self.fingertip_centered_pos)
    
            curr_plug_pos = self.plug_pos
            curr_plug_quat = self.plug_quat

            #curr_plug_pos = self.robot_plug_pos
            #curr_plug_quat = self.robot_plug_quat

            #curr_plug_pos = self.noisy_plug_pos
            #curr_plug_quat = self.noisy_plug_quat

            DeltaG_quat, DeltaG_t = tf_combine(inv_Pose_Gripper_quat, inv_Pose_Gripper_t, curr_plug_quat, curr_plug_pos)
            inv_DeltaG_quat, inv_DeltaG_t = tf_inverse(DeltaG_quat, DeltaG_t)

            return inv_DeltaG_t, inv_DeltaG_quat


        def apply_invDG_IndustReal_insert(self, num_envs, target_pos, target_quat, inv_DeltaG_t, inv_DeltaG_quat, device):

            target_Pose_Gripper_quat, target_Pose_Gripper_t = tf_combine(target_quat, target_pos, inv_DeltaG_quat, inv_DeltaG_t)
            return target_Pose_Gripper_t, target_Pose_Gripper_quat

        def save_video(self, video_frames, filename='output_video.mp4', fps=30):
            
            # Ensure there are frames to save
            if len(video_frames) == 0:
                print("No video frames to save")
                return
            
            height, width, layers = video_frames[0].shape
            size = (width, height)
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            for frame in video_frames:
                out.write(frame)
            out.release()

