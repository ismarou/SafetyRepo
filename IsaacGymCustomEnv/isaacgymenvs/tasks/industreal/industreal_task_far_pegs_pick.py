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

Can be executed with python train.py task=IndustRealTaskFarPegsPick.
"""

"""

This file contains functions in order of:
init
yaml params
acquire task tensors
refresh task tensors
    These keypoints are used as seeing the success of the task for reward generation

post physics step
Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains
Sends actions to function for applying actions as control targets

pre physics step
Step buffers. Refresh tensors. Compute observations and reward.

compute obvs
comptue rew
fill rew buffer
update_reset_buf
    This sets which environmetns which need to be reset with reset_idx

reset idx
    resets everything that needs to be reset

simulate_and_refresh
    Refreshes tensors and simulates one step

_apply_actions_as_ctrl_targets
    Takes in actions from policy which are properly sized target configurations and employs them as controls

So here what should change is:
reset no automatic grab
reset arm starting in same way as factory
reset peg starting in same random area as factory
Then rewards as a second issue
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
from isaacgymenvs.tasks.industreal.industreal_env_far_pegs import IndustRealEnvFarPegs
from isaacgymenvs.utils import torch_jit_utils


class IndustRealTaskFarPegsPick(IndustRealEnvFarPegs, FactoryABCTask):
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
        print("IndustRealTaskFarPegsPick\n\n")
        self._acquire_task_tensors()
        self.parse_controller_spec()

        # Get Warp mesh objects for SAPU and SDF-based reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        (
            self.wp_plug_meshes,
            self.wp_plug_meshes_sampled_points,
            self.wp_socket_meshes,
        ) = algo_utils.load_asset_meshes_in_warp(
            plug_files=self.plug_files,
            socket_files=self.socket_files,
            num_samples=self.cfg_task.rl.sdf_reward_num_samples,
            device=self.wp_device,
        )

        if self.viewer != None:
            self._set_viewer_params()
        
        torch.manual_seed(self.cfg_task.randomize.plug_pos_xy_initial_noise_seed)

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        ppo_path = os.path.join(
            "train/IndustRealTaskFarPegsPickPPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _acquire_task_tensors_dep(self):
        """Acquire tensors."""

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_pos_local = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    (self.cfg_task.env.socket_base_height + self.plug_grasp_offsets[i]),
                ]
                for i in range(self.num_envs)
            ],
            device=self.device,
        )
        self.gripper_goal_quat_local = self.identity_quat.clone()

        self.socket_top_pos_local = torch.tensor(
            [[0.0, 0.0, self.socket_heights[i]] for i in range(self.num_envs)],
            device=self.device,
        )
        self.socket_quat_local = self.identity_quat.clone()

        # Define keypoint tensors
        self.keypoint_offsets = (
            algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device)
            * self.cfg_task.rl.keypoint_scale
        )
        self.keypoints_plug = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_socket = torch.zeros_like(
            self.keypoints_plug, device=self.device
        )

        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )

        self.curr_max_disp = self.cfg_task.rl.initial_max_disp

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        #For each keypoint, the last value (z) is nonzero and scaled 0.0 to 1.0 across keypoints, all subtracted by 0.5

        return keypoint_offsets

    def _acquire_task_tensors(self):
        self.keypoint_offsets = algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device) * self.cfg_task.rl.keypoint_scale
        #self.keypoint_offsets = self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_gripper = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                             dtype=torch.float32,
                                             device=self.device)
        self.keypoints_plug = torch.zeros_like(self.keypoints_gripper, device=self.device)
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _refresh_task_tensors(self):
        """
        For the task, need observation information, reward information, and information for the sim
        Observation info:
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  #The pose of the peg so the position and quaternion
                       self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1]] #now in robot base frame
        Fingertip info all done in factory_base of refresh_base_tensors()
        plug_pos and plug_quat reset in reset_plug -> reset_idx -> pre_physics_step, not sure where else they are updated but this acts in the same way as nut_pos does
        They somehow compose it into the grasping frame in the pick task. Not sure what this tf_combine function is doing.
        """
        #Observation information:
        #None
        #Information for sim:
        #don't need original franka reset information as that doesn't do much
        #None
        #Reward information:
        #Keypoint vector norm difference calculated when creating reward, so need to write the current keypoint locations

        #Relatively place the location of the keypoint to where the gripper should sit to pick this up, not just the object keypoints
        #Then we will convert this to the world frame and the gripper will move towards it
        
        #self.plug_grasp_offsets[0]
        #(self.plug_heights*0.5)
        #print(torch.mean(self.plug_grasp_offsets))
        pos_local = torch.tensor([0.0, 0.0, 0.04], device=self.device).repeat((self.num_envs, 1))
        quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.plug_grasp_quat, self.plug_grasp_pos = \
            torch_jit_utils.tf_combine(self.plug_quat,self.plug_pos,quat_local,pos_local)
        
        # Compute pos of keypoints on gripper and plug in world frame
        #For each keypoint 3dim point made in keypoint offsets, position it in the world frame where the relative rotation is transfered to a different frame
        #the keypoint is repeated to cover each environment. It is just the keypoint gripper in the world frame where the position frame is y scalings, 128x3?
        #So what is happening is we are collecting 4 points by doing the current rotation unchanged and current position plus keypoint_offset [3]
        #print("fingertip pos before", self.fingertip_midpoint_pos)
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            #print("fingertip pos initial", self.fingertip_midpoint_pos)
            self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                        self.fingertip_midpoint_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]
            #print("added z", keypoint_offset)
            #print("keypoints gripper pos after", self.keypoints_gripper[:, idx])
            #torch.Size([128, 4])torch.Size([128, 3])torch.Size([128, 4])torch.Size([128, 3])
            #self.keypoints_gripper->torch.Size([128, 4, 3]), so idx indexed in the second slot makes it [128,3]

            #print("plug pos", self.plug_pos)
            #print("combined plug pos", self.plug_grasp_pos)
            #in factory they do [0,0,1] repeated as the tensor for pos, so what does composing that do? Just let the last value exist?

            #pos is 3, quat is 4. So 128 environments then 3 values for the position and 4 for the quaternion which is the rotation
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_grasp_quat,
                                                                    self.plug_grasp_pos,
                                                                    self.identity_quat,
                                                                    keypoint_offset.repeat(self.num_envs, 1))[1]
            #Basically along the positions only every num_keypoints is nonzero to be mattering in the keypoint distance calculation
            #And for quaternions only the last value counts on the keypoints. 
            #print("keypoints_gripper", self.keypoints_gripper[1])
            #keypoint_dist = self._get_keypoint_dist() * -1.0
            #print(keypoint_dist[0])

    def _refresh_task_tensors_dep(self):
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
        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )
        socket_obs_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        socket_obs_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_gripper_goal_quat,
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine(
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

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        #print("before", actions.shape)
        #set_actions = torch.tensor([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
        #actions[:, :] = set_actions.unsqueeze(0).repeat((actions.shape[0], 1))
        #print("after", actions.shape)
        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        #print("actions", actions[0])
        
        #if self.progress_buf[0] > (self.max_episode_length / 2) - 1:
        #if self.cfg_task.rl.remove_velocity == True:
          #Maybe try this instead of stop moving
          #delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
          #device=self.device)  # No hand motion
          #self._apply_actions_as_ctrl_targets(delta_hand_pose, 1.0, do_scale=False)
          #self.stop_moving()
          #self.render()
          #self.gym.simulate(self.sim)
          #print(self.dof_vel[0][0], self.dof_torque[0][0])
          #If it has velocity, then the agent doesn't actually have the full state of the system

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max, do_scale=True
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1
        
        self.rew_buf_initial()
        
        if self.progress_buf[0] == self.max_episode_length - 1:
          self._close_gripper() #it seems like gripping actions cannot be independent for different environments
          #self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
          self._lift_gripper()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    #----------------------------------------------------------------------------------------------------------------------------------------------
    
    #In totality just runs a no control on the arm except the gripper for 20 steps of shutting it, runs on every environment when this happens. I don't get how this works if the environments get offset
    def _close_gripper(self, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
          device=self.device)  # No hand motion

        self.stop_moving()

        print("sim before gripping")
        for _ in range(sim_steps):
            #print("wanted", self.ctrl_target_fingertip_centered_pos[0])
            #print("current", self.fingertip_centered_pos[0])
            self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)


        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        #for _ in range(sim_steps):
        #    self.render()
        #    self.gym.simulate(self.sim)

    def _lift_gripper(self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)

    def _check_lift_success(self, height_multiple):
        """Check if peg is above table by more than specified multiple times height of peg."""

        #print("plug poses", self.plug_pos[:, 2], "table height", self.cfg_base.env.table_height)
        lift_success = torch.where(
            self.plug_pos[:, 2] > self.cfg_base.env.table_height + (0.02 * height_multiple),
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success

    #----------------------------------------------------------------------------------------------------------------------------------------------

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors, these self. values are created in factory base so this should be exisiting correctly here
        #pose (position and quaternion) of the hand and nut, as well as the linear and angular velocity of the hand.
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.plug_grasp_quat,
                       self.plug_grasp_pos
                        ] #need to ensure these values are refreshed on refresh tensors as this all happens at once in post physics step

        #self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  #The pose of the peg so the position and quaternion
        #self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1]

        #print("plug_grasp_pos, plug_pos", self.plug_grasp_pos, self.plug_pos)

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        #self.states_buf = self.obs_buf.copy()
        #print("obs", obs_tensors[:][0])
        return self.obs_buf

    def compute_observations_dep(self):
        """Compute observations."""

        delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        # Define observations (for actor)
        obs_tensors = [
            self.arm_dof_pos,  # 7
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                1
            ],  # 4
            self.pose_world_to_robot_base(
                self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat
            )[
                1
            ],  # 4
            noisy_delta_pos,
        ]  # 3

        # Define state (for critic)
        state_tensors = [
            self.arm_dof_pos,  # 7
            self.arm_dof_vel,  # 7
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                1
            ],  # 4
            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3
            self.pose_world_to_robot_base(
                self.gripper_goal_pos, self.gripper_goal_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.gripper_goal_pos, self.gripper_goal_quat
            )[
                1
            ],  # 4
            delta_pos,  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
            noisy_delta_pos - delta_pos,
        ]  # 3

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)
        self.states_buf = torch.cat(state_tensors, dim=-1)

        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_plug - self.keypoints_gripper, p=2, dim=-1), dim=-1)

        #these keypoints initalized to zero in acquire tensors then in refresh turned to their correct values
        

        return keypoint_dist
        
    def _get_keypoint_dist_z(self):
        """Get keypoint distance."""

        keypoint_dist = self.keypoints_plug[:, :, 2] - self.keypoints_gripper[:, :, 2]
        #print("z plug", self.keypoints_plug[0, 0, 2], "\nz gripper", self.keypoints_gripper[0, 0, 2], self.keypoints_plug[0, 0, 2] - self.keypoints_gripper[0, 0, 2])
        #print(keypoint_dist[0])

        #these keypoints initalized to zero in acquire tensors then in refresh turned to their correct values

        #When gripper is above, negative, when it is below, positive, so it is swapped, want it to be above now below
        #grip_loc_success = torch.where(
        #      keypoint_dist <= 0,
        #      torch.full((self.num_envs,), 1.0, device=self.device),
        #      torch.full((self.num_envs,), 5.0, device=self.device))
        #just make it 5x bad to be below
        return keypoint_dist #torch.abs(keypoint_dist) * grip_loc_success
        
    #Same distance between all three
    def _get_keypoint_dist_std(self):
        x = torch.norm(self.keypoints_plug[:, :, 0] - self.keypoints_gripper[:, :, 0], p=2, dim=-1)
        y = torch.norm(self.keypoints_plug[:, :, 1] - self.keypoints_gripper[:, :, 1], p=2, dim=-1)
        z = torch.norm(self.keypoints_plug[:, :, 2] - self.keypoints_gripper[:, :, 2], p=2, dim=-1)
        std = torch.std(torch.stack([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=1), dim=1)
        #print(torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1).shape)
        #print("standard dev", std[0])
        #print("x, y, z", x[0], y[0], z[0])
        #print("std shape", std.shape)
        return std.squeeze() * 10.0
    

    def rew_buf_initial(self):
        self.rew_buf[:] *= 0.0
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            keypoint_dist = self._get_keypoint_dist() * -1.0
            #print("keypoint dist closeness", keypoint_dist)
            grip_loc_success = torch.where(
              torch.abs(keypoint_dist) < self.cfg_task.rl.grip_correct_tolerance,
              torch.full((self.num_envs,), 1.0, device=self.device),
              torch.full((self.num_envs,), 0.0, device=self.device))
            #print("grip_loc_success", grip_loc_success)
            self.extras['successes'] = torch.mean(grip_loc_success.float()) * 0.5
            self.rew_buf[:] += grip_loc_success * self.cfg_task.rl.grasp_pos_bonus
            
        

    def _update_rew_buf(self):
        #Compute keypoint distances using norm ||diff||
        keypoint_dist = self._get_keypoint_dist() * -1.0
        #keypoint_dist_z = self._get_keypoint_dist_z() * -1.0
        #keypoint_std = self._get_keypoint_dist_std() * -1.0
        #print("keydist:", keypoint_dist[0])
        #print("env1 plug kp:", self.keypoints_plug[0])
        #print("env1 grip kp:", self.keypoints_gripper[0])
        #Weight the reward of keypoint distance with action penalty (we don't need to do action penalty)
        self.rew_buf[:] += (keypoint_dist) * self.cfg_task.rl.keypoint_reward_scale * 0.25 #0.5 so that it doesn't change the average reward values
        #print(keypoint_dist_z[0])
        #close_enough_z = torch.where(
        #  torch.abs(keypoint_dist_z) < 0.02,
        #  torch.full((self.num_envs,), 1.0, device=self.device),
        #  torch.full((self.num_envs,), 0.0, device=self.device))
        #self.rew_buf[:] += (1.0 + keypoint_std) * close_enough_z * 0.25 #so only get this std bonus when close enough
        #print("rew_buf, z dist", self.rew_buf[0], keypoint_dist_z[0])
        #grip_loc_success_dense = torch.where(
        #  torch.abs(keypoint_dist) < self.cfg_task.rl.grip_correct_tolerance_dense,
        #  torch.full((self.num_envs,), 1.0, device=self.device),
        #  torch.full((self.num_envs,), 0.0, device=self.device))
        #self.rew_buf[:] += grip_loc_success_dense
        
        #Add the amount of success of lifting if its on the last timestep and the post physics step just lifted it
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            #grip_loc_success = torch.where(
            #  torch.abs(keypoint_dist) < self.cfg_task.rl.grip_correct_tolerance,
            #  torch.full((self.num_envs,), 1.0, device=self.device),
            #  torch.full((self.num_envs,), 0.0, device=self.device))
            #print("scale success", scale_success)
            # Check if nut is picked up and above table
            lift_success = self._check_lift_success(height_multiple=3.0)
            self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] += torch.mean((lift_success).float()) * 0.5
            print("lift success reward", self.extras['successes'])
            print(self.rew_buf[:])
        print("rewards", self.rew_buf[0])

    def _update_rew_buf_dep(self):
        """Compute reward at current timestep."""

        self.prev_rew_buf = self.rew_buf.clone()

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
        (
            low_interpen_envs,
            high_interpen_envs,
            sapu_reward_scale,
        ) = algo_utils.get_sapu_reward_scale(
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
            engagement_reward_scale = algo_utils.get_engagement_reward_scale(
                plug_pos=self.plug_pos,
                socket_pos=self.socket_pos,
                is_plug_engaged_w_socket=is_plug_engaged_w_socket,
                success_height_thresh=self.cfg_task.rl.success_height_thresh,
                device=self.device,
            )

            # Success bonus: Apply reward with reward scale
            self.rew_buf[:] += (
                engagement_reward_scale * self.cfg_task.rl.engagement_bonus
            )

            # Success bonus: Log success rate, ignoring environments with large interpenetration
            if len(high_interpen_envs) > 0:
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[
                    low_interpen_envs
                ]
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket_low_interpen.float()
                )
            else:
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket.float()
                )

            # SBC: Compute reward scale based on curriculum difficulty
            sbc_rew_scale = algo_utils.get_curriculum_reward_scale(
                cfg_task=self.cfg_task, curr_max_disp=self.curr_max_disp
            )

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

        self._reset_franka(env_ids)
        
        # Close gripper onto plug
        #self.disable_gravity()  # to prevent plug from falling
        self._reset_object(env_ids)
        #self._move_gripper_to_grasp_pose(
        #    sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        #)
        #self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        #self.enable_gravity()

        print("Both resets done for franka and objects in reset_idx")
      
        # Get plug SDF in goal pose for SDF-based reward
        """
        self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
            wp_plug_meshes=self.wp_plug_meshes,
            asset_indices=self.asset_indices,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_device=self.wp_device,
        )
        """

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Need to make sure the task .yaml has this information
        #self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

        self._reset_buffers(env_ids)

    #"""
    def _reset_franka_dep(self, env_ids):
        #Reset DOF states, DOF torques, and DOF targets of Franka.

        # Randomize DOF pos
        self.dof_pos[:] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos,
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
            ),
            dim=-1,
        ).unsqueeze(
            0
        )  # shape = (num_envs, num_dofs)

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
    #"""
    #The two important parts are setting dof_pos and setting the state tensor, I am not sure why they set this other information such as torque and stabilizing
    #Each environment is a value in the first dim of the tensor, so env_ids works to select which environments need to be reset
    #There is a tensor element for each environment in things like dof_pos, reset_buf, plug_loc, all that stuff that is contained as a self variable
    #So it seems that we don't care which env needs to be updated as this does it for all of them, it just resets them all when one needs to be reset
    #Now with this information we can more easily read the code
    #
    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = torch.cat(
            (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
                torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
                torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
            dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        #this command finds a valid space to start
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                len(multi_env_ids_int32))
                                                
        #concerned it seends to set its dof actuation force tensor
        #print("starting franka:", self.fingertip_centered_pos, self.fingertip_centered_quat)

    def _reset_object(self, env_ids):
        """Reset root state of plug and socket."""

        self._reset_socket(env_ids)
        self._reset_plug(env_ids, before_move_to_grasp=True)

    def _reset_socket(self, env_ids):
        self.socket_pos[:, 0] = (
            self.robot_base_pos[:, 0]
        )
        self.socket_pos[:, 1] = (
            self.robot_base_pos[:, 1]
        )
        self.socket_pos[:, 2] = self.cfg_base.env.table_height
        socket_rot_euler = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        )
        socket_rot_quat = torch_utils.quat_from_euler_xyz(
            socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2]
        )
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

    def _reset_socket_dep(self, env_ids):
        """Reset root state of socket."""

        # Randomize socket pos
        socket_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_noise_xy = socket_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.socket_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        socket_noise_z = torch.zeros(
            (self.num_envs), dtype=torch.float32, device=self.device
        )
        socket_noise_z_mag = (
            self.cfg_task.randomize.socket_pos_z_noise_bounds[1]
            - self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
        )
        socket_noise_z = (
            socket_noise_z_mag
            * torch.rand((self.num_envs), dtype=torch.float32, device=self.device)
            + self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
        )

        self.socket_pos[:, 0] = (
            self.robot_base_pos[:, 0]
            + self.cfg_task.randomize.socket_pos_xy_initial[0]
            + socket_noise_xy[:, 0]
        )
        self.socket_pos[:, 1] = (
            self.robot_base_pos[:, 1]
            + self.cfg_task.randomize.socket_pos_xy_initial[1]
            + socket_noise_xy[:, 1]
        )
        self.socket_pos[:, 2] = self.cfg_base.env.table_height + socket_noise_z

        # Randomize socket rot
        socket_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_rot_noise = socket_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.socket_rot_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        socket_rot_euler = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            + socket_rot_noise
        )
        socket_rot_quat = torch_utils.quat_from_euler_xyz(
            socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2]
        )
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

    def _reset_plug(self, env_ids, before_move_to_grasp):
        """Reset root state of plug."""

        """
        if before_move_to_grasp:
            # Generate randomized downward displacement based on curriculum
            curr_curriculum_disp_range = (
                self.curr_max_disp - self.cfg_task.rl.curriculum_height_bound[0]
            )
            self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[
                0
            ] + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            # Generate plug pos noise
            self.plug_pos_xy_noise = 2 * (
                torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
                - 0.5
            )
            self.plug_pos_xy_noise = self.plug_pos_xy_noise @ torch.diag(
                torch.tensor(
                    self.cfg_task.randomize.plug_pos_xy_noise,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        """



        #Then we need to set the self.plug_pos value which is used elsewhere in state understanding and such
        #Here just set the plug position of x and y given the x and y noise respectively
        plug_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        #print("plug noise", plug_noise_xy)

        plug_noise_xy = plug_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.plug_pos_xy_initial_noise, device=self.device)) #Why use a diagonal matrix here instead of just multiplying???

        #Here just set the root position of x and y given the x and y noise respectively
        self.plug_pos[env_ids, 0] = self.cfg_task.randomize.plug_pos_xy_initial[0] + plug_noise_xy[env_ids, 0]

        self.plug_pos[env_ids, 1] = self.cfg_task.randomize.plug_pos_xy_initial[1] + plug_noise_xy[env_ids, 1]

        self.plug_pos[env_ids, 2] = self.cfg_base.env.table_height# - self.bolt_head_heights.squeeze(-1)

        #Critical functions:
        #This is one of the vectors from refreshing task vectors and acquiring task vectors
        self.plug_quat[env_ids, :] = self.identity_quat.clone()

        #Set velocities to 0
        self.plug_linvel[env_ids, :] = 0.0
        self.plug_angvel[env_ids, :] = 0.0

        #Set plug root state
        plug_actor_ids_sim = self.plug_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state), #the root state is not seen anywhere in this code
            gymtorch.unwrap_tensor(plug_actor_ids_sim), #Where are the plug_ variables actually used?
            len(plug_actor_ids_sim),
        )


        """
        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        #self.plug_pos[:, :] = self.socket_pos.clone()
        far_pos = torch.tensor(self.cfg_task.randomize.plug_pos_initial)
        self.plug_pos[env_ids, :] = far_pos.unsqueeze(0).repeat((self.socket_pos.shape[0], 1))
        #print("plug_pos as new positions made", self.plug_pos) #its an array of x,y,z coordinates for each environment so how about I just keep this as a size clone then static [0.5, 0.3]
        self.plug_pos[env_ids, 2] += self.socket_heights
        self.plug_pos[env_ids, 2] -= self.curriculum_disp

        # Apply XY noise to plugs not partially inserted into sockets
        socket_top_height = self.socket_pos[:, 2] + self.socket_heights
        plug_partial_insert_idx = np.argwhere(
            self.plug_pos[env_ids, 2].cpu().numpy() > socket_top_height.cpu().numpy()
        ).squeeze()
        self.plug_pos[plug_partial_insert_idx, :2] += self.plug_pos_xy_noise[
            plug_partial_insert_idx
        ]

        self.plug_quat[env_ids, :] = self.identity_quat.clone()

        # Stabilize plug
        self.plug_linvel[env_ids, :] = 0.0
        self.plug_angvel[env_ids, :] = 0.0

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
        """

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def stop_moving(self):
        #self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos
        #self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat
        #self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        # Stabilize Franka
        self.dof_vel[:, :] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[:, :] = 0.0
        self.ctrl_target_dof_pos = self.dof_pos.clone()
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()
        #self.generate_ctrl_signals()
        

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_centered_pos = (
            self.fingertip_centered_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_centered_quat
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        #print(pos_actions)
        self.generate_ctrl_signals()

    def _move_gripper_to_grasp_pose(self, sim_steps):
        """Define grasp pose for plug and move gripper to pose."""

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = self.plug_pos.clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += self.plug_grasp_offsets

        # Set target rot
        ctrl_target_fingertip_centered_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_centered_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2],
        )

        self.move_gripper_to_target_pose(
            gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            sim_steps=sim_steps,
        )

        # Reset plug in case it is knocked away by gripper movement
        self._reset_plug(before_move_to_grasp=False)