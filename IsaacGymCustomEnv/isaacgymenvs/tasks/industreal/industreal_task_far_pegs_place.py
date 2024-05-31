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


class IndustRealTaskFarPegsPlace(IndustRealEnvFarPegs, FactoryABCTask):
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
        print("IndustRealTaskFarPegsPlace\n\n")
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
            "train/IndustRealTaskFarPegsPlacePPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        #For each keypoint, the last value (z) is nonzero and scaled 0.0 to 1.0 across keypoints, all subtracted by 0.5

        return keypoint_offsets

    def _acquire_task_tensors(self):
        self.keypoint_offsets = algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device) * self.cfg_task.rl.keypoint_scale
        #self.keypoint_offsets = self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_socket = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                             dtype=torch.float32,
                                             device=self.device)
        self.keypoints_plug = torch.zeros_like(self.keypoints_socket, device=self.device)
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _refresh_task_tensors(self):
        """
        Dense reward is distance from keypoints of peg to socket, so make the peg and socket keypoints of where the peg should in relation to the socket
        Do something slightly above the socket
        """
        
        #Make the center position of the socket we want, [0.0, 1.0, 0.0, 0.0]
        #print("socket heights", self.socket_heights, self.socket_heights.shape)

        pos_local = torch.tensor([0.0, 0.0, self.socket_heights[0]*5.0], device=self.device).repeat((self.num_envs, 1))
        quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1)) #may need to make this not rotate around
        self.socket_higher_quat, self.socket_higher_pos = \
            torch_jit_utils.tf_combine(self.socket_quat,self.socket_pos,quat_local,pos_local)
            
        pos_local_plug = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        quat_local_plug = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1)) #may need to make this not rotate around
        self.plug_higher_quat, self.plug_higher_pos = \
            torch_jit_utils.tf_combine(self.plug_quat,self.plug_pos,quat_local_plug,pos_local_plug)
        
        #Keypoint_offsets is keypoint num of x,y,z points. These scale into the world frame versions of the objects along the objects. So if the peg is rotated, it will be relative to peg then turned 
        #relative to world.
        #For each keypoint, make this scaling for each env [:
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_higher_quat,
                                                                        self.socket_higher_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]


            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_higher_quat,
                                                                    self.plug_higher_pos,
                                                                    self.identity_quat,
                                                                    keypoint_offset.repeat(self.num_envs, 1))[1]
            #keypoint_dist = self._get_keypoint_dist() * -1.0
            #print(keypoint_dist[0])
        #print("keypoints socket", self.keypoints_socket[0, :])
        #print("keypoints plug", self.keypoints_plug[0, :])
        #keypoint_dist = self._get_keypoint_dist() * -1.0
        #print("keypoint dist", keypoint_dist[0])


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
        )

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        ) #keep the gripper closed with 0.0

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward.
        This is after the one simulation step of the action"""

        self.progress_buf[:] += 1
        
        self.rew_buf_initial()
        
        #if self.progress_buf[0] == self.max_episode_length - 1:
        #  self._close_gripper() #it seems like gripping actions cannot be independent for different environments
          #self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        #  self._lift_gripper()
        #Auto idea generator from finding how the thought process works
        #For finding a behavior that works on them all at once, have a heuristic for what will work for each? Well each will have its own way it can figure it out from each spot

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors, these self. values are created in factory base so this should be exisiting correctly here
        #pose (position and quaternion) of the hand and nut, as well as the linear and angular velocity of the hand.
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.plug_higher_quat,
                       self.plug_higher_pos,
                       self.socket_higher_quat,
                       self.socket_higher_pos,
                        ] #need to ensure these values are refreshed on refresh tensors as this all happens at once in post physics step

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_plug - self.keypoints_socket, p=2, dim=-1), dim=-1)

        #these keypoints initalized to zero in acquire tensors then in refresh turned to their correct values

        return keypoint_dist
    

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
        self.rew_buf[:] += (keypoint_dist) * self.cfg_task.rl.keypoint_reward_scale * 0.25 #0.5 so that it doesn't change the average reward values
        
        #Add the amount of success of lifting if its on the last timestep and the post physics step just lifted it
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            print("lift success reward", self.extras['successes'])
            print(self.rew_buf[:])
        print("rewards", self.rew_buf[0])

    def _update_reset_buf(self):
        """Assign environments for reset if maximum episode length has been reached."""

        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        

    #self._move_gripper_to_grasp_pose(
    #    sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
    #)
    #self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
    #self.enable_gravity()
    
    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids) #reset franka then objects into franka
        
        # Close gripper onto plug
        self.disable_gravity()  # to prevent plug from falling
        self._reset_object(env_ids)

        # Close gripper onto nut
        #self.disable_gravity()  # to prevent nut from falling
        #for _ in range(self.cfg_task.env.num_gripper_close_sim_steps):
        #    self.ctrl_target_dof_pos[env_ids, 7:9] = 0.0 #I have no idea the significance of this line
            #Just no arm motion except for grasping on repeat
        #    delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
        #                                  device=self.device)  # no arm motion
        #    self._apply_actions_as_ctrl_targets(actions=delta_hand_pose,
        #                                        ctrl_target_gripper_dof_pos=0.0,
        #                                        do_scale=False)
        #    self.gym.simulate(self.sim)
        #    self.render()
        self._move_gripper_to_grasp_pose(
          sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        )
        self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        self.enable_gravity()#(gravity_mag=abs(self.cfg_base.sim.gravity[2]))
        print("Reset finished for franka and objects and Grip finished for franka in reset_idx")
        #self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps) #this moves the gripper a bit to a new pose

        self._reset_buffers(env_ids)


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
        self.simulate_and_refresh()

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

        plug_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        #print("plug noise", plug_noise_xy)

        plug_noise_xy = plug_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.plug_pos_xy_initial_noise, device=self.device)) #Why use a diagonal matrix here instead of just multiplying???

        #Here just set the root position of x and y given the x and y noise respectively
        self.plug_pos[env_ids, 0] = self.fingertip_centered_pos[env_ids, 0] #+ plug_noise_xy[env_ids, 0]

        self.plug_pos[env_ids, 1] = self.fingertip_centered_pos[env_ids, 1] #+ plug_noise_xy[env_ids, 1]

        #print(self.plug_heights[env_ids, 0].shape) #self.plug_grasp_offsets
        #print(self.plug_heights[env_ids, 0]*1.0)
        #print(self.plug_grasp_offsets)
        self.plug_pos[env_ids, 2] = self.fingertip_centered_pos[env_ids, 2] - (self.plug_heights[env_ids, 0]*0.75)# + self.plug_grasp_offsets # - self.bolt_head_heights.squeeze(-1)

        #Critical functions:
        #This is one of the vectors from refreshing task vectors and acquiring task vectors
        self.plug_quat[env_ids, :] = torch.tensor([0.14943813247359922, 0.0, 0.0, 0.9887710779360422])#self.identity_quat.clone()

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
        #self._reset_plug(before_move_to_grasp=False)