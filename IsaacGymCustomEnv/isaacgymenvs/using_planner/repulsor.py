import os
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
#sys.path.append('/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/')
sys.path.append('../../isaacgym/python/isaacgym') #"/common/home/jhd79/robotics/isaacgym/python/isaacgym/torch_utils.py"
sys.path.append('../../../isaacgym/python/isaacgym')


from isaacgymenvs.utils import torch_jit_utils
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils

import torch
from utilities import fill_buffer, fix_batch, get_scaled_quaternion, quat_diff_rad, pose_world_to_robot_base, axis_angle_to_quaternion, fix_peg_pos_state, fix_peg_pos_quat_state, quat_mul, quat_conjugate, quat_to_angle_axis, quat_from_angle_axis

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#import matplotlib.pyplot as plt #Don't have matplotlib downloadable on my env
#from mpl_toolkits.mplot3d import Axes3D
#from pytransform3d.trajectories import plot_trajectory
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.animation import FuncAnimation

import time
  

"""

Here is the concept:
  Call create_peg_points
  Call creating rectangle vertices
  Since we centerpoint the bottom, we add half the height so all the points act above the bottom and rotate about this point
  Then we interpolate between the points
  Then we add the location of them as offset

"""

# Creating points
def create_peg_points(plug_type, plug_width, plug_depth, plug_length, pos, quat, resolution, remove_top=False, remove_bottom=False, remove_walls=False):   
    
    vertices = create_rectangle_vertices(pos, quat, plug_length, plug_width, plug_depth)

    #print("vertices before: ", vertices.shape)
    vertices = interpolate_between_vertices_rectangle_less_points(vertices, resolution, remove_top = remove_top, remove_bottom = remove_bottom, remove_walls = remove_walls)

    return vertices

def create_rectangle_vertices(pos, quat, length, width, height):
    
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2
    
    # Vertices in local frame
    v_local = torch.tensor([
                              [-half_length, -half_width, -half_height],
                              [half_length, -half_width, -half_height],
                              [half_length, half_width, -half_height],
                              [-half_length, half_width, -half_height],
                              [-half_length, -half_width, half_height],
                              [half_length, -half_width, half_height],
                              [half_length, half_width, half_height],
                              [-half_length, half_width, half_height]
                                                                              ], dtype=torch.float32).to(device)

    v_local[:, 2] += half_height

    #rotate vertices as if they are all up half_height. Then put them back down so -half_height.
    v_local = rotate_vertices(v_local, quat) 
    
    return v_local


def rotate_vertices(vertices, quat): 
    
    #If the center of peg is not the pos, I can offset it to where the pos is then do the rotation
    
    #print("quat dim: ", quat.shape)
    rotation_matrix = torch_jit_utils.quaternion_to_matrix(quat)

    # rotation_matrix = torch_jit_utils.quaternion_to_rotation_matrix(quat)
    rotated_vertices = torch.matmul(vertices, rotation_matrix.T) #torch.matmul(vertices, rotation_matrix) #[8, 3] x [3, 3] 
    
    # These rotations happen individually with each value almost like a batch value. They don't have a specified centerpoint either
    # It actually rotates each point about the 0,0,0 point. 
    # That is why we keep it in its own frame when rotating. 
    
    return rotated_vertices

def update_vertices_parallel(vertices, q, pos):
    
    rotated_vertices = rotate_vertices_parallel(vertices, q)
    res_vertices = rotated_vertices + pos.unsqueeze(1) #[batch, points, 3] + [batch, 1, 3]
    return res_vertices


def rotate_vertices_parallel(vertices, q):
    
    ## Parallel quaternion
    #x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    #rotation_matrix = torch.zeros((q.shape[0], 3, 3), dtype=torch.float32).to(device)
    
    #rotation_matrix[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    #rotation_matrix[:, 0, 1] = 2*x*y - 2*z*w
    #rotation_matrix[:, 0, 2] = 2*x*z + 2*y*w
    #rotation_matrix[:, 1, 0] = 2*x*y + 2*z*w
    #rotation_matrix[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    #rotation_matrix[:, 1, 2] = 2*y*z - 2*x*w
    #rotation_matrix[:, 2, 0] = 2*x*z - 2*y*w
    #rotation_matrix[:, 2, 1] = 2*y*z + 2*x*w
    #rotation_matrix[:, 2, 2] = 1 - 2*x**2 - 2*y**2
    
    rotation_matrix = torch_jit_utils.quaternion_to_matrix(q)

    #Apply rotation parallel, [batch, points, 3] * [batch, 3, 3] = [points, 3]
    rotated_vertices = torch.matmul(vertices, torch.transpose(rotation_matrix, 1, 2)) #supposed to be Rv so when flipping transpose on the 1 2 dim
    
    return rotated_vertices


def interpolate_between_vertices_rectangle(vertices, resolution):
    
    interpolated_points = []
    
    # Bottom
    at1_1 = interpolate_between_two_points(vertices[0], vertices[1], resolution) #01
    interpolated_points.extend(interpolate_between_two_points(vertices[1], vertices[2], resolution)) #12
    at1_2 = interpolate_between_two_points(vertices[2], vertices[3], resolution) #23
    interpolated_points.extend(create_face(at1_1, at1_2, resolution))
    interpolated_points.extend(interpolate_between_two_points(vertices[3], vertices[0], resolution)) #34
    
    # Top
    at1_1 = interpolate_between_two_points(vertices[4], vertices[5], resolution)
    interpolated_points.extend(interpolate_between_two_points(vertices[5], vertices[6], resolution))
    at1_2 = interpolate_between_two_points(vertices[6], vertices[7], resolution)
    interpolated_points.extend(create_face(at1_1, at1_2, resolution))
    interpolated_points.extend(interpolate_between_two_points(vertices[7], vertices[4], resolution))
    # Right wall
    at1_1 = interpolate_between_two_points(vertices[3], vertices[7], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
    at1_2 = interpolate_between_two_points(vertices[4], vertices[0], resolution)
    interpolated_points.extend(create_face(at1_1, at1_2, resolution))
    
    #Left wall
    at1_1 = interpolate_between_two_points(vertices[2], vertices[6], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
    at1_2 = interpolate_between_two_points(vertices[5], vertices[1], resolution)
    interpolated_points.extend(create_face(at1_1, at1_2, resolution))
    
    # Back wall
    at1_1 = interpolate_between_two_points(vertices[1], vertices[5], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
    at1_2 = interpolate_between_two_points(vertices[4], vertices[0], resolution)
    interpolated_points.extend(create_face(at1_1, at1_2, resolution))
    
    #Front wall
    at1_1 = interpolate_between_two_points(vertices[2], vertices[6], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
    at1_2 = interpolate_between_two_points(vertices[7], vertices[3], resolution)
    interpolated_points.extend(create_face(at1_1, at1_2, resolution))

    return torch.stack(interpolated_points).to(device) #[1, vertices]


def interpolate_between_vertices_rectangle_less_points(vertices, resolution, remove_top=False, remove_bottom=False, remove_walls=False):
    interpolated_points = []
    #bottom
    if remove_bottom == False:
      at1_1=interpolate_between_two_points(vertices[0], vertices[1], resolution) #01
      #interpolate outputs [batch, 3] values
      interpolated_points.extend(interpolate_between_two_points(vertices[1], vertices[2], resolution)) #12
      at1_2=interpolate_between_two_points(vertices[2], vertices[3], resolution) #23
      interpolated_points.extend(create_face(at1_1, at1_2, resolution))
      interpolated_points.extend(interpolate_between_two_points(vertices[3], vertices[0], resolution)) #34
    #top
    if remove_top == False:
      at1_1=interpolate_between_two_points(vertices[4], vertices[5], resolution)
      interpolated_points.extend(interpolate_between_two_points(vertices[5], vertices[6], resolution))
      at1_2=interpolate_between_two_points(vertices[6], vertices[7], resolution)
      interpolated_points.extend(create_face(at1_1, at1_2, resolution))
      interpolated_points.extend(interpolate_between_two_points(vertices[7], vertices[4], resolution))
    if remove_walls == False:
      #right wall
      at1_1=interpolate_between_two_points(vertices[3], vertices[7], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
      at1_2=interpolate_between_two_points(vertices[4], vertices[0], resolution)
      interpolated_points.extend(create_face(at1_1, at1_2, resolution))
      #left wall
      at1_1=interpolate_between_two_points(vertices[2], vertices[6], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
      at1_2=interpolate_between_two_points(vertices[5], vertices[1], resolution)
      interpolated_points.extend(create_face(at1_1, at1_2, resolution))
      #back wall
      at1_1=interpolate_between_two_points(vertices[1], vertices[5], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
      at1_2=interpolate_between_two_points(vertices[4], vertices[0], resolution)
      interpolated_points.extend(create_face(at1_1, at1_2, resolution))
      #front wall
      at1_1=interpolate_between_two_points(vertices[2], vertices[6], resolution) #Make sure they are inverse of each other, as 3 to 7 is up and 4 to 0 is down
      at1_2=interpolate_between_two_points(vertices[7], vertices[3], resolution)
      interpolated_points.extend(create_face(at1_1, at1_2, resolution))

    return torch.stack(interpolated_points).to(device) #[1, vertices]

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0])
    else:
        return []

def create_face(points1, points2, resolution):
    new_points=[]
    for i in range(len(points1)):
      new_points.extend(interpolate_between_two_points(points1[i], points2[len(points1)-1-i], resolution))
    return new_points

def interpolate_between_two_points(p1, p2, resolution):
    new_points=[]
    for j in range(resolution+1):
        t = j / resolution #between 0 and 1
        interpolated_point = (1 - t) * p1 + t * p2 #as it goes closer to point 2 from point 1
        new_points.append(interpolated_point)
    return new_points


"""

      width: 0.015957
      depth: 0.009910
      length: 0.050

"""

"""

      width: 0.0162182
      height: 0.028
      depth: 0.023

      The table is 1.04 height and the peg at 1.04 is sitting right on the table. So the peg position value must be at its bottom so its center is +0.5*length
It rotates about its center. When we are given the peg location just add 0.5*length to it.
For the socket, the same thing. I assume this point is at the center of the socket as the x and y that line up for peg and socket mean insertion. 
So to create the rectnagles for the socket, two wider faces and two pillars
wider faces length=(l/2)-(p/2), width=w, height=h, centered at (l-length)+(length/2), w/2, h/2 and (length/2), w/2, h/2

Or make a rectangle of entire size of hole, a rectangle inside this for the size of the hole, combine them, then remove any points at the x y values of slightly inner

In my sim everything is centered at their very center
In IsaacGym everything in centered at the center of x and y and bottom of z
We have the proper peg size
We have the proper hole size
We have a guess on the outer socket size
I am not sure how 1.04 with a height of 0.028 which is 1.068 top makes 1.065 edge when the peg is against it. And the peg is 1.04 when it inserts.
This means the height must be 0.025 (the base is 0.003)
What the agent sees and what seems to really be the case is that the hole for the peg is at 1.065 which is all that matters

"""

# Socket
def create_socket(plug_type, plug_width, plug_depth, plug_length, socket_width, socket_depth, socket_length, device, pose, resolution=10):
  
  length = plug_length #- 0.004 #0.046
  width = plug_width
  height = 0.5* plug_length
  #height = 0.025


  if pose == None:
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(device) #bottom start at 1.04
    pose_l = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(device) #bottom start at 1.04
  else:
    pose_l = pose
    
  # Do this all in one pose then change to correct pose, can rotate each point later too
  pose_l = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(device)

  #In this sim we see the pos as the center so the input for z is actually halfway down so move it back up
  #pose_l[2] += height / 2.0

  #larger_box=create_peg_points(pose_l, length, width, height, resolution) #pose, length, width, height, resolution
  #larger_box = create_peg_points( pose_l, plug_length, plug_width, plug_depth, resolution, remove_bottom = True, remove_top = True )
 
  larger_box = create_peg_points(
                                    plug_type = plug_type,
                                    plug_width = plug_width,
                                    plug_depth =  plug_depth,
                                    plug_length = plug_length,
                                    pos = pose_l[0:3],
                                    quat = pose_l[3:7],
                                    resolution = resolution,
                                    remove_bottom = True,
                                    remove_top = True,
                                    remove_walls = False
                                                                 )

  tolerance_l = socket_width
  #tolerance_l=0.0162182 #This is just hole length dim1

  tolerance_w = (socket_width-plug_width) + plug_depth #This is hole width dim2
  #tolerance_w = (0.0162182-0.015957) + 0.009910 #This is hole width dim2

  pose_hole = pose_l #center of the other one
  #hole = create_peg_points(pose_hole, tolerance_l, tolerance_w, height, resolution, remove_bottom = True) #pose, length, width, height, resolution
  
  hole = create_peg_points(
                              plug_type = plug_type,
                              plug_width = socket_width,
                              plug_depth = socket_depth,
                              plug_length = socket_length,
                              pos = pose_hole[0:3],
                              quat = pose_hole[3:7],
                              resolution = resolution,
                              remove_bottom = True,
                              remove_top = False,
                              remove_walls = False
                                                                       )
                                                            
  

  # Removing the tolerance of points
  tolerance_l -= 1e-5
  tolerance_w -= 1e-5
  socket = torch.cat([larger_box, hole], axis=0)

  max_values = torch.tensor([[pose_l[0] + (tolerance_l/2.0), pose_l[1] + (tolerance_w/2.0), pose_l[2] + (plug_depth/2.0)]]).to(device) #centerpoint plus half length
  min_values = torch.tensor([[pose_l[0] - (tolerance_l/2.0), pose_l[1] - (tolerance_w/2.0), pose_l[2] - (plug_depth/2.0)]]).to(device)

  mask = torch.all((socket[:, 0:3] >= min_values) & (socket[:, 0:3] <= max_values), dim=1)
  indices = torch.nonzero(mask).squeeze()

  size = socket.shape[0]-indices.shape[0]
  mask = torch.ones_like(socket, dtype=torch.bool)
  mask[indices] = 0
  socket = torch.masked_select(socket, mask)
  socket = socket.reshape(size, 3)

  #print("pose is: ", pose)
  #socket = rotate_vertices(socket, pose[:,3:7])
  #socket += pose[:,0:3] 

  return socket


def define_trajectory_parallel(goal_pos, goal_quat, resolution, plug_lengths): 
  
  starting_pos = goal_pos.clone()
  starting_pos[:, 2] += 0.050 * 2.0 #(plug_lengths*2.0)
  trajectory = interpolate_between_two_points_parallel(starting_pos, goal_pos, resolution) #Need to make this parallel then need to make the things using the trajectory use the other things
  trajectory = torch.cat([trajectory, goal_quat.repeat(1, resolution+1, 1).to(device)], dim=-1)
  #trajectory = torch.cat([trajectory, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(trajectory.shape[0], resolution+1, 1).to(device)], dim=-1)
  return trajectory


def define_trajectory_parallel_Ismarou(goal_pos, goal_quat, init_plug_pos, resolution): 
  
  starting_pos = goal_pos.clone()
  starting_pos[:, 2] += 0.050 * 2.0 #(plug_lengths*2.0) # = init_plug_pos[:,2]
  
  #Need to make this parallel then need to make the things using the trajectory use the other things
  trajectory = interpolate_between_two_points_parallel(starting_pos, goal_pos, resolution) 
  goal_quat_repeat = goal_quat.unsqueeze(1).repeat(1, resolution + 1, 1).to(device)
  trajectory = torch.cat([trajectory, goal_quat_repeat], dim=2)
  return trajectory


def interpolate_between_two_points_parallel(p1, p2, resolution):
    
    t = torch.linspace(0, 1, steps=resolution+1, device=p1.device).unsqueeze(0).unsqueeze(-1)  #[1, resolution+1]
    interpolated_points = ((1 - t) * p1.unsqueeze(1)) + (t * p2.unsqueeze(1))  #[1, resolution+1, 1] * [batch, 1, 3] = [batch, resolution+1, 3]
    
    return interpolated_points


def quaternion_towards_in_angle_axis(a: torch.Tensor, b: torch.Tensor): #b to a, quaternion relative rotation to move
    
    #b is q1 and a is q2. This is b to a rotation
    b_conj = quat_conjugate(b)
    rotation_towards = quat_mul(b_conj, a)
    rotation_towards = rotation_towards / torch.norm(rotation_towards, dim=-1, keepdim=True)
    angle, axis = quat_to_angle_axis(rotation_towards)
    return angle, axis

#input is peg: [envs, points, 3], socket: [envs, points, 3], peg_pose: [envs, 7], trajectory: [points, 7]
def Potential_Field_actions(peg, socket, peg_pose, trajectory, repulse_dist, b_pos=0.33, b_rot=0.0): #higher b means more repulsion

  #Positional Attractive force
  dist_pos = torch.norm(peg_pose[:, None, 0:3] - trajectory[:, :, 0:3], dim=-1) #[envs, 1, positions] - [env, points, positions] = [envs, points, positions]
  
  #Closest
  min_val, min_index=torch.min(dist_pos, dim=1)
  dist_between_traj_points=torch.norm(trajectory[0, 0, 0:3] - trajectory[0, 1, 0:3], dim=0).item()
  for i in range(min_index.shape[0]):
    if dist_pos[i, min_index[i]].item() < dist_between_traj_points: #if we are closer to this point than the points are close to each other, then the target point must be the next point in the trajectory
      if min_index[i] < trajectory.shape[1]-1:
        min_index[i]+=1
  
  tar_attract = trajectory[torch.arange(len(min_index)), min_index, :]
  
  max_val, _ = torch.max(torch.abs(tar_attract[:, 0:3] - peg_pose[:, 0:3]), dim=1, keepdim=True)
  max_val[max_val==0.0] = 1.0
  
  pos_attractor_force = (tar_attract[:, 0:3] - peg_pose[:, 0:3]) / max_val
  
  #Rotational attractive force
  angle, axis = quaternion_towards_in_angle_axis(tar_attract[:, 3:7], peg_pose[:, 3:7])
  rot_attractor_force = torch.cat([axis, angle.unsqueeze(1)], dim=-1)
  attractive_force = torch.cat([pos_attractor_force, rot_attractor_force], dim=-1)
  
  #New repulsive force
  
  #Positional Repulsive force
  dist_pos=torch.norm(peg[:, None, :, :] - socket[:, :, None, :], dim=-1) #broadcast it so that each of the peg pos values go with each of the socket pos values, [envs, socket, peg]
  min_val, lowest_indices_of_pegs = torch.min(dist_pos, dim=2)
  min_val, lowest_indices_of_sockets = torch.min(min_val, dim=1)
  socket_to_peg = torch.zeros((socket.shape[0], 3), dtype=torch.float32).to(device)
  point_relative_loc = torch.zeros((socket.shape[0], 3), dtype=torch.float32).to(device)
  
  for jj in range(socket.shape[0]):
    socket_to_peg[jj] = peg[jj, lowest_indices_of_pegs[jj, lowest_indices_of_sockets[jj]], 0:3] - socket[jj, lowest_indices_of_sockets[jj], 0:3]
    point_relative_loc[jj] = peg[jj, lowest_indices_of_pegs[jj, lowest_indices_of_sockets[jj]], 0:3] - peg_pose[jj, 0:3]
  

  jacobian = jacobian_calculation_parallel(peg_pose[:, 3:7], point_relative_loc).to(device)
  
  repulsive_force = calculate_cspace_forces_parallel(jacobian, socket_to_peg).to(device)
  
  max_val, _ = torch.max(torch.abs(repulsive_force[:, 0:3]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  
  too_far = max_val.clone()
  too_far_old = max_val.clone()
  too_far[too_far<=repulse_dist] = 1.0
  too_far[too_far_old>repulse_dist] = 0.0
  
  repulsive_force[:, 0:3] = repulsive_force[:, 0:3] / max_val #this normalizes
  repulsive_force[:, :] *= too_far #this makes no force for ones too far away as we only did minimum distance so far
  repulsive_force[:, 6] /= max_val.squeeze(1)
  
  # Scaling:
  
  #Angle
  repulsive_force[:, 6] *= b_rot #0.0
  attractive_force[:, 6] *= 5.0 #import to scale to high number, think of this as the base of scaling it similar to keeping the repulsive force between -1 and 1
  attractive_force[:, 6] *= (1.0-b_rot) #1.0
  
  # Pos
  repulsive_force[:, 0:3] *= b_pos
  attractive_force[:, 0:3] *= (1.0-b_pos)

  # Adding axis angles as quaternion then convert back to axis angles that are now 3dim instead of 4dim
  repulsive_quat = quat_from_angle_axis(repulsive_force[:, 6], repulsive_force[:, 3:6])
  attractive_quat = quat_from_angle_axis(attractive_force[:, 6], attractive_force[:, 3:6])
  joined_quat = quat_mul(repulsive_quat, attractive_quat)
  #joined_quat = attractive_quat
  angle, axis = quat_to_angle_axis(joined_quat)
  joined_angle_axis_rot = prepare_actions(angle, axis)#torch.clip(prepare_actions(angle, axis), -1.0, 1.0)

  #Scale the rotations between -1 and 1 by dividing by the maximum value
  max_val, _ = torch.max(torch.abs(joined_angle_axis_rot[:, :]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  joined_angle_axis_rot /= max_val

  joined_pos = attractive_force[:, 0:3] + repulsive_force[:, 0:3]
  #joined_pos = attractive_force[:, 0:3]
  max_val, _ = torch.max(torch.abs(joined_pos[:, 0:3]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  joined_pos /= max_val
  final_actions = torch.cat([joined_pos, joined_angle_axis_rot], dim=-1)

  return final_actions



#input is peg: [envs, points, 3], socket: [envs, points, 3], peg_pose: [envs, 7], trajectory: [points, 7]
def Potential_Field_actions_Ismarou(peg, socket, peg_pose, noisy_plug_pos, noisy_plug_quat, noisy_socket_pos, noisy_socket_quat, trajectory, repulse_dist, asset_indices, wp_plug_meshes_sampled_points, wp_socket_meshes, interpen_thresh, wp_device, device, b_pos=0.33, b_rot=0.0): #higher b means more repulsion

  #Positional Attractive force
  dist_pos = torch.norm(peg_pose[:, None, 0:3] - trajectory[:, :, 0:3], dim=-1) # [envs, 1, positions] - [env, points, positions] = [envs, points, positions]
  
  #Closest
  min_val, min_index=torch.min(dist_pos, dim=1)
  #dist_between_traj_points=torch.norm(trajectory[0, 0, 0:3] - trajectory[0, 1, 0:3], dim=0).item()
  
  # Carrot Following:

  # If we are closer to this point than the points are close to each other, then the target point must be the next point in the trajectory
  for i in range(min_index.shape[0]):
    if dist_pos[i, min_index[i]].item() < dist_between_traj_points: 
    #if dist_pos[i, min_index[i]].item() < 0.002: 
      if min_index[i] < trajectory.shape[1]-1:
        min_index[i]+=1
  
  tar_attract = trajectory[torch.arange(len(min_index)), min_index, :]
  
  max_val, _ = torch.max(torch.abs(tar_attract[:, 0:3] - peg_pose[:, 0:3]), dim=1, keepdim=True)
  max_val[max_val==0.0] = 1.0
  
  pos_attractor_force = (tar_attract[:, 0:3] - peg_pose[:, 0:3]) / max_val
  
  #Rotational attractive force
  angle, axis = quaternion_towards_in_angle_axis(tar_attract[:, 3:7], peg_pose[:, 3:7])
  rot_attractor_force = torch.cat([axis, angle.unsqueeze(1)], dim=-1)
  attractive_force = torch.cat([pos_attractor_force, rot_attractor_force], dim=-1)
  
  #Repulsive force
  repulsive_fields, plug_closest_points, socket_closest_points = algo_utils.repulsive_field(
                                                                                                asset_indices = asset_indices,
                                                                                                plug_pos = noisy_plug_pos,
                                                                                                plug_quat = noisy_plug_quat,
                                                                                                socket_pos = noisy_socket_pos,
                                                                                                socket_quat = noisy_socket_quat,
                                                                                                wp_plug_meshes_sampled_points = wp_plug_meshes_sampled_points,
                                                                                                wp_socket_meshes = wp_socket_meshes, 
                                                                                                interpen_thresh = interpen_thresh,
                                                                                                wp_device = wp_device,
                                                                                                device = device
                                                                                                                                                                )

                                                    
  #point_relative_loc = noisy_plug_pos - plug_closest_points
  point_relative_loc = plug_closest_points - noisy_plug_pos

  jacobian = jacobian_calculation_parallel(peg_pose[:, 3:7], point_relative_loc).to(device)
  repulsive_force = calculate_cspace_forces_parallel(jacobian, repulsive_fields).to(device)
  
  max_val, _ = torch.max(torch.abs(repulsive_force[:, 0:3]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  
  too_far = max_val.clone()
  too_far_old = max_val.clone()
  too_far[too_far <= repulse_dist] = 1.0
  too_far[too_far_old > repulse_dist] = 0.0
  
  repulsive_force[:, 0:3] = repulsive_force[:, 0:3] / max_val
  repulsive_force[:, :] *= too_far 
  repulsive_force[:, 6] /= max_val.squeeze(1)
  
  # Scaling:
  
  if not w:

    #Angle
    repulsive_force[:, 6] *= b_rot 
    attractive_force[:, 6] *= 5.0 
    attractive_force[:, 6] *= (1.0 - b_rot) 
    
    # Pos
    repulsive_force[:, 0:3] *= b_pos
    attractive_force[:, 0:3] *= (1.0-b_pos)
  
  else:
    
    #Angle
    repulsive_force[:, 6] *= b_rot 
    attractive_force[:, 6] *= 5.0 
    attractive_force[:, 6] *= (1.0 - b_rot) 
    
    # Pos
    repulsive_force[:, 0:3] *= b_pos
    attractive_force[:, 0:3] *= (1.0-b_pos)
  


  # Adding axis angles as quaternion then convert back to axis angles that are now 3dim instead of 4dim
  repulsive_quat = quat_from_angle_axis(repulsive_force[:, 6], repulsive_force[:, 3:6])
  attractive_quat = quat_from_angle_axis(attractive_force[:, 6], attractive_force[:, 3:6])
  joined_quat = quat_mul(repulsive_quat, attractive_quat)
  angle, axis = quat_to_angle_axis(joined_quat)
  joined_angle_axis_rot = prepare_actions(angle, axis)

  #Scale the rotations between -1 and 1 by dividing by the maximum value
  max_val, _ = torch.max(torch.abs(joined_angle_axis_rot[:, :]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  joined_angle_axis_rot /= max_val

  joined_pos = attractive_force[:, 0:3] + repulsive_force[:, 0:3]
  max_val, _ = torch.max(torch.abs(joined_pos[:, 0:3]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  joined_pos /= max_val
  final_actions = torch.cat([joined_pos, joined_angle_axis_rot], dim=-1)

  return final_actions



# Input is 
#
#           plug_points: [envs, points, 3], 
#           socket_points: [envs, points, 3], 
#           noisy_plug_pos: [envs, 3],
#           noisy_plug_quat: [envs, 4], 
#           Carrot_trajectory: [points, 7]


def Attractive_Force(peg_pose, trajectory):
 
  # Attractive force
  dist_pos=torch.norm(peg_pose[:, None, 0:3] - trajectory[:, :, 0:3], dim=-1) #[envs, 1, positions] - [env, points, positions] = [envs, points, positions]
  
  #Closest
  min_val, min_index=torch.min(dist_pos, dim=1)
  dist_between_traj_points=torch.norm(trajectory[0, 0, 0:3] - trajectory[0, 1, 0:3], dim=0).item()
  for i in range(min_index.shape[0]):
    if dist_pos[i, min_index[i]].item() < dist_between_traj_points: #if we are closer to this point than the points are close to each other, then the target point must be the next point in the trajectory
      if min_index[i] < trajectory.shape[1]-1:
        min_index[i]+=1
  
  tar_attract = trajectory[torch.arange(len(min_index)), min_index, :]
  max_val, _ = torch.max(torch.abs(tar_attract[:, 0:3] - peg_pose[:, 0:3]), dim=1, keepdim=True)
  max_val[max_val==0.0] = 1.0
  pos_attractor_force = (tar_attract[:, 0:3] - peg_pose[:, 0:3]) / max_val
  
  #Rotational attractive force
  angle, axis = quaternion_towards_in_angle_axis(tar_attract[:, 3:7], peg_pose[:, 3:7])
  rot_attractor_force = torch.cat([axis, angle.unsqueeze(1)], dim=-1)
  attractive_force = torch.cat([pos_attractor_force, rot_attractor_force], dim=-1)
  
  return attractive_force


'''
def Attractive_Force(noisy_plug_pos, noisy_plug_quat, Carrot_trajectory, Carrot_threshold):
    
    # Attractive force
    dist_pos = torch.norm(noisy_plug_pos[:, None, 0:3] - Carrot_trajectory[:, :, 0:3], dim=-1)  # [envs, 1, positions] - [env, points, positions] = [envs, points, positions]

    # Closest
    min_val, min_index = torch.min(dist_pos, dim=1)

    dist_between_traj_points = torch.norm(Carrot_trajectory[:, 0, 0:3] - Carrot_trajectory[:, 1, 0:3], dim=-1)  # [envs]

    # Create a mask to identify indices that need to be incremented
    increment_mask = dist_pos[torch.arange(len(min_index)), min_index] < dist_between_traj_points[:, None]
    increment_mask &= min_index < Carrot_trajectory.shape[1] - 1

    # Increment the indices based on the mask
    if increment_mask.any():
        min_index[increment_mask] += 1

    # Gather the target attract points based on the updated min_index
    tar_attract = Carrot_trajectory[torch.arange(len(min_index)), min_index, :]

    max_val, _ = torch.max(torch.abs(tar_attract[:, 0:3] - noisy_plug_pos[:, 0:3]), dim=1, keepdim=True)
    max_val[max_val == 0.0] = 1.0
    pos_attractor_force = (tar_attract[:, 0:3] - noisy_plug_pos[:, 0:3]) / max_val

    # Rotational attractive force
    angle, axis = quaternion_towards_in_angle_axis(tar_attract[:, 3:7], noisy_plug_quat[:, 0:4])
    rot_attractor_force = torch.cat([axis, angle.unsqueeze(1)], dim=-1)

    attractive_force = torch.cat([pos_attractor_force, rot_attractor_force], dim=-1)

    return attractive_force
'''

'''
def Repulsive_Force(plug_points, socket_points, noisy_plug_pos, noisy_plug_quat, repulse_dist, device):
    
    # Positional Repulsive force
    dist_pos = torch.norm(plug_points[:, None, :, :] - socket_points[:, :, None, :], dim=-1)  # [envs, socket, peg]

    min_val, lowest_indices_of_pegs = torch.min(dist_pos, dim=2)
    min_val, lowest_indices_of_sockets = torch.min(min_val, dim=1)

    # Create a mask for the lowest indices
    mask = torch.zeros_like(dist_pos, dtype=torch.bool)
    mask[torch.arange(dist_pos.shape[0]), lowest_indices_of_sockets, lowest_indices_of_pegs[torch.arange(dist_pos.shape[0]), lowest_indices_of_sockets]] = True

    # Use the mask to select the corresponding plug and socket points
    selected_plug_points = plug_points[torch.arange(plug_points.shape[0]).unsqueeze(-1), lowest_indices_of_pegs[torch.arange(dist_pos.shape[0]), lowest_indices_of_sockets]]
    selected_socket_points = socket_points[torch.arange(socket_points.shape[0]), lowest_indices_of_sockets]

    # Compute socket_to_peg using the selected points
    socket_to_peg = selected_plug_points - selected_socket_points

    # Compute point_relative_loc using broadcasting
    point_relative_loc = selected_plug_points - noisy_plug_pos.unsqueeze(1)

    jacobian = jacobian_calculation_parallel(noisy_plug_quat[:, 0:4], point_relative_loc).to(device)
    repulsive_force = calculate_cspace_forces_parallel(jacobian, socket_to_peg).to(device)

    max_val, _ = torch.max(torch.abs(repulsive_force[:, 0:3]), dim=1, keepdims=True)
    max_val[max_val == 0.0] = 1.0

    too_far = max_val.clone()
    too_far_old = max_val.clone()
    too_far[too_far <= repulse_dist] = 1.0
    too_far[too_far_old > repulse_dist] = 0.0

    repulsive_force[:, 0:3] = repulsive_force[:, 0:3] / max_val  # this normalizes
    repulsive_force[:, :] *= too_far  # this makes no force for ones too far away as we only did minimum distance so far
    repulsive_force[:, 6] /= max_val.squeeze(1)

    return repulsive_force
'''

def Repulsive_Force(peg, socket, peg_pose, repulse_dist, device):
   
  #new repulsive force
  dist_pos=torch.norm(peg[:, None, :, :] - socket[:, :, None, :], dim=-1) #broadcast it so that each of the peg pos values go with each of the socket pos values, [envs, socket, peg]
  min_val, lowest_indices_of_pegs = torch.min(dist_pos, dim=2)
  min_val, lowest_indices_of_sockets = torch.min(min_val, dim=1)
  socket_to_peg = torch.zeros((socket.shape[0], 3), dtype=torch.float32).to(device)
  point_relative_loc = torch.zeros((socket.shape[0], 3), dtype=torch.float32).to(device)
  
  for jj in range(socket.shape[0]):
    socket_to_peg[jj] = peg[jj, lowest_indices_of_pegs[jj, lowest_indices_of_sockets[jj]], 0:3] - socket[jj, lowest_indices_of_sockets[jj], 0:3]
    point_relative_loc[jj] = peg[jj, lowest_indices_of_pegs[jj, lowest_indices_of_sockets[jj]], 0:3] - peg_pose[jj, 0:3]
  jacobian = jacobian_calculation_parallel(peg_pose[:, 3:7], point_relative_loc).to(device)
  
  repulsive_force = calculate_cspace_forces_parallel(jacobian, socket_to_peg).to(device)
  
  max_val, _ = torch.max(torch.abs(repulsive_force[:, 0:3]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  
  too_far = max_val.clone()
  too_far_old = max_val.clone()
  too_far[too_far<=repulse_dist] = 1.0
  too_far[too_far_old>repulse_dist] = 0.0
  
  repulsive_force[:, 0:3] = repulsive_force[:, 0:3] / max_val #this normalizes
  repulsive_force[:, :] *= too_far #this makes no force for ones too far away as we only did minimum distance so far
  repulsive_force[:, 6] /= max_val.squeeze(1)
  
  return repulsive_force


def Potential_Field_actions_Conditional(plug_points, socket_points, noisy_plug_pos, noisy_plug_quat, Carrot_trajectory, Carrot_threshold, repulse_dist, device, w, residual_weight, fixed_w_pos, fixed_w_rot, t, max_episode_length): 
  
  noisy_plug_poses = torch.cat([noisy_plug_pos, noisy_plug_quat], dim=-1)
  
  ### Attractive force ###
  
  attractive_force = Attractive_Force(noisy_plug_poses, Carrot_trajectory)
  
  ### Repulsive force ###
  #print("plug_points", plug_points[8])
  repulsive_force = Repulsive_Force(plug_points, socket_points, noisy_plug_poses, repulse_dist, device)
  #print(repulsive_force[8])

  # Scaling:
  if not w:

    #Angle
    repulsive_force[:, 6] *= fixed_w_rot
    attractive_force[:, 6] *= 5.0 
    attractive_force[:, 6] *= (1.0 - fixed_w_rot) 
    # Pos
    repulsive_force[:, 0:3] *= fixed_w_pos
    attractive_force[:, 0:3] *= (1.0 - fixed_w_pos)
  
  else:
    
    residual_weight += 1.0
    residual_weight *= 0.5

    #print("Repulsive shape: ", repulsive_force[:, 6].shape)
    #print("Residual weight: ", residual_weight.shape)

    #Angle
    repulsive_force[:, 6] *= residual_weight.squeeze(1) * max((t-max_episode_length)/max_episode_length)
    attractive_force[:, 6] *= residual_weight.squeeze(1) * max((t-max_episode_length)/max_episode_length) 
    attractive_force[:, 6] *= (1.0 - residual_weight.squeeze(1) * max((t-max_episode_length)/max_episode_length))
    # Pos
    repulsive_force[:, 0:3] *= residual_weight * max((t-max_episode_length)/max_episode_length)
    attractive_force[:, 0:3] *= (1.0 - residual_weight * max((t-max_episode_length)/max_episode_length))


  # Adding axis angles as quaternion then convert back to axis angles that are now 3dim instead of 4dim
  repulsive_quat = quat_from_angle_axis(repulsive_force[:, 6], repulsive_force[:, 3:6])
  attractive_quat = quat_from_angle_axis(attractive_force[:, 6], attractive_force[:, 3:6])
  joined_quat = quat_mul(repulsive_quat, attractive_quat)
  #joined_quat = attractive_quat.clone()
  angle, axis = quat_to_angle_axis(joined_quat)
  joined_angle_axis_rot = prepare_actions(angle, axis)#torch.clip(prepare_actions(angle, axis), -1.0, 1.0)

  joined_pos = attractive_force[:, 0:3] + repulsive_force[:, 0:3]

  max_val, _ = torch.max(torch.abs(joined_pos[:, 0:3]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  joined_pos /= max_val


  #Scale the rotations between -1 and 1 by dividing by the maximum value
  max_val, _ = torch.max(torch.abs(joined_angle_axis_rot[:, :]), dim=1, keepdims=True)
  max_val[max_val==0.0] = 1.0
  joined_angle_axis_rot /= max_val

  final_actions = torch.cat([joined_pos, joined_angle_axis_rot], dim=-1)
  return final_actions


"""
  Should be [-1, 1] for position and rotation for potential field. 
  The residual action can add on and we make it once again [-1, 1]
  The rotations here will have been scaled relative to each other then kept between -1 and 1 NOT CLIPPED but scaled
"""

def jacobian_calculation_parallel(q, distance_to_reference_point):

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  #quat components
    x_p, y_p, z_p = distance_to_reference_point[:, 0], distance_to_reference_point[:, 1], distance_to_reference_point[:, 2]

    # Jacobian
    jacobian = torch.zeros((q.size(0), 3, 7), dtype=torch.float32)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 2, 2] = 1.0
    jacobian[:, 0, 3] = (2*y_p*y)+(2*z_p*z)
    jacobian[:, 0, 4] = (x_p*-4*y)+(y_p*2*x)+(z_p*2*w)
    jacobian[:, 0, 5] = (x_p*-4*z)+(y_p*-2*w)+(z_p*2*x)
    jacobian[:, 0, 6] = (y_p*-2*z)+(z_p*2*y)
    jacobian[:, 1, 3] = (x_p*2*y)+(y_p*-4*x)+(z_p*-2*w)
    jacobian[:, 1, 4] = (x_p*2*x)+(z_p*2*z)
    jacobian[:, 1, 5] = (x_p*2*w)+(y_p*-4*z)+(z_p*2*y)
    jacobian[:, 1, 6] = (x_p*2*z)+(z_p*-2*x)
    jacobian[:, 2, 3] = (x_p*2*z)+(y_p*2*w)+(z_p*-4*x)
    jacobian[:, 2, 4] = (x_p*-2*w)+(y_p*2*z)+(z_p*-4*y)
    jacobian[:, 2, 5] = (x_p*2*x)+(y_p*2*y)
    jacobian[:, 2, 6] = (x_p*-2*y)+(y_p*2*x)

    return jacobian

def calculate_cspace_forces_parallel(jacobian, force_vector):

    # Transpose along 1 and 2 dim mean swap the dim 1 and 2, [batch, 7, 3], [batch, 3, 1] = [batch, 7, 1]
    cspace_actions = torch.matmul(torch.transpose(jacobian, 1, 2), force_vector.unsqueeze(-1)).squeeze(-1) 
    magnitude = torch.norm(cspace_actions[:, 3:7].squeeze(-1), dim=-1)
    cspace_actions[:, 3:7] /= magnitude.unsqueeze(-1)
    angle, axis = quat_to_angle_axis(cspace_actions[:, 3:7])
    cspace_actions[:, 3:6] = axis
    cspace_actions[:, 6] = angle
    return cspace_actions

def prepare_actions(angle, axis):
    rot_actions = angle.unsqueeze(1) * axis
    return rot_actions



"""
  ReadMe
    Creating pegs and sockets requires a pose of [7]
    Then when acting requires points of [envs, points, 3] and poses of [envs, 7] and a trajectory of [points, 3]
    The trajectory is peg centerpoint pos values rather than the peg bottom points that IsaacGym uses
    On the plot it is the same type plot, just we translate the positions to the center point positions for my simulator which match to the trajectory points
"""

# Only for Toy / Print purposes:

def define_trajectory(starting_pos=None, resolution=100, peg_height=0.050):

  if starting_pos == None:
    starting_pos = torch.tensor([0.13, 0.0, 1.09+(peg_height/2.0)]).to(device) #now these values count as peg centerpoints rather than bottom points
  goal_pos = torch.tensor([0.13, 0.0, 1.04]).to(device) #exactly at the 1.04 z value -----------------------------------------------------------------POINTS HERE MAKE ALL THE DIFFERENCE
  trajectory=interpolate_between_two_points(starting_pos, goal_pos, resolution)
  trajectory=torch.stack(trajectory)
  trajectory=torch.cat([trajectory, torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(trajectory.shape[0], 1).to(device)], dim=1)
  return trajectory

def nearest_point_plot(peg, socket, lowest_indices_of_sockets, lowest_indices_of_pegs, elev=0, azim=90):
  
  # Create a 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter(peg[:, 0], peg[:, 1], peg[:, 2], s=2)
  
  ax.scatter(socket[int(socket.shape[0]/2):int(socket.shape[0]/2)+300, 0], socket[int(socket.shape[0]/2):int(socket.shape[0]/2)+300, 1], socket[int(socket.shape[0]/2):int(socket.shape[0]/2)+300, 2], s=2)
  
  ax.scatter(socket[lowest_indices_of_sockets, 0], socket[lowest_indices_of_sockets, 1], socket[lowest_indices_of_sockets, 2], s=100)
  
  ax.scatter(peg[lowest_indices_of_pegs[lowest_indices_of_sockets], 0], peg[lowest_indices_of_pegs[lowest_indices_of_sockets], 1], peg[lowest_indices_of_pegs[lowest_indices_of_sockets], 2], s=100)

  # Set labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  ax.set_aspect("equal")
  ax.view_init(elev=elev, azim=azim)
  ax.locator_params(axis='x', nbins=1)
  ax.locator_params(axis='y', nbins=1)
  plt.savefig("NEAREST_POINTS.png")

def plot_3d_points(vec1, socket, trajectory, elev=0, azim=45):
  
  # Create a 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter(vec1[:, 0], vec1[:, 1], vec1[:, 2])
  
  ax.scatter(socket[:, 0], socket[:, 1], socket[:, 2])
  
  ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], s=100)
  
  # Set labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  ax.set_aspect("equal")
  ax.view_init(elev=elev, azim=azim)
  plt.savefig("repulsor_3d_elev"+str(elev)+"_azim"+str(azim)+".png")
  
def animate_plot_3d(vec1, socket, trajectory):
  
  # Create a 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  vec_use=socket
  ax.scatter(vec_use[:, 0], vec_use[:, 1], vec_use[:, 2])
  ax.scatter(vec1[:, 0], vec1[:, 1], vec1[:, 2])
  ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
  
  # Set labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  # Show the plot
  ax.view_init(elev=3, azim=90)
  ax.set_aspect("equal")
  ev=np.arange(0, 90)
  
  def update(frame, it):    
      # Rotate the plot by changing the viewing angle
      ax.view_init(elev=ev[it[0]], azim=frame)
      it[0] += 1
  
  # Create an animation
  it=[0]
  ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), fargs=[it], interval=50)
  
  # Show the animation
  ani.save('rotation_animation.mp4', writer='ffmpeg', dpi=100)
  plt.show()
  
def SE3_pose_trajectory_matplotlib_plot():

  #P is [steps, 7]
  #So save it as each pose in a txt then read it in
  P = torch.load("/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/trajectory_pos_noise.pt", map_location=torch.device('cpu'))
  ax = plot_trajectory(
    P=P.numpy(), s=0.3, n_frames=100, normalize_quaternions=False, lw=2, c="k")
  plt.savefig("se3_trajectory_3d.png")


    
def plot_att_function(socket, trajectory, use_x=False):

    #State is plug_pos, plug_quat, socket_pos, socket_quat, fix everything except the plug_pos's y and z
    y=0.0
    z=1.065
    state_template = torch.tensor([0.13, y, z, 0.0, 0.0, 0.0, 1.0, 1.3000e-01, 4.3656e-11, 1.0400e+00, 0.0, 0.0, 0.0, 1.0])

    num_spec = 50
    
    if use_x == False:
      x_starting_point=-0.001#0.125#-0.01 #really is y starting point for the robot
      x_ending_point=0.001#0.135#0.01
    else:
      x_starting_point=0.129#-0.01 #really is y starting point for the robot
      x_ending_point=0.131#0.01
    
    x_points=np.linspace(x_starting_point, x_ending_point, num_spec)
    y_starting_point=1.067 #really is y starting point for the robot
    y_ending_point=1.060
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
    
        resolution=10
    
        peg = create_peg(state_inp.to(device), resolution=resolution) #When I create a peg I am moving the peg location up as that is how our method works. That is how our trajectory is followed as well, at the center
    
        #So trajectory should actually have half the peg size added to it
        action=act(peg.unsqueeze(0), socket.unsqueeze(0), state_inp.unsqueeze(0).to(device), trajectory, 0.001) #[x,y,z,rotx,roty,rotz]

        if use_x == False:
          if (x_points[xx] < 0.0001306 and x_points[xx] > -0.0001306) or y_points[yy] > 1.065:
            u[xx][yy]=action[:,1].numpy()[0]
            v[xx][yy]=action[:,2].numpy()[0]
            maximum=np.abs(u[xx][yy])
    
            if np.abs(v[xx][yy]) > maximum:
              maximum=np.abs(v[xx][yy])
            if maximum == 0.0: maximum=1.0
    
            u[xx][yy] /= maximum# magnitude#divide by the average of the total vector's magnitude, divided by the same value the direction is still saved, the loss would be divided self for the action
            v[xx][yy] /= maximum# magnitude
    
          else:
            u[xx][yy]=0.0
            v[xx][yy]=0.0
    
        else:
          if (x_points[xx] < 0.1301306 and x_points[xx] > 0.1298694) or y_points[yy] > 1.065:
            u[xx][yy]=action[:,0].numpy()[0]
            v[xx][yy]=action[:,2].numpy()[0]
            maximum=np.abs(u[xx][yy])
            if np.abs(v[xx][yy]) > maximum:
              maximum=np.abs(v[xx][yy])
            if maximum == 0.0: maximum=1.0
            #print(u[xx][yy], "divided by", np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2), u[xx][yy] / np.sqrt((np.abs(u[xx][yy])**2 + np.abs(v[xx][yy]))**2))
            u[xx][yy] /= maximum #magnitude #divide by the average of the total vector's magnitude, divided by the same value the direction is still saved, the loss would be divided self for the action
            v[xx][yy] /= maximum
          else:
            u[xx][yy]=0.0
            v[xx][yy]=0.0
        
    plt.quiver(x, y, u, v, scale=40.0)#0.0025)
    #plt.gca().set_aspect('equal', adjustable='box')
    if use_x == False:
      plt.savefig("repulsor_vector_field_plot_y.png")
    else:
      plt.savefig("repulsor_vector_field_plot_x.png")
    plt.clf()


def test():
  quat_for_jacobian = torch.tensor([0.0, 0.09983341664682815, 0.0, 0.9950041652780257])
  distance_to_reference_point = torch.tensor([0.0, 0.0, 2.0])
  force_vector = torch.tensor([1.0, 0.0, 0.0])
  
  #Parallel
  jac3 = jacobian_calculation_parallel(quat_for_jacobian.repeat(2, 1), distance_to_reference_point.repeat(2, 1))
  print(jac3)
  cspace_actions = calculate_cspace_forces_parallel(jac3, force_vector.repeat(2, 1))
  #print("parallel function cspace functions", cspace_actions)

def toy_version():
  
  # Create straight up pegs
  peg=create_peg().to(device)
  socket=create_socket()
  peg_parallel=peg.unsqueeze(0).repeat(2, 1, 1)
  socket_parallel=socket.unsqueeze(0).repeat(2, 1, 1)
  
  peg_pos = torch.tensor([[0.13, 0.0, 1.061]]).repeat(2, 1)
  peg_parallel_new=update_vertices_parallel(peg_parallel, torch.tensor([[0.0, 0.0499792, 0.0, 0.9987503]]).repeat(2, 1), peg_pos)
  
  socket_parallel_new=update_vertices_parallel(socket_parallel, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1), torch.tensor([[0.13, 0.0, 1.04]]).repeat(2, 1))
  trajectory=define_trajectory_parallel(torch.tensor([[0.13, 0.0, 1.061]]).repeat(2, 1), resolution=3).to(device)
  
  #print(trajectory.shape)
  plot_3d_points(peg_parallel_new[1], socket_parallel_new[1], trajectory[1])
  plot_3d_points(peg_parallel_new[1], socket_parallel_new[1], trajectory[1], elev=0, azim=0)
  plot_3d_points(peg_parallel_new[1], socket_parallel_new[1], trajectory[1], elev=0, azim=90)
  plot_3d_points(peg_parallel_new[1], socket_parallel_new[1], trajectory[1], elev=20, azim=45)

  act(peg_parallel_new, socket_parallel_new, torch.tensor([[0.13, 0.0, 1.061, 0.0, 0.0499792, 0.0, 0.9987503]]).repeat(2, 1), trajectory, 1.0)
  
def main():
  
  #test()
  toy_version()
  return
  
  vec1=create_peg(torch.tensor([0.13, 0.0, 1.078, 0.0, 0.09983341664682815, 0.0, 0.9950041652780257]).to(device))
  #vec1=create_peg(torch.tensor([0.13, 0.0, 1.078, 0.0, 0.0, 0.0, 1.0]).to(device))
  
  socket=create_socket(torch.tensor([0.13, 0.0, 1.04, 0.0, 0.09983341664682815, 0.0, 0.9950041652780257]))
  #socket=create_socket(torch.tensor([0.13, 0.0, 1.04, 0.0, 0.0, 0.0, 1.0]))
  print(socket.shape)
  
  trajectory=define_trajectory(resolution=200).to(device)
  
  #needs rotation towards trajectory still
  wa=vec1.unsqueeze(0).repeat(2, 1, 1)
  print("hello", wa.shape)
  
  #print(torch.tensor([[0.13, 0.0, 1.065, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1).shape)
  val=act(vec1.unsqueeze(0).repeat(2, 1, 1), socket.unsqueeze(0).repeat(2, 1, 1), torch.tensor([[0.13, 0.0, 1.065, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1).to(device), trajectory, 0.3)
  print(val)
  
  #SE3_pose_trajectory_matplotlib_plot()
  #plot_att_function(socket, trajectory, use_x=False)
  plot_3d_points(vec1, socket, trajectory)
  plot_3d_points(vec1, socket, trajectory, elev=0, azim=0)
  plot_3d_points(vec1, socket, trajectory, elev=0, azim=90)
  plot_3d_points(vec1, socket, trajectory, elev=20, azim=45)
  
if __name__ == "__main__":
  main()
