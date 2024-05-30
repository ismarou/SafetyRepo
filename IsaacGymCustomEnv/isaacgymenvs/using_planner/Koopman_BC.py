from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm_
import scipy
import sys
sys.path.append('..')
sys.path.append('../..')
from utilities import fill_buffer, quat_to_angle_axis, pose_world_to_robot_base, quat_mul, quat_from_angle_axis, quat_diff_rad
#import matplotlib.pyplot as plt
#from IPython.display import clear_output

#from IL_RRT_Star import *

buffer_size = 1000000
state_dim = 7 + 7  # Fixed state dimension
action_dim = 7  # Fixed action dimension
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#state is just 14 values, the decision of which to interact and to bring to a power and such is important
class DraftedObservable(object):
    def __init__(self, state_dim):
        self.state_dim = state_dim
    
    def z(self, state):  
        """
        Inputs: hand states(pos, vel) & object states(pos, vel)
        Outputs: state in lifted space
        Note: velocity is optional
        """
        """
        obs = np.zeros(self.state_dim)
        index = 0
        #normal
        for i in range(self.state_dim):
            obs[index] = state[i]
            index += 1
        #power 2
        for i in range(self.state_dim):
            obs[index] = state[i] ** 2
            index += 1   
        #power 3
        for i in range(self.state_dim):
            obs[index] = state[i] ** 3
            index += 1
        #multiply only forward
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
              obs[index] = state[i] * state[j]
              index += 1
        #multiply forward with power 2
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
              obs[index] = state[i] ** 2 * state[j]
              index += 1  
        #multiply forward with power 3
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
              obs[index] = state[i] ** 3 * state[j]
              index += 1  
        print(obs)
        return obs
        """
        obs = []
        index = 0
        #normal
        for i in range(self.state_dim):
            obs.append(state[i])
            index += 1
        #power 2
        for i in range(self.state_dim):
            obs.append(state[i] ** 2)
            index += 1   
        #power 3
        for i in range(self.state_dim):
            obs.append(state[i] ** 3)
            index += 1
        #multiply only forward
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
              obs.append(state[i] * state[j])
              index += 1
        #multiply forward with power 2
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
              obs.append(state[i] ** 2 * state[j])
              index += 1  
        #multiply forward with power 3
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
              obs.append(state[i] ** 3 * state[j])
              index += 1  
        #print(obs)
        return np.asarray(obs)
        #They actually have two options for lifting. One is to use a neural network encoder and decoder. One is to use this and just take the slots of where the results should be in the output.
        """
        Shown here:
                    z_t = koopman_object.z(input_state).numpy()  # lift the initial states and then roll it out using Koopman operator
        for t in range(len(Test_data[k]) - 1):  # this loop is for system evolution, open loop control, no feedback
            z_t_1_computed = np.dot(koopman_matrix, z_t)
            if koopmanoption == 'Drafted':  
                x_t_1_computed = np.append(z_t_1_computed[:num_hand], z_t_1_computed[2 * num_hand: 2 * num_hand + num_obj])
                if not Velocity:
                    obj_pos_world = x_t_1_computed[28:31]  # handle pos
        """

def fix_state(state):
    #state, action, next_state, reward, done = batch
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

#@pyrallis.wrap()
def train(): #config: TrainConfig

    #read_and_process_data_RRT_Star(max_action, device)

    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("RB size:", replay_buffer.size(), replay_buffer._states.shape[0])

    #Need to import the koopman operator class and thus provide the proper lifting formulation
    #Need to have the proper usage of the state, as they have some unique things they do with the state. I am not sure how lifting works exactly so this is important
    observable=DraftedObservable(state_dim)
    num_obs=observable.z(replay_buffer._states[0]).shape[0]
    A = np.zeros((num_obs, num_obs))  
    G = np.zeros((num_obs, num_obs))
    print("num obs", num_obs)
    datapoints=replay_buffer.size()
    datapoints=20
    replay_buffer._states = fix_state(replay_buffer._states)
    replay_buffer._next_states = fix_state(replay_buffer._next_states)
    for i in range(datapoints): #num values replay_buffer._states.shape[0]/2
      if i % 1000 == 0: print(i)
      #make observable from state
      #make observable from next state
      #Is it state to state prediction? Or is it state+action prediction? I forget
      #It is state to next state from the training data
      #Then just have the target next state and I can make an eval of going to that state
      #So once we lift the values to nonlinear relations together, we can set A and G as the data, then we can solve the optimization problem with pinv.
      #Even though its linear, the lifting and dim should be enough
      state = observable.z(replay_buffer._states[i])
      next_state = observable.z(replay_buffer._next_states[i])
      A += np.outer(next_state, state)
      G += np.outer(state, state)

    M = datapoints #this just averages it, in the repo they have two loops where one is nested so they need to multiply the number of loops together
    A /= M
    G /= M
    koopman_operator = np.dot(A, scipy.linalg.pinv(G)) # do not use np.linalg.pinv, it may include all large singular values
    
    return koopman_operator
    
def test_koopman_next_state(koopman_operator):
    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("Testing, RB size:", replay_buffer.size())

    diff = 0.0
    observable=DraftedObservable(state_dim)
    for i in range(replay_buffer.size()):
      #The koopman operator is K = AGpinv where KG = A, where if G and A are column matrices then each column of G goes to each column of A, with inner product it is just same things mapped but with linear
      output_next_state = np.dot(koopman_operator, observable.z(replay_buffer._states[i]))
      temp_diff = np.average(np.absolute(torch.from_numpy(output_next_state[0:14]) - replay_buffer._next_states[i]))
      print("temp diff", temp_diff)
      diff += temp_diff
    print("Average diff", diff / float(replay_buffer.size()))
    
def test_koopman(koopman_operator):
    file_path = '/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/using_planner/interpolated_RRT_Star_4.txt'
    replay_buffer = fill_buffer()
    print("Testing, RB size:", replay_buffer.size())

    diff = 0.0
    observable=DraftedObservable(state_dim)
    for i in range(replay_buffer.size()):
      #The koopman operator is K = AGpinv where KG = A, where if G and A are column matrices then each column of G goes to each column of A, with inner product it is just same things mapped but with linear
      output_next_state = np.dot(koopman_operator, observable.z(replay_buffer._states[i]))
      temp_diff = np.average(np.absolute(output_next_state - observable.z(replay_buffer._next_states[i])))
      print("temp diff", temp_diff)
      diff += temp_diff
    print("Average diff", diff / float(replay_buffer.size()))

if __name__ == "__main__":
  koopman=train()
  np.save(
    os.path.join("./checkpoints/", f"koopman.npy"),
    koopman
  )
  test_koopman_next_state(koopman)
  #test_koopman(koopman)