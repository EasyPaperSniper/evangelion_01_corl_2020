from collections import deque
import random
import os
import math

import torch
import numpy as np
import torch.nn as nn
import gym
import daisy_kinematics
import daisy_raibert_controller

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def normalization(a, mean, std):
    return (a-mean)/std
    
def inverse_normalization(a, mean, std):
    return a*std + mean
    
class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim,delta_obs_dim, device, capacity,sim= True, save_dir = None):
        self.device = device
        self.capacity = capacity

        if type(obs_dim) == int:
            self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
            self.next_obses = np.empty((capacity, delta_obs_dim), dtype=np.float32)
        else:
            self.obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.all_mean_var = np.array([
            np.zeros(obs_dim),
            np.ones(obs_dim),
            np.zeros(action_dim),
            np.ones(action_dim),
            np.zeros(delta_obs_dim),
            np.ones(delta_obs_dim),
        ])

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save_buffer(self):
        save_dir = make_dir(os.path.join('./buffer_data'))
        np.save(save_dir+'/obses', self.obses)
        np.save(save_dir+'/actions', self.actions)
        np.save(save_dir+'/rewards', self.rewards)
        np.save(save_dir+'/next_obses', self.next_obses)
        np.save(save_dir+'/not_dones', self.not_dones)
        np.save(save_dir+'/idx',np.array([self.idx]))

        # save mean and var
        if self.full:
            self.all_mean_var[0]= np.mean(self.obses, axis = 0)
            self.all_mean_var[1]= np.std(self.obses, axis = 0)
            self.all_mean_var[2]= np.mean(self.actions, axis = 0)
            self.all_mean_var[3]= np.std(self.actions, axis = 0)
            self.all_mean_var[4]= np.mean(self.next_obses, axis = 0)
            self.all_mean_var[5]= np.std(self.next_obses, axis = 0)
        else:
            self.all_mean_var[0]= np.mean(self.obses[0:self.idx], axis = 0)
            self.all_mean_var[1]= np.std(self.obses[0:self.idx], axis = 0)
            self.all_mean_var[2]= np.mean(self.actions[0:self.idx], axis = 0)
            self.all_mean_var[3]= np.std(self.actions[0:self.idx], axis = 0)
            self.all_mean_var[4]= np.mean(self.next_obses[0:self.idx], axis = 0)
            self.all_mean_var[5]= np.std(self.next_obses[0:self.idx], axis = 0)
        np.save(save_dir+'/all_mean_var', self.all_mean_var)

    def load_mean_var(self):
        '''
        0: mean of obses; 1: var of obses; 2: mean of actions; 3: var of actions; 4: mean of next_obses; 5: var of next_obses
        '''
        self.all_mean_var = np.load('./buffer_data/all_mean_var.npy')
        return self.all_mean_var

    def load_buffer(self,):
        save_dir = './buffer_data'
        self.obses = np.load(save_dir+'/obses.npy')
        self.actions = np.load(save_dir+'/actions.npy')
        self.rewards = np.load(save_dir+'/rewards.npy')
        self.next_obses = np.load(save_dir+'/next_obses.npy')
        self.not_dones = np.load(save_dir+'/not_dones.npy')
        self.idx = np.load(save_dir+'/idx.npy')[0]

        return self.load_mean_var()
        


def gen_SC_state(state):
    pass
 
def gen_LEG_state(state, SC_action=None):
    pass




