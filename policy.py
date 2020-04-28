import math

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

import utils

class NN_actor(nn.Module):

    def __init__(self, NN_obs_dim, NN_output_dim, NN_hidden_num, NN_layer_num, device):
        '''
        Initialize the structure and options for model
        '''
        super().__init__()
        self.device = device
        modules = []
        modules.append(nn.Linear(NN_obs_dim, NN_hidden_num))
        modules.append(nn.ReLU())
        for i in range(NN_layer_num-1):
            modules.append(nn.Linear(NN_hidden_num, NN_hidden_num))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(NN_hidden_num, NN_output_dim))
        self.trunk = nn.Sequential(*modules)

    def forward(self, NN_obs):
        return self.trunk(NN_obs)
        
    def predict(self, NN_obs):
        torch_obs = torch.FloatTensor(NN_obs).to(self.device)
        torch_obs = NN_obs.unsqueeze(0)
        prediction = self.forward(torch_obs)
        return prediction.cpu().data.numpy().flatten()


class classic_control():
    '''
    # spinal cord driven joint directly with single NN
    '''
    def __init__(self, SC_obs_dim, SC_output_dim, SC_hidden_num, SC_layer_num, device, init_action = None):
        self.spinal_cord_policy = NN_actor(SC_obs_dim, SC_output_dim, SC_hidden_num, SC_layer_num, device)
        if not init_action:
            self.init_action = np.zeros(18)
        else:
            self.init_action = init_action
    
    def get_action(self, state):
        SC_state = utils.gen_SC_state(state)
        return self.spinal_cord_policy.predict(SC_state) + self.init_action

    def save_policy(self, save_dir):
        pass

    def load_policy(self, save_dir):
        pass


class modular_control(classic_control):
    '''
    spinal cord driven 6 legs seperately with every shares same NN
    '''
    def __init__(self, SC_obs_dim, SC_output_dim, SC_hidden_num, SC_layer_num, 
                    LEG_obs_dim, LEG_output_dim, LEG_hidden_num, LEG_layer_num, device, init_action=None):
        super().__init__(SC_obs_dim, SC_output_dim, SC_hidden_num, SC_layer_num, device, init_action)
        self.leg_policy = NN_actor(LEG_obs_dim, LEG_output_dim, LEG_hidden_num, LEG_layer_num, device)
        
    
    def get_action(self, state):
        SC_state = utils.gen_SC_state(state)
        SC_action = self.spinal_cord_policy.predict(SC_state)
        
        LEG_state = utils.gen_LEG_state(state, SC_action)
        LEG_action = self.leg_policy(LEG_state ) # TODO: vector output 

        return LEG_action.reshape(18)+self.init_action, SC_state, SC_action, LEG_state, LEG_action



# class modular_control_tripod(modular_control):
#     '''
#     # spinal cord driven 6 legs seperately with every shares same NN while forcing a tripod gait
#     '''
#     def __init__():
#         super().__init__()
    



       
