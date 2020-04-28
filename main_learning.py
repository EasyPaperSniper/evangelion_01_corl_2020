import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import hydra

from daisy_toolkit.daisy_API import daisy_API
import daisy_hardware.motion_library as motion_library



def main():
    # define environment 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=cfg.sim, render=False, logger=False)
    env.set_control_mode(cfg.control_mode)
    state = env.reset()
    
    if cfg.sim:
        init_state = motion_library.exp_standing(env)

    # define data_buffer

    # define agent & learning... 

    # training
    train(env, )




if __name__ == "__main__":
    main()