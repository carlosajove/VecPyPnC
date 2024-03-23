import os
import sys

import numpy as np
import datetime
import time
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import DracoEnv
cwd = os.getcwd()
sys.path.append(cwd)

model_dir = cwd + "/rl_model/PPO"



if __name__ == "__main__":
    env = DracoEnv(render=False)
    new_model = False #TODO: make funciton
    
    ## train model
    if new_model:
        model = PPO("MlpPolicy", env, verbose=1, n_steps = 256, batch_size=64, tensorboard_log="/home/carlos/Desktop/Austin/SeungHyeonProject/PyPnc_pytorch/ppo_rl_log/") #policy_kwargs=dict(net_arch=[64,64, dict(vf=[], pi=[])]), 
        startTime = time.time()
        iters = 0
        TIMESTEPS = 768
    else:
        model_path = f"{model_dir}/5376"
        model = PPO.load(model_path, env=env)
        TIMESTEPS = 768
        iters = 7


    while(True):
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, tb_log_name="second", progress_bar=True, reset_num_timesteps=False)
        endTime = time.time()
        print("Model train time: "+str(datetime.timedelta(seconds=endTime-startTime)))
        ## save the model
        save_path = '{}/{}'.format(model_dir, f"/{TIMESTEPS*iters}")
        model.save(save_path)