import os

import numpy as np
import datetime
import time

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import DracoEnv

if __name__ == "__main__":
    env = DracoEnv(render=False)

    ## train model
    model = PPO("MlpPolicy", env, verbose=1) #policy_kwargs=dict(net_arch=[64,64, dict(vf=[], pi=[])]), 
    startTime = time.time()
    model.learn(total_timesteps=1000)
    endTime = time.time()
    print("Model train time: "+str(datetime.timedelta(seconds=endTime-startTime)))

    ## save the model
    save_path = '{}/{}'.format("rl_model", "fist_try")
    model.save(save_path)