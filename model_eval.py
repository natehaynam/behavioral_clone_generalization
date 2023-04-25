import sys

import numpy as np

sys.path.append("..")

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
from stable_baselines3 import PPO
from env_simple import simpleEnv
from behavioral_clone import *

acc_dict = {}
with torch.no_grad():
    for model_name in ["model_75mec_9-5-5"]:
        device = "cpu"
        model = BC(input_size=4, output_size=5).to(device)
        model.load_state_dict(torch.load(f"mec_models/{model_name}", map_location=torch.device(device)))
        model.eval()

        for delta in range(0, 4):
            counter_acc = 0
            env = simpleEnv()
            env.DELTA = delta

            for _ in range(100):
                env.reset()
                obs = env.reset()
                # env.render()
                i = 0

                while (not env.done and i < 20):
                    env_obs = torch.Tensor([[obs[0], obs[1], obs[2], obs[3]]]).to(torch.float).to(device)
                    # print("Player: [", obs[0], obs[1], "] --- Gem: [", obs[2], obs[3], "]")
                    action = model(env_obs)

                    action = [np.argmax(x.cpu()).item() for x in action][0]
                    # print(action)
                    obs, reward, done, info = env.step(action)
                    # env.render()
                    i += 1
                print('DONE:', env.done)
                if env.done:
                    counter_acc += 1
            acc = counter_acc / 100
            acc_dict[f"model:{model_name}, DELTA:{delta}"] = acc
            print(acc)

print(acc_dict)

# results:
# DELTA = 1, Model = model_1.0
#   acc = 0.83
# DELTA = 2, Model = model_1.0
#   acc = 0.71
