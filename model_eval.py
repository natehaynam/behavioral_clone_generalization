import sys

import numpy as np

sys.path.append("..")

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
from stable_baselines3 import PPO
from Tony_env import singleEnv
from behavioral_clone import *

device = "cpu"
model = BC(input_size=4, output_size=5).to(device)
model.load_state_dict(torch.load("final_models/model_1.0"))
model.eval()

acc_dict = {}
with torch.no_grad():
    for model_name in ["model_70mec_8-5-5", "model_75mec_9-5-5"]:
        model.load_state_dic(torch.load(f"mec_models/{model_name}"))
        for delta in [1, 2, 3]:
            counter_acc = 0
            env = singleEnv(delta)
            for _ in range(100):
                env.reset()
                obs = env.reset()
                i = 0
                while (not env.done and i < 20):
                    env.render()
                    env_obs = torch.Tensor([[obs[0], obs[1], obs[2], obs[3]]]).to(torch.float).to(device)
                    print([obs[0], obs[1], obs[2], obs[3]])
                    print(env.prev_actions)
                    action = model(env_obs)

                    action = [np.argmax(x.cpu()).item() for x in action][0]
                    print(action)
                    obs, reward, done, info = env.step(action)
                    if (((obs[0] == obs[2] + 50) or (obs[0] == obs[2] - 50)) and ((obs[1] == obs[3] + 50) or (obs[1] == obs[3] - 50))):
                        env.done = True
                    i += 1
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
