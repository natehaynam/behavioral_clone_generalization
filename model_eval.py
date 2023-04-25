import sys

import numpy as np

sys.path.append("..")

import pandas as pd
import torch
from stable_baselines3 import PPO
from env import singleEnv
from behavioral_clone import *
import re
from datetime import datetime

device = "cpu"
#model = BC(input_size=4, output_size=5).to(device)
#model.load_state_dict(torch.load("final_models/model_1.0"))
#model.eval()

class BC1(torch.nn.Module):
    def __init__(self, input_size, h1_size, output_size):
        super(BC1, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, h1_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h1_size, output_size),
            torch.nn.Softmax(dim=1)
    )

    def forward(self, x):
        action = self.model(x)
        return action

class BC2(torch.nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(BC2, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, h1_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h1_size, h2_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h2_size, output_size),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        action = self.model(x)
        return action

# acc_dict = {} 
models_to_eval = ["model_20mec_3-5", "model_30mec_4-5", "model_30mec_5-5", "model_36mec_6-5",  "model_42mec_7-5",  "model_48mec_7-5-5", "model_53mec_8-5-5", "model_59mec_9-5-5", "model_63mec_9-9-5", "model_69mec_10-9-5", "model_74mec_11-7-5", "model_77mec_12-5-5"] 
deltas = [1, 2]
num_iterations = 1000
storage_array = np.zeros(shape = (len(models_to_eval), len(deltas)))
curr_model_index = 0
for model_name in models_to_eval:
    first_regex = re.search(r"mec_(.*)", model_name).group(1)
    print(first_regex)
    numbers = first_regex.split("-")
    print(numbers)
    device = "cpu"
    if len(numbers) == 2:
        model = BC1(input_size=4, h1_size=int(numbers[0]), output_size=5).to(device)
    elif len(numbers) == 3:
        model = BC2(input_size=4, h1_size=int(numbers[0]), h2_size=int(numbers[1]), output_size=5).to(device)
    model.load_state_dict(torch.load(f"mec_models/{model_name}", map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        acc_list = []
        for delta in deltas:
            counter_acc = 0
            env = singleEnv()
            for _ in range(num_iterations):
                env.eval_reset(delta)
                obs = env.eval_reset(delta)
                i = 0
                while (not env.done and i < 20):
                    env.render()
                    env_obs = torch.Tensor([[obs[0], obs[1], obs[2], obs[3]]]).to(torch.float).to(device)
                    #print([obs[0], obs[1], obs[2], obs[3]])
                    #print(env.prev_actions)
                    action = model(env_obs)

                    action = [np.argmax(x.cpu()).item() for x in action][0]
                    #print(action)
                    obs, reward, done, info = env.step(action)
                    if (((obs[0] == obs[2] + 50) or (obs[0] == obs[2] - 50)) and ((obs[1] == obs[3] + 50) or (obs[1] == obs[3] - 50))):
                        env.done = True
                        env.reached_gem = True
                    i += 1
                if env.reached_gem:
                    counter_acc += 1
            acc = counter_acc / num_iterations
            acc_list += [acc]
            #acc_dict[f"model:{model_name}, DELTA:{delta}"] = acc
            print(acc)
    storage_array[curr_model_index] = acc_list
    curr_model_index += 1

print(storage_array)
storage_table = pd.DataFrame(columns=[f"DELTA:{delta}" for delta in deltas], index=models_to_eval, data = storage_array)
print(storage_table)
now = datetime.now().strftime("%m_%d_%Y,%H:%M:%S")
print(now)
filename = f"model_eval_csvs/eval_done_at:{now}).csv"

storage_table.to_csv(filename, index = True)

# results:
# DELTA = 1, Model = model_1.0
#   acc = 0.83
# DELTA = 2, Model = model_1.0
#   acc = 0.71
# {'model:model_75mec_9-5-5, DELTA:1': 0.96, 'model:model_75mec_9-5-5, DELTA:2': 0.6}
# {'model:model_70mec_8-5-5, DELTA:1': 0.85, 'model:model_70mec_8-5-5, DELTA:2': 0.56}
