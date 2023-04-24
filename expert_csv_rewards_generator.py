import numpy as np
import pandas as pd


RENDER_SCALAR = 50
WORLD_WIDTH = 10 * RENDER_SCALAR
WORLD_HEIGHT = 10 * RENDER_SCALAR
GEM_X = 4 * RENDER_SCALAR
GEM_Y = 4 * RENDER_SCALAR


storage_array = np.zeros(shape = (10 * 10 * 4, 7))
columns = ["obs.no", "agent_x", "agent_y", "gem_x", "gem_y", "rewards", "action"]
curr_index = 0
for x in range(0, WORLD_WIDTH, RENDER_SCALAR):
    for y in range(0, WORLD_HEIGHT, RENDER_SCALAR):
        for d in range(4):
            reward = -1
            if (x < GEM_X) & (d == 1):
                reward = 1
            elif (x > GEM_X) & (d == 3):
                reward = 1
            elif (x == GEM_X) & (y < GEM_Y) & (d == 0):
                reward = 1
            elif (x == GEM_X) & (y > GEM_Y) & (d == 2):
                reward = 1
            storage_array[curr_index] = [curr_index + 1, x, y, GEM_X, GEM_Y, reward, d]
            curr_index += 1

new_expert_table = pd.DataFrame(columns = columns, data = storage_array.astype(int))
filename = f"expert_policy({WORLD_WIDTH // RENDER_SCALAR}x{WORLD_HEIGHT // RENDER_SCALAR}).csv"

new_expert_table.to_csv(filename, index = False)