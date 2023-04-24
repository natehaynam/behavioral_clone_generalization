import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

GOAL = 0


def collision_with_boundaries(player):
    if player[0] > 500 or player[0] < 0 or player[1] > 500 or player[1] < 0:
        return 1
    return 0


class simpleEnv(gym.Env):
    def __init__(self):
        super(simpleEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(4 + GOAL,), dtype=np.float32)
        self.DELTA = 0

    def step(self, action):
        self.num_steps += 1
        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.player[0] += 50
        elif button_direction == 3:
            self.player[0] -= 50
        elif button_direction == 0:
            self.player[1] += 50
        elif button_direction == 2:
            self.player[1] -= 50

        # Reward for mining Gem
        if self.player[0] == self.gem_position[0] and self.player[1] == self.gem_position[1]:
            self.score += 1
            self.total_gems += 1
            self.done = True

        # create observation:
        observation = [self.player[0], self.player[1], self.gem_position[0], self.gem_position[1]]
        observation = np.array(observation)

        return observation, 0, self.done, {}

    def render(self):
        cv2.imshow('Single_Agent_PPO', self.img)
        cv2.waitKey(15)
        # Display Grid
        self.img = np.zeros((500, 500, 3), dtype='uint8')

        # Display Gem and Player
        if (self.gem_position == self.player):
            cv2.rectangle(self.img, (self.gem_position[0], self.gem_position[1]),
                      (self.gem_position[0] + 50, self.gem_position[1] + 50), (255, 0, 255), -1)
        else:
            cv2.rectangle(self.img, (self.gem_position[0], self.gem_position[1]),
                        (self.gem_position[0] + 50, self.gem_position[1] + 50), (0, 0, 255), -1)
            cv2.rectangle(self.img, (self.player[0], self.player[1]), 
                        (self.player[0] + 50, self.player[1] + 50), (255, 0, 0), -1)
        t_end = time.time() + 0.10
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

    def reset(self):
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        self.total_gems = 0
        self.num_steps = 0
        # Initial Player and Gem position
        self.player = [random.randrange(1, 10) * 50, random.randrange(1, 10) * 50]
        self.gem_position = [random.randrange(4 - self.DELTA, 4 + self.DELTA + 1) * 50, random.randrange(4 - self.DELTA, 4 + self.DELTA + 1) * 50]
        self.score = 0

        self.done = False

        # create observation:
        observation = [self.player[0], self.player[1], self.gem_position[0], self.gem_position[1]]
        observation = np.array(observation)
        return observation

    def _get_obs(self):
        return np.array([self.player[0], self.player[1], self.gem_position[0], self.gem_position[1]])

    def _get_info(self):
        return {}