import matplotlib.pyplot as plt
import random
import numpy as np

plt.ion()

#from controller import Player
from env.controller import Player

class Terrain:
    def __init__(self):
        self.reward_locs = [[5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, -5.0]]
        self.reward_range = 1.0
        self.reward_goal = 10.0
        self.bounds_x = [-8.0, 8.0]
        self.bounds_y = [-8.0, 8.0]
        self.player = Player(0.0, 0.0, self)

    def getreward(self):
        reward = 0.0
        for x_pos, y_pos in self.reward_locs:
            max_score_temp = -(np.sqrt((self.player.x - x_pos) ** 2 + (self.player.y - y_pos) ** 2) / 20)
            reward += max_score_temp

        for x_pos, y_pos in self.reward_locs:
            #reward -= 0.15
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                reward += self.reward_goal
        return reward

    def checkepisodeend(self):
        for x_pos, y_pos in self.reward_locs:
            if abs(self.player.x - x_pos) < self.reward_range and abs(self.player.y - y_pos) < self.reward_range:
                return 1
        return 0

    def plotgame(self):
        plt.clf()
        for x_pos, y_pos in self.reward_locs:
            plt.plot([x_pos,], [y_pos,], marker='o', markersize=10, color="green")
        plt.xlim([-8, 8])
        plt.ylim([-8, 8])
        plt.plot([self.player.x,], [self.player.y,], marker='x', markersize=3, color="red")
        plt.pause(0.001)

    def resetgame(self):
        self.player = Player(0.0, 0.0, self)

"""
terrain = Terrain()
while(1):
    terrain.plotgame()
    print terrain.player.action(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
"""
