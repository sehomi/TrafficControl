# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:53:26 2022

@author: Paytakht
"""

from gym.spaces import Box, MultiDiscrete, Discrete
from gym import Env
from gym import spaces
from sumolib import checkBinary
import gym
import os
import sys
import numpy as np


#==============================================================================
    
class SingleLightEnv(Env):
    
    def __init__(self, sumo_handler, light, link, step_duration=20):
        super(SingleLightEnv, self).__init__()
        
        self.sumo_handler = sumo_handler
    
        ## We have two actions for each traffic light. It can be green (integer
        ## value of 1) or red (value 0).
        self.action_space = spaces.Discrete(2)    

        ## Here the queue length of each lane is considered as a state.
        count = self.sumo_handler.getNumLanes()
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,count), dtype=np.uint8)
        
        self.step_duration = step_duration
        self.light = light
        self.link = link
        
            
    def zero_state(self):
        
        return np.zeros(( self.sumo_handler.getNumLanes() ), dtype=np.uint8)
        
    #-------------------------------------------------------------------------     
    def reset(self):
        
        self.sumo_handler.reset()
        
        return self.zero_state()
    
    #-------------------------------------------------------------------------
    def render(self):
        pass    
    
    def reward(self, obs):
        
        vehicles = self.sumo_handler.getNumVehicles()
        halting_vehicles = np.sum(obs)
        
        if vehicles > 0:
            reward = 1 - float(halting_vehicles) / vehicles
        else:
            reward = 0
            

        return reward, reward < 0.1 and vehicles > 50

        
    #-------------------------------------------------------------------------
    def step(self, action):
        
        for i in range(self.step_duration):
            self.sumo_handler.setTrafficLight(self.light, self.link, action)
        
        self.sumo_handler.step()
        
        obs = self.sumo_handler.getAllLanesQueueLengths()
        reward, done = self.reward(obs)

        return obs, reward, done, {}        
        
        
    
