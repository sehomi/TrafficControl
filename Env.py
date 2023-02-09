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
    
class AllLightsEnv(Env):
    
    def __init__(self, sumo_handler, step_duration=20):
        super(AllLightsEnv, self).__init__()
        
        self.sumo_handler = sumo_handler
    
        self.sumo_handler.getLightsPhases()
        
        ## We have two actions for each traffic light. It can be green (integer
        ## value of 1) or red (value 0). 
        num_actions = len( self.sumo_handler.phases_counts.keys() )
        self.action_space = spaces.Box(low=0, high=self.sumo_handler.phases_counts.values()-1, shape=(num_actions,1), dtype=np.uint8)  

        ## Here the queue length of each lane is considered as a state.
        count = self.sumo_handler.getNumLanes()
        self.observation_space = spaces.Box(low=0, high=255, shape=(count,1), dtype=np.uint8)
        
        self.step_duration = step_duration
        
            
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
        self.halting_vehicles = np.sum(obs)
        
        self.avg_time = self.sumo_handler.getAverageStopTime()
        
        if vehicles > 0:
            # reward = 10*(1 - float(self.halting_vehicles) / vehicles) \
            #          - self.avg_time + self.sumo_handler.time
            # reward = - self.avg_time + self.sumo_handler.time
            reward = self.sumo_handler.getFreedAndHaltedVehicles()
        else:
            reward = 0
            

        return reward, float(self.halting_vehicles) / vehicles > 0.9 and vehicles > 50

        
    #-------------------------------------------------------------------------
    def step(self, action):
        
        self.setLightsPhases(action[0], self.step_duration)
        for i in range(self.step_duration):
            self.sumo_handler.step()
        
        obs = self.sumo_handler.getAllLanesQueueLengths()
        reward, done = self.reward(obs)

        return obs, reward, done, {}        
        
        
    
