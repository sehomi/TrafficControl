# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:53:26 2022

@author: Paytakht
"""

# !pip install stable-baselines[mpi]==2.10.2
# !pip install stable-baselines3[extra]

from gym.spaces import Box, MultiDiscrete, Discrete
from gym import Env
from gym import spaces
from sumolib import checkBinary
import gym
import traci
import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

#==============================================================================
    
class CustomEnv(Env):
    
    def __init__(self):
        # super().__init__(self, CustomEnv)
        # super(CustomEnv, self).__init__()
        self.sumoBinary = checkBinary('sumo')  # adding -gui  opens sumo-gui
        self.sumoCmd = [self.sumoBinary, "-c", "/content/TrafficControl/sumo/LnCanada2.sumocfg"]
        self.time = 0
        self.action_space = spaces.Discrete(2)
        # Set the action space to 2 discrete values i.e green, red
        # Action 1: green
        # Action 2: red
        self.observation_space = spaces.Box(low=0, high=1000,
                                            shape=(), dtype=np.uint8)

    #-------------------------------------------------------------------------       
    def reset(self):
        self.time = 0

        print("Environment Reset.")
    
    #-------------------------------------------------------------------------
    def render(self):
        pass    
    
    #-------------------------------------------------------------------------
    def step(self):

        print("Step. time: {:d}".format(self.time))

        # part1
        if self.time == 0 :
            traci.start(self.sumoCmd)

        #part2
        traci.simulationStep()
        self.time += 1
        #part3
        if self.time == 3600 :
            done = True
            traci.close()
        else :
            done  = False
            
        return done
    
        
        
#=============================================================================

print("Creating Environment.")

env = CustomEnv()
env.reset()

done = False
while not done :
    done = env.step()

print('Simulation Finished.')    


# #=============================================================================

# def trainRL():
    
    
    
# #===========================================================================

# def _prf_action(self, action):

    

# #===========================================================================

# def _cal_state(self):
   
 

   
# #===========================================================================

# def _cal_reward(self):
