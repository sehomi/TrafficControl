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
import traci
import traci.constants as tc
import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print("SUMO_HOME was found. {}".format(os.environ['SUMO_HOME']))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

#==============================================================================
    
class CustomEnv(Env):
    
    def __init__(self):
        # super().__init__(self, CustomEnv)
        # super(CustomEnv, self).__init__()
        self.sumoBinary = checkBinary('sumo-gui')  # adding -gui  opens sumo-gui
        self.sumoCmd = [self.sumoBinary, "-c", "C:/Users/a/Desktop/TrafficControl/sumo/LnCanada2.sumocfg"]
        self.action_space = spaces.Discrete(2)
        # Set the action space to 2 discrete values i.e green, red
        # Action 1: green
        # Action 2: red
        self.observation_space = spaces.Box(low=0, high=1000,
                                            shape=(), dtype=np.uint8)

    def initializeDimensions(self):
        
        self.actionSpaceDims = len(self.trafficLights)
        self.stateSpaceDims = len(self.trafficLights) + 1
        
        
    def initializeVariables(self):
        
        self.trafficLights = traci.trafficlight.getIDList()
        self.time = 0
        self.initializeDimensions()
        
                
    def getTrafficLightsQueuesLenghts(self):
    
        queues_lens = []
        for tl in self.trafficLights:
            
            lanes = traci.trafficlight.getControlledLanes(tl)
            
            tl_halting_vehicles = 0
            
            for lane in lanes:
                haltingVehicles = traci.lane.getLastStepHaltingNumber(lane)
                tl_halting_vehicles += haltingVehicles
            
            queues_lens.append(tl_halting_vehicles)
            
        return queues_lens  
    

    def setTrafficLightStates(self, actions):
        
        assert len(actions) == self.actionSpaceDims
        
        tlstates = ['gggggggggggggggg' if a else 'rrrrrrrrrrrrrrrr' for a in actions]
        
        for i, tl in enumerate(self.trafficLights):
            traci.trafficlight.setRedYellowGreenState(tl, tlstates[i])
            
    #-------------------------------------------------------------------------       
    def reset(self):
        self.time = 0
        traci.start(self.sumoCmd)
        self.initializeVariables()
        
        print("Environment Reset.")
    
    #-------------------------------------------------------------------------
    def render(self):
        pass    
    
    #-------------------------------------------------------------------------
    def step(self):

        # print("Step. time: {:d}".format(self.time))            

        traci.simulationStep()
        self.time += 1
    
        self.setTrafficLightStates([1 for i in range(self.actionSpaceDims)])
        ql = self.getTrafficLightsQueuesLenghts()
        print(ql)
        
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
