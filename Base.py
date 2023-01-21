# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:53:26 2022

@author: Paytakht
"""

from sumolib import checkBinary
import traci
import traci.constants as tc
import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


#==============================================================================
    
class SUMOHandler:
    
    def __init__(self):
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
            print("SUMO_HOME was found. {}".format(os.environ['SUMO_HOME']))
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
            
        self.sumoBinary = checkBinary('sumo-gui')
        self.sumoCmd = [self.sumoBinary, "-c", "C:/Users/a/Desktop/TrafficControl/sumo/LnCanada2.sumocfg",\
                        "--start", "--quit-on-end"]

        self.connected = False
        
        self.reset()
        
    def initializeVariables(self):
        
        trafficLights = traci.trafficlight.getIDList()
        self.lane_light_mapping = self.getLaneLightMapping(trafficLights)
        self.time = 0
        
                
    def getNumLanes(self):
        
        self.lanes = traci.lane.getIDList()
            
        return len(self.lanes)
    
        
    def getNumLights(self):
        
        tl_count = 0
        trafficLights = traci.trafficlight.getIDList()
        self.trafficLights = []
        for light in trafficLights:
            links = traci.trafficlight.getControlledLinks(light)
            
            for j, link in enumerate(links):
                if len(link) > 0:
                    
                    self.trafficLights.append({'traffic_light':light, 'link_index':j})
                    tl_count += 1
            
        
        return tl_count
    
    def getAverageStopTime(self):
        
        ids = traci.vehicle.getIDList()
        dt = 0
        for id in traci.vehicle.getIDList():
            dt += traci.vehicle.getWaitingTime(id)
        dt = dt/len(ids)
        
        return dt
    
    def getFreedAndHaltedVehicles(self):
        
        score = 0
        stopped_vehicles = []
        moving_vehicles = []
        for id in traci.vehicle.getIDList():
            if traci.vehicle.getWaitingTime(id) > 1:
                stopped_vehicles.append(id)
            else:
                moving_vehicles.append(id)
                
        if not self.last_stopped is None and not self.last_moving is None:
            for sv in stopped_vehicles:
                if sv in self.last_moving:
                    traci.vehicle.highlight(sv, color=(255, 0, 0, 255), size=-1, alphaMax=255, duration=20, type=0)
                    score -= 1
                    
            for mv in moving_vehicles:
                if mv in self.last_stopped:
                    traci.vehicle.highlight(mv, color=(0, 255, 0, 255), size=-1, alphaMax=255, duration=20, type=0)
                    score += 1
                    
        self.last_stopped = stopped_vehicles.copy()
        self.last_moving = moving_vehicles.copy()
        
        del stopped_vehicles
        del moving_vehicles
        
        return score
    
    def getNumVehicles(self):
        
        return len(traci.vehicle.getIDList())
    
    def getLaneQueueLength(self, lane):
            
        return traci.lane.getLastStepHaltingNumber(lane) 

    def getAllLanesQueueLengths(self):
          
        lens = []
        for lane in self.lanes:
            lens.append( traci.lane.getLastStepHaltingNumber(lane) )
        
        lens = np.clip( np.array(lens), 0, 255 ).astype(np.uint8)
        
        return lens
    

    def getLaneLightMapping(self, trafficLights):
        
        lane_light_mapping = {}
        
        for i, tl in enumerate(trafficLights):
            
            links = traci.trafficlight.getControlledLinks(tl)
            
            for j, link in enumerate(links):
                if len(link) > 0:
                    lane_incoming = link[0][0]
                    lane_outgoing = link[0][1]
                    lane_via = link[0][2]
                    
                    if lane_incoming not in lane_light_mapping.keys():
                        lane_light_mapping[lane_incoming] = [{'traffic_light':tl, 'link_index':j}]
                    else:
                        lane_light_mapping[lane_incoming].append({'traffic_light':tl, 'link_index':j})
        
        return lane_light_mapping
    
    
    def setTrafficLightOfLane(self, lane, state):
        
        vals = self.lane_light_mapping[lane]
        
        for val in vals:
            tl = val['traffic_light']
            index = val['link_index']
            print('setting traffic_light {}, index {:d}'.format(tl, index))
            traci.trafficlight.setLinkState(tl, index, state)
            
    def setTrafficLight(self, light, link, state_bool):
        
        state = 'r'
        if state_bool:
            state = 'g'
        
        traci.trafficlight.setLinkState(light, link, state)
        
        
    def setAllTrafficLights(self, states_bool):
        
        tl_count = 0
        trafficLights = traci.trafficlight.getIDList()
        for light in trafficLights:
            links = traci.trafficlight.getControlledLinks(light)
            for j, link in enumerate(links):
                if len(link) > 0:
                    
                    state = 'r'
                    if states_bool[tl_count]:
                        state = 'g'
                    
                    traci.trafficlight.setLinkState(light, j, state)
                    
                    tl_count += 1
                    
            
                
            
    #-------------------------------------------------------------------------       
    def reset(self):
        self.time = 0
        
        if self.connected:
            traci.close()
            
        traci.start(self.sumoCmd)
        self.connected = True
        
        # self.initializeVariables()
        
        self.last_stopped = None
        self.last_moving = None
        
        print("Environment Reset.")
    
    #-------------------------------------------------------------------------
    def render(self):
        pass    
    
    #-------------------------------------------------------------------------
    def step(self):

        # print("Step. time: {:d}".format(self.time))            

        traci.simulationStep()
        self.time += 1
    
    
        
        
#=============================================================================

if __name__=="__main__":
    print("Creating Environment.")
    
    env = SUMOHandler()
    env.reset()
    done = False
    while not done :
        done = env.step()
    
    print('Simulation Finished.')    
