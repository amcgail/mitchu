from collections import defaultdict, Counter

import numpy as np
from numpy.random import exponential
from numpy.linalg import inv
import os
import names
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import networkx as nx

class Metric:
    """
    A metric is a collection of functions which defines some way of looking at the simulation's development over time.
    Two functions should be defined by the analyst: a `measure` and a `show` function.
    The measure function returns some property of the current state of the simulation (as defined by `self.context`).
    The `show` function uses `self.snaps` to display the data gathered by this metric.
    """
    def __init__(self):
        # This context is defined when the metric is instantiated and added to a simulation
        self.context = None 

        # This keeps track of specific values of this metric through the course of the simulation
        self.snaps = {}

class Meeting:
    def __init__(self,ppl=[],resources=0):
        self.ppl = ppl
        self.resources = resources
    

class Person:
    names_already = set()
    
    def __init__(self, **kwargs):
        self.actions = {}
        self.args=kwargs
        n = None
        while 1:
            n = names.get_first_name(gender='female')
            if n not in self.names_already:
                self.names_already.add(n)
                break
        self.name = n

    def addact(self, acts):
        if type(acts) != list:
            return self.addact([acts])
        
        for a in acts:
            if not issubclass(type(a), Action):
                raise Exception("Object of type %s is not an Action" % type(a))
            self.actions[a.__class__.__name__] = a
            a.person = self
            
    def __repr__(self):
        return "[%s]"%self.name
    
    def log(self, message, context):
        context.log( "[%s]: %s" % (self.name, message))
            
            
class Action:
    def __init__(self):
        self.person = None
        self._next_t = None

    def simulate_T(self):
        if self.act_time() <= 0:
            raise Exception("Action time must be more than zero")
        if self.act_time() == float('inf'):
            return float('inf')
        return exponential( self.act_time() )
        
    @property
    def next_t(self):
        if self._next_t is None:
            self._next_t = self.simulate_T()
            
        return self._next_t
    
    def ct(self):
        self._next_t = None
        
    def log(self, message, context):
        context.log( "[%s][%s]: %s" % (self.person.name, str(self.__class__.__name__), message))
        
        
        




class sim:
    def __init__(self, ppl, debug=False):
        self.ppl = ppl
        self.metrics = {}
        self.CURRENT_WORLD_TIME = None
        self.debug = debug
        
        self.metvals = {} # (name, time) => val
        
    def addmet(self, *mets):
        for met in mets:
            mtoadd = met()
            mtoadd.context = self
            self.metrics[met.__name__] = mtoadd
    
    def run(self, MAX_T=100, INC_T=0.5):
        
        for base_t in np.arange(0, MAX_T, INC_T):
            
            #print("t=", base_t)
        
            self.CURRENT_WORLD_TIME = base_t
            # now take a look at where we are
            for mname, met in self.metrics.items():
                met.snaps[ self.CURRENT_WORLD_TIME ] = met.measure()
            
            # recompute all next times to actions.
            
            # CLEAR
            [a.ct() for p in self.ppl for a in p.actions.values()]
            
            # RECOMPUTE
            itinerary = [(a.next_t, p, a) for p in self.ppl for a in p.actions.values()]
            itinerary = [(t,p,a) for (t,p,a) in itinerary if t < INC_T]
            #itinerary = [(t,p,a) for (t,p,a) in itinerary]
            #itinerary = [(t,p,a) for (t,p,a) in itinerary]
            
            #print(itinerary)
            
            while len(itinerary):
                itinerary = sorted(itinerary, key=lambda x:-x[0])
                
                (t,p,a) = itinerary.pop()
                
                # they do it as many times as they want in that time period,
                # they recompute their propensities to act.
                
                self.CURRENT_WORLD_TIME = t+base_t
                
                a.act(self)
                
                current_t = t
                a.ct() #clear time
                if self.CURRENT_WORLD_TIME + a.next_t < base_t + INC_T:
                    itinerary.append( (self.CURRENT_WORLD_TIME + a.next_t - base_t, p, a) )
                    
                    
    def log(self, message):
        if self.debug:
            print("(%s) %s" % (self.CURRENT_WORLD_TIME, message))
        pass
    
    
    