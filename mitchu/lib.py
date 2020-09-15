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


from abc import ABC, abstractmethod  


class Metric(ABC):
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

    @abstractmethod
    def measure(self):
        return 0
    
    @abstractmethod
    def show(self):
        plt.plot(range(50), np.sin(np.linspace(0,10,50)))

class Meeting:
    """
    
    Args:
        ppl: The people present in this meeting
        resources: I honestly don't know

    Attributes:
        ppl: The people present in this meeting
        resources: I honestly don't know
    """
    def __init__(self,ppl=[],resources=0):
        self.ppl = ppl
        self.resources = resources
    

class Person:
    """
    
    Args:
        **kwargs: The people present in this meeting

    Attributes:
        actions: the actions within this person's repertoire
        args: 
            Holds all other arguments from instantiating the person. 
            Thus we can give a person some attribute by simply instantiating them as Person(color='red')
        name:
            Randomly generated female name for this person
        
    """
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
    
    def log(self, message, simulation):
        """
        Logs a message with this person's name

        Args:
            message: what you want to log
            simulation: the simulation currently running
        """
        simulation.log( "[%s]: %s" % (self.name, message))
                    
class Action(ABC):
    """
    Action is an abstract class, and should be used when constructing custom actions.

    """
    def __init__(self):
        self.person = None
        self._next_t = None

    """
    When subclassing `Action`, you must create a function `act`.
    This uses the current context as well as the actor to make 
    changes in anything in the context, 
        including people, places, memberships, relations, etc.
    """
    @abstractmethod
    def act(self, context):
        return False

    """
    When subclassing `Action`, you must create a function `act_time`.
    This returns what will be the mean of an exponential distribution describing how long, on average, a person in this state would go without acting.
    This parameter will be used to simulate a continuous time Markov process according to the transitions defined in `act`.
    """
    @abstractmethod
    def act_time(self):
        return float('inf')

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
        
    def log(self, message, simulation):
        """
        Logs a message with the person and action name

        Args:
            message: what you want to log
            simulation: the simulation currently running
        """
        simulation.log( "[%s][%s]: %s" % (self.person.name, str(self.__class__.__name__), message))
        
        
        




class sim:
    """
    This class holds all the context information of a running simulation,
        and manages the process of simulating.
    
    Args:
        ppl: A list of actors - that is, people. These actors will be associated with actions, and will drive the evolution of the simulation.
        debug(bool): Do you want diagnostic messages, passed through the log function?
    """
    def __init__(self, ppl, debug=False):
        self.ppl = ppl
        self.metrics = {}
        self.CURRENT_WORLD_TIME = None
        self.debug = debug
        
        self.metvals = {} # (name, time) => val
        
    def addmet(self, *mets):
        """
        Metrics are the recordings made as the simulation progresses which allow us to
        inspect what happened and when.
        This function should be called to add subclasses of `Metric` to be used while running the simulation.

        Parameters:
            *mets: any number of arguments of class `Metric`
        """
        for met in mets:
            mtoadd = met()
            mtoadd.context = self
            self.metrics[met.__name__] = mtoadd
    
    def run(self, MAX_T=100, INC_T=0.5):
        """
        this is the main function to run a simulation.
        The function iterates from 0 to `MAX_T`, with step size `INC_T`.
        In each iteration, 
        
        1. all metrics are measured
        2. all individuals' propensity to action are reevaluated
        3. based on this propensity, the next time to action is sampled from an exponential distribution
        4. these events are ordered, and executed in order

        Parameters:
            MAX_T: the time at which to end the simulation
            INC_T: the increment for updating propensities
        """
        
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
        """
        Logs a message with the time, conditional on `self.debug`

        Args:
            message: what you want to log
        """
        if self.debug:
            print("(%s) %s" % (self.CURRENT_WORLD_TIME, message))
        pass
    
    



"""
exponential(scale=1.0, size=None)

Draw samples from an exponential distribution.

Its probability density function is

.. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
The rate parameter is an alternative, widely used parameterization
of the exponential distribution [3]_.

The exponential distribution is a continuous analogue of the
geometric distribution.  It describes many common situations, such as
the size of raindrops measured over many rainstorms [1]_, or the time
between page requests to Wikipedia [2]_.

Parameters
----------
scale : float or array_like of floats
    The scale parameter, :math:`\\beta = 1/\\lambda`. Must be
    non-negative.
size : int or tuple of ints, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  If size is ``None`` (default),
    a single value is returned if ``scale`` is a scalar.  Otherwise,
    ``np.array(scale).size`` samples are drawn.

Returns
-------
out : ndarray or scalar
    Drawn samples from the parameterized exponential distribution.

References
----------
.. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
        Random Signal Principles", 4th ed, 2001, p. 57.
.. [2] Wikipedia, "Poisson process",
        https://en.wikipedia.org/wiki/Poisson_process
.. [3] Wikipedia, "Exponential distribution",
        https://en.wikipedia.org/wiki/Exponential_distribution
"""