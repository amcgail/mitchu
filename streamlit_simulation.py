import streamlit as st




from collections import defaultdict, Counter

import numpy as np
from numpy.random import exponential
from numpy.linalg import inv
import os
import names
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx

class Metric:
    def __init__(self, snaps = None):
        self.context = None
        if snaps is None:
            self.snaps = {}
        else:
            self.snaps = snaps

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
        while n is None or n in self.names_already:
            n = names.get_first_name(gender='female')
        self.name = n

    def addact(self, acts):
        if type(acts) != list:
            return self.addact([acts])
        
        for a in acts:
            my_new_act = a()
            self.actions[a.__name__] = my_new_act
            my_new_act.person = self
            
    def __repr__(self):
        return "[%s]"%self.name
    
    def log(self, message, context):
        context.log( "[%s]: %s" % (self.name, message))
            
            
class Action:
    def __init__(self):
        self.person = None
        self._next_t = None

    def simulate_T(self):
        return exponential( self.act_frequency() )
        
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
    def __init__(self, ppl, debug=False, progress=None):
        self.ppl = ppl
        self.metrics = {}
        self.CURRENT_WORLD_TIME = None
        self.debug = debug
        self.pbar = progress
        
        self.metvals = {} # (name, time) => val
        
    def addmet(self, met):
        mtoadd = met()
        mtoadd.context = self
        self.metrics[met.__name__] = mtoadd
    
    def run(self, MAX_T=100, INC_T=0.5):
        
        for mname, met in self.metrics.items():
            #print('recording', mname, met.measure())
            #raise
            if hasattr(met, 'measure_once'):
                met.snaps[-1] = met.measure_once()

        for base_t in np.arange(0, MAX_T, INC_T):
            if self.pbar is not None:
                self.pbar.progress(base_t/MAX_T)
            
            #print("t=", base_t)
        
            self.CURRENT_WORLD_TIME = base_t
            # now take a look at where we are
            for mname, met in self.metrics.items():
                #print('recording', mname, met.measure())
                #raise
                met.snaps[ self.CURRENT_WORLD_TIME ] = met.measure()
            
            # recompute all next times to actions.
            
            # CLEAR
            xyz = [a.ct() for p in self.ppl for a in p.actions.values()]
            
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























class degree_distribution(Metric):
    def measure(self):
        return Counter( [ len(p.friends) for p in self.context.ppl ] )
    
    def show(self, tstart=0, tstop=100, deg_range=20):
        from itertools import chain
        #print(self.snaps)
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        #for i in range(10):
        #    print("+=============================")
        #print(len(self.snaps))

        dd = pd.DataFrame(dict({
            str(deg): [self.snaps[ t ][deg] for t in alltimes]
            for deg in range(0,deg_range)
        }, t=alltimes))

        dd.plot.area(x='t')#, legend=None 
        plt.title("Degree distribution over time")
        st.write(plt.gcf())
        
class network(Metric):
    def measure(self):
        return [
            (p1.name, p2.name, p1.friends[p2]) 
            for p1i, p1 in enumerate(self.context.ppl)
            for p2i, p2 in enumerate(self.context.ppl)
            if p1i > p2i
            and p2 in p1.friends
            and p1.friends[p2] > 0
        ]
    
    def show(self, ts=[5,10], fname=None, extra={}, pos=None):
        plt.figure(figsize=(5,5))
        
        for i,t in enumerate(ts):
            #plt.subplot(2,2,i+1)
            
            snp = self.snaps[t]

            g = nx.Graph()

            g.add_edges_from([(f,t,{'weight':w}) for (f,t,w) in snp])

            edges = g.edges()
            weights = np.array([g[u][v]['weight'] for u,v in edges])

            extra_sz = np.array([extra[x] for x in g.nodes()])

            nodes = nx.draw_networkx_nodes(g, pos, node_size=extra_sz*3, node_color='w')
            # Set edge color to red
            nodes.set_edgecolor('black')

            edges = nx.draw_networkx_edges(g, edges=edges, pos=pos, width=weights*0.2)
            plt.title("network at t=%s" % t)

        st.write(plt.gcf())














"# Extraversion simulation"

class unhappiness_avg(Metric):
    def measure(self):
        return sum( abs(p.hours_per_week() - p.args['extraversion']) for p in self.context.ppl )
    
    def show(self, tstart=0, tstop=100):
        from itertools import chain
        
        #print(self.snaps)
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            'unhap': [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        dd.plot(x='t', y='unhap') # , legend=None
        plt.ylim(0,None)
        plt.title("Unhappiness over time (sum abs diff)")
        st.write(plt.gcf())

class unhappiness_max(Metric):
    def measure(self):
        return max( abs(p.hours_per_week() - p.args['extraversion']) for p in self.context.ppl )
    
    def show(self, tstart=0, tstop=100):
        from itertools import chain
        
        #print(self.snaps)
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            'unhap': [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        dd.plot(x='t', y='unhap') # , legend=None
        plt.ylim(0,None)
        plt.title("Maximum unhappiness over time (abs diff)")
        st.write(plt.gcf())



class end_friendship(Action):

    def mu(self,x):
        if x <= -1e2:
            return 0.5
        if x <= -1e2:
            return 0.5

        return abs( 1/(1 + np.exp(-x)) - 0.5 )
    
    def act_frequency(self):
        if self.person.hours_per_week() == 0 or self.person.hours_per_week() < self.person.args['extraversion']:
            return 1000
        
        step1 = self.mu( 1 - self.person.args['extraversion'] / self.person.hours_per_week() )
        step2 = 1/(step1+1e-10)
        return step2
    
    def act(self, context):
        from random import choice
        
        potential_unfriends = list(self.person.friends)
        if not len(potential_unfriends):
            #raise Exception("Person wants to remove friendship, but doesn't have any. Negative extraversion?")
            #nothing happens
            return
                    
        unfriend = choice(potential_unfriends)
        
        unfriend.friends[self.person] = 0
        self.person.friends[unfriend] = 0
        
        self.log("Unfriended %s" % unfriend.name, context)
        
        
        
class make_friend(Action):
    

    def mu(self,x):
        if x <= -1e2:
            return 0.5
        if x <= -1e2:
            return 0.5

        return abs( 1/(1 + np.exp(-x)) - 0.5 )
    
    def act_frequency(self):
        if self.person.hours_per_week() == 0: return 1
        
        if self.person.hours_per_week() >= self.person.args['extraversion']:
            return 1000
        
        step1 = self.mu( 1 - self.person.args['extraversion'] / self.person.hours_per_week() )
        step2 = 1/(step1+1e-10)
        return step2
    
    def act(self, context):
        self.log("ACT_BEGIN!", context)
        from random import random, choice
        # go out to make a friend.
        # I decide to go do it...
        self.log("trying to make a friend", context)
        
        iwant = self.person.args['extraversion'] - self.person.hours_per_week()
        if iwant < 0:
            # this shouldn't be possible
            return
        
        if random() > 0.8:
            self.log("randomly failed at making a friend", context)
            return
            
        potential_friends = {
            p for p in context.ppl
            if p != self.person and \
                p not in self.person.friends
            and p.hours_per_week() < p.args['extraversion']
        }
        
        #print([
        #    (p, self.person.friends) for p in context.ppl
        #    if p != self.person and \
        #        p not in self.person.friends
        #])
        potential_friends = list(potential_friends)
        
        if not len(potential_friends):
            self.log("No friends available... sad", context)
            return
            
        my_new_friend = choice(potential_friends)
        
        theywant = my_new_friend.args['extraversion'] - my_new_friend.hours_per_week()
        
        time_agree = np.random.uniform(iwant, theywant)
        
        my_new_friend.friends[self.person] = time_agree
        self.person.friends[my_new_friend] = time_agree
        
        self.log("Made friend w/ %s" % my_new_friend.name, context)
        
class Person(Person):
    
    def hours_per_week(self):
        return sum(list(self.friends.values()))
        
    def __init__(self, *args, **kwargs):
        self.friends = {}
        
        super().__init__(*args,**kwargs)















########   BEGIN THE GOOD / CUSTOM STUFF ##########


howlong = st.sidebar.slider('Length of simulation (t)',5,50)
N_PEOPLE = st.sidebar.selectbox('Number of people', [5,10,50,100,200,500], index=3)
RESOLUTION = st.sidebar.selectbox('Resolution', [0.25,0.5,1,5], index=2)
extraversion_max = st.sidebar.slider('Extraversion max', 0,20,5)
sim_bar = st.sidebar.progress(0)

@st.cache
def sim_cache(howlong, N_PEOPLE, RESOLUTION, extraversion_max):
    people = [ Person(extraversion=float(i)*extraversion_max/N_PEOPLE) for i in range(N_PEOPLE) ]
    xyz = [p.addact([make_friend,end_friendship]) for p in people];

    helloworld = sim(people, debug=False)
    helloworld.addmet( degree_distribution )
    helloworld.addmet( unhappiness_max )
    helloworld.addmet( unhappiness_avg )
    helloworld.addmet( network )

    helloworld.run(float(howlong), RESOLUTION)

    g = nx.Graph()
    total_edges = defaultdict(float)

    for snp in helloworld.metrics['network'].snaps.values():
        for (f,t,w) in snp:
            total_edges[(f,t)] += w
        
    g.add_edges_from([(f,t,{'weight':w}) for (f,t),w in total_edges.items()])
    pos = nx.spring_layout(g, k=0.2)

    return { 
        'metrics': {
            k: v.snaps
            for k,v in helloworld.metrics.items()
        },
        'extra': {
            p.name: p.args['extraversion']
            for p in people
        },
        'net_pos':pos
    }


res = sim_cache(howlong, N_PEOPLE, RESOLUTION, extraversion_max)
metrics = res['metrics']

#if st.sidebar.button('Rerun simulation'):
#    """## Running simulation...
#    This should run the simulation, and show a progress-bar below"""

metrics = {
    k: eval(k)(snaps=v) # revitalizes the class. weirddd
    for k,v in metrics.items()
}


"## Overall unhappiness"
metrics['unhappiness_max'].show(0, 500)
metrics['unhappiness_avg'].show(0,500)

"## Degree distribution"
deg_max = st.selectbox('Maximum degree to show',[5,10,20,50],1)
metrics['degree_distribution'].show(0,500, deg_range=deg_max+1)

if True:
    "# The network"
    t = st.slider('Time', float(0), howlong-RESOLUTION, step=RESOLUTION, value=howlong-RESOLUTION)
    metrics['network'].show([t], extra=res['extra'], pos=res['net_pos'])

# making more things!

x="""
class breakup(Action):
class moveaway(Action):
class startclub(Action):
class joinclub(Action):
""";