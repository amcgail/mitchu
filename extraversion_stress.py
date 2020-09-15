from mitchu.lib import *









def mu(x, alpha=1, minT=0, maxT=1):
    """
    
    """
    if x <= -1e2:
        return maxT
    if x <= -1e2:
        return maxT
    
    basic_shape = abs( 2/(1 + np.exp(-x/alpha)) - 1 )
    second_step = (maxT-minT)*basic_shape
    inverted = maxT - second_step
    return inverted






class unhappiness(Metric):

    def measure(self):
        return [ p.hours_per_week() / p.args['extraversion'] for p in self.context.ppl ]

    def show(self, tstart=0, tstop=100, **kwrargs):
        from itertools import chain
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        for q in np.linspace( 0, 1, 5 ):
            dd = pd.DataFrame(dict({
                'unhap': [np.quantile(self.snaps[ t ], q) for t in alltimes]
            }, t=alltimes))

            plt.plot(dd.t, dd.unhap, label="%0.3f"%q, **kwrargs)
        
        plt.xlabel("time")
        plt.ylabel("actual / desired")
        plt.legend()

class percent_oversocialized(Metric):
    

    def __init__(self, oversoc_def=1.5): # *args,**kwargs
        self.oversoc_def = oversoc_def
        
        super().__init__()
        
    def measure(self):
        return sum( p.hours_per_week() / p.args['extraversion'] > self.oversoc_def for p in self.context.ppl ) / len(self.context.ppl)
    
    def show(self, tstart=0, tstop=100, **kwrargs):
        from itertools import chain
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            'unhap': [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        plt.plot(dd.t, dd.unhap, **kwrargs)
        plt.xlabel("time")

class percent_lonely(Metric):
    

    def __init__(self, undersoc_def=0.5): # *args,**kwargs
        self.undersoc_def = undersoc_def
        
        super().__init__()
        
    def measure(self):
        return sum( p.hours_per_week() / p.args['extraversion'] < self.undersoc_def for p in self.context.ppl ) / len(self.context.ppl)
    
    def show(self, tstart=0, tstop=100, **kwrargs):
        from itertools import chain
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            'unhap': [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        plt.plot(dd.t, dd.unhap, **kwrargs)
        plt.xlabel("time")

class unhappiness_avg(Metric):
    def measure(self):
        return sum( abs(p.hours_per_week() - p.args['extraversion']) for p in self.context.ppl )
    
    def show(self, tstart=0, tstop=100, **kwrargs):
        from itertools import chain
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            'unhap': [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        plt.plot(dd.t, dd.unhap, **kwrargs)
        plt.xlabel("time")

class unhappiness_max(Metric):
    def measure(self):
        return max( abs(p.hours_per_week() - p.args['extraversion']) for p in self.context.ppl )
    
    def show(self, tstart=0, tstop=100, **kwrargs):
        from itertools import chain
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            'unhap': [self.snaps[ t ] for t in alltimes]
        }, t=alltimes))

        plt.plot(dd.t, dd.unhap, **kwrargs) # , legend=None
        plt.xlabel("time")

        
        
        
        
        
        
        
        
        
        
        
        
        

class end_friendship(Action):

    def __init__(self, minT=0.5, maxT=10, alpha=0.2): # *args,**kwargs
        self.minT = minT
        self.maxT = maxT
        self.alpha = alpha
        
        super().__init__()
        
    
    def act_time(self):
        #if self.person.hours_per_week() == 0 or self.person.hours_per_week() < self.person.args['extraversion']:
        #    return 1000
        if self.person.hours_per_week() == 0: return self.maxT
        if self.person.hours_per_week() <= self.person.args['extraversion']: return self.maxT
        
        step1 = mu(
            x=1 - self.person.args['extraversion'] / self.person.hours_per_week(),
            alpha=self.alpha, # alpha
            minT=self.minT,
            maxT=self.maxT,
        )
        return step1
    
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

    def __init__(self, minT=0.5, maxT=10, alpha=0.1): # *args,**kwargs
        self.minT = minT
        self.maxT = maxT
        self.alpha = alpha
        
        super().__init__()

    
    def act_time(self):
        if self.person.hours_per_week() == 0: return self.minT
        if self.person.hours_per_week() >= self.person.args['extraversion']: return self.maxT
        
        step1 = mu( 
            x=1 - self.person.args['extraversion'] / self.person.hours_per_week(),
            alpha=self.alpha, # alpha
            minT=self.minT,
            maxT=self.maxT
        )
        return step1
    
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