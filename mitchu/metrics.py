from .lib import Metric, Counter, nx, np, pd, plt, Path
    
class degree_distribution(Metric):
    """
    A generic degree distribution metric.
    By default the degree of each person `p` is the length of their `p.friends`.

    If you want to define a custom degree distribution metric, you can create a subclass of degree_distribution.
    The `measure` function should return a `dict` or `Counter`, which maps from degree to the number of individuals with that degree.
    """  
      
    def measure(self):
        return Counter( [ len(p.friends) for p in self.context.ppl ] )
    

    def show(self, tstart=0, tstop=100, deg_range=20):
        """
        Shows a stacked area-plot of how the degree distribution changes over time.
        The x-axis runs froom `tstart` to `tstop`, with # people on the y-axis.
        Only counts for $0 \leq deg_range < 20$ are displayed.
        """
        from itertools import chain
        
        alltimes = sorted(self.snaps.keys())
        alltimes = [x for x in alltimes if tstart<=x<=tstop]
        
        dd = pd.DataFrame(dict({
            str(deg): [self.snaps[ t ][deg] for t in alltimes]
            for deg in range(0,deg_range)
        }, t=alltimes))

        dd.plot.area(x='t')#, legend=None 
        plt.title("Degree distribution over time")
    

class prob_action(Metric):
    def __init__(self):
        self.names = []
        self.n2p = {}
        super().__init__()
        
    def measure(self):
        self.names += [ x.name for x in self.context.ppl if x.name not in self.names ]
        for p in self.context.ppl:
            if p.name not in self.n2p:
                self.n2p[p.name] = p
                
        return [
            {
                k:v.act_time()
                for k,v in self.n2p[pname].actions.items()
            }
            for pname in self.names
        ]
    
    def show(self, which, order=None, maxT=None, maxTime=None):
        alltimes = sorted(self.snaps.keys())
        
        
        if order is None:
            allpeople = range(max(len(x) for x in self.snaps.values()))
        else:
            allpeople = [ 
                self.names.index( n )
                for n in order
            ]
            
        if maxTime is not None:
            alltimes = [x for x in alltimes if x <= maxTime]
            
        mw = len(alltimes)
        mh = len(allpeople)
        
        tos = np.zeros(shape=(mh,mw))-10
        for i,t in enumerate(alltimes):
            for pi in allpeople:
                myval = -1
                thisss = self.snaps[t]
                if len(thisss) <= pi:
                    myval = -1
                else:
                    myval = thisss[pi][which]
                    
                if maxT is not None:
                    myval = min(maxT, myval)
                tos[pi,i] = myval
        
        plt.imshow(tos)
        return tos
    
class network(Metric):
    def measure(self):
        return [
            (p1, p2, p1.friends[p2]) 
            for p1i, p1 in enumerate(self.context.ppl)
            for p2i, p2 in enumerate(self.context.ppl)
            if p1i > p2i
            and p2 in p1.friends
            and p1.friends[p2] > 0
        ]
    
    def show(self, ts=None, animate=True, fname=None, nodecolor=None, weight_mod=1, maxT=None):
        
        g = nx.Graph()
        self.allnames = set()
        for t in np.linspace( min(self.snaps), max(self.snaps), 10 ):
            myt = max(tt for tt in self.snaps if tt <= t)
            snp = self.snaps[myt]
            
            g.add_edges_from([(f,t,{'weight':w}) for (f,t,w) in snp])
            self.allnames.update({f for (f,_,_) in snp})
            self.allnames.update({t for (_,t,_) in snp})
            
        g.add_nodes_from(self.allnames.difference(set(g.nodes())))

        pos = nx.spring_layout(g, k=0.2)
        
        
        if ts is not None:
            plt.figure(figsize=((3 * 7),(len(ts)//3+1)*5))
            for i,t in enumerate(ts):
                plt.subplot(len(ts)//3+1,3,i+1)
                self.plot_network(t, pos, weight_mod=weight_mod)
                plt.title("network at t=%0.3f" % t)
            plt.show()
            
        else:
            if animate:
                import imageio
                images = []
                
                mxttt = max(self.snaps)
                if maxT is not None:
                    mxttt = min(mxttt, maxT)
                for i,t in enumerate(np.linspace( min(self.snaps), mxttt, 100 )):
                    plt.figure(figsize=(7,5))
                    self.plot_network(t, pos, weight_mod=weight_mod)
                    plt.title("network at t=%0.3f" % t)
                    Path("./temp").mkdir(exist_ok=True)
                    
                    filename = "temp/%s.PNG" % i
                    
                    plt.savefig(filename)
                    images.append(imageio.imread(filename))
                    plt.close()
                    
                imageio.mimsave('movie.gif', images)
                from IPython.display import HTML,display
                display(HTML('<img src="movie.gif">'))
            else:
                raise Exception("weird combination of arguments to network 'show'")
                
    def plot_network(self, t, pos, nodecolor=None, weight_mod=1):
        myt = max(tt for tt in self.snaps if tt <= t)
        snp = self.snaps[myt]

        g = nx.Graph()
        
        g.add_nodes_from(self.allnames)
        g.add_edges_from([(f,t,{'weight':w}) for (f,t,w) in snp])

        edges = g.edges()
        weights = np.array([g[u][v]['weight'] for u,v in edges]) * weight_mod

        extra = np.array([x.args['extraversion'] for x in g.nodes()])
        mxe = np.max(extra)
        mne = np.min(extra)
        if mxe == mne:
            node_s = 15
        else:
            node_s = 30*(extra-mne+(mxe-mne)/10)/(mxe-mne)
        if nodecolor is not None:
            col = np.array([x.args[nodecolor] for x in g.nodes()])
            nodes = nx.draw_networkx_nodes(g, pos, node_size=node_s, node_color=col)
        else:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=node_s, node_color='w')
        # Set edge color to red
        if nodes:
            nodes.set_edgecolor('black')

        edges = nx.draw_networkx_edges(g, edges=edges, pos=pos, width=weights*0.2)
        
        xl = min(x for x,y in pos.values())
        xh = max(x for x,y in pos.values())
        xd = (xh - xl) / 10
        yl = min(y for x,y in pos.values())
        yh = max(y for x,y in pos.values())
        yd = (yh - yl) / 10
        
        plt.xlim(xl-xd,xh+xd)
        plt.ylim(yl-yd,yh+yd)