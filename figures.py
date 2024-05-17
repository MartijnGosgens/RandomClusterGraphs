import numpy as np
import matplotlib.pyplot as plt
from hyperspherical_community_detection.Clustering import Clustering
from rcg import *

def sizes2edges(sizes):
    import itertools as it
    return it.chain(*[
        it.combinations(range(c-s,c),2)
        for c,s in zip(it.accumulate(sizes),sizes)
    ])

def draw_cliques(n,p,name=None):
    import networkx as nx
    G = nx.Graph()
    sizes = sorted(random_sizes(n,p))
    n = sum(sizes)
    G.add_nodes_from(range(n))
    G.add_edges_from(sizes2edges(sizes))
    
    fig=plt.figure()
    pos = nx.spring_layout(G,weight=0.001,k=1/len(sizes)**0.6)
    outer_angle = 0
    outer_radius = 2*n
    pos = {}
    i = 0
    for size in sizes:
        outer_angle += 2*np.pi*(size/2)/n
        center = outer_radius*np.array([np.cos(outer_angle),np.sin(outer_angle)])
        inner_radius = 4*size
        for j in range(i,i+size):
            pos[j] = center + inner_radius*np.array([np.cos(2*np.pi*j/size),np.sin(2*np.pi*j/size)])
        outer_angle += 2*np.pi*(size/2)/n
        i+=size
    nx.draw(G,node_size=50,pos=pos)
    if name:
        plt.savefig(name+'.jpg')
    return fig

def draw_probcomplete(n_max=500,n_min=100,n_step=10,t_max=0.06,n_ts=60):
    ts=np.linspace(0,t_max,n_ts)
    p_ERs = [mp.exp(t)/(1+mp.exp(t)) for t in ts]
    #print(p_ERs)

    ns = list(range(n_min,n_max+1,n_step))
    n_mesh, t_mesh = np.meshgrid(ns,ts)
    n_max = max(ns)
    p2bells = {
        p: bell_generalized_list(n_max,p=p)
        for p in p_ERs
    }
    print('computed bells')

    probs = np.array([
        [
            float(probcomplete(n,p,bells=p2bells[p]))
            for p in p_ERs
        ]
        for n in ns
    ]).transpose()
    print('computed heatmap')

    t_critical = [
        probcomplete_inverse_nr(n,0.5) for n in ns
    ]
    print('computed critical sequence')
    ls = [
        mp.lambertw(n) for n in ns
    ]
    bells = bell_generalized_list(n_max)
    t_approx_lower = [
        (mp.log(0.5)+mp.log(bells[n]))/mp.binomial(n,2)
        for n,l in zip(ns,ls)
    ]
    t_approx_upper = [
        mp.log(mp.factorial(n))/mp.binomial(n,2)
        for n,l in zip(ns,ls)
    ]
    fig, ax = plt.subplots(1, 1)
    t2p = lambda t: 1/(1+np.exp(-t))
    t2p2 = lambda t: 1/(1+mp.exp(-t))
    c = ax.pcolormesh(n_mesh, t2p(t_mesh), probs, cmap='RdYlGn_r', shading='auto')
    plt.colorbar(c,ax=ax,label=r'$\mathbb{P}(\mathbf{C}_{n,p}=1)$')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$p$')
    #ax.set_title('Phase transition')
    ax.plot(ns,list(map(t2p2,t_critical)),label='Critical sequence')
    ax.plot(ns,list(map(t2p2,t_approx_lower)),label='Lower bound')
    ax.plot(ns,list(map(t2p2,t_approx_upper)),label='Upper bound')
    ax.set_ylim(0.5,0.514)
    ax.set_xlim(min(ns),max(ns))
    plt.legend()
    plt.savefig('probcomplete_n{}.jpg'.format(n_max), dpi=400)

def validate_golden_ratio():
    fig,axs = plt.subplots(1,2,sharey=True,figsize=(7,3))
    ns=[3,30]
    for n,ax in zip(ns,axs):
        p=1/n
        t=mp.log(p/(1-p))
        pbells = bell_generalized_list(n,p=p)
        print(n,mp.binomial(n-1,1)*mp.exp(t)*pbells[n-2]/pbells[n],1-2/(1+5**0.5))
        ax.bar([1,2,3],[
            mp.binomial(n-1,s-1)*mp.exp(t*mp.binomial(s,2))*pbells[n-s]/pbells[n]
            for s in [1,2,3]
        ])
        ax.set_xlabel(r'$s$')
        ax.set_title('$n={}$'.format(n))

    plt.yticks([0.0,1-2/(1+5**0.5),2/(1+5**0.5)],['$0.0$',r'$1-\rho^{-1}$',r'$\rho^{-1}$'])
    plt.xticks([1,2,3])
    plt.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15)

    axs[0].set_ylabel(r'$\mathbb{P}(\mathbf{S}_{n,1/n}=s)$')
    axs[0].yaxis.set_label_coords(-0.07, 0.3)

    plt.savefig('vanishing_golden_ratio.jpg'.format(n),dpi=400)

def plot_critical_sizes(n):
    t = probcomplete_inverse_nr(n)
    p = t2p(t)
    pbells = bell_generalized_list(n,p=p)
    n_cliques_of_size = [
        (mp.binomial(n,s)*mp.exp(mp.binomial(s,2)*t)*pbells[n-s])/pbells[n] 
        for s in range(1,n+1)
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [3, 1]})

    # plot the same data on both axes
    ax.bar(range(1,n+1),n_cliques_of_size)

    ax2.bar(range(1,n+1),n_cliques_of_size)

    ax.set_xlim(1, 16)
    ax2.set_xlim(n-6, n)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-3*d, +3*d), (1-d, 1+d), **kwargs)
    ax2.plot((-3*d, +3*d), (-d, +d), **kwargs)

    ax.set_xlabel('Clique size $s$')
    ax.xaxis.set_ticks([1,5,10,15])
    ax.xaxis.set_label_coords(0.75, -0.07)
    ax2.xaxis.set_ticks([n-5,n])
    ax.set_ylabel('Expected number of cliques '+r'$\mathbb{E}[\mathbf{C}^{(s)}_{n}]$')
    plt.savefig('ncliques_n{}.jpg'.format(n),dpi=400)

def community_detection_experiment():
    # Available at https://github.com/MartijnGosgens/hyperspherical_community_detection
    from hyperspherical_community_detection.random_graphs import generators as gen
    from hyperspherical_community_detection.algorithms import pair_vector as pv
    d_in,d_out = 10,10
    n,k=1000,5
    p_in = d_in/(n/k-1)
    p_out = d_out/(n-n/k)
    print(p_in,p_out)
    ppm = gen.PPM(n=n,k=k,p_in=p_in,p_out=p_out)
    ps = np.linspace(0.5,0.51,51)
    samplesize = 20
    print('sampling')
    sample = [ppm.generate(seed=s) for s in range(samplesize)]
    print('computing gammas')
    gammas = [
        bayesian_gamma(p_in,p_out,p)
        for p in ps
    ]
    print('computing heuristics')
    theta=np.median([pv.connectivity(G).meridian_angle(pv.clustering_binary(T)) for G,T in sample])
    lT = np.median([pv.clustering_binary(T).latitude() for G,T in sample])
    median_heuristic_lat=pv.heuristic_latitude(lT,theta)
    p2latdif = {
        ps[i]: abs(np.median([(pv.connectivity(G)+(1/2-float(gamma))*pv.ones).latitude() for G,T in sample])-median_heuristic_lat)
        for i,gamma in enumerate(gammas)
    }
    p2latdif2 = {
        ps[i]: abs(np.median([(pv.connectivity(G)+(1/2-float(gamma))*pv.ones).latitude() for G,T in sample])-np.pi/2)
        for i,gamma in enumerate(gammas)
    }
    
    G,T = ppm.generate(seed=0)

    from collections import defaultdict
    gamma2candidates = defaultdict(list)
    avg_grans = []
    avg_performances = []
    for gamma in gammas:
        gamma2candidates[gamma] = [
            pv.louvain_projection(pv.connectivity(G)+(1/2-float(gamma))*pv.ones)
            for G,T in sample
        ]
        avg_gran = sum([
            C.intra_pairs()/T.intra_pairs()
            for (G,T),C in zip(sample,gamma2candidates[gamma])
        ])/samplesize
        avg_grans.append(avg_gran)
        avg_performance = sum([
            mp.cos(pv.clustering_binary(C).meridian_angle(pv.clustering_binary(T)))
            for (G,T),C in zip(sample,gamma2candidates[gamma])
        ])/samplesize
        avg_performances.append(avg_performance)
        print(gamma,avg_gran,avg_performance)
    
    from matplotlib.patches import Rectangle
    t_lower = 2*(mp.log(n)-mp.log(mp.log(n))-1)/n
    t_upper = 2*(mp.log(n)-1)/n
    p_lower,p_upper=t2p(t_lower),t2p(t_upper)
    print(p_lower,p_upper)
    #ps = np.linspace(0.5,0.51,11)
    plt.plot(ps,np.array(avg_grans)*T.intra_pairs())
    plt.axhline(T.intra_pairs(),linestyle = '--',color='orange')
    plt.xticks([0.5,p_lower,p_upper,0.51],[r'$0.5$',r'$p_L$',r'$p_U$','$0.51$'])
    plt.yticks([0,T.intra_pairs(),n*(n-1)/2],[r'$0$',r'$m_T$',r'$\binom{n}{2} $'])

    ax=plt.gca()
    ymin,ymax = (ax.get_ylim())
    ax.add_patch(Rectangle((p_lower, ymin), p_upper-p_lower, ymax-ymin,facecolor='green',alpha=0.2))
    ax.set_ylim(ymin,ymax)
    plt.ylabel('#Edges in detected partition graph')
    plt.xlabel('$p$')
    plt.savefig('granularity_n{}.jpg'.format(n),dpi=400)

    plt.plot(ps,avg_performances)

    plt.xticks([0.5,p_heur,p_lower,p_upper,0.51],[r'$0.5$',r'$p_H$',r'$p_L$',r'$p_U$','$0.51$'])
    plt.ylabel(r'Average $\rho(C,T)$')
    plt.xlabel('$p$')
    ax=plt.gca()
    ymin,ymax = (ax.get_ylim())
    ax.add_patch(Rectangle((p_lower, ymin), p_upper-p_lower, ymax-ymin,facecolor='green',alpha=0.2))
    ax.set_ylim(ymin,ymax)
    #plt.axvline(p_critical,linestyle='--',color='orange')
    plt.savefig('performances_n{}.jpg'.format(n),dpi=400)

def generate_all():
    draw_cliques(50,0.25,name='sample_cliques_n50_p025')
    draw_cliques(50,0.51,name='sample_cliques_n50_p051')
    draw_cliques(50,0.53,name='sample_cliques_n50_p053')
    validate_golden_ratio()
    plot_critical_sizes(100)
    plot_critical_sizes(500)
    draw_probcomplete()
    community_detection_experiment()