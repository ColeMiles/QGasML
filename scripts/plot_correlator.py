import numpy as np
import matplotlib.pyplot as plt
import pickle
from correlator import *

d_as = pickle.load(open('../../QGasData/DatasetFullInfoAS/as_d9.0.pkl', 'rb'))["snapshots"]
d_pi = pickle.load(open('../../QGasData/DatasetFullInfoPi/pi_d9.0.pkl', 'rb'))["snapshots"]


def apply(f, d): 
    res = [] 
    for s in d:
        res.append(f(s)) 
    return res 


def makeHist(f, binsize=2): 
    prop_cycle = plt.rcParams['axes.prop_cycle'] 
    colors = prop_cycle.by_key()['color'] 
     
    fig, ax = plt.subplots(figsize=(6, 6)) 
    corr_as = np.array(apply(f, d_as)) 
    corr_pi = np.array(apply(f, d_pi)) 
    mean_as = np.mean(corr_as) 
    mean_pi = np.mean(corr_pi) 
    # Figure out the bins, in integer space 
    min_val = int(min(np.min(corr_as), np.min(corr_pi))) 
    max_val = int(max(np.max(corr_as), np.max(corr_pi))) 
    nbins = (max_val - min_val) // binsize + 1 
    bins = np.array(range(min_val, max_val+binsize, binsize)) / (15.0 * 15.0 * 16.0) 
    
    ax.hist(corr_as / (15.0 * 15.0 * 16.0), alpha=0.5, bins=bins, density=True) 
    ax.hist(corr_pi / (15.0 * 15.0 * 16.0), alpha=0.5, bins=bins, density=True) 
    ax.axvline(mean_as / (15.0 * 15.0 * 16.0), ymin=0, ymax=1, color=colors[0], lw=2) 
    ax.axvline(mean_pi / (15.0 * 15.0 * 16.0), ymin=0, ymax=1, color=colors[1], lw=2) 
    fig.set_tight_layout(True)


def dopingCorrelators():
    dopings = ['0.0', '6.0', '9.0', '12.5', '20.0']
    as_feats = { 
        "L": [], 
        "Bound": [], 
        "StringHole": [], 
        "AFMHole": [] 
    }
    pi_feats = { 
        "L": [], 
        "Bound": [], 
        "StringHole": [], 
        "AFMHole": [] 
    }
    feats = { 
        "L": corrL, 
        "Bound": corrBound, 
        "StringHole": corrStringHole, 
        "AFMHole": corrAFMHole, 
    } 

    for dope in dopings:
        d_as = pickle.load(
            open("../../QGasData/DatasetFullInfoAS/as_d{}.pkl".format(dope), "rb")
        )["snapshots"]
        d_pi = pickle.load(
            open("../../QGasData/DatasetFullInfoPi/pi_d{}.pkl".format(dope), "rb")
        )["snapshots"]
        for key, f in feats.items(): 
            as_feats[key].append(np.mean(apply(f, d_as))) 
            pi_feats[key].append(np.mean(apply(f, d_pi)))
    
    colors = ['orange', 'red', 'green', 'purple']
    dopings = list(map(float, dopings))
    plt.xlabel(r'Doping %')
    plt.ylabel('Correlator')
    for key, color in zip(feats.keys(), colors):
        plt.plot(dopings, np.array(as_feats[key]) / (15 * 15 * 16), marker='s', color=color,
                 markersize=12, linewidth=0, fillstyle='none', label=key)
        plt.plot(dopings, np.array(pi_feats[key]) / (15 * 15 * 16), marker='^', color=color,
                 markersize=12, linewidth=0, fillstyle='none')
    plt.legend()
    plt.show()

