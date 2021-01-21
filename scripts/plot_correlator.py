import numpy as np
import matplotlib.pyplot as plt
import pickle
from correlator import *

d_as = pickle.load(open('../../QGasData/DatasetFullInfoAS/as_wtest_d9.0.pkl', 'rb'))["snapshots"]
d_pi = pickle.load(open('../../QGasData/DatasetFullInfoPi/pi_wtest_d9.0.pkl', 'rb'))["snapshots"]

params = {
    'font.family': 'CMU Sans Serif',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
plt.rcParams.update(params)

def apply(f, d): 
    res = [] 
    for s in d:
        res.append(f(s)) 
    return res 


AS_COLOR = (255/255, 167/255, 86/255, 0.75)
PI_COLOR = (191/255, 119/255, 246/255, 0.75)
AS_M_COLOR = (191/255, 92/255, 0/255, 1.00)
PI_M_COLOR = (142/255, 16/255, 237/255, 1.00)

def makeHist(f, binsize=2): 
    prop_cycle = plt.rcParams['axes.prop_cycle'] 
    colors = prop_cycle.by_key()['color'] 
     
    fig, ax = plt.subplots(figsize=(5, 5)) 
    corr_as = np.array(apply(f, d_as)) 
    corr_pi = np.array(apply(f, d_pi)) 
    mean_as = np.mean(corr_as) 
    mean_pi = np.mean(corr_pi) 
    # Figure out the bins, in integer space 
    min_val = int(min(np.min(corr_as), np.min(corr_pi))) 
    max_val = int(max(np.max(corr_as), np.max(corr_pi))) 
    nbins = (max_val - min_val) // binsize + 1 
    bins = np.array(range(min_val, max_val+binsize, binsize)) / (15.0 * 15.0) 
    
    ax.hist(corr_as / (15.0 * 15.0), alpha=0.5, bins=bins, density=True, color=AS_COLOR) 
    ax.hist(corr_pi / (15.0 * 15.0), alpha=0.5, bins=bins, density=True, color=PI_COLOR) 
    ax.axvline(mean_as / (15.0 * 15.0), ymin=0, ymax=1, lw=4, color=AS_M_COLOR)
    ax.axvline(mean_pi / (15.0 * 15.0), ymin=0, ymax=1, lw=4, color=PI_M_COLOR) 
    fig.set_tight_layout(True)


def dopingCorrelators():
    dopings = ['0.0', '6.0', '9.0', '12.5', '20.0']
    as_feats = { 
        "L": [], 
        "Bound": [], 
        "StringHole": [], 
        "AFMHole": [] 
    }
    as_errs = {
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
    pi_errs = {
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
        if dope == '9.0':
            as_fname = "../../QGasData/DatasetFullInfoAS/as_wtest_d9.0.pkl"
            pi_fname = "../../QGasData/DatasetFullInfoPi/pi_wtest_d9.0.pkl"
        else:
            as_fname = "../../QGasData/DatasetFullInfoAS/as_d{}.pkl".format(dope)
            pi_fname = "../../QGasData/DatasetFullInfoPi/pi_d{}.pkl".format(dope)

        d_as = pickle.load(
            open(as_fname, "rb")
        )["snapshots"]
        d_pi = pickle.load(
            open(pi_fname, "rb")
        )["snapshots"]
        for key, f in feats.items():
            as_f = np.array(apply(f, d_as))
            pi_f = np.array(apply(f, d_pi))
            as_feats[key].append(np.mean(as_f) / (15 * 15))
            pi_feats[key].append(np.mean(pi_f) / (15 * 15))
            as_errs[key].append(np.std(as_f / (15 * 15)) / (np.sqrt(len(as_f))))
            pi_errs[key].append(np.std(pi_f / (15 * 15)) / (np.sqrt(len(pi_f))))

    colors = ['orange', 'red', 'green', 'purple']
    dopings = list(map(float, dopings))
    for key, color in zip(feats.keys(), colors):
        plt.figure(figsize=(8, 8))
        plt.xlabel(r'Doping %')
        plt.ylabel('Correlator')
        plt.errorbar(dopings, np.array(as_feats[key]), marker='s', color=color,
                 markersize=12, linewidth=0, fillstyle='none', yerr=as_errs[key], label=key)
        plt.errorbar(dopings, np.array(pi_feats[key]), marker='^', color=color,
                 markersize=12, linewidth=0, fillstyle='none', yerr=pi_errs[key])
        plt.legend()
        plt.show()


def dopingConnCorrelators():
    dopings = ['0.0', '6.0', '9.0', '12.5', '20.0']
    as_feats = {
        "L": [],
        "Bound": [],
        "StringHole": [],
        "AFMHole": []
    }
    as_errs = {
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
    pi_errs = {
        "L": [],
        "Bound": [],
        "StringHole": [],
        "AFMHole": []
    }
    feats = {
        "L": connCorrL,
        "Bound": connCorrBound,
        "StringHole": connCorrStringHole,
        "AFMHole": connCorrAFMHole,
    }

    for dope in dopings:
        print("Doping", dope)
        if dope == '9.0':
            as_fname = "../../QGasData/DatasetFullInfoAS/as_wtest_d9.0.pkl"
            pi_fname = "../../QGasData/DatasetFullInfoPi/pi_wtest_d9.0.pkl"
        else:
            as_fname = "../../QGasData/DatasetFullInfoAS/as_d{}.pkl".format(dope)
            pi_fname = "../../QGasData/DatasetFullInfoPi/pi_d{}.pkl".format(dope)

        d_as = pickle.load(
            open(as_fname, "rb")
        )["snapshots"]
        d_pi = pickle.load(
            open(pi_fname, "rb")
        )["snapshots"]
        d_as = np.stack(d_as, axis=0)
        d_pi = np.stack(d_pi, axis=0)
        for key, f in feats.items():
            as_f = symCorr(d_as, f)
            pi_f = symCorr(d_pi, f)
            if key == 'L' or key == 'AFMHole':
                as_f /= 2
                pi_f /= 2
            elif key == 'Bound':
                as_f /= 4
                pi_f /= 4
            as_feats[key].append(np.mean(as_f))
            pi_feats[key].append(np.mean(pi_f))
            as_errs[key].append(np.std(as_f) / np.sqrt(len(as_f)))
            pi_errs[key].append(np.std(pi_f) / np.sqrt(len(as_f)))

    print(as_errs)

    colors = ['orange', 'red', 'green', 'purple']
    dopings = list(map(float, dopings))
    for key, color in zip(['L', 'Bound', 'StringHole', 'AFMHole'], colors):
        plt.figure(figsize=(8, 8))
        plt.xlabel(r'Doping %')
        plt.ylabel('Correlator')
        plt.errorbar(dopings, np.array(as_feats[key]), marker='s', color=color,
                 markersize=12, linewidth=0, fillstyle='none', yerr=as_errs[key], label=key)
        plt.errorbar(dopings, np.array(pi_feats[key]), marker='^', color=color,
                 markersize=12, linewidth=0, fillstyle='none', yerr=pi_errs[key])
        plt.legend()
        plt.show()