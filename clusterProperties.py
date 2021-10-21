import fits
import calcPhysicalDist as pd
import findTrend
import math
import numpy as np
import matplotlib.pyplot as plt
import random, time, os
from astropy.stats import bootstrap
from scipy.stats import spearmanr, pearsonr, kendalltau



if(__name__ == '__main__'):
    physDict = pd.loadObject("HeCS-omnibus_PhysicalDistanceData")

    data = np.loadtxt("HeCS_omnibus_prop.txt", dtype=str)
    clusters = data[:,0]
    redshift = data[:,1].astype(np.float)
    R200 = data[:,2].astype(np.float)
    M200 = data[:,3].astype(np.float)

    #_, _, _, Dn4000Med_, DnerrMed_, RclMean_ = findTrend.findBinClusters(clusters, physDict, False, dir="")

    Dn4000Med_ = np.array(pd.loadObject("Dn4000Med"))
    DnerrMed_ = np.array(pd.loadObject("Dn4000MedErr"))
    RclMean_ = np.array(pd.loadObject("RclMean"))

    redshiftBinStart = np.min(redshift)
    redshiftBinEnd = 0.31
    redshiftBinNum = 6
    redshiftBinSize = (np.max(redshift) - np.min(redshift)) / redshiftBinNum

    massBinStart = np.min(M200)
    massBinEnd = 1842480000000000.0
    massBinNum = 6
    massBinSize = (np.max(M200) - np.min(M200)) / massBinNum

    fig, axs = plt.subplots(6, 6)
    num = 0
    colName = []
    rowName = []
    for i in range(redshiftBinNum):
        for j in range(massBinNum):
            condition = np.where(np.logical_and(np.logical_and((redshiftBinStart + i * redshiftBinSize) <= redshift, (redshiftBinStart + (i + 1) * redshiftBinSize) > redshift), np.logical_and((massBinStart + j * massBinSize) <= M200, (massBinStart + (j+1) * massBinSize > M200))))
            clustersInBin = clusters[condition]
            for cluster, R200Bin, Dn4000Med, DnerrMed, RclMean in zip(clustersInBin, R200[condition], Dn4000Med_[condition], DnerrMed_[condition], RclMean_[condition]):
                num += 1
                Dn4000, Rcl, NormalizedRcl = [], [], []
                for data in physDict[cluster]:
                    if(np.float64(data[5]) > 0 and np.float64(np.float64(data[6])) < 1):
                        Dn4000.append(np.float64(data[5]))
                        Rcl.append(np.float64(data[3]))
                        NormalizedRcl.append(np.float64(data[3]) / np.float64(R200Bin))

                #axs[i,j].scatter(Rcl, Dn4000, s = 2, alpha = 0.5)
                axs[i,j].scatter(RclMean, Dn4000Med, s = 2, alpha = 0.5)
            axs[i,j].yaxis.tick_right()
            rowName.append("{:.2e}".format((massBinStart + j * massBinSize)) + " < M200 < " + "{:.2e}".format((massBinStart + (j+1) * massBinSize)))
        colName.append(str(round(redshiftBinStart + i * redshiftBinSize, 2)) + " < z < " + str(round((redshiftBinStart + (i + 1) * redshiftBinSize), 2)))
        pass

    for ax, col in zip(axs[0], colName):
        ax.set_title(col)

    for ax, row in zip(axs[:,0], rowName):
        ax.set_ylabel(row, rotation=0, size='small', labelpad=70)

    fig.text(0.5, 0.95, 'Rcl Vs Dn4000 For All Clusters', ha='center')
    fig.text(0.5, 0.04, 'Rcl', ha='center')
    fig.text(0.92, 0.5, 'Dn4000', va='center', rotation=-90)

    plt.show()
    print(num, " ", len(clusters))
