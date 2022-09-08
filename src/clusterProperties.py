import src.fits as fits
import src.calcPhysicalDist as pd
import src.findTrend as ft
import math
import numpy as np
import matplotlib.pyplot as plt
import random, time, os
from astropy.stats import bootstrap
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.stats import linregress
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting

def linear(x, a, b):
    return a * x + b

def createDataDict(data):
    clusters = data[:,0]
    redshift = data[:,1].astype(np.float)
    R200 = data[:,2].astype(np.float)
    M200 = data[:,3].astype(np.float)

    dict = {cluster : None for cluster in clusters}
    for i in range(len(clusters)):
        dict[clusters[i]] = [redshift[i], R200[i], M200[i]]

    pd.saveObject(dict, "ClusterProperties")
    print("Data Saved.")



if(__name__ == '__main__'):
    physDict = pd.loadObject("HeCS-omnibus_PhysicalDistanceData")

    data = np.loadtxt("HeCS_omnibus_prop.txt", dtype=str)
    #createDataDict(data)
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
    massBinEnd = 1.23e15
    massBinNum = 4
    massBinSize = (1.23e15 - np.min(M200)) / massBinNum

    fig, axs = plt.subplots(6, 4, sharex=True, sharey=True)
    num = 0
    colName = []
    rowName = []
    for i in range(redshiftBinNum):
        for j in range(massBinNum):
            condition = np.where(np.logical_and(np.logical_and((redshiftBinStart + i * redshiftBinSize) <= redshift, (redshiftBinStart + (i + 1) * redshiftBinSize) > redshift), np.logical_and((massBinStart + j * massBinSize) <= M200, (massBinStart + (j+1) * massBinSize > M200))))
            clustersInBin = clusters[condition]
            RclNormMeanS = []
            Dn4000MedS = []
            DnerrMedS = []
            Dn4000TotalMed, Dn4000Totalerr, RclTotalMean = [], [], []
            for cluster, R200Bin, Dn4000Med, DnerrMed, RclMean in zip(clustersInBin, R200[condition], Dn4000Med_[condition], DnerrMed_[condition], RclMean_[condition]):
                num += 1
                Dn4000, Rcl, NormalizedRcl = [], [], []
                for data in physDict[cluster]:
                    if(np.float64(data[5]) > 0 and np.float64(np.float64(data[6])) < 1):
                        Dn4000.append(np.float64(data[5]))
                        Rcl.append(np.float64(data[3]))
                        NormalizedRcl.append(np.float64(data[3]) / np.float64(R200Bin))

                #axs[i,j].scatter(Rcl, Dn4000, s = 2, alpha = 0.5)
                RclNormMean = RclMean / np.float64(R200Bin)
                for k in range(len(RclNormMean)):
                    if (RclNormMean[k] < 1.0):
                        RclNormMeanS.append(RclNormMean[k])
                        DnerrMedS.append(DnerrMed[k])
                        Dn4000MedS.append(Dn4000Med[k])
                for r in range(len(RclMean)):
                    RclTotalMean.append(RclMean[r] / np.float64(R200Bin))
                    Dn4000TotalMed.append(Dn4000Med[r])
                    Dn4000Totalerr.append(DnerrMed[r])

                axs[i,j].scatter(RclNormMean, Dn4000Med, s = 2, alpha = 0.5)
                axs[i,j].set_ylim([0.9,2.3])
                axs[i,j].set_xlim([0,5*np.float64(R200Bin)])

            if (len(RclTotalMean) != 0):
                yy, ye, xx = ft.getMedDn4000(RclTotalMean, Dn4000TotalMed, Dn4000Totalerr)
                yyS, yeS, xxS = [], [], []
                for k in range(len(xx)):
                    if (xx[k] < 1.0):
                        yyS.append(yy[k])
                        yeS.append(ye[k])
                        xxS.append(xx[k])

                """ Code For Find On Median Data Point"""
                if(len(yyS) != 0):
                    popt, pcov = curve_fit(linear,xxS, yyS, sigma=yeS)
                    err = np.sqrt(pcov.diagonal())
                    #print("Parameters:", popt)
                    #print("Errors    :", err)
                    #print("Slope:", res.slope," Intercept: ", res.intercept)
                    x = np.linspace(0,1, 1000)
                    y = popt[1] + popt[0]*x
                    #axs[i,j].plot(x, y, c='black', label="y = ({slope:.2f} +/- {serr:.2f})x + ({intercept:.2f} +/- {ierr:.2f})".format(slope = popt[0], serr= err[0], intercept = popt[1], ierr=err[1]))
                    #axs[i,j].legend(loc="bottom")

                    axs[i,j].scatter(xx, yy, s = 4, alpha = 1, c='r')
                    axs[i,j].errorbar(xx, yy, yerr=ye, alpha = 1, c='r', linestyle='none')

            if(len(RclNormMeanS) != 0):
                """ Code For Find On All Data"""
                popt, pcov = curve_fit(linear,RclNormMeanS, Dn4000MedS)
                err = np.sqrt(pcov.diagonal())
                #print("Slope:", res.slope," Intercept: ", res.intercept)
                x = np.linspace(0,1, 1000)
                y = popt[1] + popt[0]*x
                axs[i,j].plot(x, y, c='black', label="y = ({slope:.2f} +/- {serr:.2f})x + ({intercept:.2f} +/- {ierr:.2f})".format(slope = popt[0], serr= err[0], intercept = popt[1], ierr=err[1]))
                axs[i,j].legend(loc="bottom")
            #axs[i,j].yaxis.tick_right()
            colName.append("{:.2e}".format((massBinStart + j * massBinSize)) + " < M200 < " + "{:.2e}".format((massBinStart + (j+1) * massBinSize)))
        rowName.append(str(round(redshiftBinStart + i * redshiftBinSize, 2)) + " < z < " + str(round((redshiftBinStart + (i + 1) * redshiftBinSize), 2)))
        pass
    print(len(clusters), " ", len(fits.clusters))
    for ax, col in zip(axs[0], colName):
        ax.set_title(col, size='small')

    for ax, row in zip(axs[:,0], rowName):
        ax.set_ylabel(row, rotation=0, size='small', labelpad=-1150)

    fig.text(0.5, 0.95, 'Normalized Rcl Vs Dn4000 For All Clusters (Linear fit between 0 and 1.0)', ha='center')
    fig.text(0.5, 0.04, 'Normalized Rcl', ha='center')
    fig.text(0.04, 0.5, 'Dn4000', va='center', rotation=90)

    plt.show()
    print(num, " ", len(clusters))
