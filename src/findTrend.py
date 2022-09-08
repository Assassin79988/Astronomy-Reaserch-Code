import src.fits as fits
import src.calcPhysicalDist as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random, time, os
from astropy.stats import bootstrap
from scipy.stats import spearmanr, pearsonr, kendalltau


def getMedDn4000(Rcl, Dn4000, Dnerr, useBoot=True):
        Rcl = np.array(Rcl, dtype= np.float64)
        Dn4000 = np.array(Dn4000, dtype= np.float64)
        Dnerr = np.array(Dnerr, dtype = np.float64)

        #binSize = 200 * 0.001# Mpc
        binSize = 200 * 0.001# Mpc
        RclMax = np.max(Rcl)
        binNum = int(math.ceil(RclMax/ binSize))

        indexBins = [np.where(np.logical_and(Rcl > i * binSize,  Rcl < (i+1) * binSize))  for i in range(binNum)]

        Dn4000Med = []
        DnerrMed = []
        RclMean = []
        for i in range(len(indexBins)):
            if (Dn4000[indexBins[i]].size != 0):
                Dn4000Med.append(np.median(Dn4000[indexBins[i]]))
            if (Dn4000[indexBins[i]].size > 1):
                err = []
                if(useBoot):
                    err = np.std(bootstrap(Dn4000[indexBins[i]], 500, bootfunc = np.median))
                else:
                    err = np.std(Dn4000[indexBins[i]])
                DnerrMed.append(err)
            elif (Dn4000[indexBins[i]].size != 0):
                    DnerrMed.append(Dnerr[indexBins[i]][0])
            else:
                pass
            if (Dn4000[indexBins[i]].size != 0):
                RclMean.append(np.mean(Rcl[indexBins[i]]))

        return Dn4000Med, DnerrMed, RclMean

"""
Name:

Description:

Parameter:

Return:
"""
def findBinClusters(physDict, makePlot, dir=""):
    Dn4000Med_, DnerrMed_, RclMean_ = [],  [], []
    Dn4000_, Dnerr_, Rcl_ = [],  [], []
    clusters = []
    R200 = []
    clusterProperties = pd.loadObject("ClusterProperties")
    for cluster in fits.clusters:
        if cluster not in physDict:
            continue
        clusters.append(cluster)
        Dn4000 = []
        Dnerr = []
        Rcl = []

        for data in physDict[cluster]:
            if(np.float64(data[5]) > 0 and np.float64(np.float64(data[6])) < 1):
                Dn4000.append(np.float64(data[5]))
                Dnerr.append(np.float64(data[6]))
                Rcl.append(np.float64(data[3]))

        Rcl = np.array(Rcl, dtype= np.float64)
        Dn4000 = np.array(Dn4000, dtype= np.float64)
        Dnerr = np.array(Dnerr, dtype = np.float64)

        #binSize = 200 * 0.001# Mpc
        binSize = 100 * 0.001# Mpc
        RclMax = np.max(Rcl)
        binNum = int(math.ceil(RclMax/ binSize))

        indexBins = [np.where(np.logical_and(Rcl > i * binSize,  Rcl < (i+1) * binSize))  for i in range(binNum)]

        Dn4000Med = []
        DnerrMed = []
        RclMean = []
        for i in range(len(indexBins)):
            if (Dn4000[indexBins[i]].size != 0):
                Dn4000Med.append(np.median(Dn4000[indexBins[i]]))
            if (Dn4000[indexBins[i]].size > 1):
                err = np.std(bootstrap(Dn4000[indexBins[i]], 500, bootfunc = np.median))
                DnerrMed.append(err)
            elif (Dn4000[indexBins[i]].size != 0):
                DnerrMed.append(Dnerr[indexBins[i]][0])
            else:
                pass
            if (Dn4000[indexBins[i]].size != 0):
                RclMean.append(np.mean(Rcl[indexBins[i]]))

        Dn4000Med_.append(Dn4000Med)
        DnerrMed_.append(DnerrMed)
        RclMean_.append(RclMean)
        Dn4000_.append(Dn4000)
        Dnerr_.append(Dnerr)
        Rcl_.append(Rcl)
        R200.append(clusterProperties[cluster][1])

        if(makePlot):
            fig = plt.figure(figsize=(20,10))
            plt.scatter(Rcl, Dn4000, s = 2, alpha = 0.5, c = 'r')
            plt.scatter(RclMean, Dn4000Med, s = 2)
            plt.errorbar(RclMean, Dn4000Med, yerr = DnerrMed, linestyle = 'None')
            plt.xlabel("Rcl (Mpc)")
            plt.ylabel("Dn4000")
            plt.title("Dn4000 Vs Rcl for {clust}".format(clust = cluster))
            #plt.figtext(0.02, 0.5, ps + '\n\n' + ss + '\n\n' + ks, fontsize = 14)
            plt.subplots_adjust(left=0.25)
            plt.xlim(0,1)
            #plt.savefig(dir)
            #plt.cla()
            plt.show()
            plt.close()

    pd.saveObject(Dn4000_, "Dn4000WithTrendUsingMed")
    pd.saveObject(Dnerr_, "Dn4000ErrWithTrendUsingMed")
    pd.saveObject(Rcl_, "RclWithTrendUsingMed")
    pd.saveObject(Dn4000Med_, "Dn4000MedWithTrendUsingMed")
    pd.saveObject(DnerrMed_, "Dn4000MedErrWithTrendUsingMed")
    pd.saveObject(RclMean_, "RclMeanWithTrendUsingMed")
    pd.saveObject(clusters, "ClustersWithTrendUsingMed")
    pd.saveObject(R200, "R200WithTrendUsingMed")
    print("Data Saved")
    return Dn4000_, Dnerr_, Rcl_, Dn4000Med_, DnerrMed_, RclMean_

"""
Name:

Description:

Parameter:

Return:
"""
def findClustersWithTrend(physDict, maxDnError = 1, conditon = [0,1], correlationFunc = spearmanr, useMed = True):
    clustersWithTrend,clustersWithoutTrend  = [], []
    # keeps track of how many cluster rejected the null hypothesis and how many didn't
    accpeted, rejected = 0, 0
    for cluster in fits.clusters:
        Dn4000, Dnerr,Rcl = [], [], []
        for data in physDict[cluster]:
            if(np.float64(data[5]) > 0 and np.float64(np.float64(data[6])) < maxDnError):
                Dn4000.append(np.float64(data[5]))
                Dnerr.append(np.float64(data[6]))
                Rcl.append(np.float64(data[3]))

        if(useMed):
            Dn4000, Dnerr, Rcl = getMedDn4000(Rcl, Dn4000, Dnerr)

        Dn4000 = np.array(Dn4000)
        Dnerr = np.array(Dnerr)
        Rcl = np.array(Rcl)

        indices = np.where(np.logical_and(Rcl > conditon[0], Rcl < conditon[1]))
        coeff, pValue = correlationFunc(Rcl[indices], Dn4000[indices])

        if (pValue < 0.05):
            clustersWithTrend.append(cluster)
            rejected += 1
        else:
            clustersWithoutTrend.append(cluster)
            accpeted += 1

    createCatalogue(physDict, clustersWithTrend, "clustersWithTrend" + ("UsingMed" if(useMed) else ""))
    createCatalogue(physDict, clustersWithoutTrend, "clustersWithoutTrend" + ("UsingMed" if(useMed) else ""))

    return clustersWithTrend, clustersWithoutTrend, accpeted, rejected

def createCatalogue(fullCatalogue, subClusters, name):
    subCatalogue = {cluster: None for cluster in subClusters}
    for cluster in subClusters:
        subCatalogue[cluster] = fullCatalogue[cluster]

    pd.saveObject(subCatalogue, name)


def plotClustersWithTrend():
    physDict = pd.loadObject("clustersWithTrendUsingMed")
    findBinClusters(physDict, False, dir="")

    Dn4000_ = pd.loadObject("Dn4000WithTrendUsingMed")
    Dnerr_ = pd.loadObject("Dn4000ErrWithTrendUsingMed")
    Rcl_ = pd.loadObject("RclWithTrendUsingMed")
    Dn4000Med_ = pd.loadObject("Dn4000MedWithTrendUsingMed")
    DnerrMed_ = pd.loadObject("Dn4000MedErrWithTrendUsingMed")
    RclMean_ = pd.loadObject("RclMeanWithTrendUsingMed")
    clusters = pd.loadObject("clustersWithTrendUsingMed")
    R200 = pd.loadObject("R200WithTrendUsingMed")
    i = 0
    j = 0
    n = 1;
    fig, axs = plt.subplots(5, 5, sharex=True, sharey=True)
    for cluster, Dn4000, Dnerr, Rcl, Dn4000Med, DnerrMed, RclMean, r200 in zip(clusters, Dn4000_, Dnerr_, Rcl_, Dn4000Med_, DnerrMed_, RclMean_, R200):

        axs[i,j].scatter(Rcl / r200, Dn4000, s = 2, alpha = 0.5, c = 'r')
        axs[i,j].scatter(RclMean / r200, Dn4000Med, s = 2)
        axs[i,j].errorbar(RclMean / r200, Dn4000Med, yerr = DnerrMed, linestyle = 'None')
        axs[i,j].set_title(cluster)
        #axs[i,j].set_xlim([0,2])
        axs[i,j].set_ylim([0.9,2.3])
        if( i == 4 and j == 4):
            i = 0
            j = 0
            fig.text(0.5, 0.95, 'Rcl Normalized Vs Dn4000 For All Clusters With Correlation Detected With Kendalltau', ha='center')
            fig.text(0.5, 0.04, 'Rcl Normalized', ha='center')
            fig.text(0.04, 0.5, 'Dn4000', va='center', rotation=90)
            plt.savefig("./2MpcHasTrend" + str(n) + ".png")
            plt.cla()
            plt.close()
            n += 1
            fig, axs = plt.subplots(5, 5, sharex=True, sharey=True)
        elif(j == 4):
            j = 0
            i += 1
        else:
            j += 1
    fig.text(0.5, 0.95, 'Rcl Normalized Vs Dn4000 For All Clusters With Correlation Detected With Kendalltau', ha='center')
    fig.text(0.5, 0.04, 'Rcl Normalized', ha='center')
    fig.text(0.04, 0.5, 'Dn4000', va='center', rotation=90)
    plt.savefig("./2MpcHasTrend" + str(n) + ".png")
    plt.cla()

def plotClustersWithoutTrend():
    physDict = pd.loadObject("clustersWithoutTrendUsingMed")
    #findBinClusters(physDict, False, dir="")

    Dn4000_ = pd.loadObject("Dn4000WithTrendUsingMed")
    Dnerr_ = pd.loadObject("Dn4000ErrWithTrendUsingMed")
    Rcl_ = pd.loadObject("RclWithTrendUsingMed")
    Dn4000Med_ = pd.loadObject("Dn4000MedWithTrendUsingMed")
    DnerrMed_ = pd.loadObject("Dn4000MedErrWithTrendUsingMed")
    RclMean_ = pd.loadObject("RclMeanWithTrendUsingMed")
    clusters = pd.loadObject("clustersWithTrendUsingMed")
    R200 = pd.loadObject("R200WithTrendUsingMed")
    i = 0
    j = 0
    n = 1
    fig, axs = plt.subplots(5, 5, sharex=True, sharey=True)
    #fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
    for cluster, Dn4000, Dnerr, Rcl, Dn4000Med, DnerrMed, RclMean, r200 in zip(clusters, Dn4000_, Dnerr_, Rcl_, Dn4000Med_, DnerrMed_, RclMean_, R200):
        """
        if (cluster == 'A2197' or cluster == 'A2124' or cluster == 'A2670' or cluster == 'A1773' or cluster == 'A329' or cluster == 'RMJ0041p2526'):
            axs[i,j].scatter(Rcl, Dn4000, s = 2, alpha = 0.5, c = 'r')
            axs[i,j].scatter(RclMean, Dn4000Med, s = 2)
            axs[i,j].errorbar(RclMean, Dn4000Med, yerr = DnerrMed, linestyle = 'None')
            axs[i,j].set_title(cluster)
            axs[i,j].set_xlim([0,2])
            axs[i,j].set_ylim([0.9,2.3])
            print(cluster)
            if( i == 2 and j == 1):
                i = 0
                j = 0
                fig.text(0.5, 0.95, 'Rcl Vs Dn4000 For Dropped Clusters with Kendalltau From 1 Mpc to 2 Mpc', ha='center')
                fig.text(0.5, 0.04, 'Rcl', ha='center')
                fig.text(0.04, 0.5, 'Dn4000', va='center', rotation=90)
                plt.show()
                plt.close()
                fig, axs = plt.subplots(5, 5, sharex=True, sharey=True)
            elif(j == 1):
                        j = 0
                        i += 1
            else:
                j += 1

        """
        #print("Here")
        #print(Dn4000)
        axs[i,j].scatter(Rcl / r200, Dn4000, s = 2, alpha = 0.5, c = 'r')
        axs[i,j].scatter(RclMean / r200, Dn4000Med, s = 2)
        axs[i,j].errorbar(RclMean / r200, Dn4000Med, yerr = DnerrMed, linestyle = 'None')
        axs[i,j].set_title(cluster)
        #axs[i,j].set_xlim([0,2])
        axs[i,j].set_ylim([0.9,2.3])

        if( i == 4 and j == 4):
            i = 0
            j = 0
            fig.text(0.5, 0.95, 'Rcl Normalized Vs Dn4000 For All Clusters With No Correlation Detected With Kendalltau', ha='center')
            fig.text(0.5, 0.04, 'Rcl Normalized', ha='center')
            fig.text(0.04, 0.5, 'Dn4000', va='center', rotation=90)
            plt.savefig("./2MpcNoTrend" + str(n) + ".png")
            plt.cla()
            plt.close()
            n += 1
            fig, axs = plt.subplots(5, 5, sharex=True, sharey=True)
        elif(j == 4):
            j = 0
            i += 1
        else:
            j += 1

    fig.text(0.5, 0.95, 'Rcl Normalized Vs Dn4000 For All Clusters With No Correlation Detected With Kendalltau', ha='center')
    fig.text(0.5, 0.04, 'Rcl Normalized', ha='center')
    fig.text(0.04, 0.5, 'Dn4000', va='center', rotation=90)
    plt.savefig("./2MpcNoTrend" + str(n) + ".png")
    plt.cla()
    plt.close()


if __name__ == '__main__':
        physDict = pd.loadObject("HeCS-omnibus_PhysicalDistanceData")
        #clustersWithTrend1, clustersWithoutTrend1, accpeted1, rejected1 = findClustersWithTrend(physDict, 1, [0,1], kendalltau)
        #clustersWithTrend2, clustersWithoutTrend2, accpeted2, rejected2 = findClustersWithTrend(physDict, 1, [0,2], kendalltau)
        #clustersWithTrendFull, clustersWithoutTrendFull, accpetedFull, rejectedFull = findClustersWithTrend(physDict, 1, [0,1000000], kendalltau)

        One_Two_Lose = []
        One_Full_Lose = []
        #print(clustersWithTrend1)
        #print(clustersWithTrend2)
        #for cluster in clustersWithTrend1:
        #    isIn = False
        #    for clusterFull in clustersWithTrendFull:
            #    if (cluster == clusterFull):
            #        isIn = True;
            #if (isIn == False):
            #    print(cluster)

        clustersWithTrend, clustersWithoutTrend, accpeted, rejected = findClustersWithTrend(physDict, 1, [0,2], kendalltau, False)
        print(accpeted, " Does not have trend.")
        print(rejected, " Does have trend.")
        #plotClustersWithTrend()
        #plotClustersWithoutTrend()
        #findBinClusters(physDict, 1, False, dir="")

        #plotClustersWithTrend()
        #plotClustersWithoutTrend()


        """
        ***************************************************************************
                                        OLD CODE BELOW
        ***************************************************************************

        _Trend = pd.findJPEG("./Rcl_Visual_Trend")
        Trend = []
        for string in _Trend:
            temp = string.replace("./Rcl_Visual_Trend/", "")
            Trend.append(temp.replace(".jpeg", ""))

        _noTrend = pd.findJPEG("./Rcl_Not_Visual_Trend'/")
        noTrend = []
        for string in _noTrend:
            temp = string.replace("./Rcl_Not_Visual_Trend'/", "")
            noTrend.append(temp.replace(".jpeg", ""))


        print("Number of clusters where I saw a trend: ", len(Trend))
        print("Number of clusters where I didn't see a trend: ", len(noTrend))
        """
        """"
        #Dn4000 Med Plots
        #fig = plt.figure(figsize=(20,10))
        for cluster in fits.clusters:
            Dn4000 = []
            Dnerr = []
            Rcl = []

            for data in physDict[cluster]:
                if(np.float64(data[5]) > 0 and np.float64(np.float64(data[6])) < 1):
                    Dn4000.append(np.float64(data[5]))
                    Dnerr.append(np.float64(data[6]))
                    Rcl.append(np.float64(data[3]))

            Rcl = np.array(Rcl, dtype= np.float64)
            Dn4000 = np.array(Dn4000, dtype= np.float64)
            Dnerr = np.array(Dnerr, dtype = np.float64)

            #binSize = 200 * 0.001# Mpc
            binSize = 400 * 0.001# Mpc
            RclMax = np.max(Rcl)
            binNum = int(math.ceil(RclMax/ binSize))

            indexBins = [np.where(np.logical_and(Rcl > i * binSize,  Rcl < (i+1) * binSize))  for i in range(binNum)]

            Dn4000Med = []
            DnerrMed = []
            RclMean = []
            for i in range(len(indexBins)):
                if (Dn4000[indexBins[i]].size != 0):
                    Dn4000Med.append(np.median(Dn4000[indexBins[i]]))
                    if (Dn4000[indexBins[i]].size > 1):
                        err = np.std(bootstrap(Dn4000[indexBins[i]], 500, bootfunc = np.median))
                        DnerrMed.append(err)
                    else:
                        DnerrMed.append(Dnerr[indexBins[i]][0])
                    RclMean.append(np.mean(Rcl[indexBins[i]]))
            #print(Rcl[indexBins[-1]])
            #print(RclMean)
            #X.append(np.array(RclMean))
            #Y.append(np.array(Dn4000Med))
            #Yerr.append(np.array(DnerrMed))
            ""
            coef, p = spearmanr(Rcl, Dn4000)
            One = np.where(Rcl < 1)
            Two = np.where(Rcl < 2)
            Four = np.where(Rcl < 4)

            o,po = pearsonr(Rcl[One], Dn4000[One])
            t, pt = pearsonr(Rcl[Two], Dn4000[Two])
            a, pa = pearsonr(Rcl, Dn4000)

            _o,_po = spearmanr(Rcl[One], Dn4000[One])
            _t, _pt = spearmanr(Rcl[Two], Dn4000[Two])
            _a, _pa = spearmanr(Rcl, Dn4000)

            __o,__po = kendalltau(Rcl[One], Dn4000[One])
            __t, __pt = kendalltau(Rcl[Two], Dn4000[Two])
            __a, __pa = kendalltau(Rcl, Dn4000)

            ps = '< 1 MeV Person-r:%.2f\tp-value:%.2f\n < 2 MeV Person-r:%.2f\tp-value:%.2f\nPerson-r:%.2f\tp-value:%.2f'%(o, po, t, pt, a, pa)
            ss = '< 1 MeV Spearman-r:%.2f\tp-value:%.2f\n < 2 MeV Spearman-r:%.2f\tp-value:%.2f\nSpearman-r:%.2f\tp-value:%.2f'%(_o, _po, _t, _pt, _a, _pa)
            ks = '< 1 MeV Kendalltau:%.2f\tp-value:%.2f\n < 2 MeV Kendalltau:%.2f\tp-value:%.2f\nKendalltau:%.2f\tp-value:%.2f'%(__o, __po, __t, __pt, __a, __pa)
            ""
            S.append(spearmanr(Rcl, Dn4000)[0])
            Sp.append(spearmanr(Rcl, Dn4000)[1])
            P.append(pearsonr(Rcl, Dn4000)[0])
            Pp.append(pearsonr(Rcl, Dn4000)[1])
            K.append(kendalltau(Rcl, Dn4000)[0])
            Kp.append(kendalltau(Rcl, Dn4000)[1])

            #print("\n\n\n\n\n\n\n\n\n")
            #print("--------------------- < 1MeV ----------------------")
            #print("Spearmanr: ",spearmanr(Rcl[One], Dn4000[One]))
            #print("Pearsonr: ", pearsonr(Rcl[One], Dn4000[One]))
            #print("kendalltau: ", kendalltau(Rcl[One], Dn4000[One]))

            #print("---------------------- < 2 MeV --------------------")
            #print("Spearmanr: ",spearmanr(Rcl[Two], Dn4000[Two]))
            #print("Pearsonr: ", pearsonr(Rcl[Two], Dn4000[Two]))
            #print("kendalltau: ", kendalltau(Rcl[Two], Dn4000[Two]))

            #print("---------------------- < 4 MeV --------------------")
            #print("Spearmanr: ",spearmanr(Rcl[Four], Dn4000[Four]))
            #print("Pearsonr: ", pearsonr(Rcl[Four], Dn4000[Four]))
            #print("kendalltau: ", kendalltau(Rcl[Four], Dn4000[Four]))

            fig = plt.figure(figsize=(20,10))
            plt.scatter(Rcl, Dn4000, s = 2, alpha = 0.5, c = 'r')
            plt.scatter(RclMean, Dn4000Med, s = 2)
            try:
                plt.errorbar(RclMean, Dn4000Med, yerr = DnerrMed, linestyle = 'None')
            except:
                print("\n\n\nERROR HAD OCCURED\n\n\n")
                break
            #plt.scatter(Rcl, Dn4000, s = 2, c = 'r')
            #plt.show()
            plt.xlabel("Rcl (Mpc)")
            plt.ylabel("Dn4000")
            plt.title("Dn4000 Vs Rcl for {clust}".format(clust = cluster))
            plt.figtext(0.02, 0.5, ps + '\n\n' + ss + '\n\n' + ks, fontsize = 14)
            plt.subplots_adjust(left=0.25)
            if (po > 0.05):
                accept += 1
                #print("Null accepted")
                #print("P-Value: ", po)
                #plt.show()
            else:
                reject += 1
                #print("Null rejected")
                #plt.cla()

            #plt.savefig("Rcl/" + cluster + ".jpeg")
            #plt.cla()
            #plt.close()
            plt.show()
            plt.close()

        print("Null Accepted: ",accept)
        print("Null Rejected: ",reject)
        fig, ax = plt.subplots(1,2)
        fig.suptitle("Spearman-r (All Data)")
        ax[0].hist(S)
        ax[1].hist(Sp)
        ax[0].set(xlabel = "Correlation coefficient")
        ax[1].set(xlabel="P-Value")
        plt.show()
        fig, ax = plt.subplots(1,2)
        fig.suptitle("Kendall-tau (All Data)")
        ax[0].hist(K)
        ax[1].hist(Kp)
        ax[0].set(xlabel = "Correlation coefficient")
        ax[1].set(xlabel="P-Value")
        plt.show()
        fig, ax = plt.subplots(1,2)
        fig.suptitle("Pearson-r (All Data)")
        ax[0].hist(P)
        ax[1].hist(Pp)
        ax[0].set(xlabel = "Correlation coefficient")
        ax[1].set(xlabel="P-Value")
        plt.show()

        ""
        X = np.array(X, dtype='object')
        Y = np.array(Y, dtype='object')
        Yerr = np.array(Yerr, dtype='object')

        X = X.T
        Y = Y.T
        Yerr = Yerr.T

        RclMean = []
        Dn4000Med = []
        DnerrMed = []
        for i in range(len(X)):
            print(X.T[i])
            RclMean.append(np.mean(X[i]))
            Dn4000Med.append(np.median(Y[i]))
            DnerrMed.append(np.std(bootstrap(Y[i], 500, bootfunc = np.median)))

        plt.scatter(RclMean, Dn4000Med, s = 2)
        plt.errorbar(RclMean, Dn4000Med, yerr=DnerrMed, linestyle='None')
        plt.show()
        """
        """
        plt.errorbar(Rcl, Dn4000, yerr = Dnerr, linestyle = 'None')
        plt.xlabel("Rcl (Mpc)")
        plt.ylabel("Dn4000")
        plt.title("Dn4000 vs Rcl for {clust}".format(clust = cluster))
        plt.savefig("Rcl/" + cluster + ".jpeg")
        plt.cla()
        ""

        ""
        #    Dn4000 vs Rcl Plot Code

        for cluster in fits.clusters:
            Dn4000 = []
            Dnerr = []
            Rcl = []
            for data in physDict[cluster]:
                if(np.float64(data[5]) > 0 and np.float64(data[6]) < 1):
                    Dn4000.append(np.float64(data[5]))
                    Dnerr.append(np.float64(data[6]))
                    Rcl.append(np.float64(data[3]))
            plt.scatter(Rcl, Dn4000, s = 2)
            plt.errorbar(Rcl, Dn4000, yerr = Dnerr, linestyle = 'None')
            plt.xlabel("Rcl (Mpc)")
            plt.ylabel("Dn4000")
            plt.title("Dn4000 vs Rcl for {clust}".format(clust = cluster))
            plt.savefig("Rcl/" + cluster + ".jpeg")
            plt.cla()
            #plt.show()
        ""

        ""
        Comparsion Physical Distance Plot Code

        physDists = np.array(physDists)
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        ax[0].scatter(physDists[:,0], physDists[:,1], s = 0.5)
        ax[0].set_ylabel("Jubee's Physical Distance (Mpc)")
        ax[1].scatter(physDists[:,0], physDists[:,2], s = 0.5)
        ax[1].set_ylabel("Difference (Mine - Jubee's)")
        fig.text(0.5, 0.04, 'My Physical Distance (Mpc)', ha='center', va='center')
        plt.suptitle("Mine Vs. Jubee's Physical Distance", fontsize = 20)
        plt.show()
        ""
        """
