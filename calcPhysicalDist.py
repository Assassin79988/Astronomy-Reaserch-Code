import fits
import math
from cosmocalc import cosmocalc
import numpy as np
import matplotlib.pyplot as plt
import random, time, os
from astropy.stats import bootstrap
from scipy.stats import spearmanr, pearsonr, kendalltau
import pickle

"""
Name: bootstrap(x,N,statfunc = None)
Description: Resamples a data set x N times using the bootstrap method. If user
              specify then this rountine can also compute a statistic for each resmapled set.
Parameter: x - original data set
            N - number of resamples
            statfunc - function that computes a statistic (if not specified the rountine will not compute a statistic
                       and just return the resamples a data sets)
Return: The resamples data sets if statfunc is not specified otherwise return the resamples a data sets   
def bootstrap(x,N,statfunc = None):
    # seeds the random function
    random.seed(time.gmtime())
    bootarr = [] # arr to store each resmaple set 
    # resmaples the data set, x , N times
    for i in range(N):
        temp = []
        # picks creates a new sample (same size as x) by randomly picking data from set x 
        for j in range(len(x)):
            rand = random.randint(0, len(x)-1)
            temp.append(x[rand])
        bootarr.append(temp) # append the current resample to bootarr
    # checks if user wants compute a statistic with the resmample data 
    if(statfunc != None):
        stat = [] # arr to store the statistic for each resmaple set 
        for i in range(N):
            stat.append(statfunc(bootarr[i]))
        return stat 
    else:
        return bootarr
"""

"""
Name: 
    greatCircleDist(RA1, Dec1, RA2, Dec2, units = 'r')
Description: 
    This routine determines the distance between two points on a great circle
    given there Right Ascension (R.A.) and Declination (Dec.). All outputs will be in units of
    radians.
Parameters:
        RA1   - R.A. of the first point 
        Dec1  - Dec. of the first point
        RA2   - R.A. of the second point
        Dec2  - Dec. of the second point
        units - units of the input (default is r)
            r - radians
            d - degrees
Return: Angular Distance between the two points in radians
"""
def greatCircleDist(RA1, Dec1, RA2, Dec2, units = 'r'):
    # convert the units to radians if need be
    if (units == 'd'):
        RA1 = math.radians(RA1)
        Dec1 = math.radians(Dec1)
        RA2 = math.radians(RA2)
        Dec2 = math.radians(Dec2)
    
    dist = 2.0 * math.asin(math.sqrt((math.sin((Dec2 - Dec1)/2.0)**2) + math.cos(Dec1) * math.cos(Dec2) * (math.sin((RA2 - RA1)/2.0)**2)))
    return dist

"""
Name:
    planeDist(RA1, Dec1, RA2, Dec2, units = 'r')
Description:
    This routine find the angualr distance between two points on a sphere using a 
    planar approx. NOTE: only accurate for small angular distances
Parameters:
    RA1   - R.A. of the first point
    Dec1  - Dec. of the first point
    RA2   - R.A. of the second point
    Dec2  - Dec. of the second point
    units - The units of the R.A. and Dec. of the two points (Default = 'r')
        'r' - radians
        'd' - degrees
Return:
    The Angular distance between the two points.
"""
def planeDist(RA1, Dec1, RA2, Dec2, units = 'r'):
    # converts the units to radians if need be
    if (units == 'd'):
        RA1 = math.radians(RA1)
        Dec1 = math.radians(Dec1)
        RA2 = math.radians(RA2)
        Dec2 = math.radians(Dec2)

    dist = math.sqrt(((RA1 - RA2) * math.cos(Dec1))**2 + (Dec1 - Dec2)**2)
    return dist

"""
Name:
    calcPhysicalDist(RA1, Dec1, RA2, Dec2, z, projDistMethod = 'gc', units = 'r', _H0 = 70, _WM = 0.3, _WV = 0.7, _WK = 0.0)
Description: 
    This routine finds the physical distance between an object and the supposed cluster center.
    cosmocalc is used to find the platescale (kpc / arcseconds).
Parameters:
    Required Parameters:
        RA1              - R.A. of the first object
        Dec1             - Dec. of the first object
        RA2              - R.A. of the clutser center
        Dec2             - Dec. of the cluster center
        z                - redshift of the cluster
    Optional Parameters:
        projDistMethod   - Method used to compute the projected angular distace (Default = 'gc')
                            'gc' - finds angualar distance using Haversine formula
                            'p' - planar aproxmations (small angular distances)
        units            - units of R.A.'s and Dec's (Default = 'r')
                            'r' - radians
                            'd' - degrees
        _H0              - Hubble constant 
        _WM              - Omega matter
        _WV              - Omega vacuum
        _WK              - Omega curvaturve
Return:
    The physical distance between the two objects of interest in Mpc.
"""
def calcPhysicalDist(RA1, Dec1, RA2, Dec2, z, projDistMethod = 'gc', units= 'r', _H0 = 70, _WM = 0.3, _WV = 0.7, _WK = 0.0):
    # Converts units to radians if need be
    projDist = 0
    if (projDistMethod == 'gc'):
        projDist = greatCircleDist(RA1, Dec1, RA2, Dec2, units)
    else:
        projDist = planeDist(RA1, Dec1, RA2, Dec2, untis)

    # Converts from radians to arcseconds
    projDist *= 206265
    
    # Finds the plate scale
    plateScale = cosmocalc(z, H0 = _H0, WV = _WV, WM = _WM )['PS_kpc']

    # converts angular distnace to a physical distance in kpc
    physDist = plateScale * projDist

    # Returns the physial distance in Mpc
    return physDist / 1000


def findJPEG(directory = "./", option = 's'):

    # Checks if directory is stored as a list
    if type(directory) != type([]):
        directory = [directory]

    # init the array FITS file paths
    filePaths = []
    
    # Finds all sub directories 
    if option == 'r':
        parnetDir = copy.deepcopy(directory)

        for pDir in parnetDir:
            directory.extend(fastScanDir(pDir))

    # Finds all FITS files in the specified directories
    for path in directory:
        for file in os.listdir(path):
            if file.endswith(".jpeg"):
                filePaths.append(os.path.join(path, file))

    return filePaths

def getPhysicalDistDict():
    fileDir = "../HeCS_omnibus_dn4000.fits"
    data = fits.fetchFITS(fileDir, 's')

    clusterDict = fits.getClusterDict(data)
    physDict = {cluster: None for cluster in fits.clusters} 

    centerData = np.loadtxt("../HeCS_omnibus_cluster_center.txt", dtype='str')

    physDists = []
    for i in range(len(centerData[:,0])):
        temp = []
        for cluster in clusterDict[centerData[i,0]]:

            physDist = calcPhysicalDist(cluster[3], cluster[4], float(centerData[i,1]), float(centerData[i,2]),float(centerData[i,3]), units = 'd')
            temp.append([cluster[0], cluster[2], physDist, cluster[5], physDist - cluster[5], cluster[21], cluster[22]])
            physDists.append([physDist, cluster[5], physDist - cluster[5]])

        physDict[centerData[i,0]] = np.array(temp)
        print(clusterDict[centerData[i,0]][0][0], "Done.")

        return physDict


if __name__ == '__main__':
    
    physDict = getPhysicalDict()

    _Trend = findJPEG("./Rcl_Visual_Trend")
    Trend = []
    for string in _Trend:
        temp = string.replace("./Rcl_Visual_Trend/", "")
        Trend.append(temp.replace(".jpeg", ""))

    _noTrend = findJPEG("./Rcl_Not_Visual_Trend'/")
    noTrend = []
    for string in _noTrend:
        temp = string.replace("./Rcl_Not_Visual_Trend'/", "")
        noTrend.append(temp.replace(".jpeg", ""))

    
    print(len(Trend))
    print(len(noTrend))
    print(len(fits.clusters))

    
    #Dn4000 Med Plots

    accept = 0
    reject = 0

    S, Sp = [], []
    P, Pp = [], []
    K, Kp = [], []
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

        plt.savefig("Rcl/" + cluster + ".jpeg")
        plt.cla()
        plt.close()
        #plt.show()
        
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
    
    """
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
    """

    """
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
    """

    """
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
    """
    





