import src.fits as fits
import src.calcPhysicalDist as pd
import src.findTrend as ft
import math
from threading import Thread, Lock
import numpy as np
import progressbar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random, time, os, copy
import pickle
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import iqr
from scipy.signal import find_peaks, argrelextrema
import pandas as pd
import loess
from loess.loess_2d import loess_2d
from plotbin.plot_velfield import plot_velfield
from astropy.stats import scott_bin_width

"""
*****************************************************************************
                                   Parameters
*****************************************************************************
"""
# Constants
#============================================================================
INT_MAX = 10000

# Settings
#============================================================================
showProgressBars = True                                       # Show the progress of tasks
generateCompletenessPlots = False                             # Plot completeness plot as a function of r-band mag
generateDnHist1D = False                                      #
generateDnHist2D = False                                      #
generateKDEPlots = False                                      #
generateDnWithCorrectionHist2D = False                        #
generateMaguntudeLimitPlot = False                            #

# Dn4000 Constraints
#============================================================================
Dn4000Max = 3.0                                               #
Dn4000Min = 0.0                                               #

# Cluster List
#============================================================================
fullClusterList = fits.clusters                               # List of all clusters in the dataset
clustersToIgnore = ["SDSSCGA0782"]                            # List of cluster to exclude for the analysis (only when init(smaple = None))

# File Directories
#============================================================================
spectroscopicMemberDataFileDir = "./data/HeCS_omnibus_dn4000.fits" # File path for spectrospoic members dataset
clusterObjectData3r200FileDir = "./data/HeCS_omnibus_3r200.fits"   # File path for all object within 3R200 of the cluster center
clusterDataFileDir = "./data/HeCS_omnibus_prop.txt"                # File path for infomation about the overall cluster i.e R200, M200 and redshift

# Completeness/Membership fraction calculations
#============================================================================
rbandBinSize = 1.0                                            # Rband apparent magnitude Bin Size in Mpc
completenessLimit = 0.5                                       # Completeness limit to calculate/use in the program (between 0.0-1.0) i.e. 0.5 = 50% completeness limit
rbandMax = 100                                                # Max r-band value for calculating the completeness limt
rbandMin = 15                                                 # Min r-band value for calculating the completeness limt

# Membership correction parameter
#============================================================================

# General
#----------------------------------------------------------------------------
imposeLimitOnCorrection = True                                #

# Bin Sizes
#----------------------------------------------------------------------------
rnormBinSize = 0.25                                           # Rnorm (Rcl / R200) Bin Size
MrBinSize = 0.8                                               # Abs. R-Band Magnitude Bin Size
grBinSize = 0.25                                              # g-r Color Bin Size

# Bin Constraints (Value of None the max/min values in dataset will be taken)
#----------------------------------------------------------------------------
rnormMax = None                                               # Max Rnorm (Rcl / R200) Value
rnormMin = None                                               # Min Rnorm (Rcl / R200) Value
MrMax = -20.625824886161126                                   # Max Abs. R-Band Magnitude Value
MrMin = -22.0                                                 # Min Abs. R-Band Magnitude Value
grMax = 1.4                                                   # Max g-r Color Value
grMin = 0.65                                                  # Min g-r Color Value

# KDE Parameters
#----------------------------------------------------------------------------
dn4000Binwidth = 0.1                                          #
rNormBinwidth = 0.15                                          #
showOptimalBandwidth = False                                  #

# Maguntude Limit Calculation parameter
#============================================================================
redshiftBinMin = None
redshiftBinMax = None #0.31
redshiftBinNum = 10

massBinMin = None
massBinMax = None #1842480000000000.0
massBinNum = 4

# Global variables
#============================================================================
clusterDict = None
clusterDictSpec = None
centerData = None
clusterList = None
sampleUsing = None

"""
*****************************************************************************
                              Function Definitions
*****************************************************************************
"""

# Sample - cluster sample used in analysis
#        |- redshiftSampleComplete - Comeplete redshift smaple with mag limit of -20.625824886161126
#        |- redshiftSampleIncomplete - Incomplete redshift smaple with mag limit of -20.625824886161126
#        |- massSampleComplete - Comeplete mass smaple with mag limit of -20.256521757049285 and max redshift of 0.085
#        |- massSampleIncomplete - Incomplete redshift smaple with mag limit of -20.256521757049285 and max redshift of 0.085
#        |- custom - Add custom cluster sample (_clusterList parameter) and and change the mag limit and max redshift with _MrMax and _redshiftBinMax respectively
#        |- None - Uses full clusterList except cluster set to exclude from analysis
def cleanData(sample = None, setFlag = True, _clusterList = None, _MrMax = None, _redshiftBinMax = None):
    # Declares global variables
    global clusterList
    global sampleUsing
    global MrMax
    global redshiftBinMax

    if (setFlag):
        sampleUsing = sample

    if (sample == "redshiftSampleComplete"):
        # Complete redshift smaple
        MrMax = -20.625824886161126
        redshiftBinMax = None
        clusterList = ['A1367', 'MKW11', 'Coma', 'A2197', 'A2199', 'NGC6107', 'A1314', 'A1185', 'A2063', 'A1142', 'A2147', 'HeCS_faint1', 'RXJ0137', 'A2107', 'A2593', 'A160', 'A119', 'A168', 'A957', 'SHK352', 'A671', 'A1291A', 'A757', 'A1377', 'RXCJ1022p3830', 'A85', 'A1291B', 'A2399', 'A2169', 'A2457', 'A602', 'A736', 'RXCJ1351p4622', 'A1795', 'MSPM01721', 'A1668', 'A1436', 'A2149', 'A2092', 'A2124', 'A1066', 'RXCJ1115p5426', 'A1589', 'A1767', 'A1691', 'RXCJ1053p5450', 'A744', 'A2065', 'A2064', 'A1775', 'A1831', 'A1424', 'A1190', 'A1800', 'A1205', 'A1173', 'A2670', 'RXCJ1210p0523', 'Zw1215', 'A2029', 'A1773', 'A2061', 'A1809', 'A2495', 'A2255', 'A1307', 'RXCJ1326p0013', 'MS1306', 'A2428', 'A1663', 'A1650', 'A1750', 'A1552', 'A2245', 'A2018', 'A1728', 'A2142', 'A2175', 'A2244', 'A2055', 'Zw1478', 'A1235', 'Zw8197', 'A2034', 'A2069', 'A1302', 'A1361', 'A1366', 'A2050', 'A1033', 'A655', 'A646', 'A620', 'Zw3179', 'A667', 'RXJ1720', 'A2259', 'A750', 'A1914', 'A586', 'A1204', 'MS0906', 'RMJ0835p2046', 'A2187', 'A1689', 'A383', 'Zw1883', 'A291', 'A963', 'A773', 'A2261', 'A2390', 'A267', 'RXJ2129', 'Zw2089', 'A68', 'A1758', 'A1703', 'A689', 'A697', 'A611']
    elif (sample == "redshiftSampleIncomplete"):
        # Incomplete redshift smaple
        MrMax = -20.625824886161126
        redshiftBinMax = None
        clusterList = ['MKW4', 'A1367', 'MKW11', 'A779', 'Coma', 'MKW8', 'NGC6338', 'A2197', 'A2199', 'NGC6107', 'A1314', 'A1185', 'A2063', 'A1142', 'A2147', 'HeCS_faint1', 'RXJ0137', 'A2107', 'A2593', 'A295', 'A160', 'A119', 'A168', 'A957', 'SHK352', 'A671', 'A1291A', 'A757', 'A1377', 'RXCJ1022p3830', 'A85', 'A1291B', 'A2399', 'A2169', 'A2457', 'A602', 'A736', 'RXCJ1351p4622', 'A1795', 'MSPM01721', 'A1668', 'A1436', 'A2149', 'A2092', 'A2124', 'A1066', 'RXCJ1115p5426', 'A1589', 'A1767', 'A1691', 'RXCJ1053p5450', 'A744', 'A2065', 'A2064', 'A1775', 'A1831', 'A1424', 'A1190', 'A1800', 'A1205', 'A1173', 'A2670', 'RXCJ1210p0523', 'Zw1215', 'A2029', 'A1773', 'A2061', 'A1809', 'A2495', 'A2255', 'A1307', 'RXCJ1326p0013', 'MS1306', 'A2428', 'A1663', 'A1650', 'A1750', 'A1552', 'A2245', 'A2018', 'A1728', 'A2142', 'A2175', 'A2244', 'A2055', 'A1446', 'Zw1478', 'A1235', 'Zw8197', 'A2034', 'A2069', 'A1302', 'A1361', 'A1366', 'A2050', 'A1033', 'A655', 'A646', 'A620', 'Zw3179', 'A667', 'RXJ1720', 'A2259', 'A750', 'A1914', 'A586', 'A1204', 'MS0906', 'RMJ0835p2046', 'A2187', 'A1689', 'A383', 'Zw1883', 'A291', 'A963', 'A773', 'A2261', 'A2219', 'A2390', 'A267', 'RXJ2129', 'Zw2089', 'A68', 'A1835', 'A1758', 'A1703', 'A689', 'A697', 'A611']
    elif (sample == "massSampleComplete"):
        MrMax = -20.256521757049285
        redshiftBinMax = 0.085
        clusterList = ['A2197', 'A2199', 'A1185', 'A2147', 'HeCS_faint1', 'RXJ0137', 'A2107', 'A168', 'SHK352', 'A671', 'A1291A', 'A1377', 'RXCJ1022p3830', 'A85', 'A1291B', 'A2399', 'A2457', 'MSPM01721', 'A1436', 'A2092', 'A2124', 'A1066', 'RXCJ1115p5426', 'A1767', 'A1691', 'A744', 'A2065', 'A1831', 'A1800', 'A1205', 'A1173', 'RXCJ1210p0523', 'Zw1215', 'A2029', 'A2061', 'A2255', 'RXCJ1326p0013', 'MS1306']
    elif (sample == "massSampleIncomplete"):
        MrMax = -20.256521757049285
        redshiftBinMax = 0.085
        clusterList = ['MKW4', 'A2197', 'A2199', 'A1185', 'A2147', 'HeCS_faint1', 'RXJ0137', 'A2107', 'A295', 'A168', 'SHK352', 'A671', 'A1291A', 'A1377', 'RXCJ1022p3830', 'A85', 'A1291B', 'A2399', 'A2457', 'MSPM01721', 'A1436', 'A2092', 'A2124', 'A1066', 'RXCJ1115p5426', 'A1767', 'A1691', 'A744', 'A2065', 'A1831', 'A1800', 'A1205', 'A1173', 'RXCJ1210p0523', 'Zw1215', 'A2029', 'A2061', 'A2255', 'RXCJ1326p0013', 'MS1306']
    elif (sample == "custom"):
        if (_MrMax != None):
            MrMax = _MrMax
        if (_redshiftBinMax != None):
            redshiftBinMax = _redshiftBinMax
        clusterList = _clusterList
    else:
        # All clusters
        clusterList = []
        for cluster in fullClusterList:
            if(cluster not in clustersToIgnore):
                clusterList.append(cluster)

# Fetches starting data
# Sample - Refer to cleanData() defintion
def init(sample = None, clusterList = None, MrMax = None, redshiftBinMax = None):
    # Declares global variables
    global clusterDict
    global clusterDictSpec
    global centerData

    # Spectroscopic members
    dataSpec = fits.fetchFITS(spectroscopicMemberDataFileDir, 's')
    # All members within 3*R200
    _data = fits.fetchFITS(clusterObjectData3r200FileDir, 's')

    # Generates a dictionary from the data from the fit files
    clusterDict = fits.getClusterDict(_data)
    clusterDictSpec = fits.getClusterDict(dataSpec)

    # Cluster data
    centerData = np.loadtxt(clusterDataFileDir, dtype='str')

    cleanData(sample = sample, _clusterList = clusterList, _MrMax = MrMax, _redshiftBinMax = redshiftBinMax)

# Returns an array of common values in N arrays
# Parameters are a series of arrays
def IntersecOfSets(*args):
    # Converts the arrays to sets
    sets = np.array([set(i) for i in args])

    # Check for intersection between the sets
    resultSet = sets[0]
    for s in sets[1:]:
        resultSet = resultSet.intersection(s)

    # Converts intersection set to an array
    finalList = list(resultSet)

    return finalList

# Generate a dictionary that stores the luminosity distance for each cluster
def getLumDistDict():
    # Initializing the dictionary
    lumDistDict = {cluster: None for cluster in clusterList}

    # Setting up the cosmological model
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

    # Calculates the luminosity distance for each cluster
    for cluster, redshift in zip(centerData[:,0], centerData[:,1]):
        redshift = float(redshift)
        lumDistance = cosmo.luminosity_distance(redshift)
        lumDistDict[cluster] = lumDistance.value
    return lumDistDict

# Calculates the Absolute Magnitude
def getAbsMag(m, cluster, lumDistDict):
    return m - 5.000000 * math.log10(lumDistDict[cluster] * 1.000000e6) + 5.000000

# Generate a sub dictionary that consist of just cluster members
def genMemberDict(lumDistDict):
    # Sets up the progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(marker=u'\u2588', fill='.', left='|', right='|'), "Generating Member Dictionary"]
    pbar = progressbar.ProgressBar(widgets=widgets) if(showProgressBars) else progressbar.bar.NullBar()
    # Initializing the dictionary
    memberClusterDict = {cluster: [] for cluster in clusterList}

    # Find all cluster member object in each cluster
    for cluster in pbar(clusterList):
        # Creates an empty array to store the cluster data
        c = np.empty(len(clusterDict[cluster]), dtype=object)

        # Stores the cluster object data
        for i, (data) in enumerate(clusterDict[cluster]):
            c[i] = data

        # Loops through each object in the cluster
        for objectData in c:
            # Checks if the object is a cluster member or not
            if(objectData[6] == 'Y'):
                temp = []
                temp.extend(objectData)
                # Loop through each of the spectroscopic members
                for data in clusterDictSpec[cluster]:
                    # Check if the current member and the current spectroscopic
                    # member are the same. If they are additional data is added.
                    if (data[2].strip() == objectData[1].strip()):
                        temp.extend([data[1], data[5], data[8], getAbsMag(objectData[11], cluster, lumDistDict)])
                        break
                memberClusterDict[cluster].append(temp)

    return memberClusterDict

# Calculates the completness and memberfraction for each cluster and
# stores the result in two dictionaries one with completness and the other with the
# memberfraction. Both dictionaries also stores the Rnorn bin the fraction was calculated.
def CalcCompletness():
    # Sets up the progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(marker=u'\u2588', fill='.', left='|', right='|'), "Calculating Completeness and Member Fraction for Each Cluster"]
    pbar = progressbar.ProgressBar(widgets=widgets) if(showProgressBars) else progressbar.bar.NullBar()

    # Initializing the completeness and memberfraction dictionaries
    memberfractionDict, completnessDict = {}, {}

    for cluster in pbar(clusterList):
        # Check if the cluster contains any objects
        if (cluster not in clusterDict):
            continue
        elif(len(clusterDict[cluster]) == 0):
            completnessDict[cluster] = None
            memberfractionDict[cluster] = None
            continue

        # Initializing the clusters fraction and rnrom bins to empty arrays
        memberfractionDict[cluster] = [[], []]
        completnessDict[cluster] = [[], []]

        # Sets up arrays to store the redshift, rband mag and member status
        redshift = np.empty(len(clusterDict[cluster]), dtype=float)
        rband = np.empty(len(clusterDict[cluster]), dtype=float)
        isMember = np.empty(len(clusterDict[cluster]), dtype=str)

        # Gets the redshift, rband mag and member status from clusterDict
        for i, (data) in enumerate(clusterDict[cluster]):
            redshift[i] = data[4]
            rband[i] = data[11]
            isMember[i] = data[6]

        # Gets the number of rband bins
        binSize = rbandBinSize
        RbandMax = np.max(rband)
        RbandMin = np.min(rband)
        binNum = int(math.ceil((RbandMax - RbandMin)/ binSize))

        # Gets the indices for the values in each bin of rband apparent mag
        indexBins = [np.where(np.logical_and(rband > i * binSize + RbandMin,  rband < (i+1) * binSize + RbandMin))  for i in range(binNum)]

        for indices in indexBins:
            # Finds the number of members, spectroscopic members and total objects
            numSpec, numTotal, numMem = 0, 0, 0
            for z, m in zip(redshift[indices], isMember[indices]):
                if (z > -1):
                    numSpec += 1
                if(m == 'Y'):
                    numMem += 1
                numTotal += 1

            # Calculates the completeness in the current rband bin
            if(numTotal != 0):
                completnessDict[cluster][0].append(numSpec / numTotal)
                completnessDict[cluster][1].append(np.mean(rband[indices]))
            # Calculates the member fraction in the current rband bin
            if(numSpec != 0):
                memberfractionDict[cluster][0].append(numMem / numSpec)
                memberfractionDict[cluster][1].append(np.mean(rband[indices]))

    return completnessDict, memberfractionDict

# Calculates the completeness limit
def getCompletenessLimit(completnessDict, useAbsMag=False, lumDistDict=None):
    # Sets up the progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(marker=u'\u2588', fill='.', left='|', right='|'), "Calculating the Completeness Limit for Each Cluster"]
    pbar = progressbar.ProgressBar(widgets=widgets) if(showProgressBars) else progressbar.bar.NullBar()
    # Initializing the completeness limit dictionaries
    compLimit = {}
    plotData = {}

    for cluster in pbar(clusterList):
        # Checks if the completnessDict is not None
        if(cluster not in completnessDict):
            continue
        elif(completnessDict[cluster] is None):
            compLimit[cluster] = None
            continue

        # Gets the completeness and the r-band mag values
        rband = None
        if not useAbsMag:
            # Use apparent magnitude
            rband = np.array(completnessDict[cluster][1])
        else:
            #use absolute magnitude
            if lumDistDict == None:
                # Computes lumDistDict if needed
                lumDistDict = getLumDistDict()
            rband = getAbsMag(np.array(completnessDict[cluster][1]), cluster, lumDistDict)

        completeness = np.array(completnessDict[cluster][0])

        # Gets the indices of the arrays for r-band mag values within the min/max value range
        index = np.where(np.logical_and(rband > -30.0, rband < rbandMax))

        # Initializes a spline and computes y values for ploting
        f = InterpolatedUnivariateSpline(rband[index], completeness[index])
        xdata = np.linspace(np.min(rband[index]), np.max(rband[index]), 1000)
        ydata = f(xdata)

        # Store data for plotting
        plotData[cluster] = np.array([xdata, ydata])

        try:
            # Initializes a spline for when the y values are shift down by the completeness limit
            invf = InterpolatedUnivariateSpline(rband[index], completeness[index] - completenessLimit)
            # Finds the right most root and stores the limit in the limit50 dictionary
            limit = invf.roots()[-1]
            compLimit[cluster] = limit
        except:
            # Case where the complenetess never reaches the completeness limit
            compLimit[cluster] = None

    return compLimit, plotData

# Calculates the number of additional members required
def calcAdditonalMemeber(Nphot, fcomp, fmem):
    return Nphot * (1 - fcomp) * fmem

# Generate dV for additional memebrs, return dV normilzed by the clusters velocity dispersion
def sampleDV(cluster, numSamples):
    # Gets cluster data
    _clusters,  _dispersionVel = centerData[:,0], centerData[:,4].astype(np.float)

    # Calcukate dV values
    sigmaV = None
    for i, clu in enumerate(_clusters):
        if (clu == cluster):
            sigmaV = _dispersionVel[i]
            break

    new_data = np.random.normal(loc=0.0, scale=sigmaV, size=int(numSamples))

    return new_data / sigmaV

# Gets the indices bins of Rnorm, Mr and g-r color used to generate the KDE/KDE related plots
def getKDEBins(rnorm, Mr, gr):
    # Rnorm indice bins
    _rnormMax = rnormMax if(rnormMax is not None) else np.max(rnorm)
    _rnormMin = rnormMin if(rnormMin is not None) else np.min(rnorm)
    rnormBinNum = int(round(abs(_rnormMax - _rnormMin)/ rnormBinSize))

    # Get bins indices in master array
    rnormIndexBins = [np.where(np.logical_and(rnorm > i * rnormBinSize + _rnormMin,  rnorm <= (i+1) * rnormBinSize + _rnormMin))  for i in range(rnormBinNum)]

    # Mr indice bins
    _MrMax = MrMax if(MrMax is not None) else np.max(Mr)
    _MrMin = MrMin if(MrMin is not None) else np.min(Mr)
    MrBinNum = int(math.ceil(abs(_MrMax - _MrMin)/ MrBinSize))

    # Get bins indices in master array
    MrIndexBins = [np.where(np.logical_and(Mr >= np.min(Mr), Mr <= _MrMin))]
    MrIndexBins.extend([np.where(np.logical_and(Mr > i * MrBinSize + _MrMin,  Mr <= (i+1) * MrBinSize + _MrMin))  for i in range(MrBinNum - 1)])
    MrIndexBins.extend([np.where(np.logical_and(Mr > (MrBinNum - 1) * MrBinSize + _MrMin, Mr <= _MrMax))])

    # g-r color indice bins
    _grMax = grMax if(grMax is not None) else np.max(gr)
    _grMin = grMin if(grMin is not None) else np.min(gr)
    grBinNum = int(math.ceil(abs(_grMax - _grMin)/ grBinSize))

    # Get bins indices in master array
    grIndexBins = [np.where(np.logical_and(gr > -0.5, gr <= _grMin))]
    grIndexBins.extend([np.where(np.logical_and(gr > i * grBinSize + _grMin,  gr <= (i+1) * grBinSize + _grMin))  for i in range(grBinNum)])
    grIndexBins.extend([np.where(np.logical_and(gr > _grMax, gr <= 2.5))])

    return rnormIndexBins, MrIndexBins, grIndexBins

#Finds the number of additional members requried for each cluster
def getAdditionalMemberDict(memberClusterDict, lumDistDict, compLimit):
    # Initializes two dictionaries to store the number of additional members
    # with and without the completeness limit imposed.
    additionalMembersDict, additionalMembersLimitDict = {}, {}
    tot = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    for cluster in clusterList:
        # Check if cluster is in member dict and if the number of object is greater than zero
        if(cluster not in memberClusterDict):
            continue
        if (len(memberClusterDict[cluster]) == 0):
            additionalMembersDict[cluster], additionalMembersLimitDict[cluster] = None, None
            continue

        # Initializes variables
        additionalMembersDict[cluster], additionalMembersLimitDict[cluster] = [], []
        redshift, rband, Mr, gr, rnorm, dn4000, Nphot, isMember = [], [], [], [], [], [], [], []

        # Gets the redshift, rband mag, Mr, g-r, Dn4000 and member status of each object
        for d in clusterDict[cluster]:
            redshift.append(d[4])
            rband.append(d[11])
            Mr.append(getAbsMag(d[11],cluster, lumDistDict))
            gr.append(d[9] - d[11])
            dn4000.append(d[17])
            isMember.append(d[6])

        # Converts arrays to numpy arrays
        redshift = np.array(redshift)
        rband = np.array(rband)
        Mr = np.array(Mr)
        gr = np.array(gr)
        dn4000 = np.array(dn4000)

        _, MrIndexBins, grIndexBins = getKDEBins(np.array([0]), Mr, gr)

        # Loops through each g-r bin
        for grBinNum, (grIndices) in enumerate(grIndexBins):
            # Temp array to store the additional members for the current g-r bin
            temp, tempLimit = [], []

            # Loops through each Mr bin
            for MrBinNum, (MrIndices) in enumerate(MrIndexBins):
                # Inits the number of spectropic members, members and objects
                numSpec, numMem, numTotal = 0, 0, 0
                # Inits the number of spectropic members, members and objects within the completeness limit
                numSpecL, numMemL, numTotalL = 0, 0, 0
                # Gets the common indcies between the g-r and Mr bins
                indices = IntersecOfSets(grIndices[0], MrIndices[0])
                for index in indices:
                    # Gets the number of spectropic members, members and objects within the completeness limit
                    if (cluster in compLimit and compLimit[cluster] is not None):
                        if(rband[index] <= compLimit[cluster]):
                            if (redshift[index] > -1):
                                numSpecL += 1
                            if(isMember[index] == 'Y'):
                                numMemL += 1
                            numTotalL += 1
                    # Gets the number of spectropic members, members and objects
                    if (redshift[index] > -1):
                        numSpec += 1
                    if(isMember[index] == 'Y'):
                        numMem += 1
                    numTotal += 1

                # Display Stats in each bin
                tot[MrBinNum][grBinNum][0] += numTotalL
                tot[MrBinNum][grBinNum][1] += numMemL
                tot[MrBinNum][grBinNum][2] += numSpecL

                # Calculates the number of additional members
                if (numTotal != 0 and numSpec != 0):
                    fcomp = numSpec /numTotal
                    fmem = numMem / numSpec
                    temp.append(calcAdditonalMemeber(numTotal, fcomp, fmem))
                else:
                    temp.append(None)

                # Calculates the number of additional members within the completeness limit
                if (numTotalL != 0 and numSpecL != 0):
                    fcompL= numSpecL /numTotalL
                    fmemL= numMemL / numSpecL
                    tot[MrBinNum][grBinNum][3] += calcAdditonalMemeber(numTotalL, fcompL, fmemL)
                    tempLimit.append(calcAdditonalMemeber(numTotalL, fcompL, fmemL))
                else:
                    tempLimit.append(None)

            additionalMembersDict[cluster].append(temp)
            additionalMembersLimitDict[cluster].append(tempLimit)

    return additionalMembersDict, additionalMembersLimitDict, tot

# Gets the redshift, r-band apparent/absolute mag, g-r color, Rnorm and Dn4000 for a cluster
def getParametersFromDataset(memberClusterDict, cluster):
    # Fetching list of each parameter
    c = np.array(memberClusterDict[cluster])

    # Gets required paraemeters
    redshift = c[:,4].astype(float)
    rband = c[:,11].astype(float)
    Mr = c[:,22].astype(float)
    gr = c[:,9].astype(float) - c[:,11].astype(float)
    rnorm = c[:,20].astype(float) / c[:,19].astype(float)
    dn4000 = c[:,17].astype(float)

    # Conert to numpy arrays
    redshift = np.array(redshift)
    rband = np.array(rband )
    Mr = np.array(Mr)
    gr = np.array(gr)
    rnorm = np.array(rnorm)
    dn4000 = np.array(dn4000)

    return redshift, rband, Mr, gr, rnorm, dn4000

# Gets the stacked redshift, r-band apparent/absolute mag, g-r color, Rnorm and Dn4000
def getParametersFromStackedDataset(memberClusterDict):
    # Sets up the progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(marker=u'\u2588', fill='.', left='|', right='|'), "Getting Stacked Sampled Parameters"]
    pbar = progressbar.ProgressBar(widgets=widgets) if(showProgressBars) else progressbar.bar.NullBar()

    # Initializing parameter arrays
    redshift, rband, Mr, gr, rnorm, dn4000 = [], [], [], [], [], []

    for cluster in pbar(clusterList):
        if(cluster not in memberClusterDict):
            continue
        elif(len(memberClusterDict[cluster]) == 0):
            continue

        # Gets cluster parameters
        cluRedshift, cluRband, cluMr, cluGr, cluRnorm, cluDn4000 = getParametersFromDataset(memberClusterDict, cluster)

        # Adds to stacked samples
        redshift.extend(cluRedshift)
        rband.extend(cluRband)
        Mr.extend(cluMr)
        gr.extend(cluGr)
        rnorm.extend(cluRnorm)
        dn4000.extend(cluDn4000)

    # Conert to numpy arrays
    redshift = np.array(redshift)
    rband = np.array(rband )
    Mr = np.array(Mr)
    gr = np.array(gr)
    rnorm = np.array(rnorm)
    dn4000 = np.array(dn4000)

    return redshift, rband, Mr, gr, rnorm, dn4000

# Generate 2D KDE for a given set of data
def getKDEs(data):
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)

    if(showOptimalBandwidth):
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    kde = grid.best_estimator_

    return kde

# Samples the additional members for eac bin of Mr, g-r color
def sampleAdditionalMembers(memberClusterDict, additionalMembersDict, seed = None):
    # Getting parameter arrays
    redshift, rband, Mr, gr, rnorm, dn4000 = getParametersFromStackedDataset(memberClusterDict)

    # Applies Dn4000 limits
    redshift = redshift[np.where(np.logical_and(dn4000 > Dn4000Min, dn4000 < Dn4000Max))]
    rband = rband[np.where(np.logical_and(dn4000 > Dn4000Min, dn4000 < Dn4000Max))]
    Mr = Mr[np.where(np.logical_and(dn4000 > Dn4000Min, dn4000 < Dn4000Max))]
    gr = gr[np.where(np.logical_and(dn4000 > Dn4000Min, dn4000 < Dn4000Max))]
    rnorm = rnorm[np.where(np.logical_and(dn4000 > Dn4000Min, dn4000 < Dn4000Max))]
    dn4000 = dn4000[np.where(np.logical_and(dn4000 > Dn4000Min, dn4000 < Dn4000Max))]

    # Gets the indice bins for Rnorm, Mr, and g-r color
    rnormIndexBins, MrIndexBins, grIndexBins = getKDEBins(rnorm, Mr, gr)

    # Initalizes an array that will contain all the sampled data points
    sampledData = []

    # Initalizes array to store plotting data
    plotData = []
    colName, rowName = [], []
    for MrBinNum, (MrIndices) in enumerate(MrIndexBins):
        for grBinNum, (grIndices) in enumerate(grIndexBins):
            temp = []
            # Gets row (g-r) range for ploting
            if(grBinNum == 0):
                rowName.append(str(r" -0.5 $<$ g-r $<$ 0.65"))
            elif(grBinNum == len(grIndexBins) - 1):
                rowName.append(r"1.4 $<$ g-r $<$ 2.5")
            else:
                rowName.append(str(round((grBinNum - 1) * grBinSize + grMin, 2)) + " $<$ g-r $<$ " + str(round((grBinNum) * grBinSize + grMin, 2)))

            additionalMembers = 0
            for cluster in clusterList:
                if cluster in additionalMembersDict:
                    if(additionalMembersDict[cluster] is None or (additionalMembersDict[cluster] == np.array(None)).all()):
                        continue
                    elif (len(additionalMembersDict[cluster]) <= 0):
                        continue
                    elif (additionalMembersDict[cluster][grBinNum][MrBinNum] is not None and additionalMembersDict[cluster][grBinNum][MrBinNum] != 0):
                        additionalMembers += additionalMembersDict[cluster][grBinNum][MrBinNum]
            if(len(IntersecOfSets(MrIndices[0], grIndices[0])) < 5):
                sampledData.append([None] * len(IntersecOfSets(MrIndices[0], grIndices[0])))
                continue

            data = np.vstack([rnorm[IntersecOfSets(MrIndices[0], grIndices[0])], dn4000[IntersecOfSets(MrIndices[0], grIndices[0])]]).T

            # Stores raw data plotting infomation
            temp.append([rnorm[IntersecOfSets(MrIndices[0], grIndices[0])], dn4000[IntersecOfSets(MrIndices[0], grIndices[0])]])

            # Generate the kde
            if (seed != None):
                kde = getKDEs(data, random_state = seed)
            else:
                kde = getKDEs(data)

            # Stores probability distribution plotting infomation
            xgrid = np.linspace(0, 3, 100)
            ygrid = np.linspace(0.9, 2.5, 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            log_dens = kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T)
            dens = np.exp(log_dens)
            temp.append(dens.reshape(Xgrid.shape))

            # Gets new data
            new_data = kde.sample(int(additionalMembers), random_state=0)
            sampledData.append(new_data)

            # Stroe plotting data
            plotData.append(temp)

        # Gets col (Mr) range for ploting
        if(MrBinNum == 0):
            colName.append(str(round(np.min(Mr),2)) + " $<$ $M_{r}$ $<$ " + str(round(MrMin, 2)))
        else:
            colName.append(str(round((MrBinNum - 1) * MrBinSize + MrMin, 2)) + " $<$ $M_{r}$ $<$ " + str(round((MrBinNum) * MrBinSize + MrMin, 2)))

    return sampledData, plotData, rowName, colName

# Assignes Additional Member to each cluster and returns a dictionary
# that consist of the original data and the data with the correction
# applied.
def assignAdditionalMembers(memberClusterDict, additionalMembersDict, sampledData):
    # Inits a dictionary to caontain the uncorrected and corrected data for
    # each cluster
    plotDataDict = {cluster: [] for cluster in fits.clusters}

    index = {i: {j: 0 for j in range(len(additionalMembersDict[clusterList[0]]))} for i in range(len(additionalMembersDict[clusterList[0]][0]))}

    for cluster in clusterList:
        if(cluster not in memberClusterDict):
            continue
        elif (len(memberClusterDict[cluster]) == 0):
            continue

        additionalMembers = []
        if cluster in additionalMembersDict:
            for rnum, row in enumerate(additionalMembersDict[cluster]):
                for i, (numberofAdditionalMember) in enumerate(row):
                    if (numberofAdditionalMember == None):
                        continue
                    elif (len(sampledData[i * len(additionalMembersDict[cluster]) + rnum]) == 0):
                        continue
                    elif ((np.array(sampledData[i * len(additionalMembersDict[cluster]) + rnum]) == None).all()):
                        continue
                    new_dV = sampleDV(cluster, numberofAdditionalMember)
                    for memberIndex in range(int(numberofAdditionalMember)):
                        element = sampledData[i * len(additionalMembersDict[cluster]) + rnum][index[i][rnum]]
                        index[i][rnum] += 1
                        element = np.append(element, [1])
                        if (len(element) < 3): continue
                        additionalMembers.append([element[1], element[0], element[2], None, None, new_dV[memberIndex]])

        # Gets the clusters redshift, Mr, Rnorm and Dn4000
        redshift, _ , Mr, _, rnorm, dn4000 = getParametersFromDataset(memberClusterDict, cluster)

        _clusters, _redshift,  _dispersionVel = centerData[:,0], centerData[:,1].astype(np.float), centerData[:,4].astype(np.float)

        zc, sigmaV = None, None
        for i, clu in enumerate(_clusters):
            if clu == cluster:
                zc, sigmaV = _redshift[i], _dispersionVel[i]

        c = 299792458  * 0.001 # km/s, speed of light
        dV = c * abs((redshift - zc)) / (1.0 + zc)

        dV /= sigmaV

        # Adds the data without correction to the dataset
        rawData = np.vstack([dn4000, rnorm, [0] * len(rnorm), Mr, redshift, dV]).T
        plotDataDict[cluster].append(rawData)

        # Gets the data with the correction
        newData = None
        if(len(additionalMembers) != 0):
            newData = np.concatenate((rawData, additionalMembers), axis = 0)
        else:
            newData = rawData
        plotDataDict[cluster].append(newData)

        # Adds the data with just the corrected objects to the dataset
        plotDataDict[cluster].append(np.array(additionalMembers))

    return plotDataDict

#
def getStackedSample(plotDataDict, dataIndex=1):
    _clusters = centerData[:,0]
    _redshift = centerData[:,1].astype(np.float)
    _M200 = centerData[:,3].astype(np.float)
    _R200 = centerData[:,2].astype(np.float)
    _dispersionVel = centerData[:,4].astype(np.float)

    clusters, redshift, M200, R200, Mr, zo, dispersionVel = [], [], [], [], [], [], []

    medDn4000, meanRnorm = [], []

    Dn400Stacked, rnornStacked, MrStacked, zoStacked, dVStacked = [], [], [], [], []

    for cluster in clusterList:
        if cluster not in plotDataDict:
            continue
        elif (len(plotDataDict[cluster]) == 0):
            continue
        for index, c in enumerate(_clusters):
            if(cluster == c):
                clusters.append(_clusters[index])
                redshift.append(_redshift[index])
                M200.append(_M200[index])
                R200.append(_R200[index])
                dispersionVel.append(_dispersionVel[index])

        rnorm = np.array(plotDataDict[cluster][dataIndex][:,1])
        Dn4000 = np.array(plotDataDict[cluster][dataIndex][:,0])
        Mr = np.array(plotDataDict[cluster][dataIndex][:,3])
        zo = np.array(plotDataDict[cluster][dataIndex][:,4])
        dV = np.array(plotDataDict[cluster][dataIndex][:,5])

        c1 = np.where(np.logical_and(Dn4000 > 0, Dn4000 < 3))

        Dn4000 = Dn4000[c1]
        zo = zo[c1]
        rnorm = rnorm[c1]
        Mr = Mr[c1]
        dV = dV[c1]

        c4 = np.where(rnorm > 0)

        Dn4000 = Dn4000[c4]
        rnorm = rnorm[c4]
        Mr = Mr[c4]
        zo = zo[c4]
        dV = dV[c4]

        binSize = 200 * 0.001 # Mpc
        RclMax = np.max(rnorm)
        binNum = int(math.ceil(RclMax/ binSize))

        indexBins = [np.where(np.logical_and(rnorm > i * binSize,  rnorm < (i+1) * binSize))  for i in range(binNum)]

        Dn400Stacked.append(Dn4000)
        rnornStacked.append(rnorm)
        MrStacked.append(Mr)
        zoStacked.append(zo)
        dVStacked.append(dV)

        medDn4000_ = []
        meanRnorm_ = []
        for i in range(len(indexBins)):
                if (Dn4000[indexBins[i]].size != 0):
                    medDn4000_.append(np.median(Dn4000[indexBins[i]]))

                if (Dn4000[indexBins[i]].size != 0):
                    meanRnorm_.append(np.mean(rnorm[indexBins[i]]))

        medDn4000.append(medDn4000_)
        meanRnorm.append(meanRnorm_)

    Dn4000Stacked = np.array(Dn400Stacked, dtype=object)
    rnornStacked = np.array(rnornStacked, dtype=object)
    MrStacked = np.array(MrStacked, dtype=object)
    zoStacked = np.array(zoStacked, dtype=object)
    dVStacked = np.array(dVStacked, dtype=object)

    medDn4000 = np.array(medDn4000, dtype=object)
    meanRnorm = np.array(meanRnorm, dtype=object)
    clusters= np.array(clusters)

    return clusters, redshift, M200, R200, Mr, zo, Dn400Stacked, rnornStacked, MrStacked, zoStacked, medDn4000, meanRnorm, dispersionVel, dVStacked

def getBinning(completnessDict, binType = 'equal'):
    plotData = []
    limit50 = getCompletenessAtMagLimit(completnessDict, useAbsMag=True)

    centerData = np.loadtxt(clusterDataFileDir, dtype='str')
    cluster, redshift, M200 = centerData[:,0], centerData[:,1].astype(np.float), centerData[:,3].astype(np.float)

    x,y = [], []
    x1,y1 = [], []
    x2,y2 = [], []
    completeClusters = []
    ncClusters = []
    tot, com = 0, 0
    te = 0
    for clu, z, m in zip(cluster, redshift, M200):
        if clu not in limit50:
            continue

        x.append(m)
        y.append(z)

        x1.append(m)
        y1.append(limit50[clu])
        if (limit50[clu] is not None and limit50[clu] >= 0.5):
            completeClusters.append(clu)
            com += 1
        if (limit50[clu] is not None and limit50[clu] < 0.5):
            ncClusters.append(clu)
            tot += 1
        if (limit50[clu] is not None):
            tot += 1

        if (m > 3.440100e+14):
            te += 1


        x2.append(z)
        y2.append(limit50[clu])


    #print("Percent Above: ", (com / tot) * 100.0, "%")
    #print("Percent Below: ", ((tot - com) / tot) * 100.0, "%")
    cleanCond1 = np.where(np.logical_and(x1 != np.array(None), y1 != np.array(None)))
    cleanCond2 = np.where(np.logical_and(x2 != np.array(None), y2 != np.array(None)))
    cleanCond3 = np.where(np.logical_and(x != np.array(None), y != np.array(None)))

    x = np.array(x)[cleanCond3]
    y = np.array(y)[cleanCond3]
    x1 = np.array(x1)[cleanCond1]
    y1 = np.array(y1)[cleanCond1]
    x2 = np.array(x2)[cleanCond2]
    y2 = np.array(y2)[cleanCond2]

    q = None
    if (binType == 'equal'):
        q = np.array([[1e14, 0.11], [1e15, 0.20]])
    else:
        q = mquantiles(np.vstack([x, y]).T, axis=0)

    plotData.append([x, y])
    plotData.append([x1, y1])
    plotData.append([x2, y2])

    return q, plotData

def linear(x, a, b):
    return a * (x) + b

def residual(params, y, x):
    param1 = params['one']
    param2 = params['two']
    param3 = params['three']
    param4 = params['four']

    p = param1 * x + param2
    p[np.where(param3 < x)] = param4

    return p - y

def fitDn4000vsRnorm(data, fitMode='linear'):
    x = data[0]
    y = data[1]
    yerr = data[2]

    if (fitMode == 'linear'):
        popt, pcov = curve_fit(linear, x, y, sigma=yerr)
    elif(fitMode == 'elbow'):
        from lmfit import minimize, Parameters, Model
        params = Parameters()
        params.add('one', value=0.0)
        params.add('two', value=1.0)
        params.add('three', value=1.0)
        params.add('four', value=1.5)
        out = minimize(residual, params,args=(y,x))
        return out
    else:
        popt, pcov = None

    return popt, pcov

def getMedDn4000(Rcl, Dn4000, Dnerr, useBoot=True):
        Rcl = np.array(Rcl, dtype= np.float64)
        Dn4000 = np.array(Dn4000, dtype= np.float64)
        Dnerr = np.array(Dnerr, dtype = np.float64)

        #binSize = 200 * 0.001# Mpc
        binSize = 200 * 0.001# Mpc
        RclMax = np.max(Rcl)
        RclMin = np.min(Rcl)
        binNum = int(math.ceil((RclMax - RclMin)/ binSize))

        indexBins = [np.where(np.logical_and(Rcl > i * binSize + RclMin,  Rcl < (i+1) * binSize + RclMin))  for i in range(binNum)]

        Dn4000Med = []
        DnerrMed = []
        RclMean = []
        for i in range(len(indexBins)):
            if (Dn4000[indexBins[i]].size <= 1):
                continue
            if (Dn4000[indexBins[i]].size != 0):
                Dn4000Med.append(np.median(Dn4000[indexBins[i]]))
            if (Dn4000[indexBins[i]].size > 1):
                err = []
                if(useBoot):
                    err = np.std(ft.bootstrap(Dn4000[indexBins[i]], 500, bootfunc = np.median))
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

# dataIndex = 1 for data with correction
# dataIndex = 0 for data without correction
# !!!! Note: Function Only wotks for current even binning z: 0.11, 0.20 and 1e14 < m200 < 1e15 !!!!
# !!!! If Binning Needs to be chnaged condition in if/else clause need to be modified  !!!!
# !!!! Contact me if this needs to be more general !!!!
def generateDn4000vsRnormPlots(plotDataDict, completnessDict, errorMethod = "std", xproj = 'Rnorm', dataIndex = 1):
    q, _ = getBinning(completnessDict, binType = 'equal')

    clusters, redshift, M200, R200, Mr, zo, Dn4000Stacked, rnornStacked, MrStacked, zoStacked, medDn4000, meanRnorm, dispersionVel, dVStacked = getStackedSample(plotDataDict, dataIndex = dataIndex)

    redshiftBinStart = redshiftBinMin if (redshiftBinMin is not None) else np.min(redshift)
    redshiftBinEnd = redshiftBinMax if (redshiftBinMax is not None) else np.max(redshift)
    redshiftBinSize = abs(redshiftBinEnd - redshiftBinStart) / 4

    massBinStart = massBinMin if (massBinMin is not None) else np.min(M200)
    massBinEnd = massBinMax if (massBinMax is not None) else np.max(M200)
    massBinSize = abs(massBinEnd - massBinStart) / 3


    plotData = []
    colName = []
    R200 = np.array(R200)
    Dn4000Stacked = np.array(Dn4000Stacked)
    slope = []
    serr = []
    intercept = []
    ierr = []
    setcol = True
    mmm = np.array([1e14, 1e15])
    minMembers = 1e99
    for i in range(3):
        for j in range(1):
            Mmax = [1e15, 1e14]
            if j == 1 or i == 3:
                continue
            if (i ==  0 and j ==0):
                condition = np.where(np.logical_and(q[:,1][0] >= redshift, np.logical_and(mmm[0] < M200, mmm[1] > M200)))
            elif (i ==0 and j == 2):
                condition = np.where(np.logical_and(q[:,1][0] >= redshift, q[:,0][1]  <= M200))
            elif (i == 2 and j == 0):
                condition = np.where(np.logical_and(q[:,1][1] <= redshift, np.logical_and(mmm[0]  < M200, mmm[1]  > M200)))
            elif (i ==2 and j ==2):
                condition = np.where(np.logical_and(q[:,1][1] <= redshift, q[:,0][1] <= M200))
            elif (j == 0):
                condition = np.where(np.logical_and(np.logical_and(q[:,1][i - 1] <= redshift, q[:,1][i] > redshift), np.logical_and(mmm[0]  < M200, mmm[1]  > M200)))
            elif (i == 0):
                condition = np.where(np.logical_and(q[:,1][0] >= redshift, np.logical_and(q[:,0][j - 1] <= M200, q[:,0][j] > M200)))
            elif (i == 2):
                condition = np.where(np.logical_and(q[:,1][1] <= redshift, np.logical_and(q[:,0][j - 1] <= M200, q[:,0][j] > M200)))
            elif (j == 2):
                condition = np.where(np.logical_and(np.logical_and(q[:,1][i - 1] <= redshift, q[:,1][i] > redshift), q[:,0][1]  <= M200))
            else:
                condition = np.where(np.logical_and(np.logical_and(q[:,1][i - 1] <= redshift, q[:,1][i] > redshift), np.logical_and(q[:,0][j - 1] <= M200, q[:,0][j] > M200)))

            clustersInBin = clusters[condition]
            numMem = 0
            for Mr, RnormS, Dn4000S, dVS, clu in zip(MrStacked[condition], rnornStacked[condition], Dn4000Stacked[condition], dVStacked[condition], clustersInBin):
                RnormS = np.array(RnormS, dtype= np.float64)
                dVS = np.array(dVS, dtype= np.float64)

                if (xproj == 'Rnorm'):
                    pass
                elif (xproj == 'LogRnorm'):
                    RnormS = np.log10(RnormS)
                elif (xproj == 'InfallTime'):
                    RnormS = RnormS * dVS
                elif (xproj == 'LogInfallTime'):
                    RnormS = np.log10(RnormS * dVS)
                else:
                    print("!!!Invaild x-axis projection!!!")
                    return

                MrCopy = copy.deepcopy(Mr)
                Dn4000SCopy = copy.deepcopy(Dn4000S)
                RnormSCopy = copy.deepcopy(RnormS)

                addedindeices = []

                for index, (_Mr, _Rn, _Dn) in enumerate(zip(Mr, RnormS, Dn4000S)):
                    if (_Mr == None or _Rn == None or _Dn == None):
                        addedindeices.append(index)
                    elif (np.isnan(_Rn) or np.isinf(_Rn)):
                        addedindeices.append(index)

                Mr = np.delete(Mr, addedindeices)
                RnormS = np.delete(RnormS, addedindeices)
                Dn4000S = np.delete(Dn4000S, addedindeices)

                # Above
                cond = np.where(Mr < MrMax)
                #Below
                #cond = np.where(Mr > MrMin)

                numMem += len(RnormS[cond])

                cond2 = np.where(MrCopy == None)
                for _data in RnormSCopy[cond2]:
                    if (np.isnan(_data) or np.isinf(_data)):
                        pass
                    else:
                        numMem += 1

            if (numMem < minMembers):
                minMembers = numMem

    for i in range(3):
        if i > 7:
            continue
        rowName = []
        tempSlope = []
        tempIntercept = []
        tempserr = []
        tempierr = []
        for j in range(1):
            if j == 1 or i == 3:
                continue
            if (i ==  0 and j ==0):
                rowName.append("{:.2e}".format(1e14) + " $M_\odot$ $<$ $M_{200}$ $<$"  + "{:.2e}".format(1e15) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(np.min(redshift), 2)) + " $<$ z $<$ " + str(round(q[:,1][0], 2)))
                    setcol = False
                condition = np.where(np.logical_and(q[:,1][0] >= redshift, np.logical_and(mmm[0]  < M200, mmm[1]  > M200)))
            elif (i ==0 and j == 2):
                rowName.append("{:.2e}".format(q[:,0][1]) + " $M_\odot$ $<$ $M_{200}$ $<$ " + "{:.2e}".format(np.max(M200)) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(np.min(redshift), 2)) + " $<$ z $<$ " + str(round(q[:,1][0], 2)))
                    setcol = False
                condition = np.where(np.logical_and(q[:,1][0] >= redshift, q[:,0][1]  <= M200))
            elif (i == 2 and j == 0):
                rowName.append("{:.2e}".format(1e14) + " $M_\odot$ $<$ $M_{200}$ $<$"  + "{:.2e}".format(1e15) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(q[:,1][1], 2)) + " $<$ z $<$ " + str(round(np.max(redshift), 2)))
                    setcol = False
                condition = np.where(np.logical_and(q[:,1][1] <= redshift, np.logical_and(mmm[0]  < M200, mmm[1]  > M200)))
            elif (i ==2 and j ==2):
                rowName.append("{:.2e}".format(q[:,0][1]) + " $M_\odot$ $<$ $M_{200}$ $<$ " + "{:.2e}".format(np.max(M200)) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(q[:,1][1], 2)) + " $<$ z $<$ " + str(round(np.max(redshift), 2)))
                    setcol = False
                condition = np.where(np.logical_and(q[:,1][1] <= redshift, q[:,0][1] <= M200))
            elif (j == 0):
                rowName.append("{:.2e}".format(1e14) + " $M_\odot$ $<$ $M_{200}$ $<$"  + "{:.2e}".format(1e15) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(q[:,1][i - 1], 2)) + " $<$ z $<$ " + str(round(q[:,1][i], 2)))
                    setcol = False
                condition = np.where(np.logical_and(np.logical_and(q[:,1][i - 1] <= redshift, q[:,1][i] > redshift), np.logical_and(mmm[0]  < M200, mmm[1]  > M200)))
            elif (i == 0):
                rowName.append("{:.2e}".format(q[:,0][j - 1])  + " $M_\odot$ $<$ $M_{200}$ $<$ " + "{:.2e}".format(q[:,0][j]) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(np.min(redshift), 2)) + " $<$ z $<$ " + str(round(q[:,1][0], 2)))
                    setcol = False
                condition = np.where(np.logical_and(q[:,1][0] >= redshift, np.logical_and(q[:,0][j - 1] <= M200, q[:,0][j] > M200)))
            elif (i == 2):
                rowName.append("{:.2e}".format(q[:,0][j - 1])  + " $M_\odot$ $<$ $M_{200}$ $<$ " + "{:.2e}".format(q[:,0][j]) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(q[:,1][1], 2)) + " $<$ z $<$ " + str(round(np.max(redshift), 2)))
                    setcol = False
                condition = np.where(np.logical_and(q[:,1][1] <= redshift, np.logical_and(q[:,0][j - 1] <= M200, q[:,0][j] > M200)))
            elif (j == 2):
                rowName.append("{:.2e}".format(q[:,0][1]) + " $M_\odot$ $<$ $M_{200}$ $<$ " + "{:.2e}".format(np.max(M200)) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(q[:,1][i - 1], 2)) + " $<$ z $<$ " + str(round(q[:,1][i], 2)))
                    setcol = False
                condition = np.where(np.logical_and(np.logical_and(q[:,1][i - 1] <= redshift, q[:,1][i] > redshift), q[:,0][1]  <= M200))
            else:
                rowName.append("{:.2e}".format(q[:,0][j - 1])  + " $M_\odot$ $<$ $M_{200}$ $<$ " + "{:.2e}".format(q[:,0][j]) + " $M_\odot$")
                if (setcol == True):
                    colName.append(str(round(q[:,1][i - 1], 2)) + " $<$ z $<$ " + str(round(q[:,1][i], 2)))
                    setcol = False
                condition = np.where(np.logical_and(np.logical_and(q[:,1][i - 1] <= redshift, q[:,1][i] > redshift), np.logical_and(q[:,0][j - 1] <= M200, q[:,0][j] > M200)))

            temp = []
            clustersInBin = clusters[condition]
            Dn4000TotalMed, Dn4000Totalerr, RclTotalMean = [], [], []
            Dn4000, Rnorm = [], []
            _Dn4000, _Rnorm = [], []
            numAddMem = 0
            numRealMem = 0
            ttt = 0
            for Mr, RnormS, Dn4000S, dVS, clu in zip(MrStacked[condition], rnornStacked[condition], Dn4000Stacked[condition], dVStacked[condition], clustersInBin):
                for index, (_Mr, _Rn, _Dn) in enumerate(zip(Mr, RnormS, Dn4000S)):
                    if (_Mr == None or _Rn == None or _Dn == None):
                        numAddMem += 1
                    else:
                        numRealMem += 1
                RnormS = np.array(RnormS, dtype= np.float64)
                dVS = np.array(dVS, dtype= np.float64)

                if (xproj == 'Rnorm'):
                    pass
                elif (xproj == 'LogRnorm'):
                    RnormS = np.log10(RnormS)
                elif (xproj == 'InfallTime'):
                    RnormS = RnormS * dVS
                elif (xproj == 'LogInfallTime'):
                    RnormS = np.log10(RnormS * dVS)
                else:
                    print("!!!Invaild x-axis projection!!!")
                    return

                MrCopy = copy.deepcopy(Mr)
                Dn4000SCopy = copy.deepcopy(Dn4000S)
                RnormSCopy = copy.deepcopy(RnormS)

                addedindeices = []

                for index, (_Mr, _Rn, _Dn) in enumerate(zip(Mr, RnormS, Dn4000S)):
                    if (_Mr == None or _Rn == None or _Dn == None):
                        addedindeices.append(index)
                    elif (np.isnan(_Rn) or np.isinf(_Rn)):
                        addedindeices.append(index)

                Mr = np.delete(Mr, addedindeices)
                RnormS = np.delete(RnormS, addedindeices)
                Dn4000S = np.delete(Dn4000S, addedindeices)

                # Above
                cond = np.where(Mr < MrMax)
                #Below
                #cond = np.where(Mr > MrMin)
                _Rnorm.extend(RnormS[cond])
                _Dn4000.extend(Dn4000S[cond])

                cond2 = np.where(MrCopy == None)
                for _data1, _data2 in zip(RnormSCopy[cond2], Dn4000SCopy[cond2]):
                    if (np.isnan(_data1) or np.isinf(_data1)):
                        pass
                    else:
                        _Rnorm.append(_data1)
                        _Dn4000.append(_data2)


            indices = random.sample([i for i in range(len(_Rnorm))], minMembers)

            for ind in indices:
                Rnorm.append(_Rnorm[ind])
                Dn4000.append(_Dn4000[ind])

            temp.append([Rnorm, Dn4000])

            if (len(Rnorm) != 0):
                _, drange, _ = getMedDn4000(Rnorm, Dn4000, [0 for i in Dn4000], False)
                dmed, dmederr, rmean = getMedDn4000(Rnorm, Dn4000, [0 for i in Dn4000], True)
                xx, yy, ye = [], [] ,[]

                for data in range(len(dmed)):
                    #if(rmean[data] < 0.0 and rmean[data] > -1.45):
                    if (xproj == 'Rnorm' or xproj == 'InfallTime'):
                        if(rmean[data] >= 0.0 and rmean[data] < 3.0):
                            xx.append(rmean[data])
                            yy.append(dmed[data])
                            ye.append(dmederr[data])
                    elif(xproj == 'LogRnorm' or xproj == 'LogInfallTime'):
                        if(rmean[data] < 0.47712125472):
                            xx.append(rmean[data])
                            yy.append(dmed[data])
                            ye.append(dmederr[data])

                popt, pcov = fitDn4000vsRnorm([xx, yy, ye], fitMode='linear')
                err =  np.sqrt(pcov.diagonal())

                x = None
                if (xproj == 'LogRnorm' or xproj == 'LogInfallTime'):
                    x = np.linspace(np.min(rmean), 0.47712125472, 1000)
                elif (xproj == 'Rnorm' or xproj == 'InfallTime'):
                    x = np.linspace(0.0, 3.0, 1000)
                y = linear(x, popt[0], popt[1])

                # Stores plotting data
                temp.append([rmean, dmed, dmederr, drange])
                temp.append([x, y])

                tempSlope.append(popt[0])
                tempserr.append(err[0])
                tempIntercept.append(popt[1])
                tempierr.append(err[1])
            else:
                tempSlope.append(None)
                tempIntercept.append(None)
                tempserr.append(None)
                tempierr.append(None)

        plotData.append(temp)
        slope.append(tempSlope)
        intercept.append(tempIntercept)
        serr.append(tempserr)
        ierr.append(tempierr)
        setcol = True
    slope = np.array(slope)
    intercept = np.array(intercept)
    serr = np.array(serr)
    ierr = np.array(ierr)

    return slope, serr, intercept, ierr, rowName, colName, plotData

def processFitResult(slopes, serr, intercepts, ierr):
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    slopes_new = [[np.median(slopes[:,0,0])], [np.median(slopes[:,1,0])], [np.median(slopes[:,2,0])]]
    intercepts_new = [[np.median(intercepts[:,0,0])], [np.median(intercepts[:,1,0])], [np.median(intercepts[:,2,0])]]
    iqrs = [[iqr(slopes[:,0,0], rng=(16, 84))], [iqr(slopes[:,1,0], rng=(16, 84))], [iqr(slopes[:,2,0], rng=(16, 84))]]
    iqri = [[iqr(intercepts[:,0,0], rng=(16, 84))], [iqr(intercepts[:,1,0], rng=(16, 84))], [iqr(intercepts[:,2,0], rng=(16, 84))]]
    errs = [[np.std(ft.bootstrap(slopes[:,0,0], 500, bootfunc = np.median))], [np.std(ft.bootstrap(slopes[:,1,0], 500, bootfunc = np.median))], [np.std(ft.bootstrap(slopes[:,2,0], 500, bootfunc = np.median))]]
    erri = [[np.std(ft.bootstrap(intercepts[:,0,0], 500, bootfunc = np.median))], [np.std(ft.bootstrap(intercepts[:,1,0], 500, bootfunc = np.median))], [np.std(ft.bootstrap(intercepts[:,2,0], 500, bootfunc = np.median))]]

    return slopes_new, iqrs, errs, intercepts_new, iqri, erri

# Calculates the alternate Absolute Magnitude calculation
def getAbsMag2(m, z):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    lumDistance = cosmo.luminosity_distance(z).value
    return m - 5.000000 * math.log10(lumDistance * 1.000000e6) + 5.000000

# Calculates the apparent Magnitude
def getAppMag(M, z):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    lumDistance = cosmo.luminosity_distance(z).value
    return M + 5.000000 * math.log10(lumDistance * 1.000000e6) - 5.000000

def f(x, a, b):
    return a * np.log(x) + b


def fitfunc(data):
    x = data[0]
    y = data[1]
    #yerr = data[2]
    popt, pcov = curve_fit(f, x, y)

    return popt, pcov

def getStackedSampleFromMagLimit(plotDataDict, limit50, lumDistDict, dataIndex=1):
    # Sets up the progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(marker=u'\u2588', fill='.', left='|', right='|'), "Preparing The Stacked Sample"]
    pbar = progressbar.ProgressBar(widgets=widgets) if(showProgressBars) else progressbar.bar.NullBar()

    _clusters = centerData[:,0]
    _redshift = centerData[:,1].astype(np.float)
    _M200 = centerData[:,3].astype(np.float)
    _R200 = centerData[:,2].astype(np.float)
    _dispersionVel = centerData[:,4].astype(np.float)

    clusters, redshift, M200, R200, Mr, zo, dispersionVel, MrComp = [], [], [], [], [], [], [], []

    medDn4000, meanRnorm = [], []

    Dn400Stacked, rnornStacked, MrStacked, zoStacked = [], [], [], []

    for cluster in pbar(clusterList):
        if cluster not in plotDataDict:
            continue
        elif (len(plotDataDict[cluster]) == 0):
            continue
        for index, c in enumerate(_clusters):
            if(cluster == c):
                clusters.append(_clusters[index])
                redshift.append(_redshift[index])
                M200.append(_M200[index])
                R200.append(_R200[index])
                dispersionVel.append(_dispersionVel[index])
                #print(index, ": ", limit50[c])
                if limit50[c] is not None:
                    MrComp.append(getAbsMag(limit50[c], c, lumDistDict))
                else:
                    MrComp.append(None)


        rnorm = np.array(plotDataDict[cluster][dataIndex][:,1])
        Dn4000 = np.array(plotDataDict[cluster][dataIndex][:,0])
        Mr = np.array(plotDataDict[cluster][dataIndex][:,3])
        zo = np.array(plotDataDict[cluster][dataIndex][:,4])

        c1 = np.where(np.logical_and(Dn4000 > 0, Dn4000 < 3))

        Dn4000 = Dn4000[c1]
        zo = zo[c1]
        rnorm = rnorm[c1]
        Mr = Mr[c1]

        c4 = np.where(rnorm > 0)

        Dn4000 = Dn4000[c4]
        rnorm = rnorm[c4]
        Mr = Mr[c4]
        zo = zo[c4]

        binSize = 200 * 0.001 # Mpc
        RclMax = np.max(rnorm)
        binNum = int(math.ceil(RclMax/ binSize))

        indexBins = [np.where(np.logical_and(rnorm > i * binSize,  rnorm < (i+1) * binSize))  for i in range(binNum)]

        Dn400Stacked.append(Dn4000)
        rnornStacked.append(rnorm)
        MrStacked.append(Mr)
        zoStacked.append(zo)

        medDn4000_ = []
        meanRnorm_ = []
        for i in range(len(indexBins)):
                if (Dn4000[indexBins[i]].size != 0):
                    medDn4000_.append(np.median(Dn4000[indexBins[i]]))

                if (Dn4000[indexBins[i]].size != 0):
                    meanRnorm_.append(np.mean(rnorm[indexBins[i]]))

        medDn4000.append(medDn4000_)
        meanRnorm.append(meanRnorm_)

    Dn4000Stacked = np.array(Dn400Stacked, dtype=object)
    rnornStacked = np.array(rnornStacked, dtype=object)
    MrStacked = np.array(MrStacked, dtype=object)
    zoStacked = np.array(zoStacked, dtype=object)

    medDn4000 = np.array(medDn4000, dtype=object)
    meanRnorm = np.array(meanRnorm, dtype=object)
    clusters= np.array(clusters)

    return clusters, redshift, M200, R200, Mr, zo, Dn400Stacked, rnornStacked, MrStacked, zoStacked, medDn4000, meanRnorm, dispersionVel, MrComp

def magnitudeLimit(plotDataDict, completnessDict):
    lumDistDict = getLumDistDict()
    limit50, _ = getCompletenessLimit(completnessDict, useAbsMag=False)

    plotData = []
    clusters, redshift, M200, R200, Mr, zo, Dn400Stacked, rnornStacked, MrStacked, zoStacked, medDn4000, meanRnorm, dispersionVel, _ = getStackedSampleFromMagLimit(plotDataDict, limit50, lumDistDict)

    redshiftBinStart = redshiftBinMin if (redshiftBinMin is not None) else np.min(redshift)
    redshiftBinEnd = redshiftBinMax if (redshiftBinMax is not None) else np.max(redshift)
    redshiftBinSize = abs(redshiftBinEnd - redshiftBinStart) / redshiftBinNum

    massBinStart = massBinMin if (massBinMin is not None) else np.min(M200)
    massBinEnd = massBinMax if (massBinMax is not None) else np.max(M200)
    massBinSize = abs(massBinEnd - massBinStart) / massBinNum


    MrStacked = np.array(MrStacked)
    zoStacked = np.array(zoStacked)
    MrS, zoS = [], []
    zl = []
    ml = []
    cl = []
    sx, sy = [], []
    for m, z in zip(MrStacked, zoStacked):
        zoSCleaned = list(filter(None, z))
        MoSCleaned = list(filter(None, m))
        MrS.extend(MoSCleaned)
        zoS.extend(zoSCleaned)

    for c, z in zip(clusters, redshift):
        # Clusters below 50% completeness after magnitude correction
        #if (c in ['MKW4', 'A779', 'MKW8', 'NGC6338', 'A295', 'A1446', 'A2219', 'A1835']):
        #    sx.append(z)
        #    sy.append(getAbsMag(limit50[c], c, lumDistDict))
        #    continue
        if (limit50[c] == None):
            continue
        zl.append(z)
        ml.append(getAbsMag(limit50[c], c, lumDistDict))
        cl.append(c)

    zl = np.array(zl)
    ml = np.array(ml)
    cl = np.array(cl)

    MrS = np.array(MrS)
    zoS = np.array(zoS)

    binSize = 0.02
    zMin = np.min(zoS)
    zMax = np.max(zoS)
    binNum = int(math.ceil(abs(zMax - zMin )/ binSize))
    indexBins = []
    indexBins.extend([np.where(np.logical_and(zoS > i * binSize + zMin,  zoS <= (i+1) * binSize + zMin))  for i in range(binNum)])

    colors = iter(cm.rainbow(np.linspace(0, 1, len(indexBins) + 1)))

    limit = []
    x = []
    l1 = None
    x1 = None
    for indices in indexBins:
        m = MrS[indices]
        p90 = np.percentile(m, 90)
        if (l1 == None):
            l1 = p90
            x1 = np.min(zoS[indices])
        limit.append(p90)
        x.append(np.min(zoS[indices]))
        col = next(colors)


    popt, pcov = fitfunc([zl[np.where(zl <= 0.1)],ml[np.where(zl <= 0.1)]])
    err =  np.sqrt(pcov.diagonal())

    lim = f(0.1, popt[0], popt[1])
    cList = []
    for c, m, z in zip(cl, ml, zl):
        if(True): #z < 0.1
            # Below blue line
            rlm = f(z, popt[0], popt[1])
            if (m >= rlm):
                cList.append(c)
        else:
            # Below red curve
            if(m > lim):
                cList.append(c)
    #print("Clusters: ", len(cList))
    #print(cList)

    xx = np.linspace(0.02,0.1, 1000)
    yy = f(xx, popt[0], popt[1])
    plotData.append([zl, ml]) # Clusters
    plotData.append([zoS, MrS]) # Galaxies
    plotData.append([xx, yy]) # Curve fit 0 < z < 0.1

    m = getAppMag(-20.625824886161126, 0.1)

    x = np.linspace(0.02, 0.1, 1000)
    y = []
    for xx in x:
        y.append(getAbsMag2(m, xx))

    return lim, plotData

def onSegment(p, q, r):

    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True

    return False

def orientation(p, q, r):

    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))

    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
    return null

def doIntersect(p1, q1, p2, q2):

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True

    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True

    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True

    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True

    return False

def isInsidePolygon(points: list, p: list) -> bool:

    n = len(points)

    if n < 3:
        return False

    extreme = (INT_MAX, p[1])
    count = i = 0

    while True:
        next = (i + 1) % n

        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):

            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])

            count += 1

        i = next

        if (i == 0):
            break

    return (count % 2 == 1)

def generateRelativeVelocityPlots(plotDataDict, zmin, zmax, massMin, massMax, useCorrection = True, useMagLimit = True):
    plotData = []
    _dataIndex = 1 if (useCorrection) else 0
    clusters, redshift, M200, R200, Mr, zo, Dn4000Stacked, rnornStacked, MrStacked, zoStacked, medDn4000, meanRnorm, dispersionVel, dVStacked = getStackedSample(plotDataDict, dataIndex = _dataIndex)
    clusterZ = np.array(redshift)
    c = 299792458  * 0.001 # km/s, speed of light
    dVS = []
    RnS = []
    dn4000 = []

    for zo, zc, rnorm, dn, clu, sigmaV, m200, MrS, dVc in zip(zoStacked, clusterZ, rnornStacked, Dn4000Stacked, clusters, dispersionVel, M200, MrStacked, dVStacked):
        condition = []
        for i, (mr, rn, dn4, dvc) in enumerate(zip(MrS, rnorm, dn, dVc)):
            if (mr == None):
                dn4000.append(dn4)
                RnS.append(rn)
                dVS.append(dvc)
            elif(mr < MrMax or not useMagLimit):
                condition.append(i)
        condition = np.array(condition, dtype=int)


        if (zc < zmin or zc > zmax):
            continue
        if (m200 < massMin or m200 > massMax):
            continue

        dn = np.array(dn)
        zo = np.array(zo)
        rnorm = np.array(rnorm)

        dn = dn[condition]
        zo = zo[condition]
        rnorm = rnorm[condition]

        dn4000.extend(dn)
        RnS.extend(rnorm)
        zo = np.array(zo)


        rnorm = np.array(rnorm)
        dV = c * abs((zo - zc)) / (1.0 + zc)
        dVS.extend(dV / sigmaV)

        rnormBinStart = 0
        rnormBinEnd = 3
        rnormBinNum = 30
        rnormBinSize = (rnormBinEnd - rnormBinStart) / rnormBinNum

        dVmed = []
        rmed = []
        for i in range(rnormBinNum):
            condition = np.where(np.logical_and((rnormBinStart + i * rnormBinSize) <= rnorm, (rnormBinStart + (i + 1) * rnormBinSize) > rnorm))
            dVmed.append(np.median(dV[condition]))
            rmed.append(np.mean(rnorm[condition]))

    rnormBinStart = 0
    rnormBinEnd = 3
    rnormBinNum = 20 # Try to reduce ~25/ couple
    rnormBinSize = (rnormBinEnd - rnormBinStart) / rnormBinNum

    dVBinStart = 0
    dVBinEnd = 3.5
    dVBinNum = 20
    dVBinSize = (dVBinEnd - dVBinStart) / dVBinNum

    RnS = np.array(RnS)
    dVS = np.array(dVS)
    dn4000 = np.array(dn4000)

    dVmed = []
    rmed = []
    x = []
    y = []
    d = []

    r1 = [(0.0, 0.0), (0.0, 1.97), (0.5, 0.0)]
    r2 = [(0.37, 0.5), (1.38, 0.5), (1.52, 0.0), (0.5, 0.0)]
    r3 = [(0.37, 0.5), (1.13, 0.5), (0.88, 1.52), (0.13, 1.52)]
    r4 = [(0.0, 1.97), (0.0, 2.52), (0.38, 2.52), (0.63, 1.52), (0.13, 1.52)]
    r5 = [(0.38, 2.52), (3.0, 1.0), (3.0, 0.0), (1.52, 0.0), (1.38, 0.5), (1.13, 0.5), (0.88, 1.52), (0.63, 1.52)]

    r1Dn4000 = []
    r2Dn4000 = []
    r3Dn4000 = []
    r4Dn4000 = []
    r5Dn4000 = []
    r6Dn4000 = []

    rEData = []

    c = 0
    tx = []
    ty = []
    tx1 = []
    ty1 = []
    for i in range(rnormBinNum):
        for j in range(dVBinNum):
            condition = np.where(np.logical_and(np.logical_and((rnormBinStart + i * rnormBinSize) <= RnS, (rnormBinStart + (i + 1) * rnormBinSize) > RnS), np.logical_and((dVBinStart + j * dVBinSize) <= dVS, (rnormBinStart + (j + 1) * dVBinSize) > dVS)))
            dVmed.append(np.median(dVS[condition]))
            rmed.append(np.mean(RnS[condition]))
            _x = []
            _y = []
            _d = []
            for dv, rnorm, dn400 in zip(dVS[condition], RnS[condition], dn4000[condition]):
                if (dn400 > 0.9 and dn400 < 3.0):
                    _x.append(rnorm)
                    _y.append(dv)
                    _d.append(dn400)

                    if (isInsidePolygon(r1, [rnorm, dv])):
                        r1Dn4000.append(dn400)
                    elif (isInsidePolygon(r2, [rnorm, dv])):
                        r2Dn4000.append(dn400)
                    elif (isInsidePolygon(r3, [rnorm, dv])):
                        r3Dn4000.append(dn400)
                    elif (isInsidePolygon(r4, [rnorm, dv])):
                        r4Dn4000.append(dn400)
                    elif (isInsidePolygon(r5, [rnorm, dv])):
                        rEData.append([rnorm, dv, dn400])
                        if (dn400 > 1.8):
                            pass
                        else:
                            tx1.append(rnorm)
                            ty1.append(dv)
                        r5Dn4000.append(dn400)
                    else:
                        tx.append(rnorm)
                        ty.append(dv)
                        r6Dn4000.append(dn400)

            x.append(np.mean(_x))
            y.append(np.mean(_y))
            d.append(np.median(_d))

    rEData = np.array(rEData)

    r1Dn4000, r2Dn4000, r3Dn4000, r4Dn4000, r5Dn4000, r6Dn4000 = [], [], [], [], [], []
    import math

    for dn, v, r in zip(d, y, x):
        #if (math.isnan(dn)):
        #    continue
        if (np.isnan(dn)):
            pass
        elif (isInsidePolygon(r1, [r, v])):
            r1Dn4000.append(dn)
        elif (isInsidePolygon(r2, [r, v])):
            r2Dn4000.append(dn)
        elif (isInsidePolygon(r3, [r, v])):
            r3Dn4000.append(dn)
        elif (isInsidePolygon(r4, [r, v])):
            r4Dn4000.append(dn)
        elif (isInsidePolygon(r5, [r, v])):
            r5Dn4000.append(dn)
        else:
            r6Dn4000.append(dn)

    # loess smoothing
    x = np.array(x)
    y = np.array(y)
    d = np.array(d)

    cond = np.where(np.logical_and(np.logical_and(~np.isnan(x), ~np.isnan(y)), np.logical_and(~np.isnan(d), y < 300.0)))

    x = x[cond]
    y = y[cond]
    d = d[cond]

    xx = x[np.where(y < 2.7)]
    yy = y[np.where(y < 2.7)]
    dd = d[np.where(y < 2.7)]

    zout, wout = loess_2d(xx, yy, dd, degree=1)

    plotData.append([xx, yy, zout])
    plotData.append(r1Dn4000)
    plotData.append(r2Dn4000)
    plotData.append(r3Dn4000)
    plotData.append(r4Dn4000)
    plotData.append(r5Dn4000)
    plotData.append(r6Dn4000)

    return plotData

def getCompletenessAtMagLimit(completnessDict, useAbsMag=False, lumDistDict=None):
    # Sets up the progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(marker=u'\u2588', fill='.', left='|', right='|'), "Calculating the Completeness Limit for Each Cluster"]
    pbar = progressbar.ProgressBar(widgets=widgets) if(showProgressBars) else progressbar.bar.NullBar()
    # Initializing the completeness limit dictionaries
    compLimit = {}

    for cluster in pbar(clusterList):
        # Checks if the completnessDict is not None
        if(cluster not in completnessDict):
            continue
        elif(completnessDict[cluster] is None):
            compLimit[cluster] = None
            continue

        # Gets the completeness and the r-band mag values
        rband = None
        if not useAbsMag:
            # Use apparent magnitude
            rband = np.array(completnessDict[cluster][1])
        else:
            #use absolute magnitude
            if lumDistDict == None:
                # Computes lumDistDict if needed
                lumDistDict = getLumDistDict()
            #print(rband)
            rband = getAbsMag(np.array(completnessDict[cluster][1]), cluster, lumDistDict)
            #print("Yo")
            #print(rband)
        completeness = np.array(completnessDict[cluster][0])

        # Gets the indices of the arrays for r-band mag values within the min/max value range
        index = np.where(np.logical_and(rband > -24.0, rband < rbandMax))

        if (generateCompletenessPlots):
            # Initializes a spline and computes y values for ploting
            f = InterpolatedUnivariateSpline(rband[index], completeness[index])
            xdata = np.linspace(np.min(rband[index]), np.max(rband[index]), 1000)
            ydata = f(xdata)

            # Plots the completeness as a function of member fraction
            plt.plot(xdata, ydata)
            plt.xlabel("r-band mag")
            plt.ylabel("completeness")
            plt.xlim(rbandMin, rbandMax)
            plt.title(cluster)


        try:
            # Initializes a spline for when the y values are shift down by the completeness limit
            invf = InterpolatedUnivariateSpline(rband[index], completeness[index])
            compLimit[cluster] = invf(MrMax)
        except:
            # Case where the complenetess never reaches the completeness limit
            compLimit[cluster] = None

        if (generateCompletenessPlots):
            plt.savefig(completenessPlotsDir + "/" + cluster + ".png")

    return compLimit
