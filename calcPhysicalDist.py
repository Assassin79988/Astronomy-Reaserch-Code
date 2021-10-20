import fits
import math
from cosmocalc import cosmocalc
import numpy as np
import matplotlib.pyplot as plt
import random, time, os
from astropy.stats import bootstrap
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


"""
Name: findJPEG(directory = "./", option = 's')

Description: This method parses the folders and returns the name of each
             JPEG file.
Parameters: directory - the parnet directory to search.
            option - Recursively serch a folder or not.
                s - only look for jpeg in current folders
                r - look for jpeg in current folder and all sub folders

Return: List of cluster where I visually saw a trend.
"""
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

"""
Name: saveObject(obj, name)

Description: This method takes in any object as an input and
             save the object as a pickel file. This method assumes
             a ./obj folder currently exist.

Parameters:
    obj - object to be saved.
    name - name of the folder

Return: None.
"""
def saveObject(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

"""
Name: loadObject(name)

Description: This method load a object from a pickel file.

Parameters:
    name - name of the file to be loaded.

Return: Object store in the pickle file
"""
def loadObject(name):
    with open('obj/' + name + '.pkl', 'rb') as file:
        return pickle.load(file)

"""
Name: generateCatalogue(fileName):

Description: This method generate a catalogue of each cluster of interest
             with there physical distance data.

Parameters:
    fileDir - name and directory of the file where the .fits file is located.

Return:
    A dict containing each cluster name and there physical distance.
"""
def generateFullCatalogue(fileDir):
    data = fits.fetchFITS(fileDir, 's')

    clusterDict = fits.getClusterDict(data)
    physDict = {cluster: None for cluster in fits.clusters}

    centerData = np.loadtxt("./HeCS_omnibus_cluster_center.txt", dtype='str')

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
    physDict = generateFullCatalogue("./HeCS_omnibus_dn4000.fits")
    saveObject(physDict, "HeCS-omnibus_PhysicalDistanceData")
    print("Dict Complied and Saved!")
    #physDict = loadObject("HeCS-omnibus_PhysicalDistanceData")
