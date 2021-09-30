import fits
import math
from cosmocalc import cosmocalc
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':

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





    """
    Dn4000 vs Rcl Plot Code
    
    for cluster in fits.clusters:
        Dn4000 = []
        Dnerr = []
        Rcl = []
        for data in physDict[cluster]:
            if(np.float64(data[5]) > 0):
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
    





