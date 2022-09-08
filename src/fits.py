import os, copy
import csv
from includes.cosmocalc import cosmocalc
from astropy.io import fits
import astropy.units as u
from specutils import Spectrum1D
from specutils.manipulation import gaussian_smooth, median_smooth, box_smooth
import numpy as np
import matplotlib.pyplot as plt


clusters = ["MKW4","A1367","MKW11","A779","Coma","NGC4325","RXCJ2214p1350","MKW8","NGC6338","A2197","Zw1665","A2199","NGC6107","A1314","A1185","A2063","A1142","A2147","HeCS_faint1","A76","RXJ0137","PegasusII","A2107","A2593","A295","A160","A119","A168","A957","SHK352","A193","IC1365","A671","A1291A","A757","A1377","MSPM02035","RXCJ1022p3830","A2665","A85","A1291B","A2399","A2256","A2169","A2457","A602","A736","RXCJ1351p4622","A1795","MSPM01721","A1668","A1436","A2149","A311","A2092","A1035A","A2124","A1066","RXCJ1115p5426","A1589","A1767","A1691","RXCJ1053p5450","A744","A2065","A2067","A2064","A1775","A1831","A1424","A1190","A1800","A1205","A1173","A2670","RXCJ1210p0523","Zw1215","A2029","A1773","A2061","A1809","RXJ2344m0422","A2495","A2255","A1035B","SDSS_C4_DR3_3144","A1307","RXCJ1326p0013","MS1306","A2428","A1663","A1650","A1750","A1552","A2249","A2245","A2018","A1885","MSPM06300","SDSSCGA0782","RXJ1540p1752","A1728","A2142","A2440","A971","A954","A21","A2175","A2110","A2244","A2055","A1446","Zw1478","A7","A1235","A98S","A1925","A2220","A2443","Zw8197","A2034","A2069","A1302","A1361","A1366","A1986","A2051","A2050","A1033","Zw0836","A655","A961","A646","A1930","A620","A1437","A743","A1132","RMJ0022p2317","A1272","A795","WARPJ1416p2315","A1068","A1918","A329","A1413","A990","Zw3179","RMJ0041p2526","A2409","A667","A1978","A2009","A657","MS2348p2929","A980","Zw8284","RXJ1720","A2259","A1902","A750","A2454","A1914","A1553","A1201","WHLJ134901p491840","A344","A586","A1204","MS0906","RMJ0835p2046","Zw8193","A665","RMJ0727p4227","A2187","A1689","RMJ0751p1737","A383","A115","A2396","A1246","Zw1883","RMJ2259p3102","A2623","A291","A625","RMJ0001p1203","A963","RMJ0727p3846","RMJ0826p3108","RMJ0737p3517","A1423","Zw2701","RXJ1504","RMJ0756p3839","A773","A2261","A2219","Zw1693","A1682","RMJ2326p2921","A2390","A267","A2111","RMJ0758p2641","A2355","A1763","RXJ2129","Zw2089","RMJ2201p1118","RMJ0108p2758","RMJ0230p0247","RMJ0051p2617","A68","A1835","A2645","Zw348","RMJ0830p3224","Zw7160","A1758","A2631","A1703","A689","A697","A611","Zw3146","A2537"]

"""
Name:
    fetchFITS(filePaths = "./", option = 'r')
Description:
    This routine uses the astropy.fits routines to fetch the data from FITS file for
    a single or multiple FITS file(s). With no parameters this routine will return the
    infomation for all FITS files in the current directory.

Parameters:
    filePaths - The path of the file(s) or directory that you want gather FITS file(s) data from (Default = ./)
    option - Store a char that tells the routine what mechanism will be used to file FITS files
        s - Gathers infomation for a specified FITS files
        r - Finds all FITS files in just the filePath (Default option)
        rr - Recursively finds all FITS file in the filePath
Return:
    This routine returns an array (or single) opened FIT files using astropy's fit.open routine
"""
def fetchFITS(filePaths = "./", option = 'r'):

    # inits/gathers the file paths
    if (option == 'r' or option == 'rr'):
        filePaths = findFITS(filePaths, 's' if option == 'r' else 'r')
    else:
        if type(filePaths) != type([]):
            filePaths = [filePaths]

    # inits a array to store the data from FITS files
    data = []

    # collects data from the FITS files
    for file in filePaths:
        try:
           hdul = fits.open(file)
        except FileNotFoundError:
            print("WARNING: " + file + " WAS NOT FOUND.")
            continue
        data.append(hdul)

    return data

"""
Name:
    findFITS(directory = ./, option = 's')
Description:r
    This routine finds all FITS files in directory. Without any parameter this routine
    returns all FITS files in just the current directory.
Parameters:
    directory - The directory where the FITS files fill be found in.
    option - How FITS will be found in the directory
        s - FITS will be found in the directories specified only (Default)
        r - FITS will be found in the specified directories and all sub directories
Return:
    This routine returns an array of the FITS files directories
"""
def findFITS(directory = "./", option = 's'):

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
            if file.endswith(".fits"):
                filePaths.append(os.path.join(path, file))

    return filePaths

"""
Name:
    fastScanDir(directory)
Description:
    This routine finds all sub-directories in directory.
Parameters:
    directory - The parnet directory
Return:
    A list of a sub-directories of the parnet directory
"""
def fastScanDir(directory):

    # Gets list of sub Folders
    subFolders = [f.path for f in os.scandir(directory) if f.is_dir()]

    # Loops through each folder and adds all subsequent sub Folders
    for directory in list(subFolders):
        subFolders.extend(fastScanDir(directory))

    return subFolders

"""
Name:
    getVar(FitFiles, var)
Description:

Parameters:

Return:

"""
def getVar(FitFiles, var):
    variables = []

    for data in FitFiles:
        variables.append(data[0].header[var])

    return variables

"""
Name:

Description:

Parameters:

Return:

"""
def getFlux(data):
    Fluxes, err = [], []
    for d in data:
        Fluxes.append(d[0].data[0])
        err.append(d[0].data[1])
    return Fluxes, err

"""
Name:

Description:

Parameters:

Return:

"""
def getWavelength(data):

    CRVAL1List = getVar(data, "CRVAL1")
    CD1_1List = getVar(data, "CD1_1")
    pixels = [i for i in range(len(data[0][0].data[0]))]

    wavelengths = []

    for CRVAL1, CD1_1 in zip(CRVAL1List, CD1_1List):
        temp = []
        for pixel in pixels:
            temp.append(CRVAL1 + CD1_1 * pixel)
        wavelengths.append(temp)

    return wavelengths

"""
Name:
    getDn4000(data)
Description:

Parameters:

Return:

"""
def getDn4000(data, func = np.mean):

    # gathers required data to compute the Dn4000
    wavelength = getWavelength(data)
    flux, fluxErr = getFlux(data)
    Dn4000 = []
    Derr = []
    b = []
    r = []
    mb = []
    mr = []
    # Loops through each dataset
    for W, F, Ferr in zip(wavelength, flux, fluxErr):

        # inits blue/red wavelength flux sets
        blue, red = [], []
        berr, rerr = [], []

        # Loops through each flux/wavelength value
        for w, f, ferr in zip(W, F, Ferr):
            if (w > 3850 and w < 3950):
                blue.append(f)
                berr.append(ferr)
            elif (w > 4000 and w < 4100):
                red.append(f)
                rerr.append(ferr)

        # Computes Dn4000 and it's error
        medBlue = func(blue)
        medBerr = np.sqrt(np.sum(np.array(berr) ** 2) / len(berr)**2)
        b.append(np.sum(blue))
        mb.append(medBlue)

        medRed = func(red)
        medRerr = np.sqrt(np.sum(np.array(rerr) ** 2) / len(rerr)**2)
        r.append(np.sum(red))
        mr.append(medRed)

        Derr.append(np.sqrt((medRerr**2 / medBlue**2) + ((medRed**2 * medBerr**2)/medBlue**4)))
        Dn4000.append(medRed / medBlue)

    return Dn4000, Derr

"""
Name:

Description:

Parameters:

Return:
"""
def closeFITS(data):
    for file in data:
        file.close()

"""
Name:

Description:

Parameters:

Return:
"""
def saveSpectra(data, directory = "./", plotError = True, lbound = 0, ubound = 4000):

    # Gathers reqired infomation for spectra plot
    flux, fluxErr = np.array(getFlux(data))
    wavelength = np.array(getWavelength(data))
    name = getVar(data, "FILENAME")
    for f, ferr, w, n, i in zip(flux, fluxErr, wavelength, name, [i for i in range(len(name))]):
        spec = Spectrum1D(spectral_axis=w * u.AA, flux = f * u.Unit('erg cm-2 s-1 AA-1'))
        spec = median_smooth(spec, width = 51)
        indices = np.where(w < 8000)[0]
        plt.plot(spec.spectral_axis[indices], spec.flux[indices], c = 'b' ,label = "Spectrum")
        if(plotError):
            plt.scatter(w[indices], ferr[indices], s = 0.1, c= 'r', label = "Flux Error")
        plt.ylabel("Flux")
        plt.xlabel("Wavelength (Angstrom)")
        plt.title(n)
        plt.legend(loc='best')
        _n = n.replace("/", "_")
        #plt.show()
        plt.savefig(directory + ("/" if directory[-1] != "/" else "") + str(i) + "_" + _n + ".jpeg")
        plt.cla()
        print(n + " Spectra Saved.")


def getClusterDict(data, index = 1):
    clusterInfo = []
    clusterDict = {cluster: [] for cluster in clusters}
    for clusterData in data[0][index].data:
        clusterDict[clusterData[0]].append(clusterData)
    return clusterDict


"""

Test Code

"""
if (__name__ == '__main__'):
    fileDir = "../HeCS_omnibus_dn4000.fits"
    data = fetchFITS(fileDir, 's')
    print(data[0][1].header)
    clusterDict = getClusterDict(data)
    print(clusterDict[clusters[0]][0])
    print(cosmocalc(0.020430, H0 = 70, WM = .3, WV = 0.7))
    """
    data = fetchFITS(fileDir, 'r')
    #saveSpectra(data, "../A1767_spectra/Spectra" ,plotError = False)
    Dn4000 = getDn4000(data)
    files = findFITS(fileDir, 's')

    f = open("Colby_Dn4000.txt", "w")
    writer = csv.writer(f)

    writer.writerow(["Filename","Total Blue Flux","Mean Blue Flux","Total Red Flux","Mean Red Flux", "Dn4000"])
    for i in range(len(files)):
        row = [files[i].replace("../A1767_spectra/",""), Dn4000[2][i], Dn4000[3][i], Dn4000[4][i], Dn4000[5][i], Dn4000[0][i]]
        writer.writerow(row)
    """
    """
    files = findFITS(fileDir, 's')
    differ = []
    Jfiles = np.loadtxt("dn.txt", dtype=str)[:,0]
    JDn4000 = (np.loadtxt("dn.txt", dtype=str)[:,3]).astype(float)
    JDerr = (np.loadtxt("dn.txt", dtype=str)[:,4]).astype(float)
    x = []
    g = []
    #print("Filename\t\t", "Jubee Dn4000\t", "Jubee Dn400 err\t", "My D4000\t", "My Dn4000 err")
    for file in files:
        data = fetchFITS(file, 's')
        Dn4000, Derr = getDn4000(data, func = np.mean)
        Dn4000 = Dn4000[0]
        Derr = Derr[0]
        file = file.replace("../A1767_spectra/","")
        for Jfile, JDn, JDnerr in zip(Jfiles, JDn4000, JDerr):
            if (Jfile == file):
                g.append(Dn4000)
                #print(file, "\t", JDn, "\t", JDnerr ,"\t" ,Dn4000, "\t", Derr)
                differ.append(Dn4000 - JDn)
                x.append(int(file[0:3]))


    fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Vertically stacked subplots')
    axs[0].scatter(g, JDn4000)
    axs[0].set(ylabel = "Jubee's Computed Dn4000")
    fig.suptitle("Mine Vs. Jubee's Computed Dn4000", fontsize="x-large")
    axs[1].scatter(g, differ)
    axs[1].set(ylabel = "Difference", xlabel = "My Computed Dn4000")
    plt.show()
    #plt.scatter(differ, JDn4000, c = 'b', s = 0.5)
    #plt.xlabel("File Number")
    #plt.ylabel("Dn4000 Absolute Difference")
    #plt.title("Mine Vs. Jubee's Dn4000 Calculation Difference")
    #plt.show()
    """
    closeFITS(data)
