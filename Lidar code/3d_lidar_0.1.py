#------IMPORTS-----
#Packages for ETA backend
import json
import etabackend.eta #Available at: https://github.com/timetag/ETA, https://eta.readthedocs.io/en/latest/
import etabackend.tk as etatk

#Packages used for analysis
import numpy as np
from pathlib import Path
import os
import time as t
from scipy.signal import find_peaks

#Packages used for curve fitting
import lmfit as lm
from lmfit.models import GaussianModel, ConstantModel, SkewedGaussianModel
from lmfit import Parameters

#Packages for plotting
from matplotlib import pyplot as plt
from colour import Color
import random

import LIDAR_lib as lidar  #Contains functions for 3D analysis

def ToF_analysis(timetag_file, recepie_file, ch_sel):
    load_start = t.time()
    #Load the recipe from seperate ETA file
    with open(recepie_file, 'r') as filehandle:
        recipe_obj = json.load(filehandle)

    eta_engine = etabackend.eta.ETA()
    eta_engine.load_recipe(recipe_obj)

    #Set parameters in the recipe
    eta_engine.recipe.set_parameter("binsize", str(binsize))
    eta_engine.recipe.set_parameter("bins", str(bins))
    eta_engine.recipe.set_parameter("sync_delay", str(sync_delay))
    eta_engine.load_recipe()
    load_time = t.time() - load_start

    #---File handling---
    file = Path(timetag_file)

    #------ETA PROCESSING-----
    START = t.time()
    """
    Code block loads the time-tagging data and runs the ETA script to genereate ToF histograms
    """
    TOF_anal_start = t.time()
    print("Starting TOF analysis")
    cutfile = eta_engine.clips(file)
    result = eta_engine.run({"timetagger1":cutfile}, group='quTAG') #Runs the time tagging analysis and generates histograms
    histogram=result[ch_sel] #Selects the intended output from ETA, in this case it returns a 2-d array. The y axis for all ToF histograms. X axis must be recreated seperatly
    TOF_anal_time =  t.time() - TOF_anal_start

    return histogram, START, load_time, TOF_anal_time

def scatter(data):
    ##### Extract the data for plotting ####
    Xdata, Ydata, Zdata = [], [], []
    for direction in data:
        for peak in data[direction]:
            x,y,z = lidar.XYZ(np.abs(peak), direction[0], direction[1])
        
            Xdata.append(x)
            Ydata.append(y)
            Zdata.append(z)

    print("Number of data points: ", len(data))
    print("Number of points to plot: ", len(Xdata))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111,projection="3d")

    
    ax.scatter(Ydata,Zdata,Xdata,s=2) #Plots the data
    #ax.azim = int(z_rot) #sets a rototion of the plot if needed, not used much
    #ax.elev = int(x_rot)
    
    #ax.set_title("3D reconstruction", fontsize = 40)
    ax.set_xlabel("X mm", fontsize = 40, labelpad=40)
    ax.set_ylabel("Y mm", fontsize = 40, labelpad=40)
    ax.set_zlabel("Z mm", fontsize = 40, labelpad=40)
    
    ax.tick_params(labelsize=30)

    plt.show()

def multi_gauss(x,y, ref, background = 6):
    binsize = x[1]-x[0]
    peaks, _ = find_peaks(y, height = background, distance = 50)

    #print("Peaks: ", x[peaks])

    sigma = 22
    c = 3
    
    supermodel = ConstantModel()
    supermodel.set_param_hint('c', value = c)
    for i in range(len(peaks)):
        pref = 'f'+str(i)+'_'
        model = GaussianModel(prefix = pref)
        supermodel = supermodel + model

        supermodel.set_param_hint(pref+'amplitude', value = y[peaks[i]]*sigma*np.sqrt(2*np.pi) )
        supermodel.set_param_hint(pref+'center', value = x[peaks[i]])
        supermodel.set_param_hint(pref+'sigma', value = sigma)

    #Results
    result = supermodel.fit(y, x=x)
    #result.plot()
    #plt.show()
    
    bestparam = result.params
    X = np.arange(np.min(x),np.max(x), (np.max(x)-np.min(x))/1e6)
    Y = supermodel.eval(params=bestparam, x=X)

    C = []
    for i in range(len(peaks)):
        pref = f'f{i}_center'
        C.append(bestparam[pref].value)
        #print(f"Center of peak {i}: {center} ps")

    D = [lidar.calcDistance(c, ref*binsize) for c in C]
    return D, peaks

def multi_peak(x, y, ref):
    binsize = x[1] - x[0]
    peaks, _ = find_peaks(y, height = 6, distance = 50)

    D = [lidar.calcDistance(p*binsize, ref*binsize) for p in peaks]
    return D, peaks
    

#--------------- Analysing the ToF histograms -----------
def analyse_3d(histogram, index_cutoff, index_ref, x_deg, y_deg, dimX = 100, dimY = 100):
    print("Starting 3D analysis")
    d_data = {}  #used to store the distance data
    average_peak = 0  #some fun values to keep track of
    average_failed_peak = 0 #some fun values to keep track of
    
    num_peaks = {0:0, 1:0, 2:0, 3:0}   #Used to keep track of the number of peaks per pixel
    start = t.time() #Evaluate the time efficiency of the algorithms
    
    background = 10
    avg_num_of_peaks = 0
    """
    Code block loops through all histograms. Removes background/internal reflections and calculates the distance to a reference value that must be measured separately (but is reused for all scans)
    """
    for i in range(0,dimY):
        print(i,"/",dimY)
        for j in range(0,dimX):
            h = histogram[j][i]
            h[:index_cutoff] = 0 #Cuts away the internal reflections, background_cutoff is assigned in ETA frontend and is based on a background measurement. 

            distances, peaks = multi_gauss(time, h, index_ref)
            #distances, peaks = multi_peak(time, h, index_ref)
            
            d_data[(x_deg[i],y_deg[j])] = distances #TODO

            avg_num_of_peaks += len(peaks)
            average_peak += np.sum(h[peaks])

            try:
                num_peaks[len(peaks)] += 1
            except KeyError:
                num_peaks[len(peaks)] = 1

    print("Average number of peaks: ", num_peaks)
                
    stop = t.time()
    print("Average peak: ", average_peak/(dimY*dimX - num_peaks[0]))
    print("3D analysis time: ", stop-start)
    return d_data

"""Set Parameters for analysis and plotting"""
recipe = "C:/Users/staff/Documents/Lidar LF/ETA_recipes/quTAG_LiDAR_1.1.eta"
#.timeres file to analysed
file = "C:/Users/staff/Documents/Lidar LF/Data/211101/hand_JU_10ms_VL_10MHz_400kHzCts_-17.5uA_[3,4,-6,-6]_100x100_211101.timeres"
file = "C:/Users/staff/Documents/Lidar LF/Data/210505/Cup_10ms_10MHz_1.8MHzCts_1550nm_[8,8,-8,-8]_100x100_210505.timeres"
anal_method = "gauss"

#Parameters for etabackend to generate histograms
binsize = 16 #Histogram binsize in ps
bins = 6250 #6250 #Number of bins in the histogram: bins*binsize should equal 1/f where f is the repition rate of the laser in use
ch_sel = 't1' #Selects a specific histogram
records_per_cut = 2e5 #Number of events to be used per evalution cycle in ETA, not important in this code

#Time offsets for different signals [ps]
sync_delay = 0 #40000  #All events of the sync channel is delayed by 40000 ps (not necessary)
bw_delay = 0
fw_delay = 0

histogram, START, load_time, TOF_anal_time = ToF_analysis(file, recipe, ch_sel)
time = (np.arange(0,bins)*binsize) #Recreate time axis
file = Path(file)

#----------------- Scan variables ---------------------
#Variable that were used when scanning. What region was scanned and how many points
#Scanning variables
rect = [8,8,-8,-8] #[3,4,-6,-6] #Voltage range of scan, linear to angle
dimX = 100 # number of steps in the scan steps == resolution
dimY = 100
x_deg, y_deg = lidar.angles(rect,dimX,dimY)

#Analysis parameters
index_cutoff = 4060 #5220 #2040 #Removes the background noise. This value depends on the specifics of the setup and the delays. Must be optimised for new setups
index_ref = 4000 #5100 #Time index of the mirrors position, used as origin when calculating 3D point clouds. Not at zero because the laser must first travel to the optical setup. Mus be measured seperatly

#Plotting parameters
coff = 3  #Removes outliers for plotting purposes. Simply to avoid squished plots
z_rot = 270 #Angle of veiw in 3D plot
x_rot = 20

#lidar.save_pixel_array(histogram, file, binsize) #To save some raw data for troubleshooting
#lidar.save_all_pixels(histogram, file, dimX, dimY, binsize)
d_data = analyse_3d(histogram, index_cutoff, index_ref, x_deg, y_deg, dimX, dimY)

print("Loading time: ", load_time)
print("TOF analysis time: ", TOF_anal_time)
print("Total Analysis time: ", t.time()-START)
#-------------------- Save code -------------
print("Saving Images")
coff = int(coff) # prevents the images from being to squished

scatter(d_data)
#lidar.scatter(d_data, file, coff, anal_method + "_Fit_")
#lidar.save_data(d_data, file, anal_method + '_')

print("Job Done!")

