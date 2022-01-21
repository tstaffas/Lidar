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

import LIDAR_lib as lidar  #Contains functions for 3D analysis
import intensity_map

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

#--------------- Analysing the ToF histograms -----------
def analyse_3d(histogram, index_cutoff, index_ref, x_deg, y_deg, dimX = 100, dimY = 100, background = 3):
    print("Starting 3D analysis")
    d_data = {}  #used to store the distance data
    i_data = {}
    average_peak = 0  #some fun values to keep track of
    average_failed_peak = 0 #some fun values to keep track of
    
    F = 0   #Used to keep track of the number of pixels without peaks
    E = 0
    start = t.time() #Evaluate the time efficiency of the algorithms
    
    """
    Code block loops through all histograms. Removes background/internal reflections and calculates the distance to a reference value that must be measured separately (but is reused for all scans)
    """
    for i in range(0,dimY):
        print(i,"/",dimY)
        for j in range(0,dimX):
            h = histogram[j][i]
            h[:index_cutoff] = 0 #Cuts away the internal reflections, background_cutoff is assigned in ETA frontend and is based on a background measurement. 
            
            peak = np.amax(h) #Identifies the target peak
            if peak > background:  #removes pixels with only noise, noise threshold can be modified
                d, h = lidar.gauss(time,h,index_ref) #Gaussian algorithm
                #d = lidar.getDistance(index_ref, h, binsize = binsize)  #Peak finding Algorithm
                if d != np.NaN: #Gaussian algorithm can return np.NaN if unable to fit a curve to data, very unlikely after filtering away peaks with. It's a relic and might be obselete (but it's not hurting anyone)
                    x,y,z = lidar.XYZ(np.abs(d),x_deg[i],y_deg[j])
                    d_data[(x,y)] = z
                    i_data[(x,y)] = h

                elif d == np.NaN:
                    E+=1
                average_peak += peak
                
            else:
                x,y,z = lidar.XYZ(1,x_deg[i],y_deg[j])
                d_data[(x,y)] = np.NaN
                i_data[(x,y)] = np.NaN

                F +=1
                average_failed_peak += peak
                
    stop = t.time()
    print("Empty pixels: ", F)
    print("Failed pixels: ", E)
    print("Average peak: ", average_peak/(dimY*dimX - F))
    if F!=0:
        print("Average failed peak: ", average_failed_peak/F)

    print("3D analysis time: ", stop-start)
    return d_data, i_data

"""Set Parameters for analysis and plotting"""
recepie = "C:/Users/staff/Documents/Lidar LF/ETA_recipes/quTAG_LiDAR_1.1.eta"

#.timeres file to analysed
file = "C:/Users/staff/Documents/Lidar LF/Data/211101/hand_JU_10ms_VL_10MHz_400kHzCts_-17.5uA_[3,4,-6,-6]_100x100_211101.timeres"
file = "C:/Users/staff/Documents/Lidar LF/Data/210505/Cup_10ms_10MHz_1.8MHzCts_1550nm_[8,8,-8,-8]_100x100_210505.timeres"
file = 'C:/Users/staff/Documents/Lidar LF/Data/210602/spaceship_10ms_10MHz_200KHzCts_17uA_[8,8,-8,-8]_100x100_210602.timeres'
file = "C:/Users/staff/Documents/Lidar LF/Data/210426/other/Psyduck_10ms_870kHz_10MHz_1550nm_[8,8,-8,-8]_100x100_210426.timeres"
anal_method = "gauss"

#Parameters for etabackend to generate histograms
binsize = 16 #Histogram binsize in ps
bins = 6250 #Number of bins in the histogram: bins*binsize should equal 1/f where f is the repition rate of the laser in use
ch_sel = 't1' #Selects a specific histogram
records_per_cut = 2e5 #Number of events to be used per evalution cycle in ETA, not important in this code

#Time offsets for different signals [ps]
sync_delay = 0 #40000  #All events of the sync channel is delayed by 40000 ps (not necessary)
bw_delay = 0
fw_delay = 0

histogram, START, load_time, TOF_anal_time = ToF_analysis(file, recepie, ch_sel)
time = (np.arange(0,bins)*binsize) #Recreate time axis


#----------------- Variables ---------------------
#Scanning variables
rect = [8,8,-8,-8] #[3,4,-6,-6] #Voltage range of scan, linear to angle
dimX = 100 # number of steps in the scan steps == resolution
dimY = 100
x_deg, y_deg = lidar.angles(rect,dimX,dimY)

#Analysis parameters
index_cutoff = 4500 #5220 #Removes the background noise. This value depends on the specifics of the setup and the delays. Must be optimised for new setups
index_ref = 4500 #5120 #Time index of the mirrors position, used as origin when calculating 3D point clouds. Not at zero because the laser must first travel to the optical setup. Mus be measured seperatly

#Plotting parameters
coff = 3  #Removes outliers for plotting purposes. Simply to avoid squished plots
z_rot = 270 #Angle of veiw in 3D plot
x_rot = 20


#lidar.save_pixel_array(histogram, file, binsize) #To save some raw data for troubleshooting
#lidar.save_all_pixels(histogram, file, dimX, dimY, binsize)
d_data, i_data = analyse_3d(histogram, index_cutoff, index_ref, x_deg, y_deg, dimX, dimY, background = 3)
file = Path(file)

print("Loading time: ", load_time)
print("TOF analysis time: ", TOF_anal_time)
print("Total Analysis time: ", t.time()-START)
#-------------------- Save code -------------

print("Saving Images")
coff = int(coff) # prevents the images from being to squished

intensity_map.heatmap(i_data)

#lidar.scatter(d_data, file, cutoff = coff, name = anal_method + "_Fit_", show = True)
#lidar.save_data(i_data, file, name = 'intensity_gauss' + '_')

print("Job Done!")

