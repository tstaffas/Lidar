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

def ToF_analysis(timetag_file, recipe_file, ch_sel, **kwargs):
    load_start = t.time()
    #Load the recipe from seperate ETA file
    with open(recipe_file, 'r') as filehandle:
        recipe_obj = json.load(filehandle)

    eta_engine = etabackend.eta.ETA()
    eta_engine.load_recipe(recipe_obj)

    #Set parameters in the recipe
    for arg in kwargs:
        eta_engine.recipe.set_parameter(arg, str(kwargs[arg]))

    """
    eta_engine.recipe.set_parameter("binsize", str(binsize))
    eta_engine.recipe.set_parameter("bins", str(bins))
    eta_engine.recipe.set_parameter("sync_delay", str(sync_delay))
    eta_engine.recipe.set_parameter("dimX", str(dimX))
    eta_engine.recipe.set_parameter("dimX", str(dimY))
    """
    
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

    print(f"Number of histograms produced: {result['pixelnumber']+1}")
    
    return histogram, START, load_time, TOF_anal_time

#--------------- Analysing the ToF histograms -----------
#--------------- Analysing the ToF histograms -----------
def analyse_3d(histogram, index_cutoff, index_ref, x_deg, y_deg, time, method = 'peak', background = 6, delay = 0):
    METHODS = {'peak': lidar.peak_distance, 'gauss': lidar.gauss}
    anal = METHODS[method]
    print("Starting 3D analysis")
    
    d_data = {}  #used to store the distance data
    i_data = {}
    average_peak = 0  #some fun values to keep track of
    average_failed_peak = 0 #some fun values to keep track of
    
    F = 0   #Used to keep track of the number of failed pixels
    start = t.time() #Evaluate the time efficiency of the algorithms
    
    """
    Code block loops through all histograms. Removes background/internal reflections and calculates the distance to a reference value that must be measured separately (but is reused for all scans)
    """
    dimX = len(x_deg)
    dimY = len(y_deg)
    for i in range(0,dimY):
        print(i,"/",dimY)
        for j in range(0,dimX):
            h = histogram[j][i]
            #h = ToF_histogram_offset(h,delay, binsize)
            h[:index_cutoff] = 0 #Cuts away the internal reflections, background_cutoff is assigned in ETA frontend and is based on a background measurement. 
            
                
            peak = np.amax(h) #Identifies the target peak
            if peak > background:  #removes pixels with only noise, noise threshold can be modified
                #d, _ = lidar.skewedgauss(time,h,index_ref) #Gaussian algorithm
                #d, _ = lidar.gauss(time,h,index_ref) #Gaussian algorithm
                #d = lidar.peak_distance(time,h, index_ref)#d = lidar.getDistance(index_ref, h, binsize = binsize)  #Peak finding Algorithm
                d, h = anal(time, h, index_ref)
                if d != np.NaN: #Gaussian algorithm can return np.NaN if unable to fit a curve to data, very unlikely after filtering away peaks with. It's a relic and might be obselete (but it's not hurting anyone)
                    x,y,z = lidar.XYZ(np.abs(d),x_deg[i],y_deg[j])
                    d_data[(x,y)] = z
                    i_data[(x,y)] = h
                average_peak += peak
                
            else:
                x,y,z = lidar.XYZ(1,x_deg[i],y_deg[j])
                #x,y,z, = np.NaN, np.NaN, np.NaN
                d_data[(x,y)] = np.NaN
                i_data[(x,y)] = np.NaN

                F +=1
                average_failed_peak += peak
                
    stop = t.time()
    print("Failed pixels: ", F)
    print("Average peak: ", average_peak/(dimY*dimX - F))
    if F!=0:
        print("Average failed peak: ", average_failed_peak/F)

    print("3D analysis time: ", stop-start)
    return d_data, i_data

def main():

    """Set Parameters for analysis and plotting"""
    recipe = "C:/Users/staff/Documents/Lidar LF/ETA_recipes/quTAG_LiDAR_1.2.eta" #ETA_recipe file

    #.timeres file to analysed
    file = 'C:/Users/staff/Documents/Lidar LF/Data/220216/Fredrik_10.0ms_300kHz_-17.9uA_[4, 9, -5, -4.5]_200x200_220216.timeres'
    anal_method = "peak"

    #Parameters for etabackend to generate histograms
    base_binsize = 16 #Histogram binsize in ps
    base_bins = 6250 #Number of bins in the histogram: bins*binsize should equal 1/f where f is the repition rate of the laser in use
    ch_sel = 't1' #Selects a specific histogram
    records_per_cut = 2e5 #Number of events to be used per evalution cycle in ETA, not important in this code

    #Time offsets for different signals [ps]
    base_sync_delay = 0 #40000  #All events of the sync channel is delayed by 40000 ps (not necessary)
    base_delay = 10560 #16800 #

    base_dimX = 200 # number of steps in the scan steps == resolution
    base_dimY = 200

    histogram, START, load_time, TOF_anal_time = ToF_analysis(file, recipe, ch_sel,
                                                              bins = base_bins, binsize=base_binsize,
                                                              dimX=base_dimX, dimY=base_dimY, sync_delay = base_sync_delay)
    time = (np.arange(0,base_bins)*base_binsize) #Recreate time axis

    #----------------- Variables ---------------------
    #Scanning variables
    rect = [4, 9, -5, -4.5] #Voltage range of scan, linear to angle
    x_deg, y_deg = lidar.angles(rect,base_dimX,base_dimY)

    #Analysis parameters
    index_cutoff = 4520 #5220 #Removes the background noise. This value depends on the specifics of the setup and the delays. Must be optimised for new setups
    index_ref = 4500 #5150 #Time index of the mirrors position, used as origin when calculating 3D point clouds. Not at zero because the laser must first travel to the optical setup. Mus be measured seperatly

    #Plotting parameters
    coff = 4  #Removes outliers for plotting purposes. Simply to avoid squished plots
    z_rot = 270 #Angle of wiev in 3D plot
    x_rot = 20

    ##lidar.save_pixel_array(histogram, file, dimX, dimY, binsize) #To save some raw data for troubleshooting
    #lidar.save_all_pixels(histogram, file, dimX, dimY, binsize)
    d_data, i_data = analyse_3d(histogram, index_cutoff, index_ref, x_deg, y_deg, time, method = anal_method,  background = 6, delay = base_delay)
    file = Path(file)

    print("Loading time: ", load_time)
    print("TOF analysis time: ", TOF_anal_time)
    print("Total Analysis time: ", t.time()-START)

    print(f"Length of d_data: {len(d_data)}| i_data: {len(i_data)}")
    #-------------------- Save code -------------
    print("Saving Images")
    coff = int(coff) # prevents the images from being to squished

    intensity_map.heatmap(i_data)
    #lidar.scatter(d_data, file, cutoff = coff, name = anal_method + "_Fit_", show = True)#, ylim = (300,600), xlim=(-200,200))
    #lidar.save_data(d_data, file, anal_method + '_')
    print("Job Done!")

if __name__ == '__main__':
    main()

