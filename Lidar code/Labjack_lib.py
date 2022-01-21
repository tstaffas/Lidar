from labjack import ljm  #Allows communication with the labjack
import time
import numpy as np

from ctypes import*
import os

def scan_rectangle(rectangle, dim, t_delay):
    #rectangle = [x_max, y_max, x_min, y_min], dim = [dimX, dimY], t_delay = [us]

    start = time.time()
    wait_address = "WAIT_US_BLOCKING" #"61590"
    wait_value = t_delay
    x_address = "TDAC1" #"30002"
    y_address = "TDAC0" #"30000"
    stop_address = "DAC1"
    start_address = "DAC0"

    dimX, dimY = dim[0], dim[1]
    Xmax, Xmin, Ymax, Ymin = rectangle[0], rectangle[2], rectangle[1], rectangle[3] 

    aNames = [wait_address]
    aValues = [wait_value]

    y_voltages = []
    for i in range(0,dimY):
        y_voltages.append(np.round(Ymax - i*(Ymax-Ymin)/(dimY-1),4) )    

    x_voltages = []
    for i in range(0,dimX):
        x_voltages.append(np.round(Xmax - i*(Xmax-Xmin)/(dimX-1),4))    

    aNames_up_sweep = []
    aValues_up_sweep = []
    aNames_down_sweep = []
    aValues_down_sweep = []
    
    L = len(x_voltages)
    for i in range(L):
        aValues_up_sweep.append(x_voltages[i])
        aValues_down_sweep.append(x_voltages[L-1-i])

        
        aNames_up_sweep.append(x_address)
        aNames_down_sweep.append(x_address)
        #Up sweep
        aNames_up_sweep.append(start_address)
        aValues_up_sweep.append(4)
        aNames_up_sweep.append(start_address)
        aValues_up_sweep.append(0)
        
        aNames_up_sweep.append(wait_address)
        aValues_up_sweep.append(wait_value)

        aNames_up_sweep.append(stop_address)
        aValues_up_sweep.append(4)
        aNames_up_sweep.append(stop_address)
        aValues_up_sweep.append(0)
        
        #Down sweep
        aNames_down_sweep.append(start_address)
        aValues_down_sweep.append(4)
        aNames_down_sweep.append(start_address)
        aValues_down_sweep.append(0)
        
        aNames_down_sweep.append(wait_address)
        aValues_down_sweep.append(wait_value)

        aNames_down_sweep.append(stop_address)
        aValues_down_sweep.append(4)
        aNames_down_sweep.append(stop_address)
        aValues_down_sweep.append(0)

    t = 0
    for y in y_voltages:
        if t%2 == 1:
            aNames = aNames_up_sweep
            aValues = aValues_up_sweep
        elif t%2 == 0:
            aNames = aNames_down_sweep
            aValues = aValues_down_sweep 
        
        ljm.eWriteNames(handle, len([y] + aNames), [y_address]+ aNames, [y] + aValues)
        t+=1
        print("Percent scanned: ", round(t/len(y_voltages),4))


    stop = time.time()
    print("Scan time: ", stop - start)

    ljm.eWriteAddress(handle, 30000, ljm.constants.FLOAT32, 0)
    ljm.eWriteAddress(handle, 30002, ljm.constants.FLOAT32, 0)

if __name__ == '__main__':

    handle = ljm.openS("T7", "ANY", "ANY")
    #info = ljm.getHandleInfo(handle)

    rectangle = [8,8,-8,-8] # Sets the voltage square that is scanned. [Xmax,Ymax,Xmin,Ymin]
    dim = [100,100] #The "resolution" or #steps in X and Y
    time_per_pixel =1*1000 #Integration time per pixel in micro seconds

    scan_rectangle(rectangle, dim, time_per_pixel)
