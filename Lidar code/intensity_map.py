from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np


def heatmap(d_data):
    x,y,i = [], [], []
    for p in d_data:
        x.append(p[0])
        y.append(p[1])
        i.append(d_data[p])
    
    x = np.array(x)
    y = np.array(y)
    N = int(len(x)**.5)

    i = np.array(i)
    i = i.reshape(N,N)

    plt.imshow(i, extent = (np.min(x), np.max(x), np.min(y), np.max(y)),
           cmap = cm.brg, norm = LogNorm())

    plt.colorbar()
    plt.show()



def main():
    """
    x = [1,2,1,2]
    y = [1,1,2,2]
    i = [1,np.NaN,3,4]

    x = np.array(x)
    y = np.array(y)
    N = int(len(x)**.5)
    
    i = np.array(i)
    i = i.reshape(N,N)
    print(i)

    plt.imshow(i, extent = (np.min(x), np.max(x), np.min(y), np.max(y)),
           cmap = cm.brg)#, norm = LogNorm())

    plt.colorbar()
    plt.show()
    """
    t = 1

if __name__ == "__main__":
    main()



file = "C:/Users/staff/Documents/Lidar LF/Data/211101/3d data/Gauss_hand_JU_10ms_VL_10MHz_400kHzCts_-17.5uA_[3,4,-6,-6]_100x100_211101.txt"
x,y,z = np.loadtxt(file, unpack = True)
