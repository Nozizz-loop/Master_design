import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
## 将mat文件中data1位置对应的数据读取出来
datafile = 'E:/Python/Practice/0310/matdata.mat'
mldata = io.loadmat(datafile)
# 方法二重新存储mat文件里面的数据为num_data.npy
data1 = mldata['d2']
data2 = mldata['d1']
#data = [[data1], [data2]]
data = np.array([data1, data2])
#numpy_data = np.transpose(data)
np.save('matdata.npy',data)
"""

#print(type(mldata))

pydata = np.load('E:/Python/Practice/0310/pydata.npy')  #这个是python的
#print(type(pydata))
print(pydata.shape)
ns = pydata[0,:]
ew = pydata[1,:]
#print(pydata)

#print(mldata['DataFFT'].shape)
#mldata_t = np.transpose(mldata['DataFFT'])
#np.save('DataFFT_t.npy', mldata_t)

#np.save('DataFFT.npy', mldata)  # 将.mat文件存为npy文件

matdata = np.load('E:/Python/Practice/0310/matdata.npy') #这个是matlab的数据
mdns = matdata[0,:]
mdew = matdata[1,:]
#print(type(matdata))

error_ns = ns - mdns
error_ew = ew - mdew

ns_max = np.max(error_ns)
ns_min = np.min(error_ns)
ew_max = np.max(error_ew)
ew_min = np.min(error_ew)
print(f'error_ns_max = {ns_max}\nerror_ns_min = {ns_min}')
print(f'error_ew_max = {ns_max}\nerror_ew_min = {ns_min}')



import numpy as np
import os
# import math
# from scipy.signal import kaiserord, firwin, filtfilt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from vlf_fenyi import filename2header
# from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits



sample_rate = 250000.00

# filename='EWNS,Trig,fenyi_bb,1497,250000.00,10s_50s,20200229_085700,27.91N,114.70E.cos'
N_max = 5000000  # set larger numbers and truncate the zeros
indata = np.zeros(N_max, dtype=np.uint16)

# fid=open(infile_dir+filename,'r+b') # read and write in binary mode


EWdata = indata[0:N_max:2]  # every althernative 东西
NSdata = indata[1:N_max:2]  # 南北

# channel filter design 300 Hz - 50000 Hz
thousand =1000.
million = 1000000.
fs = 250000.0  # sampling rate
fnq = 0.5 * fs  # nyquest frequency[通信] 奈奎斯特频率
# cutoff frequency for bandpass filter
# fcuts = [300, 500, 50000, 51000]
# changed into symmetric transition width  DY 2020.03.02 转变为对称过渡宽度
fcuts = [300, 500, 50000, 50200]
f_cutoff=[500,50000]


Ap = 1.  # passband attenuation [dB]通带衰减
As = 40. # stopband attenutation [dB]阻带衰减

R_db = As # Ribble in dB

width = 1000.0/fnq  # need to be calculated

Signal_NS = NSdata - np.mean(NSdata)  # remove the directional current
Signal_EW = EWdata - np.mean(EWdata)
N_fft = 2048  # number of data in each segment每个段中的数据量
N_NS = len(Signal_NS)  # 一个通道数据的长度
N_EW = len(Signal_EW)

win_hamm = np.hamming(N_fft)  # create a hamming window tapering创建一个渐缩的汉明窗口
N_step_NS = int(np.floor(N_NS/N_fft))  # 包含的数据段数largest integer not greater than x.最大的整数，不大于x。
N_step_EW = int(np.floor(N_EW/N_fft))  # largest integer not greater than x.

T_seg = 1./fs*N_fft  # Duration of each segment频谱中的时间分辨率
dt = 1/fs  # sample interval
#dt = 1./fs*N_fft
#dt = 1/fs
z_NS = np.zeros((N_fft//2, N_step_NS))  # amplitude for FFT transform振幅为快速傅里叶变换
# A_EW = np.zeros((N_fft, N_step_NS), dtype=complex)
z_EW = np.zeros((N_fft//2, N_step_EW))

freq = np.fft.fftfreq(N_fft, d=dt)  # calculate frequency
Freq_NS = freq[0:N_fft//2]
Freq_EW = freq[0:N_fft//2]


fig,(ax0,ax1) = plt.subplots(nrows = 2)
Time_NS = np.arange(N_step_NS)* T_seg
xgrid, ygrid = Time_NS,Freq_NS[0:N_fft//2]/thousand
xmesh, ymesh = np.meshgrid(xgrid,ygrid)
levels = MaxNLocator(nbins = 5).tick_values(ns_max, ns_min)

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = mpl.cm.jet
norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)

img = ax0.pcolormesh(xmesh,ymesh, error_ns,cmap = cmap,norm = norm)
fig.colorbar(img,ax = ax0)
ax0.set_title('NS error')
ax0.set_xlim(0,10)
ax0.set_xlabel('Time(s)')
ax0.set_ylim(0,50)
ax0.set_ylabel('Frequency (kHz)')



Time_EW = np.arange(N_step_EW) * T_seg
xgrid, ygrid = Time_EW, Freq_EW/thousand

#xgrid, ygrid = Time_EW,Freq_EW[0:N_fft//2]/thousand
xmesh, ymesh = np.meshgrid(xgrid, ygrid)

levels = MaxNLocator(nbins = 5).tick_values(ew_max, ew_min)

cmap = mpl.cm.jet
norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)

img = ax1.pcolormesh(xmesh, ymesh, error_ew, cmap = cmap, norm = norm)
fig.colorbar(img, ax = ax1)
ax1.set_title('EW error')
ax1.set_xlim(0, 10)  # Time
ax1.set_xlabel('Time(s)')
ax1.set_ylim(0, 50)  # frequency kHz
ax1.set_ylabel('Frequency (kHz)')


fig.tight_layout()
plt.show()

