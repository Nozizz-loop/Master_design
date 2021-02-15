# http://mpastell.com/pweave/index.html
# scientific reporting with pyton
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

file_suffix = '.cos'  # file suffix文件后缀
infile_dir = 'D:/Jupyter/practice_cos/'
filename = 'EWNS,Trig,Wuhan,2095,250000.00,10s_50s,20170425_110000,30.54N,114.37E.cos'
header = filename2header(filename)

# 'EWNS,Trig,fenyi_bb,1437,250000.00,10s_10s,20200229_075900,27.91N,114.70E.cos'
instrument = 'Fenyi_bb'
sample_rate = 250000.00
date = 20200229

# filename='EWNS,Trig,fenyi_bb,1497,250000.00,10s_50s,20200229_085700,27.91N,114.70E.cos'
N_max = 5000000  # set larger numbers and truncate the zeros
indata = np.zeros(N_max, dtype=np.uint16)

# fid=open(infile_dir+filename,'r+b') # read and write in binary mode

with open(infile_dir+filename,'r+b') as fid:
   fid.readinto(indata)
indata.size

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
# 
# https://scipy-cookbook.readthedocs.io/items/FIRFilter.html
# The Nyquist rate of the signal.
# nyq_rate = sample_rate / 2.0


# The desired attenuation in the stop band, in dB.
# ripple_db = 60.0
# Compute the order and Kaiser parameter for the FIR filter.
# N, beta = kaiserord(ripple_db, width)
# Compute the order and Kaiser parameter for the FIR filter.


Ap = 1.  # passband attenuation [dB]通带衰减
As = 40. # stopband attenutation [dB]阻带衰减
# passband deviation delta_p
# or alpha_p[dB] = -20log(1-delta_p)
# passband amplitude in range of [1+delta_p, 1-delta_p] 
# stopband deviation delta_s 
# or alpha_s [dB] = -20 log delta_s
# or delta_s = 10**(-alpha_s/20)
# stopband amplitude in range of [0, delta_s]


R_db = As # Ribble in dB

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 1000 Hz transition width.
# transition band width in faction of nyquest frenquency
# 从通过到停止的过渡所需的宽度，相对于奈奎斯特速率。我们将设计一个过渡宽度为1000赫兹的滤波器。
width = 1000.0/fnq  # need to be calculated 
# number of taps, beta
# N, beta = kaiserord(R_db, width)
# h_LP = firwin(N,f_cutoff,window=('kaiser', beta),fs=fs,pass_zero='bandpass',scale=False) #,pass_zero=False, scale=False
# data=filtfilt(h_LP,1,NSdata) # filtered data with low-pass
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


for i1 in range(N_step_NS):  # 0,N_step_NS-1
   tmp1 = abs(np.fft.fft(win_hamm * Signal_NS[N_fft*i1:N_fft*(i1+1)], N_fft))
   tmp2 = 2/N_fft*tmp1[0:N_fft//2]  #
   tmp2[0] = tmp2[0] / 2.0
   z_NS[:, i1] = 20.*np.log10(tmp2)  # into dB
print(z_NS.shape)
fig,(ax0,ax1) = plt.subplots(nrows = 2)
Time_NS = np.arange(N_step_NS)* T_seg
xgrid, ygrid = Time_NS,Freq_NS[0:N_fft//2]/thousand
xmesh, ymesh = np.meshgrid(xgrid,ygrid)
levels = MaxNLocator(nbins = 12).tick_values(z_NS.max()-50, z_NS.max())

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = mpl.cm.jet
norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)

img = ax0.pcolormesh(xmesh,ymesh,z_NS,cmap = cmap,norm = norm)
fig.colorbar(img,ax = ax0)
ax0.set_title('NS Ampitude (dB) @ Fenyi')
ax0.set_xlim(0,10)  # Time 
ax0.set_xlabel('Time(s)')
ax0.set_ylim(0,50)  #  frequency kHz
ax0.set_ylabel('Frequency (kHz)')

hdu1 = fits.ImageHDU(z_NS)


for i1 in range(N_step_EW):  # 0,N_step_NS-1;1,N_step_EW-1
   tmp1 = abs(np.fft.fft(win_hamm * Signal_EW[N_fft*i1:N_fft*(i1+1)], N_fft))
   tmp2 = 2/N_fft*tmp1[0:N_fft//2]  #
   tmp2[0] = tmp2[0] / 2.0
   z_EW[:, i1] = 20.*np.log10(tmp2)  # into dB


Time_EW = np.arange(N_step_EW) * T_seg
xgrid, ygrid = Time_EW, Freq_EW/thousand


#xgrid, ygrid = Time_EW,Freq_EW[0:N_fft//2]/thousand
xmesh, ymesh = np.meshgrid(xgrid, ygrid)

levels = MaxNLocator(nbins = 12).tick_values(z_EW.max()-50, z_EW.max())

cmap = mpl.cm.jet
norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)

img = ax1.pcolormesh(xmesh, ymesh, z_EW, cmap = cmap, norm = norm)
fig.colorbar(img, ax = ax1)
ax1.set_title('EW Ampitude (dB) @ Fenyi')
ax1.set_xlim(0, 10)  # Time
ax1.set_xlabel('Time(s)')
ax1.set_ylim(0, 50)  # frequency kHz
ax1.set_ylabel('Frequency (kHz)')

hdu2 = fits.ImageHDU(z_EW)


fig.tight_layout()
plt.show()

hdr = filename2header(filename)
hdu0 = fits.PrimaryHDU(data=indata,header=hdr)

(newfn, extension) = os.path.splitext(filename)

hdu = fits.HDUList([hdu0,hdu1,hdu2])
hdu.writeto(infile_dir + newfn + '.fits', overwrite=True)

print(repr(hdr))
hdu.info()
#print(type(z_EW))

pydata = np.array([z_NS,z_EW])
np.save('D:/Jupyter/practice_cos/pydata.npy',pydata)
#print()

