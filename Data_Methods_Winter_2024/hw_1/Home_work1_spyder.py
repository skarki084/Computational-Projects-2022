# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objs as go

data_path = 'subdata.npy'

d = np.load(data_path) # huge matrix of size 262144 x 49 (columns contain flattened 3d matrix of size 64x64x64)

L = 10; # length of spatial domain (cube of side L = 2*10)
N_grid = 64; # number of grid points/Fourier modes in each direction
xx = np.linspace(-L, L, N_grid+1) #spatial grid in x dir
x = xx[0:N_grid]
y = x # same grid in y,z direction
z = x

times = np.linspace(0,24,49)

K_grid = (2*np.pi/(2*L))*np.linspace(-N_grid/2, N_grid/2 -1, N_grid) # frequency grid for one coordinate

xv, yv, zv = np.meshgrid( x, y, z) # generate 3D meshgrid for plotting


#Below, I hand filter the subdata. I normalze and take the absolute value. And then I find when the 
#data is above a certain threshold. I plot the points in te data that are above the threshold for each measurement.
#This is actually a decent way to find where the submarine is without doing any of the fft stuff. This could be used as
#an alternative way to find the submarine. This can also be a good figure to start off the report with, to sort of motivate the 
#rest of the project.

#FIST FIGURE

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlim([-L, L])
ax.set_ylim([-L, L])
ax.set_zlim([-L, L])
for j in range(0,49,5):
    
    signal = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
    normal_sig_abs = np.abs(signal)/np.abs(signal).max() #normalize the signal
    
    threshold = 0.80
    
    indices = np.where(normal_sig_abs > threshold)
    
    in_x, in_y, in_z = indices
    
    values = normal_sig_abs[in_x, in_y, in_z]
    
    scatter = ax.scatter(x[in_x], y[in_y], x[in_z], c=[(j/50,1-j/50,0)]*len(in_x), marker='o')
    
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,1,0), markersize=10, label='t = 0'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1,0,0), markersize=10, label='t = 24')
], loc='upper right')

plt.title('Extreme Acoustic Data at Various Times')

plt.show()


#What I actually need to do:
    #Take the Fourier transform off all the data sets. Average these data sets.
    #This averaged Fourier transform will reveal the important frequency of the submarine.
    #Then, with this frequency, I will apply a Gaussian centered at this frequency to all the FFt'd measurements
    #When I ifft, the data should be very clear, and the submarine will be easilly visable.
    


#Step 1) Take the Fourier Transform of some slices of the first data measurement to make a nice plot
#I want to plot 4 different plots, of the FFTd data set, without any filtering, see what it looks like
#these 4 plots will be at K_grid[0], K_grid[13], K_grid[31],K_grid[47],K_grid[63] 
 
#SECOND FIGURE


index_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,63]

fig,axs = plt.subplots(1,len(index_list),figsize=(30,30))
_slice = np.reshape(d[:, 0], (N_grid, N_grid, N_grid)) #first measurement
fft_slice = np.fft.fftn(_slice)
fft_slice = np.fft.fftshift(fft_slice)

nabfft_slice = np.abs(fft_slice)/np.max(np.abs(fft_slice))

for j in range(len(index_list)):
    axs[j].imshow(nabfft_slice[:,:,index_list[j]],cmap='magma',vmin=0, vmax=1)
    axs[j].set_title(f'kz = {K_grid[index_list[j]]:.1f}', fontsize=25)
    axs[j].axis('off')
    
fig.suptitle('Fourier Transform of t = 0 Measurement', fontsize=40, y = 0.56)
plt.tight_layout()
plt.show()



#THRID FIGURE
#Now, I want to take the FFT of everytime step, and then flatten that data. Then average the FFT data acroos all measurements

fft_mat = 0*d #has same dimension as original data. Will have flattened fftshift data.

for j in range(len(times)):
    _slice = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
    fft_slice = np.fft.fftn(_slice)
    fft_slice = np.fft.fftshift(fft_slice)
    
    fft_mat[:,j] = fft_slice.flatten()

fft_mean = np.mean(fft_mat, axis=1) #this is the average of all the fft measurements
fft_mean = fft_mean.reshape(64,64,64)

nabs_fft_mean = np.abs(fft_mean)/ np.max(np.abs(fft_mean))

#Now with this fft_mean data, I can make the same plot as before, with this new averaged fft data, as opposed to the noisy initial measurement
index_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,63]

fig,axs = plt.subplots(1,len(index_list),figsize=(30,30))
for j in range(len(index_list)):
    #labfft_slice = np.log(np.abs(fft_mean))
    
    axs[j].imshow(nabs_fft_mean[:,:,index_list[j]],cmap='magma',vmin=0, vmax=1)
    axs[j].set_title(f'kz = {K_grid[index_list[j]]:.1f}', fontsize=25)
    axs[j].axis('off')
    
fig.suptitle('Average Fourier Transform of all Measurements', fontsize=40, y = 0.56)
plt.tight_layout()
plt.show()

#Okay, Great! We can from the Average of the Fourier Transform see that there is a 
#bright spot around kx = 5, ky = -5, kz = -7, Next thing I need to do is find the exact location of this
#that is not very hard.

loud_freq_in = np.unravel_index(np.argmax(nabs_fft_mean), nabs_fft_mean.shape)
loud_freq_loc = np.array([K_grid[loud_freq_in[0]],K_grid[loud_freq_in[1]],K_grid[loud_freq_in[2]]])

#the exact location of the the loud frequency is kx = 2.2, 5.3, -6.9

#Now, I want to take each frame of the FFT data, apply a gaussian that is centered at
# the loud frequency (given above) with sigma of around 3, and then Iffshift and ifft the gaussian*fft data. I will put each unfiltered time step into
# a new filtered dataset

#define gaussian:
def g(x,y,z): # 2D Gaussian filter centered at loud freq, with sigma = 5
  s = 5
  val = np.exp( - ( ((x - 2.2)**2 + (y - 5.3)**2 + (z + 6.9)**2)/(s**2)  ))

  return val

KX, KY, KZ = np.meshgrid(K_grid, K_grid,K_grid)
Gfilter = g(KX, KY, KZ)

#take flattened fft data--> unflatten--> apply gaussian --> iiftshift --> iffn --> flatten --> put into filtered datalist
filtered_data = 0*d
for i in range(len(times)):
    slice_ = np.reshape(fft_mat[:,i],(N_grid, N_grid, N_grid))
    f_slice = Gfilter*slice_
    f_slice = np.fft.ifftshift(f_slice)
    f_slice = np.fft.ifftn(f_slice)
    filtered_data[:,i] = f_slice.flatten()
    
#Okay, I have the filtered data. I just need to plot the x y location of the submarine at every timestep:
    
#FIGURE 4
x_list = 0*times #these are the list of x and y values of the submarine
y_list = 0*times

for i in range(len(times)):
    slice_ = np.reshape(filtered_data[:,i],(N_grid, N_grid, N_grid))
    
    indexs = np.unravel_index(np.argmax(slice_), slice_.shape)
    x_list[i] = x[indexs[0]]
    y_list[i] = y[indexs[1]]
    
    plt.scatter(x_list[i],y_list[i], color = (i/50,1-i/50,0))
    
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,1,0), markersize=10, label='t = 0'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1,0,0), markersize=10, label='t = 24')
], loc='upper right')
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.title('Path in x-y Plane Method A')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#One last thing, Before, I had previously sort of plotted the path of the submarine by just by taking the original data given
#taking the abs value, normaliziing, and then plotting the largest values of each measurement

#FIGURE 5

#I will try that again, this time plotting the xy values, and seeing if just taking the max is enough.

for j in range(0,49):
    
    signal = np.reshape(d[:, j], (N_grid, N_grid, N_grid))
    normal_sig_abs = np.abs(signal)/np.abs(signal).max() #normalize the signal
    
    indices = np.unravel_index(np.argmax(signal), signal.shape)
    
    in_x, in_y, in_z = indices
    
    plt.scatter(x[in_x], y[in_y], color=(j/50,1-j/50,0), marker='o')
    
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,1,0), markersize=10, label='t = 0'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1,0,0), markersize=10, label='t = 24')
], loc='upper right')
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.title('Path in x-y Plane Method B')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



#One more thing I'd like to do-
#I will take the first raw measurement, then average across, then for each x y location, I will average all the z values,
#so in a way I am taking the projection of the 3d data onto the xy plane. Will I be able to find the submarine location?
#I can also do this for the filtered data.
#FIGURE 6
slice_ = np.abs(np.reshape(d[:, 0], (N_grid, N_grid, N_grid)))
fslice = np.abs(np.reshape(filtered_data[:,0],(N_grid, N_grid, N_grid)))
basic_pic = np.mean(slice_, axis = 2)
filtered_pic = np.mean(fslice, axis = 2)

fig,axs = plt.subplots(1,2,figsize=(20,20))
axs[0].imshow(basic_pic, cmap='viridis')
axs[0].set_title('Unfiltered',fontsize=25)
axs[0].axis('off')
axs[1].imshow(filtered_pic, cmap = 'viridis')
axs[1].set_title('Filtered',fontsize=25)
axs[1].axis('off')

fig.suptitle('Initial Measurement x-y Projection', fontsize=30, y = 0.74)

