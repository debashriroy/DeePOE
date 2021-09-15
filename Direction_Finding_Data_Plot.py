
##############################################################
#  Radio Fingerprinting in RFML Environment                 #
#        Neural Network for Direction Finding Data 2020     #
#         Author: Debashri Roy                              #
#############################################################

############ IMPORTING NECESSARY PACKAGES ################
import numpy as np # Package for numerical computation
np.set_printoptions(threshold=np.inf) # To print each elements
import time # Package is for computing execution time
import sys # Package to get command line arguments
import tensorflow as tf
from sklearn.model_selection import train_test_split
from array import array

#   by setting env variables before Keras import you can set up which backend
import os,random
#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=cuda0, dnn.enabled=False"
import theano
#theano.config.mode = ""




import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.models import Sequential
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers import Embedding
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, Conv1D, Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D
from keras.regularizers import *
from keras.optimizers import adam, Nadam, Adadelta
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import rmsprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
#from keras.regularizers import l2, activity_l2
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU, PReLU
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers import GRU, RNN, SimpleRNN, LSTM, GRUCell, SimpleRNNCell, LSTMCell

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import keras
import itertools
import scipy

from sklearn.linear_model import LinearRegression
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

data_file_loc1 = '/Users/debashri/Desktop/DirectionFinding_Data/Outdoor/DataJune9Out30Degree/5ft/0_5ft_06_09_2020_914MHz.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 ='/Users/debashri/Desktop/DirectionFinding_Data/Outdoor/DataJune9Out30Degree/5ft/+30_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = '/Users/debashri/Desktop/DirectionFinding_Data/Outdoor/DataJune9Out30Degree/5ft/+60_5ft_06_09_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = '/Users/debashri/Desktop/DirectionFinding_Data/Outdoor/DataJune9Out30Degree/5ft/+90_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

#dtype_all= np.dtype([('raw-iq0', 'c16')]) # gr_complex is '32fc' --> make any sense?

dtype_all= scipy.dtype([('raw-iq', scipy.complex64)]) # gr_complex is '32fc' --> make any sense?

# print("Total number of i/q samples for REEF BACK:")
# print(scipy.fromfile(open(data_file_loc1), dtype=dtype_all).shape[0])
#
# print("Total number of i/q samples for REEF FRONT LEFT:")
# print(scipy.fromfile(open(data_file_loc3), dtype=dtype_all).shape[0])
#
# print("Total number of i/q samples for REEF FRONT RIGHT:")
# print(scipy.fromfile(open(data_file_loc2), dtype=dtype_all).shape[0])

sample_size = 512 # CHANGE
no_of_samples = 4000 # CHANGE
iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all, count = sample_size * no_of_samples)

# iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all) # DATA COLLECTED at UCF
# iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all) # DATA COLLECTED at UCF
# iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all) # DATA COLLECTED at UCF
# iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all) # DATA COLLECTED at UCF


start_time = time.time()  # Taking start time to calculate overall execution time
no_of_loc1 = iqdata_loc1.shape[0]
no_of_loc2 = iqdata_loc2.shape[0]
no_of_loc3 = iqdata_loc3.shape[0]
no_of_loc4 = iqdata_loc4.shape[0]

# USING ONLY LAST N SAMPLES OF DATA
number_of_data_to_read = sample_size * no_of_samples
extra_rows_loc1 = no_of_loc1 - number_of_data_to_read
extra_rows_loc2 = no_of_loc2 - number_of_data_to_read
extra_rows_loc3 = no_of_loc3 - number_of_data_to_read
extra_rows_loc4 = no_of_loc4 - number_of_data_to_read



xdata_loc1 = np.delete(iqdata_loc1, np.s_[:extra_rows_loc1], 0)
xdata_loc2 = np.delete(iqdata_loc2, np.s_[:extra_rows_loc2], 0)
xdata_loc3= np.delete(iqdata_loc3, np.s_[:extra_rows_loc3], 0)
xdata_loc4= np.delete(iqdata_loc4, np.s_[:extra_rows_loc4], 0)

# PREPARING THE DATA WITHOUT TIME INFORMATION
no_of_data_loc1 = iqdata_loc1.shape[0]
no_of_data_loc2 = iqdata_loc2.shape[0]
no_of_data_loc3 = iqdata_loc3.shape[0]
no_of_data_loc4 = iqdata_loc4.shape[0]




xdata_loc1= np.concatenate([xdata_loc1['raw-iq'].real.reshape(number_of_data_to_read,1), xdata_loc1['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)


xdata_loc3= np.concatenate([xdata_loc3['raw-iq'].real.reshape(number_of_data_to_read,1), xdata_loc3['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)


xdata_loc2= np.concatenate([xdata_loc2['raw-iq'].real.reshape(number_of_data_to_read,1), xdata_loc2['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)


xdata_loc4= np.concatenate([xdata_loc4['raw-iq'].real.reshape(number_of_data_to_read,1), xdata_loc4['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)


print("Total number of I/Q samples LOC 1:")
print(xdata_loc1.shape[0])

print("Total number of I/Q samples for LOC 2:")
print(xdata_loc3.shape[0])

print("Total number of I/Q sample LOC 3:")
print(xdata_loc2.shape[0])

print("Total number of I/Q samples for LOC 4:")
print(xdata_loc4.shape[0])


######################################################################
#                                                                    #
########                  PLOTTING THE DATA                      #####
#							                                         #
######################################################################
xdata_loc1 = xdata_loc1.tolist()
xdata_loc3 = xdata_loc3.tolist()
xdata_loc2 = xdata_loc2.tolist()
print("TEST1")
#xdata_loc4 = xdata_loc4.tolist()
#xdata_loc5 = xdata_loc5.tolist()
print("TEST2")
#xdata_uah = xdata_uah.tolist()

############ PLOTTING W_E Direction  DATA ###################
#
#  Storing data for all rows
data_x = []
data_y = []
timestamps = []
row = 0
totalRow = len(xdata_loc1)
#totalRow = 40000 # Just a random number of samples to print
while row < totalRow:
 col = 0
 while col < len(xdata_loc1[0])-1:
    data_x.append(xdata_loc1[row][col])
    data_y.append(xdata_loc1[row][col+1])
    timestamps.append(row+1)
    col +=2
 row +=1

print("Size of I values:", len(data_x))
print("Size of Q values:", len(data_y))
print("Size of Timestamp:", len(timestamps))


#  PLOTTING PREDICTIONS FOR PRESENCE TIMES
fig = plt.figure('Training Data', [6, 6])
plt.scatter(data_x, data_y, marker = "o", s=1, c="k", label='Data')
plt.grid(b=True, which='major', axis='both', color='r', linestyle=':', linewidth=0.4)
plt.ylabel('I')
plt.xlabel('Q')
plt.legend()
plt.tight_layout()
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/0_5ft.png')  # CHANGE
plt.close(fig)


############ PLOTTING W_W Direction DATA ###################
#
#  Storing data for all rows
data_x = []
data_y = []
timestamps = []
row = 0
totalRow = len(xdata_loc3)
#totalRow = 40000 # Just a random number of samples to print
while row < totalRow:
 col = 0
 while col < len(xdata_loc3[0])-1:
    data_x.append(xdata_loc3[row][col])
    data_y.append(xdata_loc3[row][col+1])
    timestamps.append(row+1)
    col +=2
 row +=1

print("Size of I values:", len(data_x))
print("Size of Q values:", len(data_y))
print("Size of Timestamp:", len(timestamps))


#  PLOTTING PREDICTIONS FOR PRESENCE TIMES
fig = plt.figure('Training Data', [6, 6])
plt.scatter(data_x, data_y, marker = "o", s=1, c="k", label='Data')
plt.grid(b=True, which='major', axis='both', color='r', linestyle=':', linewidth=0.4)
plt.ylabel('I')
plt.xlabel('Q')
plt.legend()
plt.tight_layout()
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/+30_5ft.png')  # CHANGE
plt.close(fig)


############ PLOTTING W_N Direction DATA ###################
#
#  Storing data for all rows
data_x = []
data_y = []
timestamps = []
row = 0
totalRow = len(xdata_loc2)
#totalRow = 40000 # Just a random number of samples to print
while row < totalRow:
 col = 0
 while col < len(xdata_loc2[0])-1:
    data_x.append(xdata_loc2[row][col])
    data_y.append(xdata_loc2[row][col+1])
    timestamps.append(row+1)
    col +=2
 row +=1

print("Size of I values:", len(data_x))
print("Size of Q values:", len(data_y))
print("Size of Timestamp:", len(timestamps))


#  PLOTTING PREDICTIONS FOR PRESENCE TIMES
fig = plt.figure('Training Data', [6, 6])
plt.scatter(data_x, data_y, marker = "o", s=1, c="k", label='Data')
plt.grid(b=True, which='major', axis='both', color='r', linestyle=':', linewidth=0.4)
plt.ylabel('I')
plt.xlabel('Q')
plt.legend()
plt.tight_layout()
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/+60_5ft.png')  # CHANGE
plt.close(fig)


############ PLOTTING  W_S Direction  DATA ###################
#
print(xdata_loc4[0][0])
print(xdata_loc4[0][1])
#  Storing data for all rows
data_x = []
data_y = []
timestamps = []
row = 0
totalRow = len(xdata_loc4)
#totalRow = 40000 # Just a random number of samples to print
while row < totalRow:
 col = 0
 while col < len(xdata_loc4[0])-1:
    data_x.append(xdata_loc4[row][col])
    data_y.append(xdata_loc4[row][col+1])
    timestamps.append(row+1)
    col +=2
 row +=1

print("Size of I values:", len(data_x))
print("Size of Q values:", len(data_y))
print("Size of Timestamp:", len(timestamps))


#  PLOTTING PREDICTIONS FOR PRESENCE TIMES
fig = plt.figure('Training Data', [6, 6])
plt.scatter(data_x, data_y, marker = "o", s=1, c="k", label='Data')
plt.grid(b=True, which='major', axis='both', color='r', linestyle=':', linewidth=0.4)
plt.ylabel('I')
plt.xlabel('Q')
plt.legend()
plt.tight_layout()
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/+90_5ft.png')  # CHANGE
plt.close(fig)

