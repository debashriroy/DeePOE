

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

dtype_all= scipy.dtype([('raw-iq', scipy.complex64)]) # gr_complex is '32fc' --> make any sense?

# print("Total number of i/q samples for REEF BACK:")
# print(scipy.fromfile(open(data_file_loc1), dtype=dtype_all).shape[0])
#
# print("Total number of i/q samples for REEF FRONT LEFT:")
# print(scipy.fromfile(open(data_file_loc3), dtype=dtype_all).shape[0])
#
# print("Total number of i/q samples for REEF FRONT RIGHT:")
# print(scipy.fromfile(open(data_file_loc2), dtype=dtype_all).shape[0])

# sample_size = 512 # CHANGE
# no_of_samples = 1000 # CHANGE
# iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all, count = sample_size * no_of_samples)
# iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all, count = sample_size * no_of_samples)
# iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all, count = sample_size * no_of_samples)
# iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all, count = sample_size * no_of_samples)
# iqdata_loc5 = scipy.fromfile(open(data_file_loc5), dtype=dtype_all, count = sample_size * no_of_samples)
# iqdata_loc6 = scipy.fromfile(open(data_file_loc6), dtype=dtype_all, count = sample_size * no_of_samples)

iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all) # DATA COLLECTED at UCF
iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all) # DATA COLLECTED at UCF
iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all) # DATA COLLECTED at UCF
iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all) # DATA COLLECTED at UCF

start_time = time.time()  # Taking start time to calculate overall execution time
no_of_loc1 = iqdata_loc1.shape[0]
no_of_loc2 = iqdata_loc2.shape[0]
no_of_loc3 = iqdata_loc3.shape[0]
no_of_loc4 = iqdata_loc4.shape[0]

print("Number of (I,Q) value pairs in 5FT 0 Degree: ", no_of_loc1)
print("Number of (I,Q) value pairs in 5FT +30 Degree: ", no_of_loc2)
print("Number of (I,Q) value pairs in 5FT +60 Degree: ", no_of_loc3)
print("Number of (I,Q) value pairs in 5FT +90 Degree: ", no_of_loc4)
