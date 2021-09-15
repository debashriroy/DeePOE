# TRANSMITTER HAS A DIRECTIONAL ANTENNA -  POINTED IN 12 DIFFERENT POSES
# RECEIVER HAS AN OMNI DIRECTIONAL ANTENNA
# DISTANCE BETWEEN RECEIVER AND TRANSMITTER - (5, 10, 15) FEET
# IMPEMENTING HIERARCHICAL MACHINE LEARNING
# IMPLEMENTING TRANSFER LEARNING
# DATA COLLECTED IN OUTDOOR ENVIRONMENT

#############################################################
#  Pose Estimation and Ranging the RF Transmitter           #
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

from keras.models import load_model

########## FUNCTIONS TO CALCULATE F SCORE OF THE MODEL ###############
from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
######################################################################


################# THE WEIGHT MATRIX #################3
W = np.matrix([[np.cos(1*(np.pi/8)), np.sin(1*(np.pi/8))],
[np.cos(2*(np.pi/8)), np.sin(2*(np.pi/8))],
[np.cos(3*(np.pi/8)), np.sin(3*(np.pi/8))],
[np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
[np.cos(5*(np.pi/8)), np.sin(5*(np.pi/8))],
[np.cos(6*(np.pi/8)), np.sin(6*(np.pi/8))],
[np.cos(7*(np.pi/8)), np.sin(7*(np.pi/8))],
[np.cos(8*(np.pi/8)), np.sin(8*(np.pi/8))]]
)

# W = np.matrix([[np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))]]
# )

print(W)

# variables
dtype_all= scipy.dtype([('raw-iq', scipy.complex64)])


sample_size = 1024  # CHANGE AND EXPERIMENT -1024
no_of_samples = 2000 # CHANGE AND EXPERIMENT - 4000
no_of_features= 8 # CHANGE AND EXPERIMENT
number_of_data_to_read = sample_size * no_of_samples
folder_path_outdoor = '/Users/debashri/Desktop/DirectionFinding_Data/Outdoor/' # CHANGE AS PER YOUR COMPUTER FOLDER PATH

#######################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE 5FT DATA                                   #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/0_5ft_06_09_2020_914MHz.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path_outdoor + 'DataJune9Out30Degree/5ft/+30_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/+60_5ft_06_09_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/+90_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/+120_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path_outdoor + 'DataJune9Out30Degree/5ft/+150_5ft_06_09_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/180_5ft_06_09_2020_914MHz.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/-150_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/-120_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path_outdoor + 'DataJune9Out30Degree/5ft/-90_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/-60_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path_outdoor + 'DataJune9Out30Degree/5ft/-30_5ft_06_09_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER



iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all, count = sample_size * no_of_samples)

iqdata_loc5 = scipy.fromfile(open(data_file_loc5), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc6 = scipy.fromfile(open(data_file_loc6), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc7 = scipy.fromfile(open(data_file_loc7), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc8 = scipy.fromfile(open(data_file_loc8), dtype=dtype_all, count = sample_size * no_of_samples)


iqdata_loc9 = scipy.fromfile(open(data_file_loc9), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc10 = scipy.fromfile(open(data_file_loc10), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc11 = scipy.fromfile(open(data_file_loc11), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc12 = scipy.fromfile(open(data_file_loc12), dtype=dtype_all, count = sample_size * no_of_samples)



# PREPARING THE DATA WITHOUT TIME INFORMATION
no_of_data_loc1 = iqdata_loc1.shape[0]
no_of_data_loc2 = iqdata_loc2.shape[0]
no_of_data_loc3 = iqdata_loc3.shape[0]
no_of_data_loc4 = iqdata_loc4.shape[0]

no_of_data_loc5 = iqdata_loc5.shape[0]
no_of_data_loc6 = iqdata_loc6.shape[0]
no_of_data_loc7 = iqdata_loc7.shape[0]
no_of_data_loc8 = iqdata_loc8.shape[0]

no_of_data_loc9 = iqdata_loc9.shape[0]
no_of_data_loc10 = iqdata_loc10.shape[0]
no_of_data_loc11 = iqdata_loc11.shape[0]
no_of_data_loc12 = iqdata_loc12.shape[0]



################################################################################################################
# CONCATINATING THE I AND Q VALUES VERTICALLY OF (I, Q) SAMPLE. -- note the axis argument is set to 1 (means vertical stacking)
# SIMULATNEOUSLY MULTIPLYING WITH THE WEIGHT MATRIX - TO REFLECT THE MULTI-ANGULAR PROJECTION

xdata_loc1= np.concatenate([iqdata_loc1['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc1['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc1 = np.matmul(xdata_loc1, np.transpose(W))


xdata_loc2= np.concatenate([iqdata_loc2['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc2['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc2 = np.matmul(xdata_loc2, np.transpose(W))


xdata_loc3= np.concatenate([iqdata_loc3['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc3['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc3 = np.matmul(xdata_loc3, np.transpose(W))


xdata_loc4= np.concatenate([iqdata_loc4['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc4['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc4 = np.matmul(xdata_loc4, np.transpose(W))




xdata_loc5= np.concatenate([iqdata_loc5['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc5['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc5 = np.matmul(xdata_loc5, np.transpose(W))

xdata_loc6= np.concatenate([iqdata_loc6['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc6['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc6 = np.matmul(xdata_loc6, np.transpose(W))


xdata_loc7= np.concatenate([iqdata_loc7['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc7['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc7 = np.matmul(xdata_loc7, np.transpose(W))


xdata_loc8= np.concatenate([iqdata_loc8['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc8['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc8 = np.matmul(xdata_loc8, np.transpose(W))




xdata_loc9= np.concatenate([iqdata_loc9['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc9['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc9 = np.matmul(xdata_loc9, np.transpose(W))


xdata_loc10= np.concatenate([iqdata_loc10['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc10['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc10 = np.matmul(xdata_loc10, np.transpose(W))


xdata_loc11= np.concatenate([iqdata_loc11['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc11['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc11 = np.matmul(xdata_loc11, np.transpose(W))


xdata_loc12= np.concatenate([iqdata_loc12['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc12['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc12 = np.matmul(xdata_loc12, np.transpose(W))




# RESHAPING THE XDATA
xdata_loc1= xdata_loc1.T.reshape(no_of_data_loc1//(sample_size), sample_size*no_of_features)
xdata_loc2 = xdata_loc2.T.reshape(no_of_data_loc2//(sample_size), sample_size*no_of_features)
xdata_loc3 = xdata_loc3.T.reshape(no_of_data_loc3//(sample_size), sample_size*no_of_features)
xdata_loc4 = xdata_loc4.T.reshape(no_of_data_loc4//(sample_size), sample_size*no_of_features)

xdata_loc5= xdata_loc5.T.reshape(no_of_data_loc5//(sample_size), sample_size*no_of_features)
xdata_loc6 = xdata_loc6.T.reshape(no_of_data_loc6//(sample_size), sample_size*no_of_features)
xdata_loc7 = xdata_loc7.T.reshape(no_of_data_loc7//(sample_size), sample_size*no_of_features)
xdata_loc8 = xdata_loc8.T.reshape(no_of_data_loc8//(sample_size), sample_size*no_of_features)

xdata_loc9= xdata_loc9.T.reshape(no_of_data_loc9//(sample_size), sample_size*no_of_features)
xdata_loc10 = xdata_loc10.T.reshape(no_of_data_loc10//(sample_size), sample_size*no_of_features)
xdata_loc11 = xdata_loc11.T.reshape(no_of_data_loc11//(sample_size), sample_size*no_of_features)
xdata_loc12 = xdata_loc12.T.reshape(no_of_data_loc12//(sample_size), sample_size*no_of_features)


# STORING THE XDATA FOR DIFFERENT POSES
xdata_pose1 = xdata_loc1
xdata_pose2 = xdata_loc2
xdata_pose3 = xdata_loc3
xdata_pose4 = xdata_loc4

xdata_pose5 = xdata_loc5
xdata_pose6 = xdata_loc6
xdata_pose7 = xdata_loc7
xdata_pose8 = xdata_loc8

xdata_pose9 = xdata_loc9
xdata_pose10 = xdata_loc10
xdata_pose11 = xdata_loc11
xdata_pose12 = xdata_loc12



#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_5ft = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_5ft =xdata_5ft.astype(np.float)

# REMOVING THE NANS
xdata_5ft = np.nan_to_num(xdata_5ft)



################################################################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE 10FT DATA                                   #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/0_10ft_06_10_2020_914MHz.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path_outdoor + 'DataJune10Out30Degree/10ft/+30_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/+60_10ft_06_10_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/+90_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/+120_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path_outdoor + 'DataJune10Out30Degree/10ft/+150_10ft_06_10_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/180_10ft_06_10_2020_914MHz.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/-150_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/-120_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path_outdoor + 'DataJune10Out30Degree/10ft/-90_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/-60_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path_outdoor + 'DataJune10Out30Degree/10ft/-30_10ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER



iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all, count = sample_size * no_of_samples)

iqdata_loc5 = scipy.fromfile(open(data_file_loc5), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc6 = scipy.fromfile(open(data_file_loc6), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc7 = scipy.fromfile(open(data_file_loc7), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc8 = scipy.fromfile(open(data_file_loc8), dtype=dtype_all, count = sample_size * no_of_samples)


iqdata_loc9 = scipy.fromfile(open(data_file_loc9), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc10 = scipy.fromfile(open(data_file_loc10), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc11 = scipy.fromfile(open(data_file_loc11), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc12 = scipy.fromfile(open(data_file_loc12), dtype=dtype_all, count = sample_size * no_of_samples)



# PREPARING THE DATA WITHOUT TIME INFORMATION
no_of_data_loc1 = iqdata_loc1.shape[0]
no_of_data_loc2 = iqdata_loc2.shape[0]
no_of_data_loc3 = iqdata_loc3.shape[0]
no_of_data_loc4 = iqdata_loc4.shape[0]

no_of_data_loc5 = iqdata_loc5.shape[0]
no_of_data_loc6 = iqdata_loc6.shape[0]
no_of_data_loc7 = iqdata_loc7.shape[0]
no_of_data_loc8 = iqdata_loc8.shape[0]

no_of_data_loc9 = iqdata_loc9.shape[0]
no_of_data_loc10 = iqdata_loc10.shape[0]
no_of_data_loc11 = iqdata_loc11.shape[0]
no_of_data_loc12 = iqdata_loc12.shape[0]



################################################################################################################
# CONCATINATING THE I AND Q VALUES VERTICALLY OF (I, Q) SAMPLE. -- note the axis argument is set to 1 (means vertical stacking)
# SIMULATNEOUSLY MULTIPLYING WITH THE WEIGHT MATRIX - TO REFLECT THE MULTI-ANGULAR PROJECTION

xdata_loc1= np.concatenate([iqdata_loc1['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc1['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc1 = np.matmul(xdata_loc1, np.transpose(W))


xdata_loc2= np.concatenate([iqdata_loc2['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc2['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc2 = np.matmul(xdata_loc2, np.transpose(W))


xdata_loc3= np.concatenate([iqdata_loc3['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc3['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc3 = np.matmul(xdata_loc3, np.transpose(W))


xdata_loc4= np.concatenate([iqdata_loc4['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc4['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc4 = np.matmul(xdata_loc4, np.transpose(W))




xdata_loc5= np.concatenate([iqdata_loc5['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc5['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc5 = np.matmul(xdata_loc5, np.transpose(W))

xdata_loc6= np.concatenate([iqdata_loc6['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc6['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc6 = np.matmul(xdata_loc6, np.transpose(W))


xdata_loc7= np.concatenate([iqdata_loc7['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc7['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc7 = np.matmul(xdata_loc7, np.transpose(W))


xdata_loc8= np.concatenate([iqdata_loc8['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc8['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc8 = np.matmul(xdata_loc8, np.transpose(W))




xdata_loc9= np.concatenate([iqdata_loc9['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc9['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc9 = np.matmul(xdata_loc9, np.transpose(W))


xdata_loc10= np.concatenate([iqdata_loc10['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc10['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc10 = np.matmul(xdata_loc10, np.transpose(W))


xdata_loc11= np.concatenate([iqdata_loc11['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc11['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc11 = np.matmul(xdata_loc11, np.transpose(W))


xdata_loc12= np.concatenate([iqdata_loc12['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc12['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc12 = np.matmul(xdata_loc12, np.transpose(W))




# RESHAPING THE XDATA
xdata_loc1= xdata_loc1.T.reshape(no_of_data_loc1//(sample_size), sample_size*no_of_features)
xdata_loc2 = xdata_loc2.T.reshape(no_of_data_loc2//(sample_size), sample_size*no_of_features)
xdata_loc3 = xdata_loc3.T.reshape(no_of_data_loc3//(sample_size), sample_size*no_of_features)
xdata_loc4 = xdata_loc4.T.reshape(no_of_data_loc4//(sample_size), sample_size*no_of_features)

xdata_loc5= xdata_loc5.T.reshape(no_of_data_loc5//(sample_size), sample_size*no_of_features)
xdata_loc6 = xdata_loc6.T.reshape(no_of_data_loc6//(sample_size), sample_size*no_of_features)
xdata_loc7 = xdata_loc7.T.reshape(no_of_data_loc7//(sample_size), sample_size*no_of_features)
xdata_loc8 = xdata_loc8.T.reshape(no_of_data_loc8//(sample_size), sample_size*no_of_features)

xdata_loc9= xdata_loc9.T.reshape(no_of_data_loc9//(sample_size), sample_size*no_of_features)
xdata_loc10 = xdata_loc10.T.reshape(no_of_data_loc10//(sample_size), sample_size*no_of_features)
xdata_loc11 = xdata_loc11.T.reshape(no_of_data_loc11//(sample_size), sample_size*no_of_features)
xdata_loc12 = xdata_loc12.T.reshape(no_of_data_loc12//(sample_size), sample_size*no_of_features)


# STORING THE XDATA FOR DIFFERENT POSES
xdata_pose1 = np.concatenate([xdata_pose1, xdata_loc1], axis=0)
xdata_pose2 = np.concatenate([xdata_pose2, xdata_loc2], axis=0)
xdata_pose3 = np.concatenate([xdata_pose3, xdata_loc3], axis=0)
xdata_pose4 = np.concatenate([xdata_pose4, xdata_loc4], axis=0)

xdata_pose5 = np.concatenate([xdata_pose5, xdata_loc5], axis=0)
xdata_pose6 = np.concatenate([xdata_pose6, xdata_loc6], axis=0)
xdata_pose7 = np.concatenate([xdata_pose7, xdata_loc7], axis=0)
xdata_pose8 = np.concatenate([xdata_pose8, xdata_loc8], axis=0)

xdata_pose9 = np.concatenate([xdata_pose9, xdata_loc9], axis=0)
xdata_pose10 = np.concatenate([xdata_pose10, xdata_loc10], axis=0)
xdata_pose11 = np.concatenate([xdata_pose11, xdata_loc11], axis=0)
xdata_pose12 = np.concatenate([xdata_pose12, xdata_loc12], axis=0)



#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_10ft = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_10ft =xdata_10ft.astype(np.float)

# REMOVING THE NANS
xdata_10ft = np.nan_to_num(xdata_10ft)

################################################################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE 15FT DATA                                   #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/0_15ft_06_10_2020_914MHz.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path_outdoor + 'DataJune10Out30Degree/15ft/+30_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/+60_15ft_06_10_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/+90_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/+120_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path_outdoor + 'DataJune10Out30Degree/15ft/+150_15ft_06_10_2020_914MHz.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/180_15ft_06_10_2020_914MHz.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/-150_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/-120_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path_outdoor + 'DataJune10Out30Degree/15ft/-90_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/-60_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path_outdoor + 'DataJune10Out30Degree/15ft/-30_15ft_06_10_2020_914MHz.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER



iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all, count = sample_size * no_of_samples)

iqdata_loc5 = scipy.fromfile(open(data_file_loc5), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc6 = scipy.fromfile(open(data_file_loc6), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc7 = scipy.fromfile(open(data_file_loc7), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc8 = scipy.fromfile(open(data_file_loc8), dtype=dtype_all, count = sample_size * no_of_samples)


iqdata_loc9 = scipy.fromfile(open(data_file_loc9), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc10 = scipy.fromfile(open(data_file_loc10), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc11 = scipy.fromfile(open(data_file_loc11), dtype=dtype_all, count = sample_size * no_of_samples)
iqdata_loc12 = scipy.fromfile(open(data_file_loc12), dtype=dtype_all, count = sample_size * no_of_samples)



# PREPARING THE DATA WITHOUT TIME INFORMATION
no_of_data_loc1 = iqdata_loc1.shape[0]
no_of_data_loc2 = iqdata_loc2.shape[0]
no_of_data_loc3 = iqdata_loc3.shape[0]
no_of_data_loc4 = iqdata_loc4.shape[0]

no_of_data_loc5 = iqdata_loc5.shape[0]
no_of_data_loc6 = iqdata_loc6.shape[0]
no_of_data_loc7 = iqdata_loc7.shape[0]
no_of_data_loc8 = iqdata_loc8.shape[0]

no_of_data_loc9 = iqdata_loc9.shape[0]
no_of_data_loc10 = iqdata_loc10.shape[0]
no_of_data_loc11 = iqdata_loc11.shape[0]
no_of_data_loc12 = iqdata_loc12.shape[0]



################################################################################################################
# CONCATINATING THE I AND Q VALUES VERTICALLY OF (I, Q) SAMPLE. -- note the axis argument is set to 1 (means vertical stacking)
# SIMULATNEOUSLY MULTIPLYING WITH THE WEIGHT MATRIX - TO REFLECT THE MULTI-ANGULAR PROJECTION

xdata_loc1= np.concatenate([iqdata_loc1['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc1['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc1 = np.matmul(xdata_loc1, np.transpose(W))


xdata_loc2= np.concatenate([iqdata_loc2['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc2['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc2 = np.matmul(xdata_loc2, np.transpose(W))


xdata_loc3= np.concatenate([iqdata_loc3['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc3['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc3 = np.matmul(xdata_loc3, np.transpose(W))


xdata_loc4= np.concatenate([iqdata_loc4['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc4['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc4 = np.matmul(xdata_loc4, np.transpose(W))




xdata_loc5= np.concatenate([iqdata_loc5['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc5['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc5 = np.matmul(xdata_loc5, np.transpose(W))

xdata_loc6= np.concatenate([iqdata_loc6['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc6['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc6 = np.matmul(xdata_loc6, np.transpose(W))


xdata_loc7= np.concatenate([iqdata_loc7['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc7['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc7 = np.matmul(xdata_loc7, np.transpose(W))


xdata_loc8= np.concatenate([iqdata_loc8['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc8['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc8 = np.matmul(xdata_loc8, np.transpose(W))




xdata_loc9= np.concatenate([iqdata_loc9['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc9['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc9 = np.matmul(xdata_loc9, np.transpose(W))


xdata_loc10= np.concatenate([iqdata_loc10['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc10['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc10 = np.matmul(xdata_loc10, np.transpose(W))


xdata_loc11= np.concatenate([iqdata_loc11['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc11['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc11 = np.matmul(xdata_loc11, np.transpose(W))


xdata_loc12= np.concatenate([iqdata_loc12['raw-iq'].real.reshape(number_of_data_to_read,1), iqdata_loc12['raw-iq'].imag.reshape(number_of_data_to_read,1)], axis=1)
xdata_loc12 = np.matmul(xdata_loc12, np.transpose(W))




# RESHAPING THE XDATA
xdata_loc1= xdata_loc1.T.reshape(no_of_data_loc1//(sample_size), sample_size*no_of_features)
xdata_loc2 = xdata_loc2.T.reshape(no_of_data_loc2//(sample_size), sample_size*no_of_features)
xdata_loc3 = xdata_loc3.T.reshape(no_of_data_loc3//(sample_size), sample_size*no_of_features)
xdata_loc4 = xdata_loc4.T.reshape(no_of_data_loc4//(sample_size), sample_size*no_of_features)

xdata_loc5= xdata_loc5.T.reshape(no_of_data_loc5//(sample_size), sample_size*no_of_features)
xdata_loc6 = xdata_loc6.T.reshape(no_of_data_loc6//(sample_size), sample_size*no_of_features)
xdata_loc7 = xdata_loc7.T.reshape(no_of_data_loc7//(sample_size), sample_size*no_of_features)
xdata_loc8 = xdata_loc8.T.reshape(no_of_data_loc8//(sample_size), sample_size*no_of_features)

xdata_loc9= xdata_loc9.T.reshape(no_of_data_loc9//(sample_size), sample_size*no_of_features)
xdata_loc10 = xdata_loc10.T.reshape(no_of_data_loc10//(sample_size), sample_size*no_of_features)
xdata_loc11 = xdata_loc11.T.reshape(no_of_data_loc11//(sample_size), sample_size*no_of_features)
xdata_loc12 = xdata_loc12.T.reshape(no_of_data_loc12//(sample_size), sample_size*no_of_features)

# STORING THE XDATA FOR DIFFERENT POSES
xdata_pose1 = np.concatenate([xdata_pose1, xdata_loc1], axis=0)
xdata_pose2 = np.concatenate([xdata_pose2, xdata_loc2], axis=0)
xdata_pose3 = np.concatenate([xdata_pose3, xdata_loc3], axis=0)
xdata_pose4 = np.concatenate([xdata_pose4, xdata_loc4], axis=0)

xdata_pose5 = np.concatenate([xdata_pose5, xdata_loc5], axis=0)
xdata_pose6 = np.concatenate([xdata_pose6, xdata_loc6], axis=0)
xdata_pose7 = np.concatenate([xdata_pose7, xdata_loc7], axis=0)
xdata_pose8 = np.concatenate([xdata_pose8, xdata_loc8], axis=0)

xdata_pose9 = np.concatenate([xdata_pose9, xdata_loc9], axis=0)
xdata_pose10 = np.concatenate([xdata_pose10, xdata_loc10], axis=0)
xdata_pose11 = np.concatenate([xdata_pose11, xdata_loc11], axis=0)
xdata_pose12 = np.concatenate([xdata_pose12, xdata_loc12], axis=0)



#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_15ft = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_15ft =xdata_15ft.astype(np.float)

# REMOVING THE NANS
xdata_15ft = np.nan_to_num(xdata_15ft)


################################################################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                    HIERARCHICAL TRAINING- FIRST STEP                                         #######
########                                    FIRST CLASSIFYING THE DATA BASED ON DISTANCES                             #######
########                                  PREDICTING DISTANCE BETWEEN THE RECEIVER AND TRANSMITTER                    #######
########                                                                                                              #######
#############################################################################################################################


xdata_ranging = np.concatenate([xdata_5ft, xdata_10ft, xdata_15ft], axis= 0 )

# CREATING LABEL FOR THE DATASETS
ydata_range1 = np.full(xdata_5ft.shape[0], 0, dtype=int)
ydata_range2 = np.full(xdata_10ft.shape[0], 1, dtype=int)
ydata_range3 = np.full(xdata_15ft.shape[0], 2, dtype=int)

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata_ranging = np.concatenate([ydata_range1, ydata_range2, ydata_range3], axis=0)

#################### NORMALIZE THE X DATA #######################


standard = preprocessing.StandardScaler().fit(xdata_ranging)  # Normalize the data with zero mean and unit variance for each column
xdata_ranging = standard.transform(xdata_ranging)



############### SEPARATING TRAIN AND TEST DATA #######################
print("############## STARTING THE TRAINING TO PREDICT THE RANGE BETWEEN RECEIVER AND TRANSMITTER ##########################")

xtrain_ranging, xtest_ranging, ytrain_ranging, ytest_ranging = train_test_split(xdata_ranging, ydata_ranging, test_size=0.2, shuffle = True, random_state=42)  # Randomly shuffling and 80/20 is train/test size
print("XTRAIN AND XTEST SHAPE:", xtrain_ranging.shape, xtest_ranging.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain_ranging.shape, ytest_ranging.shape)

# RESHAPING THE DATA FROM 2 DIMENSIONAL TO 4 DIMENSIONAL SHAPE - NEEDED TO APPLY TO USE 2D-CONVOLUTION
# reshape to be [samples][width][height][channels]
xtrain_ranging = xtrain_ranging.reshape((xtrain_ranging.shape[0], no_of_features, sample_size, 1)).astype('float32')
xtest_ranging = xtest_ranging.reshape((xtest_ranging.shape[0], no_of_features, sample_size, 1)).astype('float32')


num_classes = 3  # TOTAL NUMBER OF RANGES



# Convert labels to categorical one-hot encoding
ytrain_ranging_one_hot = to_categorical(ytrain_ranging, num_classes=num_classes)  # DEFINE THE NUMBER OF TOTAL CLASSES IN LABEL
ytest_ranging_one_hot = to_categorical(ytest_ranging, num_classes=num_classes)


print("XTRAIN AND XTEST SHAPE:", xtrain_ranging.shape, xtest_ranging.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain_ranging_one_hot.shape, ytest_ranging_one_hot.shape)

############################################################
#                                                          #
########    Building a 2D Convolutional Neural Network #####
#							                               #
############################################################

dr = 0.6  # dropout rate (%)
batch_size = 128  # Mini batch size
nb_epoch = 100  # Number of Epoch (Give a higher number to get better accuracy)

classes = ["5ft", "10ft", "15ft"] # CHANGE LABEL
in_shp = list(xtrain_ranging.shape[1:])  # Input Dimension
print(in_shp)
# model = models.Sequential()
timesteps=1
data_dim=xtrain_ranging.shape[1]



# print ("AFTER RESHAPE")
ytrain_ranging_one_hot = np.reshape(ytrain_ranging_one_hot, (ytrain_ranging_one_hot.shape[0], num_classes))  # Used in training
ytest_ranging_one_hot = np.reshape(ytest_ranging_one_hot, (ytest_ranging_one_hot.shape[0], num_classes))  # Used in training

start_time = time.time()  # Taking start time to calculate overall execution time

# Modeling the CNN
model_ranging = Sequential()

# FIRST CONVOLUTIONAL LAYER
model_ranging.add(Conv2D(128, (2, 2), input_shape=(no_of_features, sample_size, 1), activation='relu')) # CHANGE # Stride (1, 1)
model_ranging.add(MaxPooling2D())  # Pool size: (2, 2) and stride (2, 2)
model_ranging.add(Dropout(0.2))

# SECOND CONVOLUTIONAL LAYER
model_ranging.add(Conv2D(64, (2, 2), activation='relu'))
model_ranging.add(MaxPooling2D())
model_ranging.add(Dropout(dr))

model_ranging.add(Flatten())

# FIRST DENSE LAYER
model_ranging.add(Dense(256, activation='relu'))

# SECOND DENSE LAYER
model_ranging.add(Dense(128, activation='relu'))

# OUTPUT LAYER
model_ranging.add(Dense(num_classes, activation='softmax'))

# Compile model
# For a multi-class classification problem
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # Multiclass classification with rmsprop

#model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['acc', f1_m, precision_m, recall_m])  # Multiclass classification with rms adam optimizer # CHANGE

model_ranging.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

model_ranging.summary()
filepath = '/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/direction_data_ranging_2D_CNN_Mapping.wts.h5'
print("The dropout rate was: ")
print(dr)


# Fit the model
# history= model.fit(xtrain, ytrain_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_data = (xtest, ytest_one_hot), callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'), keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])
history = model_ranging.fit(xtrain_ranging, ytrain_ranging_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])



# SAVING THE MODEL FOR TRANSFER LEARNING
saved_file = '/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/2D_CNN_ranging_classifier.h5'
model_ranging.save(saved_file) # SAVING THE MODEL FOR TRANSFER LEARNING



# Evaluate the model
loss, accuracy, f1_score, precision, recall = model_ranging.evaluate(xtest_ranging, ytest_ranging_one_hot, batch_size=batch_size) # CHANGE
print("\nTest Loss: %s: %.2f%%" % (model_ranging.metrics_names[0], loss * 100)) # CHANGE
print("\nTest Accuracy: %s: %.2f%%" % (model_ranging.metrics_names[1], accuracy * 100)) # CHANGE
print("\nTest F1 Score: %s: %.2f" % (model_ranging.metrics_names[2], f1_score)) # CHANGE
print("\nTest Precision: %s: %.2f%%" % (model_ranging.metrics_names[3], precision * 100)) # CHANGE
print("\nTest Recall: %s: %.2f%%" % (model_ranging.metrics_names[4], recall * 100)) # CHANGE

# Calculating total execution time
end_time = time.time()  # Taking end time to calculate overall execution time
print("\n Total Execution Time (Minutes): ")
print(((end_time - start_time) / 60))

#### SET PLOTTING PARAMETERS #########
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
plt.rcParams.update(params)


# Show Accuracy Curves
fig = plt.figure()
# plt.title('Training Performance')
plt.plot(history.epoch, history.history['acc'], label='Training Accuracy', linewidth=2.0, c='b')
plt.plot(history.epoch, history.history['val_acc'], label='Validation Accuracy', linewidth=2.0, c='r')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/direction_ranging_acc_2D_CNN_Mapping.png')  # save the figure to file
plt.close(fig)


# plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.YlGnBu, labels=[], normalize=False, filedest = ''):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    else:
        cm = cm.astype('int')
    # print('Confusion matrix, without normalization')
    plt.rcParams.update(params) # ADDED
    fig = plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i,"{:,}".format(cm[i, j]),
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize="xx-large",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    # ax.plot([0, 1, 2], [10, 20, 3])
    plt.tight_layout()
    fig.savefig(filedest)  # save the figure to file
    plt.close(fig)


# plt.show()



# Plot confusion matrix
test_Y_hat = model_ranging.predict(xtest_ranging, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, xtest_ranging.shape[0]):
    j = list(ytest_ranging_one_hot[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
plot_confusion_matrix(conf, labels=classes, normalize=False, filedest='/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/direction_ranging_conf_mat_2D_CNN_Mapping.png')


#############################################################################################################################
########                                                                                                              #######
########                                    HIERARCHICAL TRAINING- SECOND STEP                                        #######
########                                     CLASSIFYING THE DATA BASED ON POSES OF TRANSMITER ANTENNA                #######
########                                  PREDICTING REALTIVE POSES OF TRANSMITER ANTENNA                             #######
#############################################################################################################################


xdata_pose = np.concatenate([xdata_pose1, xdata_pose2, xdata_pose3, xdata_pose4, xdata_pose5, xdata_pose6, xdata_pose7, xdata_pose8, xdata_pose9, xdata_pose10, xdata_pose11, xdata_pose12], axis= 0 )

# CREATING LABEL FOR THE DATASETS
ydata_loc1 = np.full(xdata_pose1.shape[0], 0, dtype=int)
ydata_loc2 = np.full(xdata_pose2.shape[0], 1, dtype=int)
ydata_loc3 = np.full(xdata_pose3.shape[0], 2, dtype=int)
ydata_loc4 = np.full(xdata_pose4.shape[0], 3, dtype=int)

ydata_loc5 = np.full(xdata_pose5.shape[0], 4, dtype=int)
ydata_loc6 = np.full(xdata_pose6.shape[0], 5, dtype=int)
ydata_loc7 = np.full(xdata_pose7.shape[0], 6, dtype=int)
ydata_loc8 = np.full(xdata_pose8.shape[0], 7, dtype=int)

ydata_loc9 = np.full(xdata_pose9.shape[0], 8, dtype=int)
ydata_loc10 = np.full(xdata_pose10.shape[0], 9, dtype=int)
ydata_loc11 = np.full(xdata_pose11.shape[0], 10, dtype=int)
ydata_loc12 = np.full(xdata_pose12.shape[0], 11, dtype=int)

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata_pose = np.concatenate([ydata_loc1, ydata_loc2, ydata_loc3, ydata_loc4, ydata_loc5, ydata_loc6, ydata_loc7, ydata_loc8, ydata_loc9, ydata_loc10, ydata_loc11, ydata_loc12], axis=0)


#################### NORMALIZE THE X DATA #######################


standard = preprocessing.StandardScaler().fit(xdata_pose)  # Normalize the data with zero mean and unit variance for each column
xdata_pose = standard.transform(xdata_pose)


print("############## STARTING THE TRAINING TO PREDICT THE POSES OF TRANSMITTER ANTENNA WITH (5/10/15) FT DISTANCE FROM RECEIVER ##########################")



############### SEPARATING TRAIN AND TEST DATA #######################

xtrain_pose, xtest_pose, ytrain_pose, ytest_pose = train_test_split(xdata_pose, ydata_pose, test_size=0.2, shuffle = True, random_state=42)  # Randomly shuffling and 80/20 is train/test size
print("XTRAIN AND XTEST SHAPE:", xtrain_pose.shape, xtest_pose.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain_pose.shape, ytest_pose.shape)

# RESHAPING THE DATA FROM 2 DIMENSIONAL TO 4 DIMENSIONAL SHAPE - NEEDED TO APPLY TO USE 2D-CONVOLUTION
# reshape to be [samples][width][height][channels]
xtrain_pose = xtrain_pose.reshape((xtrain_pose.shape[0], no_of_features, sample_size, 1)).astype('float32')
xtest_pose = xtest_pose.reshape((xtest_pose.shape[0], no_of_features, sample_size, 1)).astype('float32')


num_classes = 12  # TOTAL NUMBER OF RANGES



# Convert labels to categorical one-hot encoding
ytrain_pose_one_hot = to_categorical(ytrain_pose, num_classes=num_classes)  # DEFINE THE NUMBER OF TOTAL CLASSES IN LABEL
ytest_pose_one_hot = to_categorical(ytest_pose, num_classes=num_classes)


print("XTRAIN AND XTEST SHAPE:", xtrain_pose.shape, xtest_pose.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain_pose_one_hot.shape, ytest_pose_one_hot.shape)

############################################################
#                                                          #
########    Building a 2D Convolutional Neural Network #####
#							                               #
############################################################

dr = 0.6  # dropout rate (%)
batch_size = 128  # Mini batch size
nb_epoch = 100  # Number of Epoch (Give a higher number to get better accuracy)

classes = ["0", "+30", "+60", "+90", "+120", "+150", "180", "-150", "-120", "-90", "-60", "-30"] # CHANGE LABEL
in_shp = list(xtrain_pose.shape[1:])  # Input Dimension
print(in_shp)
# model = models.Sequential()
timesteps=1
data_dim=xtrain_pose.shape[1]



# print ("AFTER RESHAPE")
ytrain_pose_one_hot = np.reshape(ytrain_pose_one_hot, (ytrain_pose_one_hot.shape[0], num_classes))  # Used in training
ytest_pose_one_hot = np.reshape(ytest_pose_one_hot, (ytest_pose_one_hot.shape[0], num_classes))  # Used in training

start_time = time.time()  # Taking start time to calculate overall execution time

#IMPLEMENTING THE TRANSFER LEARNING
#loading the previously saved model
source_model = load_model(saved_file, custom_objects={
        "f1_m": f1_m,
        "precision_m": precision_m,
        "recall_m": recall_m
    })

model_pose = Sequential()
for layer in source_model.layers[:-3]: # go through until last layer
    model_pose.add(layer)

 ####################  -  THIS NOT WORKING #############
# model_pose = load_model(saved_file, custom_objects={
#         "f1_m": f1_m,
#         "precision_m": precision_m,
#         "recall_m": recall_m
#     })

# POPING THE LAST THREE DENSE LAYERs (2 Hidden  + 1 Output)
# model_pose.layers.pop()
# model_pose.layers.pop()
# model_pose.layers.pop()
 ####################  -  THIS NOT WORKING #############

# FIRST DENSE LAYER
model_pose.add(Dense(256, activation='relu'))

# SECOND DENSE LAYER
model_pose.add(Dense(128, activation='relu'))

# # THIRD DENSE LAYER - ADDING NEW LAYER DID NOT WORK
# model_pose.add(Dense(64, activation='relu'))

# ADDING OUTPUT LAYER
model_pose.add(Dense(num_classes, activation='softmax'))

# Compile model
# For a multi-class classification problem
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # Multiclass classification with rmsprop

#model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['acc', f1_m, precision_m, recall_m])  # Multiclass classification with rms adam optimizer # CHANGE

model_pose.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

model_pose.summary()
filepath = '/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/direction_data_12_poses_2D_CNN_Mapping.wts.h5'
print("The dropout rate was: ")
print(dr)


# Fit the model
# history= model.fit(xtrain, ytrain_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_data = (xtest, ytest_one_hot), callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'), keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])
history = model_pose.fit(xtrain_pose, ytrain_pose_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])


# Evaluate the model
loss, accuracy, f1_score, precision, recall = model_pose.evaluate(xtest_pose, ytest_pose_one_hot, batch_size=batch_size) # CHANGE
print("\nTest Loss: %s: %.2f%%" % (model_pose.metrics_names[0], loss * 100)) # CHANGE
print("\nTest Accuracy: %s: %.2f%%" % (model_pose.metrics_names[1], accuracy * 100)) # CHANGE
print("\nTest F1 Score: %s: %.2f" % (model_pose.metrics_names[2], f1_score)) # CHANGE
print("\nTest Precision: %s: %.2f%%" % (model_pose.metrics_names[3], precision * 100)) # CHANGE
print("\nTest Recall: %s: %.2f%%" % (model_pose.metrics_names[4], recall * 100)) # CHANGE

# Calculating total execution time
end_time = time.time()  # Taking end time to calculate overall execution time
print("\n Total Execution Time (Minutes): ")
print(((end_time - start_time) / 60))

#### SET PLOTTING PARAMETERS #########
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
plt.rcParams.update(params)


# Show Accuracy Curves
fig = plt.figure()
# plt.title('Training Performance')
plt.plot(history.epoch, history.history['acc'], label='Training Accuracy', linewidth=2.0, c='b')
plt.plot(history.epoch, history.history['val_acc'], label='Validation Accuracy', linewidth=2.0, c='r')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/direction_12_poses_acc_2D_CNN_Mapping.png')  # save the figure to file
plt.close(fig)


# plt.show()


# Plot confusion matrix
test_Y_hat = model_pose.predict(xtest_pose, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, xtest_pose.shape[0]):
    j = list(ytest_pose_one_hot[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
plot_confusion_matrix(conf, labels=classes, normalize=False, filedest= '/Users/debashri/Desktop/DirectionFinding_Plots/Outdoor/direction_12_poses_conf_mat_2D_CNN_Mapping.png')

