# TRANSMITTER HAS A DIRECTIONAL ANTENNA -  POINTED IN 12 DIFFERENT POSES
# RECEIVER HAS AN DIRECTIONAL DIRECTIONAL ANTENNA - POINTED TO 4 DIFFERENT DIRECTIONS (ORTHOGONAL TO EACH OTHER)
# DISTANCE BETWEEN RECEIVER AND TRANSMITTER - (5) FEET
# IT IS A 48-CLASS CLASSIFICATION
# DATA COLLECTED IN INDOOR ENVIRONMENT

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


sample_size = 512  # CHANGE AND EXPERIMENT - 1024
no_of_samples = 4000 # CHANGE AND EXPERIMENT - 4000
no_of_features= 8 # CHANGE AND EXPERIMENT
number_of_data_to_read = sample_size * no_of_samples
folder_path = '/Users/debashri/Desktop/DirectionFinding_Data/Indoor/Directional/' # CHANGE AS PER YOUR COMPUTER FOLDER PATH

#######################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE 0 DEGREE RECEIVER DATA                     #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path + '0R/0T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path + '0R/+30T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path + '0R/+60T_0R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path + '0R/+90T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path + '0R/+120T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path + '0R/+150T_0R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path + '0R/180T_0R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path + '0R/-150T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path + '0R/-120T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path + '0R/-90T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path + '0R/-60T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path + '0R/-30T_0R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER



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


# # STORING THE XDATA FOR DIFFERENT POSES
# xdata_pose1 = xdata_loc1
# xdata_pose2 = xdata_loc2
# xdata_pose3 = xdata_loc3
# xdata_pose4 = xdata_loc4
#
# xdata_pose5 = xdata_loc5
# xdata_pose6 = xdata_loc6
# xdata_pose7 = xdata_loc7
# xdata_pose8 = xdata_loc8
#
# xdata_pose9 = xdata_loc9
# xdata_pose10 = xdata_loc10
# xdata_pose11 = xdata_loc11
# xdata_pose12 = xdata_loc12

# CREATING LABEL FOR THE DATASETS
ydata_loc1 = np.full(xdata_loc1.shape[0], 0, dtype=int)
ydata_loc2 = np.full(xdata_loc2.shape[0], 1, dtype=int)
ydata_loc3 = np.full(xdata_loc3.shape[0], 2, dtype=int)
ydata_loc4 = np.full(xdata_loc4.shape[0], 3, dtype=int)

ydata_loc5 = np.full(xdata_loc5.shape[0], 4, dtype=int)
ydata_loc6 = np.full(xdata_loc6.shape[0], 5, dtype=int)
ydata_loc7 = np.full(xdata_loc7.shape[0], 6, dtype=int)
ydata_loc8 = np.full(xdata_loc8.shape[0], 7, dtype=int)

ydata_loc9 = np.full(xdata_loc9.shape[0], 8, dtype=int)
ydata_loc10 = np.full(xdata_loc10.shape[0], 9, dtype=int)
ydata_loc11 = np.full(xdata_loc11.shape[0], 10, dtype=int)
ydata_loc12 = np.full(xdata_loc12.shape[0], 11, dtype=int)

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata_0R = np.concatenate([ydata_loc1, ydata_loc2, ydata_loc3, ydata_loc4, ydata_loc5, ydata_loc6, ydata_loc7, ydata_loc8, ydata_loc9, ydata_loc10, ydata_loc11, ydata_loc12], axis=0)




#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_0R = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_0R =xdata_0R.astype(np.float)

# REMOVING THE NANS
xdata_0R = np.nan_to_num(xdata_0R)



################################################################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE -90 DEGREE RECEIVER DATA                   #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path + '-90R/0T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path + '-90R/+30T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path + '-90R/+60T_-90R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path + '-90R/+90T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path + '-90R/+120T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path + '-90R/+150T_-90R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path + '-90R/180T_-90R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path + '-90R/-150T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path + '-90R/-120T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path + '-90R/-90T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path + '-90R/-60T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path + '-90R/-30T_-90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER


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


# CREATING LABEL FOR THE DATASETS
addIndex = 12
ydata_loc1 = np.full(xdata_loc1.shape[0], (addIndex+0), dtype=int)
ydata_loc2 = np.full(xdata_loc2.shape[0], (addIndex+1), dtype=int)
ydata_loc3 = np.full(xdata_loc3.shape[0], (addIndex+2), dtype=int)
ydata_loc4 = np.full(xdata_loc4.shape[0], (addIndex+3), dtype=int)

ydata_loc5 = np.full(xdata_loc5.shape[0], (addIndex+4), dtype=int)
ydata_loc6 = np.full(xdata_loc6.shape[0], (addIndex+5), dtype=int)
ydata_loc7 = np.full(xdata_loc7.shape[0], (addIndex+6), dtype=int)
ydata_loc8 = np.full(xdata_loc8.shape[0], (addIndex+7), dtype=int)

ydata_loc9 = np.full(xdata_loc9.shape[0], (addIndex+8), dtype=int)
ydata_loc10 = np.full(xdata_loc10.shape[0], (addIndex+9), dtype=int)
ydata_loc11 = np.full(xdata_loc11.shape[0], (addIndex+10), dtype=int)
ydata_loc12 = np.full(xdata_loc12.shape[0], (addIndex+11), dtype=int)

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata_m90R = np.concatenate([ydata_loc1, ydata_loc2, ydata_loc3, ydata_loc4, ydata_loc5, ydata_loc6, ydata_loc7, ydata_loc8, ydata_loc9, ydata_loc10, ydata_loc11, ydata_loc12], axis=0)




#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_m90R = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_m90R =xdata_m90R.astype(np.float)

# REMOVING THE NANS
xdata_m90R = np.nan_to_num(xdata_m90R)

################################################################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE 180 DEGREE RECEIVER DATA                   #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path + '180R/0T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path + '180R/+30T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path + '180R/+60T_180R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path + '180R/+90T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path + '180R/+120T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path + '180R/+150T_180R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path + '180R/180T_180R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path + '180R/-150T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path + '180R/-120T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path + '180R/-90T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path + '180R/-60T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path + '180R/-30T_180R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER




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


# CREATING LABEL FOR THE DATASETS
addIndex = 24
ydata_loc1 = np.full(xdata_loc1.shape[0], (addIndex+0), dtype=int)
ydata_loc2 = np.full(xdata_loc2.shape[0], (addIndex+1), dtype=int)
ydata_loc3 = np.full(xdata_loc3.shape[0], (addIndex+2), dtype=int)
ydata_loc4 = np.full(xdata_loc4.shape[0], (addIndex+3), dtype=int)

ydata_loc5 = np.full(xdata_loc5.shape[0], (addIndex+4), dtype=int)
ydata_loc6 = np.full(xdata_loc6.shape[0], (addIndex+5), dtype=int)
ydata_loc7 = np.full(xdata_loc7.shape[0], (addIndex+6), dtype=int)
ydata_loc8 = np.full(xdata_loc8.shape[0], (addIndex+7), dtype=int)

ydata_loc9 = np.full(xdata_loc9.shape[0], (addIndex+8), dtype=int)
ydata_loc10 = np.full(xdata_loc10.shape[0], (addIndex+9), dtype=int)
ydata_loc11 = np.full(xdata_loc11.shape[0], (addIndex+10), dtype=int)
ydata_loc12 = np.full(xdata_loc12.shape[0], (addIndex+11), dtype=int)

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata_180R = np.concatenate([ydata_loc1, ydata_loc2, ydata_loc3, ydata_loc4, ydata_loc5, ydata_loc6, ydata_loc7, ydata_loc8, ydata_loc9, ydata_loc10, ydata_loc11, ydata_loc12], axis=0)




#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_180R = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_180R =xdata_180R.astype(np.float)

# REMOVING THE NANS
xdata_180R = np.nan_to_num(xdata_180R)


################################################################################################################################

#############################################################################################################################
########                                                                                                              #######
########                                                       READING THE +90 DEGREE RECEIVER DATA                   #######
########                                                                                                              #######
#############################################################################################################################

data_file_loc1 = folder_path + '+90R/0T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER DIRECTLY POINTING TO THE RECEIVER
data_file_loc2 =folder_path + '+90R/+30T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE LEFT TO THE RECEIVER
data_file_loc3 = folder_path + '+90R/+60T_+90R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 60 DEGREE LEFT TO THE RECEIVER
data_file_loc4 = folder_path + '+90R/+90T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE LEFT TO THE RECEIVER

data_file_loc5 = folder_path + '+90R/+120T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE LEFT TO THE RECEIVER
data_file_loc6 =folder_path + '+90R/+150T_+90R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS 150 DEGREE LEFT TO THE RECEIVER
data_file_loc7 = folder_path + '+90R/180T_+90R_5ft_06_22_2020_914MHz_Indoor.dat'# TRANSMITTER ANTENNA IS DIRECTLY POINTED AWAY FROM THE RECEIVER
data_file_loc8 = folder_path + '+90R/-150T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 30 DEGREE RIGHT TO THE RECEIVER

data_file_loc9 = folder_path + '+90R/-120T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 60 DEGREE RIGHT TO THE RECEIVER
data_file_loc10 =folder_path + '+90R/-90T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 90 DEGREE RIGHT TO THE RECEIVER
data_file_loc11 = folder_path + '+90R/-60T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 120 DEGREE RIGHT TO THE RECEIVER
data_file_loc12 = folder_path + '+90R/-30T_+90R_5ft_06_22_2020_914MHz_Indoor.dat' # TRANSMITTER ANTENNA IS 150 DEGREE RIGHT TO THE RECEIVER


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


# CREATING LABEL FOR THE DATASETS
addIndex = 36
ydata_loc1 = np.full(xdata_loc1.shape[0], (addIndex+0), dtype=int)
ydata_loc2 = np.full(xdata_loc2.shape[0], (addIndex+1), dtype=int)
ydata_loc3 = np.full(xdata_loc3.shape[0], (addIndex+2), dtype=int)
ydata_loc4 = np.full(xdata_loc4.shape[0], (addIndex+3), dtype=int)

ydata_loc5 = np.full(xdata_loc5.shape[0], (addIndex+4), dtype=int)
ydata_loc6 = np.full(xdata_loc6.shape[0], (addIndex+5), dtype=int)
ydata_loc7 = np.full(xdata_loc7.shape[0], (addIndex+6), dtype=int)
ydata_loc8 = np.full(xdata_loc8.shape[0], (addIndex+7), dtype=int)

ydata_loc9 = np.full(xdata_loc9.shape[0], (addIndex+8), dtype=int)
ydata_loc10 = np.full(xdata_loc10.shape[0], (addIndex+9), dtype=int)
ydata_loc11 = np.full(xdata_loc11.shape[0], (addIndex+10), dtype=int)
ydata_loc12 = np.full(xdata_loc12.shape[0], (addIndex+11), dtype=int)

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata_90R = np.concatenate([ydata_loc1, ydata_loc2, ydata_loc3, ydata_loc4, ydata_loc5, ydata_loc6, ydata_loc7, ydata_loc8, ydata_loc9, ydata_loc10, ydata_loc11, ydata_loc12], axis=0)




#CONCATINATING THE DIFFERENT POSE DATA HORIZONTALLY (ROWWISE)
xdata_90R = np.concatenate([xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4, xdata_loc5, xdata_loc6, xdata_loc7, xdata_loc8, xdata_loc9, xdata_loc10, xdata_loc11, xdata_loc12], axis=0)

# PREPROCESSING X AND Y DATA
xdata_90R =xdata_90R.astype(np.float)

# REMOVING THE NANS
xdata_90R = np.nan_to_num(xdata_90R)

################################################################################################################################


#############################################################################################################################
########                                                                                                              #######
########                                  PREDICTING POSES WITH DIRECTIONAL THE RECEIVER AND TRANSMITTER              #######
########                                                                                                              #######
#############################################################################################################################


xdata = np.concatenate([xdata_0R, xdata_m90R, xdata_180R, xdata_90R], axis= 0 )

#CONCATINATING THE DIFFERENT POSE LABELS HORIZONTALLY (ROWWISE)
ydata = np.concatenate([ydata_0R, ydata_m90R, ydata_180R, ydata_90R], axis=0)

#################### NORMALIZE THE X DATA #######################


standard = preprocessing.StandardScaler().fit(xdata)  # Normalize the data with zero mean and unit variance for each column
xdata = standard.transform(xdata)



############### SEPARATING TRAIN AND TEST DATA #######################
print("############## STARTING THE TRAINING TO PREDICT THE RELATIVE POSE OF RECEIVER AND TRANSMITTER ##########################")

xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2, shuffle = True, random_state=42)  # Randomly shuffling and 80/20 is train/test size
print("XTRAIN AND XTEST SHAPE:", xtrain.shape, xtest.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

# RESHAPING THE DATA FROM 2 DIMENSIONAL TO 4 DIMENSIONAL SHAPE - NEEDED TO APPLY TO USE 2D-CONVOLUTION
# reshape to be [samples][width][height][channels]
xtrain = xtrain.reshape((xtrain.shape[0], no_of_features, sample_size, 1)).astype('float32')
xtest = xtest.reshape((xtest.shape[0], no_of_features, sample_size, 1)).astype('float32')


num_classes = 48  # TOTAL NUMBER OF RANGES



# Convert labels to categorical one-hot encoding
ytrain_one_hot = to_categorical(ytrain, num_classes=num_classes)  # DEFINE THE NUMBER OF TOTAL CLASSES IN LABEL
ytest_one_hot = to_categorical(ytest, num_classes=num_classes)


print("XTRAIN AND XTEST SHAPE:", xtrain.shape, xtest.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain_one_hot.shape, ytest_one_hot.shape)

############################################################
#                                                          #
########    Building a 2D Convolutional Neural Network #####
#							                               #
############################################################

dr = 0.6  # dropout rate (%)
batch_size = 128  # Mini batch size
nb_epoch = 100  # Number of Epoch (Give a higher number to get better accuracy)

classes = ["0_0R", "+30_0R", "+60_0R", "+90_0R", "+120_0R", "+150_0R", "180_0R", "-150_0R", "-120_0R", "-90_0R", "-60_0R", "-30_0R",
           "0_-90R", "+30_-90R", "+60_-90R", "+90_-90R", "+120_-90R", "+150_-90R", "180_-90R", "-150_-90R", "-120_-90R", "-90_-90R", "-60_-90R", "-30_-90R",
           "0_180R", "+30_180R", "+60_180R", "+90_180R", "+120_180R", "+150_180R", "180_180R", "-150_180R", "-120_180R", "-90_180R", "-60_180R", "-30_180R",
           "0_90R", "+30_90R", "+60_90R", "+90_90R", "+120_90R", "+150_90R", "180_90R", "-150_90R", "-120_90R", "-90_90R", "-60_90R", "-30_90R",] # CHANGE LABEL
in_shp = list(xtrain.shape[1:])  # Input Dimension
print(in_shp)
# model = models.Sequential()
timesteps=1
data_dim=xtrain.shape[1]



# print ("AFTER RESHAPE")
ytrain_one_hot = np.reshape(ytrain_one_hot, (ytrain_one_hot.shape[0], num_classes))  # Used in training
ytest_one_hot = np.reshape(ytest_one_hot, (ytest_one_hot.shape[0], num_classes))  # Used in training

start_time = time.time()  # Taking start time to calculate overall execution time

# Modeling the CNN
model_ranging = Sequential()

# FIRST CONVOLUTIONAL LAYER
model_ranging.add(Conv2D(256, (2, 2), input_shape=(no_of_features, sample_size, 1), activation='relu')) # CHANGE # Stride (1, 1)
model_ranging.add(MaxPooling2D())  # Pool size: (2, 2) and stride (2, 2)
model_ranging.add(Dropout(0.2))

# SECOND CONVOLUTIONAL LAYER
model_ranging.add(Conv2D(128, (2, 2), activation='relu'))
model_ranging.add(MaxPooling2D())
model_ranging.add(Dropout(dr))

model_ranging.add(Flatten())

# FIRST DENSE LAYER
model_ranging.add(Dense(256, activation='relu'))

# SECOND DENSE LAYER
model_ranging.add(Dense(128, activation='relu'))


# # THIRD DENSE LAYER -  EXTRA LAYER DID NOT WORK
# model_ranging.add(Dense(64, activation='relu'))

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
filepath = '/Users/debashri/Desktop/DirectionFinding_Plots/Indoor/double_direction_2D_CNN.wts.h5'
print("The dropout rate was: ")
print(dr)


# Fit the model
# history= model.fit(xtrain, ytrain_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_data = (xtest, ytest_one_hot), callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'), keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])
history = model_ranging.fit(xtrain, ytrain_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])



# SAVING THE MODEL FOR TRANSFER LEARNING
saved_file = '/Users/debashri/Desktop/DirectionFinding_Plots/Indoor/double_direction_classifier.h5'
model_ranging.save(saved_file) # SAVING THE MODEL FOR TRANSFER LEARNING



# Evaluate the model
loss, accuracy, f1_score, precision, recall = model_ranging.evaluate(xtest, ytest_one_hot, batch_size=batch_size) # CHANGE
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
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}
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
fig.savefig('/Users/debashri/Desktop/DirectionFinding_Plots/Indoor/double_direction_5ft_acc_indoor.png')  # save the figure to file
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
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i,"{:,}".format(cm[i, j]),
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize="xx-small",
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
test_Y_hat = model_ranging.predict(xtest, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, xtest.shape[0]):
    j = list(ytest_one_hot[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
plot_confusion_matrix(conf, labels=classes, normalize=False, filedest='/Users/debashri/Desktop/DirectionFinding_Plots/Indoor/double_direction_5ft_conf_mat_indoor.png')

