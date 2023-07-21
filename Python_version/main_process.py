## Main Program 

# Quincy(Qiyuan) Ma
# 01/03/2021
# 


import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Processing import bandpower


## Initial
# load EEG_data 
# Data: channels(data+trigger) X data length
# All data were preprocessed: filtered with 1-45Hz band filter and downsampled 

raw_data = sio.load('EEG.mat')

# config
chan_num = len(raw_data,1)-1  
raw_sample_rate = 1000
target_sample_rate = 250

video_length = 300
game_length = 300

frontal_channel = [0,1,2,3,4,7,8,9]
temporal_channnel = [10,11,12,13,14,15,19,20,21,22,23,24,30,31,32,33,38,39,40,41,47,48]
central_channel = [16,17,18,25,26,27,28,29]
parietal_channel = [34,35,36,37,42,43,44,45,46]
occipital_channel = [49,50,51,54,55,56,57,58]
behindEar_channel = [5,6,52,53,60,61,62,63]



# Epoch



## Extract Features


for chan_i in chan_num:
    pass

##


