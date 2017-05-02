from dev_config import *
## GLOBALS / CONSTANTS ##
PCA_N_COMPONENTS = 100
MLP_ID = 1
SVM_ID = 2
NO_MATCH = "no_match"
DEBUG = DEV_DEBUG # this is ghetto sorry
USE_CACHED_MODEL = True
SMALL_MODEL = True
SMILE_DATA_PATH = "data/smile_home"
PREPROCESSOR_CACHE_PATH = "data/preprocessor_cache"

BEST_SMALL_MODEL = {'hidden_layer_sizes':(2,1), 'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'batch_size': 80}
BEST_MODEL = {'hidden_layer_sizes':(20,8), 'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'batch_size': 80}

## these are supposed to be standard dimensions of img
## not sure if discrepency actually affects things
## or if i can just brute resize everything either
processed_width = 100
processed_height = 100

config = {
            'h'             : 250,
            'w'             : 250,
            'min_faces'     : 10,
            'aspect_ratio'  : 0.4
         }

def setup_config(h = 250, w = 250, min_faces = 70, aspect_ratio = 0.4):
    config['h'] = h
    config['w'] = w
    config['min_faces'] = min_faces
    config['aspect_ratio'] = aspect_ratio
