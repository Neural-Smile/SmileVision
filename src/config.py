## GLOBALS / CONSTANTS ##
PCA_N_COMPONENTS = 150
MLP_ID = 1
SVM_ID = 2
NO_MATCH = "no_match"
DEBUG = True
USE_CACHED_MODEL = True

## these are supposed to be standard dimensions of img
## not sure if discrepency actually affects things
## or if i can just brute resize everything either
W = 37
H = 50

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
