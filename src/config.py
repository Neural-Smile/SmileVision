## GLOBALS / CONSTANTS ##
PCA_N_COMPONENTS = 150
MLP_ID = 1
SVM_ID = 1
DEBUG = True

config = {
            'h'             : 250,
            'w'             : 250,
            'min_faces'     : 70,
            'aspect_ratio'  : 0.4
         }

def setup_config(h = 250, w = 250, min_faces = 70, aspect_ratio = 0.4):
    config['h'] = h
    config['w'] = w
    config['min_faces'] = min_faces
    config['aspect_ratio'] = aspect_ratio
