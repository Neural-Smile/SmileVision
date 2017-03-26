from time import time
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from config import *

##################
## CLASSIFIERS ###
##################

## SUPPORT VECTOR MACHINE ##
class SVM:
    def __init__(self, search=True, param_grid = []):
        self.search = search
        self.param_grid = param_grid
        #TODO: param passing, default for now
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        if search:
            self.clf = GridSearchCV(
                SVC(kernel='rbf', class_weight='balanced'), param_grid)
        else:
            #TODO: passing params
            self.clf = SVC(kernel='rbf', class_weight='balanced')


    def train(self, x_train, y_train):
        if DEBUG:
            t0 = time()
        self.clf.fit(x_train, y_train)
        if DEBUG:
            print("done in %0.3fs" % (time() - t0))
            if self.search:
                print(self.clf.best_estimator_)

    def predict(self, x_test):
        #Debug info?
        return self.clf.predict(x_test)


    def predict_prob(self, x_test):
        return self.clf.predict_proba(x_test) 



## MULTI-LAYER PERCEPTRON ##
class MLP:
    def __init__(self, search=True, param_grid = []):
        self.search = search
        self.param_grid = param_grid
        #TODO: param passing, default for now
        param_grid = {'hidden_layer_sizes' : [(50, 3)], 'alpha' : [0.1, 1, 10]}
        if self.search:
            self.clf = GridSearchCV(
                MLPClassifier(activation="relu", solver="adam"), param_grid)
        else:
            #TODO: passing params
            self.clf = MLPClassifier(activation="relu", solver="adam")


    def train(self, x_train, y_train):
        if DEBUG:
            t0 = time()
        self.clf.fit(x_train, y_train)
        if DEBUG:
            print("done in %0.3fs" % (time() - t0))
            if self.search:
                print(self.clf.best_estimator_)


    def predict(self, x_test):
        #Debug info?
        return self.clf.predict(x_test)


    def predict_prob(self, x_test):
        return self.clf.predict_proba(x_test) 


########################## ....
## CLASSIFIER SELECTION ##    .
##########################    v
clf_menu =              {
                            1 : MLP,
                            2 : SVM 
                        }

def choose_clf(choice, args = None):
    #TODO: double check i'm expanding this right
    if args:
        return clf_menu[choice](**args)
    else:
        return clf_menu[choice]()
