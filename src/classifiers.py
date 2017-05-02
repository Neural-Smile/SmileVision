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

class SVM:
    def __init__(self, search=False, param_grid = []):
        self.search = search
        self.param_grid = param_grid
        #TODO: param passing, default for now
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        if search:
            self.clf = GridSearchCV(
                SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)
        else:
            #TODO: passing params
            self.clf = SVC(kernel='rbf', class_weight='balanced', probability=True)


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

    def cache_string(self):
        return "SVM"


class MLP:
    def __init__(self, search=False, param_grid = []):
        self.search = search
        self.param_grid = param_grid
        #TODO: param passing, default for now
        if self.search:
            self.clf = GridSearchCV(
                MLPClassifier(activation="relu", solver="adam"), param_grid)
        else:
            #TODO: passing params
            if SMALL_MODEL:
                self.clf = MLPClassifier(activation="relu", solver="adam", **BEST_SMALL_MODEL)
            else:
                self.clf = MLPClassifier(activation="relu", solver="adam", **BEST_MODEL)


    def train(self, x_train, y_train):
        if DEBUG:
            t0 = time()
        self.clf.fit(x_train, y_train)
        if DEBUG:
            print("done in %0.3fs" % (time() - t0))
            if self.search:
                print(self.clf.best_estimator_)


    def predict(self, x_test):
        return self.clf.predict(x_test)


    def predict_prob(self, x_test):
        return self.clf.predict_proba(x_test)


    def cache_string(self):
        return "MLP"

clf_classes = {
    1 : MLP,
    2 : SVM
}

def choose_clf(choice, args = None):
    #TODO: double check i'm expanding this right
    if args:
        return clf_classes[choice](**args)
    else:
        return clf_classes[choice]()
