import helpers
import classifiers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from config import *
import os.path
import pickle

# Interface between the server/local helper and the neural network
class Model(object):
    def __init__(self, preprocessor, clf_id=MLP_ID, db=None):
        self.preprocessor = preprocessor
        self.clf_id = clf_id
        self.clf = classifiers.choose_clf(clf_id)
        self.db = db

    def initialize(self):
        cache_path = self.clf.cache_string()
        if LOAD_CACHE and os.path.isfile(cache_path):
            print("Loading model from %s" % cache_path)
            with open(cache_path, 'rb') as f:
                self.clf = pickle.load(f)
        else:
            print("Initializing model")
            if self.db:
                print("Loading data from DB")
                data = self.preprocessor.data_from_db()
            else:
                print("Loading data from sklearn")
                data = self.preprocessor.get_data()

            (X_train, 
            X_test, \
            y_train, \
            y_test, \
            target_names, \
            H, \
            W
            ) = data

            pca = RandomizedPCA(n_components=PCA_N_COMPONENTS, whiten=True).fit(X_train)
            eigenfaces = pca.components_.reshape((PCA_N_COMPONENTS, H, W))
            X_train_processed = pca.transform(X_train)
            self.train(X_train_processed, y_train)
            self.save_model()

    def save_model(self):
        with open(self.clf.cache_string(), 'wb') as f:
            pickle.dump(self.clf, f)


    ## image is formatted and processed correctly by this point
    def verify(self, img):
        helpers.save_image(img)
        y_prob = self.clf.predict_prob(img)
        if self.has_match(y_prob):
            return self.clf.predict(img)
        return NO_MATCH


    def train(self, img, name):
       self.clf.train(img, name) 


    def confidence_title(self, y_pred, y_prob, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        confidence = y_prob[i][y_pred[i]]
        return 'predicted: %s\nconfidence:     %s'%(pred_name, confidence)

    def has_match(self, y_prob):
        return y_prob.max() > 0.7


    def display_confidence(self, x_test, y_pred, target_names):
        y_prob = self.clf.predict_prob(x_test) 
        for i in range(y_prob.shape[0]):
            print(self.confidence_title(y_pred, y_prob, target_names, i))
            if(self.has_match(y_prob[i])):
                print("SHOULD BE NEW PERSON")



    def eval_validation(self, x_test, y_test, target_names):
        y_pred = self.clf.predict(x_test)
        self.display_confidence(x_test, y_pred, target_names)
        print(classification_report(
            y_test, y_pred, target_names=target_names))
        print('Confusion Matrix')
        #Make a data frame so we can have some nice labels
        cm = confusion_matrix(y_test, y_pred, labels=range(target_names.shape[0]))
        df = pd.DataFrame(cm, columns = target_names, index = target_names)
        print(df)

        prediction_titles = [title(y_pred, y_test, target_names, i)
                             for i in range(y_pred.shape[0])]
        return prediction_titles


    def scripted_run(self):
        ## DATA INGEST ##
        (
            X_train,
            X_test, \
            y_train, \
            y_test, \
            target_names, \
            H, \
            W
        ) = self.preprocessor.get_data()

        ## ???: Can't wrap pca construction and take return from func, errors.
        #pca = prepocessor.pca_eigenfaces(PCA_N_COMPONENTS, H, W)
        pca = RandomizedPCA(
            n_components=PCA_N_COMPONENTS, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((PCA_N_COMPONENTS, H, W))
        X_train_processed = pca.transform(X_train)

        self.train(X_train_processed, y_train)

        X_test_pca = pca.transform(X_test)
        prediction_titles = self.eval_validation(X_test_pca, y_test, target_names)
        plot_gallery(X_test, prediction_titles, H, W, 6, 4)


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99,
                        top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        plt.savefig("file.png")

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:     %s'%(pred_name, true_name)


