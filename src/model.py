import helpers
import classifiers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from config import *
import numpy as np
import pickle
import os

# Interface between the server/local helper and the neural network
class Model(object):
    def __init__(self, preprocessor, clf_id=MLP_ID, db=None):
        self.preprocessor = preprocessor
        self.clf_id = clf_id
        self.clf = classifiers.choose_clf(clf_id)
        self.db = db
        self.initialized = False
        self.target_names = None
        self.pca = None

    def cache_path(self):
        filename = self.clf.cache_string()
        return "./data/models/{}".format(filename)

    def get_data(self):
        if self.db:
            print("Using training data from DB")
            return self.preprocessor.data_from_db()
        else:
            return self.preprocessor.get_data()

    def get_pca_for(self, X_train):
        if self.pca is None:
            self.pca = PCA(n_components=PCA_N_COMPONENTS, whiten=True, svd_solver='randomized').fit(X_train)
        return self.pca

    def get_face_embeddings(self, faces):
        pca = self.get_pca_for(faces)
        n_components = min(PCA_N_COMPONENTS, pca.components_.shape[0])
        if n_components != PCA_N_COMPONENTS:
            print("WARNING: COULD NOT DECOMPOSE INTO {} COMPONENTS. USING {} INSTEAD.".format(PCA_N_COMPONENTS, n_components))
        pca.components_.reshape((n_components, processed_height, processed_width))
        embeddings = pca.transform(faces)
        return embeddings

    def initialize_from_cache(self):
        print("Loading cached model from %s" % self.cache_path())
        self.load_model(self.cache_path())

    def initialize_from_data(self, data):
        (X_train, X_test, y_train, y_test, target_names) = data
        self.target_names = target_names
        X_train_embeddings = self.get_face_embeddings(X_train)
        self.train(X_train_embeddings, y_train)
        self.save_model(self.cache_path())

    def initialize(self, force=False):
        if self.initialized and not force:
            return

        print("Initializing model")
        if not force and USE_CACHED_MODEL and self.cached_files_present():
            self.initialize_from_cache()
        else:
            data = self.get_data()
            self.initialize_from_data(data)
        self.initialized = True

    def cached_files_present(self, basename=None):
        if basename is None:
            basename = self.cache_path()

        model = "{}_model.pkl".format(basename)
        targets = "{}_figs.pkl".format(basename)
        pca = "{}_pca.pkl".format(basename)

        res = os.path.exists(model)
        res = res and os.path.exists(targets)
        res = res and os.path.exists(pca)

        return res

    def save_model(self, basename=None):
        if basename is None:
            basename = self.cache_path()

        model = "{}_model.pkl".format(basename)
        targets = "{}_figs.pkl".format(basename)
        pca = "{}_pca.pkl".format(basename)

        with open(model, 'wb') as f:
            pickle.dump(self.clf, f)
        with open(targets, 'wb') as f:
            pickle.dump(self.target_names, f)
        with open(pca, 'wb') as f:
            pickle.dump(self.pca, f)

    def load_model(self, basename=None):
        if basename is None:
            basename = self.cache_path()

        model = "{}_model.pkl".format(basename)
        targets = "{}_figs.pkl".format(basename)
        pca = "{}_pca.pkl".format(basename)

        with open(model, 'rb') as f:
            self.clf = pickle.load(f)
        with open(targets, 'rb') as f:
            self.target_names = pickle.load(f)
        with open(pca, 'rb') as f:
            self.pca = pickle.load(f)

    def verify(self, embedding):
        y_prob = self.clf.predict_prob(embedding)
        y_pred = self.clf.predict(embedding)
        print("Predicted: %s\n Confidence: %s" % (self.target_names[y_pred], y_prob[0][y_pred]))
        if self.has_match(y_prob):
            return self.target_names[y_pred][0]
        return NO_MATCH

    def train(self, embeddings, labels):
       self.clf.train(embeddings, labels)

    def train_new_identity(self, identity, imgs):
        faces = []
        processed = np.array(map(lambda img: self.preprocessor.process(img), imgs))
        for p in processed:
            if len(p) == 1:
                faces.append(p[0])

        embeddings = self.get_face_embeddings(faces)
        labels = [len(self.target_names) for _ in range(len(embeddings))]
        self.target_names = np.append(self.target_names, [identity])

        self.train(embeddings, labels)
        self.save_model()

        return True

    def confidence_title(self, y_pred, y_prob, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        confidence = y_prob[i][y_pred[i]]
        return 'predicted: %s\nconfidence:     %s'%(pred_name, confidence)

    def has_match(self, y_prob):
        return y_prob.max() > 0.85

    def display_confidence(self, x_test, y_pred, target_names):
        y_prob = self.clf.predict_prob(x_test)
        for i in range(y_prob.shape[0]):
            print(self.confidence_title(y_pred, y_prob, target_names, i))

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

    def initialize_and_test(self):
        self.initialize()
        if DEBUG:
            (X_train, X_test, y_train, y_test, target_names) = self.get_data()
            X_test_pca = self.get_face_embeddings(X_test)
            prediction_titles = self.eval_validation(X_test_pca, y_test, target_names)
            plot_gallery(X_test, prediction_titles, processed_height, processed_width, 6, 4)


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
