from time import time
import matplotlib.pyplot as plt
 
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

PCA_N_COMPONENTS = 150

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


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:     %s'%(pred_name, true_name)


def SVM_training(x_train, y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
     
    clf = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(x_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf 


def MLP_training(x_train, y_train):
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'hidden_layer_sizes' : [(50, 2), (100, 2), (50, 3), (100, 3)], 'alpha' : [0.001, 0.1, 1, 2, 10]}
    clf = GridSearchCV(
            MLPClassifier(activation="relu", solver="adam"), param_grid)
    clf.fit(x_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf


def get_data():
    people = fetch_lfw_people(
        './data', min_faces_per_person=70, resize=0.4)
    n_samples, h, w = people.images.shape
    X = people.data
    n_features = X.shape[1]
    y = people.target
    target_names = people.target_names
    n_classes = target_names.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test, target_names, h, w


## PREPROCESS : reduce dimensions / feature scaling
def pca_eigenfaces(X_train, h, w):
    pca = RandomizedPCA(
        n_components=150, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((150, h, w))
    return pca

 
def eval_validation(clf, X_test, y_test, target_names):
    y_pred = clf.predict(X_test)
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


def main():
    ## DATA INGEST ##
    ## TODO: build a state struct, config defaults in main
    (
        X_train, 
        X_test, \
        y_train, \
        y_test, \
        target_names, \
        H, \
        W
    ) = get_data()

    ## ???: Can't wrap pca construction and take return from func, errors.
    #pca = pca_eigenfaces(PCA_N_COMPONENTS, H, W)
    pca = RandomizedPCA(
        n_components=150, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((150, H, W))

    X_train_pca = pca.transform(X_train)

    ##########################
    ## CLASSIFIER SELECTION ##
    ##########################

    #clf = SVM_training(X_train_pca, y_train)
    clf = MLP_training(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)
    prediction_titles = eval_validation(clf, X_test_pca, y_test, target_names)
    plot_gallery(X_test, prediction_titles, H, W, 6, 4)

if __name__ == "__main__":
    main()
