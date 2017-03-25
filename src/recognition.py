from time import time
import matplotlib.pyplot as plt
 
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
import pandas as pd

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

people = fetch_lfw_people(
    './data', min_faces_per_person=70, resize=0.4)

n_samples, h, w = people.images.shape
 
X = people.data
n_features = X.shape[1]
 
y = people.target
target_names = people.target_names
n_classes = target_names.shape[0]

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)
 
# Compute the PCA (eigenfaces) on the face dataset
n_components = 150
 
pca = RandomizedPCA(
    n_components=n_components, whiten=True).fit(X_train)
 
eigenfaces = pca.components_.reshape((n_components, h, w))
X_train_pca = pca.transform(X_train)

#Train a SVM classification model
 
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
         'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
 
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


#Quantitative evaluation of the model quality on the test set
#Validate the data
X_test_pca = pca.transform(X_test)
y_pred = clf.predict(X_test_pca)
 
print(classification_report(
    y_test, y_pred, target_names=target_names))
 
print('Confusion Matrix')
#Make a data frame so we can have some nice labels
cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
df = pd.DataFrame(cm, columns = target_names, index = target_names)
print(df)


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:     %s'%(pred_name, true_name)
 
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
 
plot_gallery(X_test, prediction_titles, h, w, 6, 4)
