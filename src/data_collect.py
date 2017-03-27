import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, log_loss
from classifiers import *
from model import *
from config import *
import numpy as np

def test_case(param, values, x_train, y_train, x_test, y_test):
    stats = []
    y1 = []
    y2 = []
    xTicks = [str(val) for val in values]
    x = np.array(range(len(values)))
    for val in values:
        accuracy, loss, time = test_single_var(param, val, x_train, y_train, x_test, y_test)
        y1.append(accuracy)
        y2.append(loss)
        stats.append({ 'param'     : param,
                        'value'    : val, 
                        'accuracy' : accuracy,
                        'loss'     : loss,
                        'tain_time'  : time })
    
    fig, ax = plt.subplots()
    plt.xticks(x, xTicks)
    line1, = ax.plot(x, y1, '-', label='Accuracy')
    plt.xticks(x, xTicks)
    line2, = ax.plot(x, y2, '-', label='Loss')
    ax.legend(loc = 'lower right')
    #plt.show()
    plt.savefig("graphs/" + "%s_%s" % (param, values[0]))

def test_single_var(arg, value, x_train, y_train, x_test, y_test):
    mlp = MLPClassifier(activation="relu", solver="adam", **{arg : value})
    t0 = time()
    mlp.fit(x_train, y_train)
    train_time = (time() - t0)
    ## STATS ##
    y_pred = mlp.predict(x_test)
    y_proba = mlp.predict_proba(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    return accuracy, loss, train_time

people = fetch_lfw_people(
    './data', min_faces_per_person=config['min_faces'], resize=config['aspect_ratio'])
n_samples, H, W = people.images.shape
X = people.data 
y = people.target
target_names = people.target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

pca = RandomizedPCA(n_components=PCA_N_COMPONENTS, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((PCA_N_COMPONENTS, H, W))
x_train = pca.transform(X_train)
x_test = pca.transform(X_test)

test_case("hidden_layer_sizes", [(50, 3), (60, 3), (40, 4)], x_train, y_train, x_test, y_test)


