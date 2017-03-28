from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, log_loss
from classifiers import *
from model import *
from config import *
import numpy as np
from preprocessor import Preprocessor
from sklearn.preprocessing import StandardScaler

DELTA_ERROR = 0.01

def has_converged(y):
    return Math.abs(y[-1] - y[-2]) <= DELTA_ERROR and Math.abs(y[-2] - y[-3]) <= DELTA_ERROR

def test_case(param, values, n_classes, x_train, y_train, x_test, y_test, data_name, other_args={}):
    y1 = []
    y2 = []
    losses = []
    accuracies = []
    times = []
    best_acc = 0
    best_val = None

    try:
        for val in values:
            accuracy, loss, time = test_single_var(param, val, n_classes, x_train, y_train, x_test, y_test, other_args)
            print("Param: %s, value: %s, accuracy: %s, loss: %s" % (param, val, accuracy, loss))
            accuracies.append(accuracy)
            losses.append(loss)
            times.append(time)
            if accuracy > best_acc:
                best_val = val
                best_acc = accuracy
            y1.append(accuracy)
            y2.append(loss)
    except KeyboardInterrupt:
        pass

    x = np.array(range(len(y1)))

    stats = {'n_samples'   : x_train.shape[0],
             'n_classes'   : n_classes,
             'n_test'      : x_test.shape[0],
             'param'       : param,
             'values'      : values,
             'accuracies'  : accuracies,
             'losses'      : losses,
             'tain_times'  : times,
             'best_acc'    : best_acc,
             'best_val'    : best_val,
             'other_args'  : other_args}

    save_test_stats(stats, data_name)
    max_y1 = float(max(y1))
    max_y2 = float(max(y2))
    y1 = list(map(lambda x: x/max_y1, y1))
    y2 = list(map(lambda x: x/max_y2, y2))
    save_graph(x, y1, y2, param, values, data_name)

    print("Best %s: %s\nAccuracy %s" % (param, best_val, best_acc))

def save_graph(x, y1, y2, param, values, data_name):
    xTicks = [str(values[i]) for i in range(len(x))]
    fig, ax = plt.subplots()
    plt.xticks(x, xTicks)
    line1, = ax.plot(x, y1, '-', label='Accuracy')
    plt.xticks(x, xTicks)
    line2, = ax.plot(x, y2, '-', label='Loss')
    ax.legend(loc = 'lower right')
    ax.set_xlabel(param)
    plt.show()
    fig.autofmt_xdate(rotation=80)
    plt.savefig("data/graphs/" + "%s.png" % (data_name))


def test_single_var(test_arg, value, n_classes, x_train, y_train, x_test, y_test, other_args = {}):
    all_args = {'activation':"relu", 'solver':"adam", 'max_iter':100, 'batch_size':30}
    all_args = {test_arg : value}
    all_args.update(other_args)
    mlp = MLPClassifier(**all_args)
    t0 = time()
    mlp.fit(x_train, y_train)
    train_time = (time() - t0)

    y_pred = mlp.predict(x_test)
    y_proba = mlp.predict_proba(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # different number of classess 156, 158
    loss = log_loss(y_test, y_proba, labels=[i for i in range(n_classes)])

    return accuracy, loss, train_time

def save_test_stats(stats, data_name):
    with open("./data/stats/" + "%s" % data_name, 'a') as f:
        print("##TESTCASE##", file=f)
        for k,v in stats.items():
            print("%s: %s" % (k,v), file=f)

p = Preprocessor()
X_train, X_test, y_train, y_test, target_names = p.get_data()
print("LOADED DATA")
n_classes = target_names.shape[0]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

pca = RandomizedPCA(n_components=200, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((200, 50, 37))
x_train = pca.transform(X_train)
x_test = pca.transform(X_test)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Drop into an ipython session to experiment
from IPython import embed
embed()
