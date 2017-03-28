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

DELTA_ERROR = 0.01

def has_converged(y):
    return Math.abs(y[-1] - y[-2]) <= DELTA_ERROR and Math.abs(y[-2] - y[-3]) <= DELTA_ERROR

def test_case(param, values, n_classes, x_train, y_train, x_test, y_test, fig_name, other_args={}):
    y1 = []
    y2 = []
    losses = []
    accuracies = []
    times = []
    best_acc = 0
    best_val = None

    ## ITER ##
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
        if len(y1) > 2 and has_converged(y):
            print("Converged")
            break

    x = np.array(range(len(y1)))

    ## STATS ##
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
    save_test_stats(stats)
    save_graph(x, y1, y2, param, values)

    print("Best %s: %s\nAccuracy %s" % (param, best_val, best_acc))

def save_graph(x, y1, y2, param, values, fig_name):
    xTicks = [str(values[i]) for i in range(len(x))]
    fig, ax = plt.subplots()
    plt.xticks(x, xTicks)
    line1, = ax.plot(x, y1, '-', label='Accuracy')
    plt.xticks(x, xTicks)
    line2, = ax.plot(x, y2, '-', label='Loss')
    ax.legend(loc = 'lower right')
    #plt.show()
    fig.autofmt_xdate(rotation=80)
    plt.savefig("data/graphs/" + "%s.png" % (fig_name))


def test_single_var(test_arg, value, n_classes, x_train, y_train, x_test, y_test, other_args = {}):
    all_args = {test_arg : value}
    all_args.update(other_args)
    mlp = MLPClassifier(activation="relu", solver="adam", **all_args)
    t0 = time()
    mlp.fit(x_train, y_train)
    train_time = (time() - t0)
    ## STATS ##
    y_pred = mlp.predict(x_test)
    y_proba = mlp.predict_proba(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # different number of classess 156, 158
    loss = log_loss(y_test, y_proba, labels=[i for i in range(n_classes)])

    return accuracy, loss, train_time

def save_test_stats(stats):
    with open("./data/stats/" + "%s_%s" % (stats['param'], stats['values'][0]), 'a') as f:
        print("##TESTCASE##", file=f)
        for k,v in stats.items():
            print("%s: %s" % (k,v), file=f)

p = Preprocessor()
X_train, X_test, y_train, y_test, target_names = p.get_data()
n_classes = target_names.shape[0]

pca = RandomizedPCA(n_components=200, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((200, 50, 37))
x_train = pca.transform(X_train)
x_test = pca.transform(X_test)

from IPython import embed
embed()

#test_case("hidden_layer_sizes", [(50, 3), (60, 3), (40, 4)], n_classes, x_train, y_train, x_test, y_test)
#test_case("hidden_layer_sizes", [(40, 4), (60, 4), (40, 5), (60, 5), (40, 6), (40, 7)], n_classes, x_train, y_train, x_test, y_test)
#test_case("hidden_layer_sizes", [(40, 7), (40, 10), (40, 12), (20, 20), (40, 20), (80, 20)], n_classes, x_train, y_train, x_test, y_test)
#test_case("hidden_layer_sizes", [(50, 20), (40, 30), (40, 40), (40, 50), (40, 100), (40, 200)], n_classes, x_train, y_train, x_test, y_test)
#test_case("hidden_layer_sizes", [(i, 160) for i in range(30,55)], n_classes, x_train, y_train, x_test, y_test)
#test_case("hidden_layer_sizes", [(i, 160) for i in range(48,55)], n_classes, x_train, y_train, x_test, y_test)
""" ok found (52, 160) to be pretty good, setting model to this """
#test_case("alpha", [0.0001, 0.001, 0.01, 0.1, 1, 10], n_classes, x_train, y_train, x_test, y_test, other_args = {'hidden_layer_sizes': (52, 160)})
#test_case("alpha", [1, 1.1, 1.2, 1.3, 2, 3, 4, 5], n_classes, x_train, y_train, x_test, y_test, other_args = {'hidden_layer_sizes': (52, 160)})
"""alpha 1-1.3 all jump around same accuracy.. idk"""
#test_case("beta_1", [0.7, 0.8, 0.9, 0.92, 0.94], n_classes, x_train, y_train, x_test, y_test, other_args = {'hidden_layer_sizes': (52, 160), 'alpha': 1.1})
#test_case("beta_1", [0.88,0.89,0.9,0.91,0.92], n_classes, x_train, y_train, x_test, y_test, other_args = {'hidden_layer_sizes': (52, 160), 'alpha': 1.1})
"""stick to beta_1 0.9"""
#test_case("hidden_layer_sizes", [(50,160), (45, 160), (55, 160), (50, 130), (50, 190)], n_classes, x_train, y_train, x_test, y_test, other_args = {'hidden_layer_sizes': (52, 160), 'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'adaptive'})
#test_case("hidden_layer_sizes", [(50,130), (50, 100), (50, 50), (50, 120), (50, 125)], n_classes, x_train, y_train, x_test, y_test, other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'adaptive'})
"""adaptive learning isn't doing anything for me, and very sporadic results"""
#test_case("hidden_layer_sizes", [(50, 50),(50, 100),(50, 120), (50, 125),(50,130)], n_classes, x_train, y_train, x_test, y_test, other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant'}) ## with max_iter = 3000, batch_size = 3000
#test_case("hidden_layer_sizes", [(i,1) for i in range(1,100,2)], n_classes, x_train, y_train, x_test, y_test, "hi_iter_batch_(i,1-100)", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 3000}) #0 variance in acc basically
#test_case("hidden_layer_sizes", [(i,3) for i in range(1,500,10)], n_classes, x_train, y_train, x_test, y_test, "hi_iter_batch_(i,1-500)", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 3000})
test_case("hidden_layer_sizes", [(20,i) for i in range(3,200,5)], n_classes, x_train, y_train, x_test, y_test, "hi_iter_batch_(20,3-200)", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 3000})
