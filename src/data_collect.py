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

def train_model(x_train, y_train, args):
    mlp = MLPClassifier(**args)
    mlp.fit(x_train, y_train)
    return mlp

def verify_img(clf, img, label, target_names):
    #just to get rid of unsuppressable stupid sklearn warnings
    img = img.reshape(1, -1)
    y_prob = clf.predict_proba(img)
    y_pred = clf.predict(img)
    print("Predicted: %s\n Confidence: %s\n Actually: %s" % (target_names[y_pred], y_prob[0][y_pred], label))
    if y_prob.max() > 0.85:
        return "MATCH"
    else:
        return "NO_MATCH"

p = Preprocessor()
X_train, X_test, y_train, y_test, target_names = p.get_data()
print("LOADED DATA")
n_classes = target_names.shape[0]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

pca = PCA(n_components=PCA_N_COMPONENTS, whiten=True, svd_solver='randomized').fit(X_train)
eigenfaces = pca.components_.reshape((PCA_N_COMPONENTS, processed_width, processed_height))
x_train = pca.transform(X_train)
x_test = pca.transform(X_test)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## This is grabbing img/target_label_name for single person
new_person_imgs, target_labels = p.load_test_data("data/Nikola")
np_imgs = pca.transform(new_person_imgs)
np_imgs = scaler.transform(np_imgs)

## Testing for NO_MATCH on new person
args = {'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'hidden_layer_sizes':(20,8), 'batch_size': 80}
mlp = train_model(x_train, y_train, args)
results = []
for i in range(np_imgs.shape[0]):
    outcome = verify_img(mlp, np_imgs[i], target_labels[i], target_names)
    results.append(outcome)
correct = results.count("NO_MATCH")
accuracy = correct/float(len(results))
strings = (correct, len(results), accuracy, target_labels[0])
print("Correct: %s, Total: %s, Accuracy: %s, Test subject: %s" % strings)


# Drop into an ipython session to experiment
#from IPython import embed
#embed()

#test_case("hidden_layer_sizes", [(20,i) for i in range(3,200,5)], n_classes, x_train, y_train, x_test, y_test, "hi_iter_batch_(20,3-200)", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 3000})
