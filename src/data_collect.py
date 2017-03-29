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
    max_y2 = float(max(y2))
    y2 = list(map(lambda x: x/max_y2, y2))
    print("Best %s: %s\nAccuracy %s" % (param, best_val, best_acc))
    save_graph(x, y1, y2, param, values, data_name)


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
    max_y2 = float(max(y2))
    y2 = list(map(lambda x: x/max_y2, y2))
    save_graph(x, y1, y2, param, values, data_name)
    print("Best %s: %s\nAccuracy %s" % (param, best_val, best_acc))
    return best_val


def test_no_match(x_train, y_train, x_test, param, values, data_name, other_args = {}):
    accs = []
    best_acc = 0
    best_val = None
    for val in values:
        all_args = {param: val}
        all_args.update(other_args)
        acc = _test_no_match(x_train, y_train, x_test, all_args)
        print("Param: %s, value: %s, accuracy: %s" % (param, val, acc))
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_val = val

    y1 = np.array(accs)
    x = np.array(range(len(y1)))
    y2 = [1-a for a in accs]
    print("Best %s: %s\nAccuracy %s" % (param, best_val, best_acc))
    save_graph(x, y1, y2, param, values, data_name)
def _test_no_match(x_train, y_train, x_test, args):
    mlp = train_model(x_train, y_train, args)
    ids = [verify_img(mlp,x) for x in x_test]
    correct = ids.count("NO_MATCH")
    accuracy = correct/float(len(ids))
    return accuracy


def save_graph(x, y1, y2, param, values, data_name):
    xTicks = [str(values[i]) for i in range(len(x))]
    fig, ax = plt.subplots()
    plt.xticks(x, xTicks)
    line1, = ax.plot(x, y1, '-', label='Accuracy')
    plt.xticks(x, xTicks)
    line2, = ax.plot(x, y2, '-', label='Loss')
    ax.legend(loc = 'lower right')
    ax.set_xlabel(param)
    fig.autofmt_xdate(rotation=80)
    #plt.show()
    plt.savefig("data/graphs/" + "%s.png" % (data_name))


def test_single_var(test_arg, value, n_classes, x_train, y_train, x_test, y_test, other_args = {}):
    all_args = {test_arg : value}
    all_args.update(other_args)
    mlp = MLPClassifier(**all_args)
    t0 = time()
    mlp.fit(x_train, y_train)
    train_time = (time() - t0)
    y_pred = mlp.predict(x_test)
    y_proba = mlp.predict_proba(x_test)
    accuracy = accuracy_score(y_test, y_pred)
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


def verify_img(clf, img, label=None, target_names=None):
    #just to get rid of unsuppressable stupid sklearn warnings
    img = img.reshape(1, -1)
    y_prob = clf.predict_proba(img)
    y_pred = clf.predict(img)
    if label and target_names:
        print("Predicted: %s\n Confidence: %s\n Actually: %s" % (target_names[y_pred], y_prob[0][y_pred], label))
    if y_prob.max() > 0.85:
        return "MATCH"
    else:
        return "NO_MATCH"


p = Preprocessor()
x_train, x_test, y_train, y_test, target_names = p.get_data()
print("LOADED DATA")
n_classes = target_names.shape[0]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
PCA_N = 120
#pca = PCA(n_components=PCA_N_COMPONENTS, whiten=True, svd_solver='randomized').fit(X_train)
pca = PCA(n_components=PCA_N, whiten=True, svd_solver='randomized').fit(x_train)
#n_components = min(PCA_N_COMPONENTS, pca.components_.shape[0])
n_components = min(PCA_N, pca.components_.shape[0])
eigenfaces = pca.components_.reshape((n_components, processed_height, processed_width))
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
#scaler = StandardScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

print("N Classes: ", n_classes)
print("N Samples: ", x_train.shape)
print("N Test: ", x_test.shape)
print("PCA_N: %s, N_COMP: %s" % (PCA_N,n_components))

## This is grabbing img/target_label_name for single person
#new_person_imgs, target_labels = p.load_test_data("data/Nikola")
#np_imgs = pca.transform(new_person_imgs)
#np_imgs = scaler.transform(np_imgs)

#test_no_match(x_train, y_train, np_imgs, "hidden_layer_sizes", [(20,i) for i in range (1,30,2)], "small_(20,1-30)", {'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'batch_size': 80})
#test_no_match(x_train, y_train, np_imgs, "hidden_layer_sizes", [(i,2) for i in range (3,60,4)], "small_(3-60,2)", {'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'batch_size': 80}) #(7,2), (11,2), (47,2) all 100%
#test_no_match(x_train, y_train, np_imgs, "hidden_layer_sizes", [(i,1) for i in range (1,30)], "small_(1-30,1)", {'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'batch_size': 80})
#test_no_match(x_train, y_train, np_imgs, "hidden_layer_sizes", [(i,1) for i in range (1,3)], "small_(1-30,1)", {'alpha':1.1, 'beta_1':0.9, 'learning_rate':'constant', 'max_iter':3000, 'batch_size': 80})


# Drop into an ipython session to experiment
#from IPython import embed
#embed()

#just to keep template for quick test.. dont be upset
#best_val = test_case("learning_rate_init", [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 2], n_classes, x_train, y_train, x_test, y_test, "POS_mlplearnrate[0.0001-2]", other_args = {'hidden_layer_sizes':(20,8), 'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 80}) #0.003
#best_val = test_case("learning_rate_init", [0.0009, 0.001, 0.002, 0.003], n_classes, x_train, y_train, x_test, y_test, "POS_mlplearnrate[0.0009-0.003]", other_args = {'hidden_layer_sizes':(20,8), 'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 80}) #0.003

best_val = test_case("hidden_layer_sizes", [(100,i) for i in range(1,30)], n_classes, x_train, y_train, x_test, y_test, "POS_mlp(100, 1-30", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 80, 'learning_rate_init':0.003}) # 10,14 and then leveled out

#best_val = test_case("hidden_layer_sizes", [(i,5) for i in range(1,100,2)], n_classes, x_train, y_train, x_test, y_test, "POS_mlp(1-100, 5)", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 80, 'learning_rate_init':0.003}) # 10,14 and then leveled out
#test_case("hidden_layer_sizes", [(i,best_val) for i in range(2,60,2)], n_classes, x_train, y_train, x_test, y_test, "POS_mlp(2-60, best)", other_args = {'alpha': 1.1, 'beta_1': 0.9, 'learning_rate': 'constant', 'max_iter' : 3000, 'batch_size' : 80})
