print(__doc__)
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2}]

labels = ["constant learning-rate", "with momentum",
          "with Nesterov's momentum"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'}]


def plot_on_dataset(X, y):
    # for each dataset, plot learning for each learning strategy
    X = X.tocsr()
    X = X.todense()
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    max_iter=400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    plt.figure(0)
    for mlp, label, args in zip(mlps, labels, plot_args):
            plt.plot(mlp.loss_curve_, label=label, **args)
    plt.legend(labels, loc="upper right")
    plt.savefig('plots/mlp_result.png', format='png')

# load data
X,y = sk.datasets.load_svmlight_file('test_data')

plot_on_dataset(X, y) 

