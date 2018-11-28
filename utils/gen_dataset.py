# Generate the data for the regression model
# From https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week1/Programming-Assignments/Regularization/reg_utils.py


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_planar_dataset(seed, m=400):
    """
    Generates a dataset of points in the 2d space.

    seed is the random seed, m is the number of examples


    Returns X,Y, where:

    X is a matrix m x D where...

    Y is a matrix of m X 1 labels that is 0 if the point is red,
    1 if the point is blue.
    """

    np.random.seed(seed)
    
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        val = np.c_[r*np.sin(t), r*np.cos(t)]
        X[ix] = val
        Y[ix] = j

    return X, Y


def plot(X,Y):
    colors = {0 : [[],[]],
              1 : [[],[]]}
    alias = {0 : 'r', 1 : 'b'}


    for i in range(len(X)):
        color_index = Y[i][0]

        (x, y) = X[i]
        color_list = colors[color_index]
        color_list[0].append(x)
        color_list[1].append(y)

    for (color_index, lists) in colors.iteritems():
        plt.scatter(lists[0], lists[1], color=alias[color_index]) 
    plt.show()


def write_dataset(X, Y, out_dir, dataset_prefix):
    """ Write the dataset in the dataset_file as output.

    Follow the csv format of mlpack
    """

    # csv file for data
    # - each row is a sample, each column a dimension
    data_file = os.path.join(out_dir, dataset_prefix + ".csv")
    # tab separated value for labels
    label_file = os.path.join(out_dir, dataset_prefix + "_label.tsv")

    print len(X)

    assert(len(X) == len(Y))

    with open(data_file, 'w') as data_out:
        with open(label_file, 'w') as label_out:
            for i in range(len(X)):
                assert len(X[i]) == 2
                x,y = X[i]

                data_out.write("%f %f\n" % (x,y))
                label_out.write("%f " % Y[i])
            label_out.close()
        data_out.close()

def create_datasets():
    n = 500

    X,Y = load_planar_dataset(1, n)

    # plot(X, Y)

    write_dataset(X[:400], Y[:400], ".", "training")
    write_dataset(X[400:], Y[400:], ".", "validation")

create_datasets();

