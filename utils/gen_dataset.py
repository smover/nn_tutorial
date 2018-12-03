# Generate the data for the regression model
# From https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week1/Programming-Assignments/Regularization/reg_utils.py


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import random
import optparse
import logging


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
              1 : [[],[]],
              2 : [[],[]]}
    alias = {0 : 'r', 1 : 'b', 2 : 'g'}

    for i in range(len(X)):
        color_index = Y[i][0]

        (x, y) = X[i]
        color_list = colors[color_index]
        color_list[0].append(x)
        color_list[1].append(y)

    for (color_index, lists) in colors.iteritems():
        plt.scatter(lists[0], lists[1], color=alias[color_index]) 
    plt.show()


def write_dataset(X, out_dir, dataset_prefix):
    """ Write the dataset in the dataset_file as output.

    Follow the csv format of mlpack
    """

    # csv file for data
    # - each row is a sample, each column a dimension
    data_file = os.path.join(out_dir, dataset_prefix + ".csv")
    # tab separated value for labels
    label_file = os.path.join(out_dir, dataset_prefix + "_label.csv")

    with open(data_file, 'w') as data_out:
        data_out.write("label,xcord,ycord\n")

        for i in range(len(X)):
            # csv, comma separated
            d = X[i]
            data_out.write("%d,%f,%f\n" % (d[0],d[1],d[2]))
        data_out.close()

def create_datasets():
    n = 500

    X,Y = load_planar_dataset(1, n)

    assert(len(X) == len(Y))

    data = []
    for i in range(len(X)):
        x,y = X[i]
        data.append([int(Y[i]),x,y])

    data = random.sample( data, len(data) )

    write_dataset(data[:400], ".", "training")
    write_dataset(data[400:], ".", "test")


def main(input_args=None):
    p = optparse.OptionParser()

    p.add_option('-m', '--mode', type='choice',
                 choices= ["gen","plotinput","plotres"],default="gen")

    p.add_option('-d', '--dataset',
                 help="File containing the dataset")

    if (input_args is None):
        input_args = sys.argv[1:]
    opts, args = p.parse_args(input_args)

    if (opts.mode == "gen"):
        create_datasets();
    elif (opts.mode == "plotinput" or opts.mode == "plotres"):
        if (not opts.dataset):
            sys.exit(1)


        is_res = opts.mode == "plotres"
        with open(opts.dataset,"r") as f:
            X_py = []
            Y_py = []
            W_py = []
            first = True
            for line in f.readlines():
                if first:
                    first = False
                else:
                    line = line.strip()
                    data = line.split(",")
                    label = int(data[0])
                    x = data[1]
                    y = data[2]
                    
                    if (is_res):
                        real_label = int(data[3])
                        X_py.append([x,y])

                        if (real_label != label):
                            Y_py.append([2])
                        else:
                            Y_py.append([label])
                    else:
                        X_py.append([x,y])
                        Y_py.append([label])

            plot(np.array(X_py), np.array(Y_py))

            f.close()


if __name__ == '__main__':
    main()

