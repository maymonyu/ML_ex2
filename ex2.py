import numpy as np
import scipy.stats


def create_training_set(size, labels):
    training_set = []
    for label in labels:
        X = np.random.normal(2*label, size=size)
        training_set.extend([{'x': x, 'y': label} for x in X])
    return training_set

def p_y_is_i_given_x(x_t, i):
    global b, w, k
    sum_of_grades = np.sum([np.exp(w[j]*x_t + b[j]) for j in range(k)])
    return np.exp(w[i-1]*x_t + b[i-1]) / sum_of_grades


def get_loss_value(w, b):
    global training_set
    loss = 0

    for example in training_set:
        loss -= np.log(p_y_is_i_given_x(example.x, example.y))
    return loss


def update_rule(old_w):

    return old_w


def estimated_y_function(x):
    return p_y_is_i_given_x(x, 1)


def real_y_function(x):


m = 100 #training set size
d = 1
k = 3
w = np.zeros((d, k))
b = np.zeros(k)
training_set = create_training_set(m, range(1, k+1))

while get_loss_value(w, b) > 0.1:
    w -= update_rule(w)
