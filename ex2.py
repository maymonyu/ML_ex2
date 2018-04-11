import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def create_training_set(size, labels):
    training_set_array = []
    for label in labels:
        X = np.random.normal(2 * label, size=size)
        training_set_array.extend([{'x': x, 'y': label} for x in X])
    return training_set_array


def p_y_is_i_given_x(x_t, i):
    global b, w, k
    sum_of_grades = np.sum([np.exp(w.transpose()[j] * x_t + b[j]) for j in range(k)])
    return np.exp((w.transpose()[i - 1] * x_t)[0] + b[i - 1]) / sum_of_grades


def get_loss_value():
    global training_set
    loss = 0
    for example in training_set:
        loss -= np.log(p_y_is_i_given_x(example['x'], example['y']))
    return loss


def update_rule(old_w):
    return old_w


def train_logistic_regression():
    global w

    while get_loss_value() > 0.1:
        print w
        print get_loss_value()
        w = np.array(np.matrix(w) - np.matrix(update_rule(w)))


def estimated_y_function(x):
    return p_y_is_i_given_x(x, 1)


def conditional_density(x, label):
    return scipy.stats.norm.pdf(x, 2 * label, 1)


def real_y_function(x):
    sum_of_labels_probabilities = 0
    for label in range(1, k+1):
            sum_of_labels_probabilities += conditional_density(x, label)
    return conditional_density(x, 1) / sum_of_labels_probabilities


m = 100  # training set size
d = 1
k = 3
w = np.zeros((d, k))
b = np.zeros(k)
training_set = create_training_set(m, range(1, k + 1))
# train_logistic_regression()

x = np.linspace(0, 10, 100)
y1 = real_y_function(x)
# y2 = estimated_y_function(x)
plt.plot(x, y1)
# plt.plot(x, y2)
plt.show()
