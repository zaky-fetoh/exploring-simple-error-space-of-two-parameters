import numpy as np
import matplotlib.pyplot as plt


def get_art_dt(beta=[5, 9], num=100):
    epse = np.random.normal(0, .5, size=(num,))
    x = np.linspace(0, 1, num)
    y = np.sum([bt * (x ** i) for i, bt in enumerate(beta)], axis=0)
    return x, (y + epse)


def model(X, w, b):
    return w * X + b


def mse(Y_act, Y_pre):
    return np.mean((Y_act - Y_pre) ** 2)


def mae(Y_act, Y_pre):
    return np.mean(np.abs(Y_act - Y_pre))


def compute_param_error(w, b, X, Y, err_list=[mae]):
    Y_pre = model(X, w, b)
    return [e(Y, Y_pre) for e in err_list]


def get_error_map(X, Y, num=1000, param_low=-20, param_high=20, err_list=[mse]):
    er_map = np.zeros((len(err_list), num, num))
    w = np.linspace(param_low, param_high, num)
    b = np.linspace(param_low, param_high, num)
    for i in range(num):
        for j in range(num):
            er_map[:, i, j] = compute_param_error(w[i], b[j], X, Y, err_list=err_list)
    return er_map


Normalize = lambda X: (X - X.mean()) / X.std()

X, Y = get_art_dt([.5, .5])
X = Normalize(X)

err_list = [mae, mse]
im = get_error_map(X, Y, param_low=-5, param_high=5, err_list=err_list)
for i in range(len(err_list)):
    plt.subplot(1, len(err_list), i + 1)
    plt.imshow(im[i], cmap='hot')
    plt.axis('off')
    plt.colorbar(orientation='horizontal')
