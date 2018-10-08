import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import handyTools.denoise as denoise


def autocorrelation(y):

    y = np.array(y)

    m = np.mean(y)
    s = np.std(y)
    y2 = np.append(y, y[::-1])  # Pad the signal symmetrically to avoid numeric problems when tau -> n
    n = y.size
    tau = range(1, n + 1)
    ac = np.array(list(
        [
            (1 / ((2 * n - t) * s**2) * np.sum(((y2[:-t] if t > 0 else y2[:]) - m) * (y2[t:] - m)))
            if s > 0 else 1 for t in tau
        ]
    ))
    return ac


def fvar(y, error):
    """Fractional variation"""

    y = np.array(y)
    error = np.array(error)

    m = np.mean(y)
    n = y.size
    if np.sum((y - m)**2) - np.sum(error**2) >= 0:
        return (np.sum((y - m)**2) - np.sum(error**2))**0.5 / (n**0.5 * m)
    else:
        return 0


def trapezoids(t, y):
    # Integral approximation by trapezoid areas

    t = np.array(t)
    y = np.array(y)

    delta_t = np.diff(t)
    area = np.sum(0.5 * delta_t * (y[:-1] + y[1:]))
    return area


def hist(y, npoints=100, norm=True, smooth=True):

    y = np.array(y)

    # Evaluation points
    length = max(y) - min(y)
    delta = length / npoints
    yp = np.linspace(min(y) - delta, max(y) + delta, npoints + 2)

    # cumulative probability distribution
    p_acc = np.array([np.sum(y < i) for i in yp])

    if smooth:
        p_acc = denoise.smooth(p_acc)

    # "Probability" density distribution ('continuous histogram')
    p_diff = 0.5 * (np.diff(p_acc)[:-1] / np.diff(yp)[:-1] + np.diff(p_acc)[1:] / np.diff(yp)[1:])

    # Normalise so the integral equals one
    if norm:
        integ = trapezoids(yp[1:-1], p_diff)
        p_diff /= integ

    # output
    return yp[1:-1], p_diff


def find_local_max(y):

    y = np.array(y)

    i = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1
    return i, y[i]


def find_local_min(y):

    y = np.array(y)

    i = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1
    return i, y[i]


def power_law_fit(t, y):

    """Fit to y = f0 * (t - t0)**(-c)"""

    t = np.array(t)
    x = np.array(y)
    if t.size != y.size:
        raise ValueError('t and y must be the same size')

    ind = ((y > 0) * (t > 0)).nonzero()
    t1 = t[ind]
    y1 = y[ind]

    n = t1.size

    b = (n * np.sum(np.log(t1) * np.log(y1)) - np.sum(np.log(t1)) * np.sum(np.log(y1)))
    b = b / (n * np.sum(np.log(t1)**2) - np.sum(np.log(t1))**2)

    a = (np.sum(np.log(y1)) - b * np.sum(np.log(t1))) / n

    return -b, np.exp(a)


def confusion_matrix(true_class, predicted_class, normalize=False):
    n_classes = int(max(np.max(true_class), np.max(predicted_class))) + 1

    cm = np.zeros((n_classes, n_classes))

    for i, j in zip(true_class, predicted_class):
        cm[i, j] += 1

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


def cpm(true_class:np.ndarray, probs:np.ndarray) -> np.ndarray:
    """
    Conditional Probability matrix
    :param true_class: [n_samples, 1] indices of true classes
    :param probs: [n_samples, n_classes] probaiility for each class
    :return m: numpy.ndarray [n_classes, n_classes] conditional probability matrix
    """

    if true_class.shape[0] != probs.shape[0]:
        raise ValueError('First dimension must be the same')

    n_classes = probs.shape[-1]
    n_samples = probs.shape[0]

    if np.max(true_class) > n_classes:
        raise ValueError('Found index greater than n_classes in true_class')

    m = np.zeros((n_classes, n_classes))

    for i in range(n_samples):
        j = true_class[i]
        m[j,:] += probs[i,:]

    class_ratios = np.array(list([np.sum(true_class == i) for i in range(n_classes)]))
    class_ratios = class_ratios / n_samples
    class_ratios = class_ratios.reshape((n_classes, 1))

    m  = m / class_ratios / n_samples

    return m


def plot_cpm(true_class:np.ndarray, probs:np.ndarray, class_names=None, **kwargs):

    m = cpm(true_class, probs)
    # n_samples = probs.shape[0]
    n = m.shape[0]

    fig = plt.figure()
    plt.imshow(
        m,
        interpolation='nearest',
        **kwargs
    )

    if class_names is not None:
        cls = class_names
    else:
        cls = range(n)

    plt.xticks(range(n), cls)
    plt.yticks(range(n), cls)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    fmt = '{:.2f}'
    thr = 0.5
    for k in range(m.size):
        j = k % n
        i = k // n
        plt.text(j, i, fmt.format(m[i, j]),
                 horizontalalignment='center',
                 color='k' if m[i, j] > thr else 'w')

    plt.colorbar()
    plt.clim(0, 1)

    return m, fig


def plot_confusion_matrix(true_classes, predicted_classes, class_names=None, normalize=False, **kwargs):

    cm = confusion_matrix(true_classes, predicted_classes, normalize=normalize)
    cm_norm = confusion_matrix(true_classes, predicted_classes, normalize=True)
    n = cm.shape[0]

    fig = plt.figure()
    plt.imshow(
        cm_norm,
        interpolation='nearest',
        **kwargs
    )

    if class_names is not None:
        cls = class_names
    else:
        cls = range(n)

    plt.xticks(range(n), cls)
    plt.yticks(range(n), cls)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    if normalize:
        fmt = '{:.1%}'
    else:
        fmt = '{:d}'

    # thr = 0.5 * (np.max(cm) + np.min(cm))
    thr = 0.5

    for k in range(cm.size):
        j = k % n
        i = k // n
        plt.text(j, i, fmt.format(cm[i, j] if normalize else int(cm[i, j])),
                 horizontalalignment='center',
                 color='k' if cm_norm[i, j] > thr else 'w')

    # if normalize:
    #
    #     def fmtfunc(x, pos):
    #         return '{:.0%}'.format(x)
    #
    #     plt.colorbar(format=ticker.FuncFormatter(fmtfunc))
    #     plt.clim(0, 1)
    # else:
    #     plt.colorbar()
    #     plt.clim(vmin=0)

    def fmtfunc(x, pos):
        return '{:.0%}'.format(x)

    plt.colorbar(format=ticker.FuncFormatter(fmtfunc))
    plt.clim(0, 1)

    # plt.tight_layout()

    return cm, fig


def rebin(t, x, err=None, n=100):
    tmax = t[-1]
    tmin = t[0]

    tout = np.linspace(tmin, tmax, n + 1)
    tout = 0.5 * (tout[1:] + tout[:-1])
    delta = 0.5 * min(np.diff(tout))
    xout = tout * 0
    errout = tout * 0

    for i in range(tout.size):
        ind = ((t >= tout[i] - delta) * (t <= tout[i] + delta)).nonzero()
        if len(ind) >= 1:
            xout[i] = np.mean(x[ind])
            if err is not None:
                errout[i] = (np.sum(err[ind]**2))**0.5 / err[ind].size

    if err is not None:
        return tout, xout, errout
    else:
        return tout, xout


def plot_correlation(x:np.ndarray, names=None) -> (np.ndarray, plt.figure):

    m = np.corrcoef(x.T)
    n = m.shape[0]
    fig = plt.figure()
    plt.imshow(m)

    if names is not None:
        plt.xticks(range(n), names, rotation=90)
        plt.yticks(range(n), names)

    fmt = '{:.1f}'
    thr = 0
    for k in range(m.size):
        j = k % n
        i = k // n
        plt.text(j, i, fmt.format(m[i, j]),
                 horizontalalignment='center',
                 color='k' if m[i, j] > thr else 'w')

    plt.colorbar()
    plt.clim(-1, 1)

    # plt.tight_layout()

    return m, fig
